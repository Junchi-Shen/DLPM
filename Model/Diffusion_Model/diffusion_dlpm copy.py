# DLPM扩散模型适配器
# 与现有的diffusion_with_condition接口兼容

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm.auto import tqdm
from functools import partial

from .Utils import exists, default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from .DLPM.generative_levy_process import GenerativeLevyProcess
from .DLPM.dlpm_core import ModelMeanType, ModelVarType


class DLPMDiffusion1D(nn.Module):
    """
    DLPM扩散模型，与GaussianDiffusion1D接口兼容
    """
    def __init__(
        self,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        alpha=1.7,  # DLPM参数：Lévy稳定分布的alpha参数 (1 < alpha <= 2)
        objective='pred_noise',
        beta_schedule='cosine',  # 保留用于兼容性，实际使用DLPM的调度
        ddim_sampling_eta=0.,
        auto_normalize=True,
        channels=None,
        self_condition=None,
        model,  # Unet1D模型
        channel_first=True,
        condition_network: nn.Module = None,
        isotropic=True,  # DLPM参数：是否各向同性
        rescale_timesteps=True,  # DLPM参数：是否重新缩放时间步
        scale='scale_preserving',  # DLPM参数：调度类型
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.channels = default(channels, lambda: self.model.channels)
        self.self_condition = default(self_condition, getattr(self.model, 'self_condition', False))

        self.channel_first = channel_first
        self.seq_index = -2 if not channel_first else -1

        self.seq_length = seq_length
        self.objective = objective
        self.learnable_alpha = nn.Parameter(torch.tensor(float(alpha)))  # Lévy稳定分布的alpha参数

        self.condition_network = condition_network
        self.has_condition_network = exists(condition_network)

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise, pred_x0, or pred_v'

        # 初始化DLPM的GenerativeLevyProcess
        device = next(model.parameters()).device
        self.generative_process = GenerativeLevyProcess(
            alpha=alpha,
            device=device,
            reverse_steps=timesteps,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED,
            time_spacing='linear',
            rescale_timesteps=rescale_timesteps,
            isotropic=isotropic,
            scale=scale,
            input_scaling=False,
        )

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.num_timesteps = timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        
        # 默认使用DDIM采样（DLIM方法），可以大幅减少采样步数
        if self.sampling_timesteps < timesteps:
            print(f"✅ 使用DDIM采样（DLIM方法），采样步数: {self.sampling_timesteps} (训练步数: {timesteps})")

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = identity
        self.train_num_steps = kwargs.get('train_num_steps', 100000)
        self.warmup_steps = int(self.train_num_steps * kwargs.get('warmup_ratio', 0.25))
        self.ema_beta = kwargs.get('ema_beta', 0.99)
        for name in ['global_vol', 'heavy_tail', 'vol_clustering',"jump_limit","jump_density","quantile"]:
            self.register_buffer(f'ema_{name}', torch.tensor(0.0))
        self.auto_normalize = auto_normalize # 存一下这个布尔值
        if auto_normalize:
            self.normalize = normalize_to_neg_one_to_one
            # 核心：这里不能是 identity，必须是对抗 normalize 的逆运算
            self.unnormalize = unnormalize_to_zero_to_one 
        else:
            self.normalize = identity
            self.unnormalize = identity
    def model_predictions(self, x, t, self_cond=None, cond_input=None, clip_x_start=False, 
                         model_forward_kwargs: dict = dict()):
        """模型预测"""
        model_output = self.model(x, t, cond_input=cond_input, **model_forward_kwargs)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.generative_process.dlpm.predict_xstart(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            pred_noise = self.generative_process.dlpm.predict_eps(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.generative_process.dlpm.predict_eps(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.generative_process.dlpm.predict_xstart(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.generative_process.dlpm.predict_eps(x, t, x_start)

        return pred_noise, x_start, None

    def _get_condition(self, cond_input):
        if not exists(cond_input):
            return None
        if self.has_condition_network:
            return self.condition_network(cond_input)
        return cond_input
    
    def p_sample(self, x, t, self_cond=None, cond_input=None, model_forward_kwargs: dict = dict()):
        """
        单步采样：增强了自条件处理与数值稳定性
        """
        # 1. 统一获取条件编码 (避免重复编码 Bug)
        processed_cond = self._get_condition(cond_input)
    
        # 2. 构造传给 UNet 的参数
        model_kwargs = model_forward_kwargs.copy()
        model_kwargs['cond_input'] = processed_cond
    
        # 只有当 self_condition 开启时，才把上一步预测的 x_start 喂给模型
        if self.self_condition:
            model_kwargs['y_self_cond'] = self_cond 

        # 3. 确保 DLPM 内部状态对齐
        if self.generative_process.dlpm.A is None or self.generative_process.dlpm.Sigmas is None:
            self.generative_process.dlpm.sample_A(x.shape, self.num_timesteps)
            self.generative_process.dlpm.compute_Sigmas()

        # 4. 执行 DLPM 的核心推理
        # 注意：这里我们调用 generative_process 的 p_sample
        out = self.generative_process.p_sample(
            self.model, 
            x, 
            t,
            clip_denoised=False, # 金融数据不建议在此时硬 clip，除非 Std=1 对齐得很好
            model_kwargs=model_kwargs,)

        # 5. 核心修复：取出预测的 pred_xstart (即 x0)
        # 这个值将作为下一轮采样循环中的 self_cond 传进去
        pred_x_start = out.get("pred_xstart", None)
    
        # 6. 数值安全防护：如果预测的 x_start 已经 inf/nan 了，及时止损
        if pred_x_start is not None:
            pred_x_start = torch.nan_to_num(pred_x_start, nan=0.0)
            # 这里的裁剪范围应与你训练时的 clamp 对齐 (比如 -2.5, 2.5)
            pred_x_start = torch.clamp(pred_x_start, min=-2.5, max=2.5)

        return out["sample"], pred_x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise=False, model_forward_kwargs: dict = dict()):
        device = self.generative_process.device
        
        # 统一编码条件：全过程只编这一次
        processed_cond = self._get_condition(model_forward_kwargs.get('cond_input'))
        
        # 修复 Bug 2: 统一噪声源，只保留一个循环
        noise0 = self.generative_process.dlpm.barsigmas[-1] * \
                 self.generative_process.dlpm.gen_eps.generate(size=shape)
        img = noise0
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop'):
            self_cond = x_start if self.self_condition else None
            mask = model_forward_kwargs.get('mask')
            
            # 每步都强制对齐 Mask 噪声
            if exists(mask):
                img = img * mask + noise0 * (1 - mask)
            
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 调用 p_sample，传递编码后的 processed_cond
            img, x_start = self.p_sample(
                img, t_tensor, 
                self_cond=self_cond, 
                cond_input=processed_cond, 
                model_forward_kwargs=model_forward_kwargs
            )

            # 专家建议的软约束逻辑：直接集成在主循环末尾，不另开循环
            if t < 100:
                img = torch.clamp(img, min=-2.5, max=2.5)

        img = self.unnormalize(img)
        if exists(mask):
            img = img * mask + noise0 * (1 - mask)
        return (img, noise0) if return_noise else img
   

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True, model_forward_kwargs: dict = dict(), return_noise=False):
        # 修复 Bug 3: DDIM 也要固定同一宇宙的噪声
        noise0 = self.generative_process.dlpm.barsigmas[-1] * \
                 self.generative_process.dlpm.gen_eps.generate(size=shape)
        
        processed_cond = self._get_condition(model_forward_kwargs.get('cond_input'))
        
        ddim_model_kwargs = model_forward_kwargs.copy()
        ddim_model_kwargs['cond_input'] = processed_cond

        # 如果你的 ddim_sample_loop 支持传入初始 noise，务必传入 noise0
        img = self.generative_process.ddim_sample_loop(
            self.model,
            shape=shape,
            noise=noise0, # 关键：让 DDIM 从这个噪声开始走
            clip_denoised=clip_denoised,
            model_kwargs=ddim_model_kwargs,
            eta=self.ddim_sampling_eta,
            sampling_timesteps=self.sampling_timesteps
        )
        
        img = self.unnormalize(img)
        mask = model_forward_kwargs.get('mask')
        if exists(mask):
            img = img * mask + noise0 * (1 - mask)
        return (img, noise0) if return_noise else img

    @torch.no_grad()
    def sample(self, batch_size=16, cond_input=None, mask=None, return_noise=False, 
              model_forward_kwargs: dict = dict()):
        """采样接口"""
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        shape = (batch_size, channels, seq_length) if self.channel_first else (batch_size, seq_length, channels)
        
        if exists(cond_input):
            model_forward_kwargs['cond_input'] = cond_input
        if exists(mask):
            model_forward_kwargs['mask'] = mask
            
        return sample_fn(shape, return_noise=return_noise, model_forward_kwargs=model_forward_kwargs)

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        return self.generative_process.q_sample(x_start, t, noise)

    def forward(self, img, cond_input=None, mask=None, global_step=0, *args, **kwargs):
        """
        前向传播方法：负责【条件的唯一一次编码】并【返回最终损失】
        """
        b, c, n = img.shape
        device = img.device
        
        # 1. 唯一的一次条件预处理入口 (这里面已经处理了 condition_network)
        # 得到的 processed_cond_input 直接就是给 U-Net 用的特征张量
        processed_cond_input = self._get_condition(cond_input)
        
        # 2. 随机采样扩散时间步
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # 3. 调用 p_losses 计算损失
        # 关键点：直接把上面处理好的 processed_cond_input 传下去，
        # 这样 p_losses 内部就不用再调用 condition_network 了
        loss = self.p_losses(
            x_start=img,
            t=t,
            cond_input=processed_cond_input, 
            mask=mask,
            global_step=global_step,
            **kwargs
        )
        
        # 4. 最终返回 loss 给 trainer 
        return loss

    def _masked_statistics(self, x, mask, eps=1e-6):
        if mask is None:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            return mean, torch.sqrt(var + eps)
        if x.dim() > mask.dim():
            mask = mask.unsqueeze(1).expand_as(x)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        masked_x = x * mask
        mean = masked_x.sum(dim=-1, keepdim=True) / count
        var = ((masked_x - mean)**2 * mask).sum(dim=-1, keepdim=True) / count
        return mean, torch.sqrt(var + eps)

    def _masked_kurtosis(self, x, mask, eps=1e-6):
        mean, std = self._masked_statistics(x, mask, eps)
        if mask is None:
            z = (x - mean) / (std + eps)
            k = torch.mean(z**4, dim=-1) - 3.0
            return k.clamp(min=0.0)
        z = (x - mean) / (std + eps)
        z_masked = z * mask
        count = mask.sum(dim=-1).clamp(min=1)
        z4_sum = (z_masked**4).sum(dim=-1)
        k = (z4_sum / count) - 3.0
        return k.clamp(min=0.0)

    def _get_annealed_weights(self, global_step):
        warmup_steps = getattr(self, 'warmup_steps', 25000)
        global_scale = min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0
        return {
            'vol_clustering': 0.5 * global_scale,
            'heavy_tail': 1 * global_scale,
            'global_vol': 1 * global_scale,
            "jump_limit": 3 * global_scale,
            'jump_density': 2 * global_scale,
            'quantile': 1 * global_scale
        }
    



    def p_losses(self, x_start, t, cond_input=None, noise=None, mask=None, global_step=0, **kwargs):
        """
        计算训练损失：
        1. 修复了条件重复编码问题 (Bug 1)
        2. 引入 SNR Gate 解决数值爆炸 (防爆机制)
        3. 采用 tanh 平滑引导 Alpha (解决贴边震荡)
        """
        # [1] 动态更新 Alpha 数值安全
        current_alpha = torch.clamp(self.learnable_alpha, 1.5, 1.95)
        self.generative_process.dlpm.alpha = current_alpha
        
        # [2] 生成并约束扩散噪声 (防止 Lévy 分布产生过大的初始离群点)
        if noise is None:
            noise = self.generative_process.dlpm.gen_eps.generate(size=x_start.shape)
        noise = torch.clamp(noise, min=-5.0, max=5.0)

        # [3] 前向扩散采样
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        if isinstance(x_t, (list, tuple)): x_t = x_t[0]
        
        # 极端值防爆兜底
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=20.0, neginf=-20.0)
        x_t = torch.clamp(x_t, min=-20.0, max=20.0)

        # [4] 模型预测
        # 修复 Bug 1：直接使用 forward 传进来的已编码 cond_input，绝不再调用 condition_network
        model_kwargs = {'cond_input': cond_input} if exists(cond_input) else {}
        # 训练阶段默认不使用 self_cond 以保证效率，如有需要可在此按概率开启
        model_kwargs['y_self_cond'] = None 

        model_output = self.model(x_t, t, **model_kwargs)
        model_output = torch.nan_to_num(model_output, nan=0.0, posinf=20.0, neginf=-20.0)

        # [5] 反推原始信号 pred_x_start
        pred_x_start_raw = self.generative_process.dlpm.predict_xstart(x_t, t, model_output)
        
        # [6] 关键：信噪比闸门 (SNR Gate)
        # 解决数值爆炸的根源：仅在信号可见度高的样本 (t 较小) 上计算统计 Loss
        snr = (self.generative_process.dlpm.bargammas[t]**2) / (self.generative_process.dlpm.barsigmas[t]**2)
        stat_gate = (snr > 0.05).float().reshape(-1, 1, 1) 

        # [7] 物理边界惩罚 (Overflow Loss)
        # 只有在 SNR 足够时才惩罚溢出，防止大 t 时的数学误差干扰梯度
        overflow_loss = torch.mean(torch.relu(torch.abs(pred_x_start_raw) - 2.5)**2 * stat_gate) * 100
        
        # 限制 pred_x_start 范围用于后续指标计算，防止 Kurtosis 算成 Inf
        pred_x_start = torch.clamp(pred_x_start_raw, min=-2.5, max=2.5)

        # [8] 基础重构损失 (Base Loss)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else: # pred_v
            target = self.generative_process.dlpm.predict_v(x_start, t, noise)
            
        base_loss_unreduced = 2.0 * F.smooth_l1_loss(model_output, target, beta=1.0, reduction='none')
        
        if exists(mask):
            # 确保 mask 维度对齐 [B, C, L]
            mask_expanded = mask.unsqueeze(1).expand_as(base_loss_unreduced) if mask.dim() < base_loss_unreduced.dim() else mask
            base_loss = (base_loss_unreduced * mask_expanded).sum() / mask_expanded.sum().clamp(min=1e-5)
        else:
            base_loss = base_loss_unreduced.mean()

        # [9] 统计特征对齐损失
        custom_losses_raw = {}
        eps = 1e-6
        
        # A. 波动率对齐
        _, p_std = self._masked_statistics(pred_x_start, mask)
        _, t_std = self._masked_statistics(x_start, mask)
        custom_losses_raw['global_vol'] = F.l1_loss(p_std, t_std)

        # B. 峰度对齐 (L1 损失比 MSE 更稳)
        p_kurt = self._masked_kurtosis(pred_x_start, mask)
        t_kurt = self._masked_kurtosis(x_start, mask)
        custom_losses_raw['heavy_tail'] = F.l1_loss(p_kurt, t_kurt)

        # C. 99分位点对齐 (捕捉极端收益率分布)
        # 注意：由于 torch.quantile 不支持 mask，这里仅对整个 batch 计算
        p_q99 = torch.quantile(pred_x_start, 0.99)
        t_q99 = torch.quantile(x_start, 0.99)
        custom_losses_raw['quantile'] = F.mse_loss(p_q99, t_q99)

        # [10] 汇总总损失并进行 EMA 量级归一化
        weights = self._get_annealed_weights(global_step)
        total_loss = base_loss + overflow_loss
        
        for name, raw_val in custom_losses_raw.items():
            ema_buf = getattr(self, f'ema_{name}')
            # 更新 EMA 缓冲区
            with torch.no_grad():
                if ema_buf == 0: ema_buf.copy_(raw_val.detach())
                else: ema_buf.copy_(ema_buf * self.ema_beta + raw_val.detach() * (1 - self.ema_beta))
            
            w = weights.get(name, 0.0)
            if w > 0:
                # 归一化 Loss 确保不同指标在同一个数量级，并受 SNR 闸门保护
                norm_loss = raw_val / (ema_buf + eps)
                total_loss += (w * norm_loss * stat_gate).mean()

        # [11] Alpha 综合引导损失 (修复 Bug H：改为 tanh 平滑逻辑)
        # 获取误差方向
        kurt_error = p_kurt.mean() - t_kurt.mean().detach()
        p_diff_mean = torch.abs(pred_x_start[..., 1:] - pred_x_start[..., :-1]).mean()
        t_diff_mean = torch.abs(x_start[..., 1:] - x_start[..., :-1]).mean().detach()
        jump_error = p_diff_mean - t_diff_mean
        
        # 综合推力计算
        combined_direction = (
            1.0 * torch.tanh(kurt_error / 5.0) + 
            1.5 * torch.tanh(jump_error / 0.05)
        )
        
        alpha_guidance_loss = 0
        if global_step > getattr(self, 'guidance_warmup_threshold', 4400):
            # 通过负向梯度诱导 learnable_alpha 自动挪动
            alpha_guidance_loss = - (self.learnable_alpha * combined_direction * 0.1)
            total_loss += alpha_guidance_loss

        # [12] 周期性打印详细诊断日志 (每100步)
        if global_step % 100 == 0:
            print(f"\n[STEP {global_step}] BaseLoss: {base_loss:.4f} | Overflow: {overflow_loss:.4f} | Alpha: {current_alpha.item():.4f}")
            print(f"指标对标 -> Vol: P{p_std.mean():.4f}/T{t_std.mean():.4f} | Kurt: P{p_kurt.mean():.2f}/T{t_kurt.mean():.2f}")

        return total_loss
        
       
