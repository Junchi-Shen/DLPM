# DLPM扩散模型适配器
# 与现有的diffusion_with_condition接口兼容

import torch
import torch.nn as nn
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
        self.alpha = alpha  # Lévy稳定分布的alpha参数

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

    def p_sample(self, x, t, self_cond=None, cond_input=None, model_forward_kwargs: dict = dict()):
        """单步采样"""
        model_kwargs = {}
        
        # 处理条件输入：如果存在条件网络，先通过条件网络处理
        processed_cond_input = None
        if exists(cond_input):
            if self.has_condition_network:
                processed_cond_input = self.condition_network(cond_input)
            else:
                processed_cond_input = cond_input
            model_kwargs['cond_input'] = processed_cond_input
        
        # 确保A和Sigmas已初始化
        if self.generative_process.dlpm.A is None or self.generative_process.dlpm.Sigmas is None:
            shape = x.shape
            self.generative_process.dlpm.sample_A(shape, self.num_timesteps)
            self.generative_process.dlpm.compute_Sigmas()

        out = self.generative_process.p_sample(
            self.model, x, t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        return out["sample"], None

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise=False, model_forward_kwargs: dict = dict()):
        """采样循环"""
        device = self.generative_process.device
        batch = shape[0]
        
        # 初始化A和Sigmas
        self.generative_process.dlpm.sample_A(shape, self.num_timesteps)
        self.generative_process.dlpm.compute_Sigmas()
        
        # 初始噪声
        noise = self.generative_process.dlpm.barsigmas[-1] * \
                self.generative_process.dlpm.gen_eps.generate(size=shape)
        img = noise
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), 
                     desc='sampling loop time step', 
                     total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            cond_input = model_forward_kwargs.get('cond_input', None)
            mask = model_forward_kwargs.get('mask', None)
            
            if exists(mask):
                img_masked = img * mask
                noise_masked = noise * (1 - mask)
                img = img_masked + noise_masked
            
            t_tensor = torch.full((batch,), t, device=device, dtype=torch.long)
            img, x_start = self.p_sample(img, t_tensor, self_cond, cond_input, 
                                        model_forward_kwargs=model_forward_kwargs)

        img = self.unnormalize(img)
        if exists(mask):
            img = img * mask + noise * (1 - mask)
        return (img, noise) if return_noise else img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True, model_forward_kwargs: dict = dict(), return_noise=False):
        """DDIM采样（DLIM方法）"""
        device = self.generative_process.device
        batch = shape[0]
        eta = self.ddim_sampling_eta
        
        # 处理条件输入（在DDIM采样循环之前）
        cond_input = model_forward_kwargs.get('cond_input', None)
        processed_cond_input = None
        if exists(cond_input):
            if self.has_condition_network:
                processed_cond_input = self.condition_network(cond_input)
            else:
                processed_cond_input = cond_input
        
        # 更新model_kwargs
        ddim_model_kwargs = {}
        if exists(processed_cond_input):
            ddim_model_kwargs['cond_input'] = processed_cond_input
        
        # 使用DDIM采样循环（DLIM方法）
        # 这会自动处理A和Sigmas的初始化以及跳步采样
        img = self.generative_process.ddim_sample_loop(
            self.model,
            shape=shape,
            noise=None,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=ddim_model_kwargs,
            progress=True,
            eta=eta,
            get_sample_history=False,
            sampling_timesteps=self.sampling_timesteps
        )
        
        # 处理mask
        mask = model_forward_kwargs.get('mask', None)
        noise = self.generative_process.dlpm.barsigmas[-1] * \
                self.generative_process.dlpm.gen_eps.generate(size=shape)
        
        img = self.unnormalize(img)
        if exists(mask):
            img = img * mask + noise * (1 - mask)
        return (img, noise) if return_noise else img

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
        前向传播方法，与trainer兼容
        """
        b, c, n = img.shape
        device = img.device
        
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # 处理条件输入
        processed_cond_input = None
        if exists(cond_input):
            if self.has_condition_network:
                processed_cond_input = self.condition_network(cond_input)
            else:
                processed_cond_input = cond_input
        
        # 计算损失
        loss = self.p_losses(
            x_start=img,
            t=t,
            cond_input=processed_cond_input,
            mask=mask,
            global_step=global_step,
            **kwargs
        )
        
        return loss

    def p_losses(self, x_start, t, cond_input=None, noise=None, mask=None, global_step=0, model_forward_kwargs: dict = dict()):
        """计算训练损失"""
        if noise is None:
            noise = self.generative_process.dlpm.gen_eps.generate(size=x_start.shape)

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_kwargs = {}
        if exists(cond_input):
            model_kwargs['cond_input'] = cond_input

        # 使用DLPM的训练损失
        loss_dict = self.generative_process.training_losses(
            models={'default': self.model},
            x_start=x_start,
            model_kwargs=model_kwargs,
            **model_forward_kwargs
        )

        loss = loss_dict['loss']
        
        # 如果提供了mask，应用mask到损失
        if exists(mask):
            # DLPM的损失已经是标量，这里简单返回
            # 如果需要更精细的mask处理，可以在generative_process中实现
            pass
        
        return loss

