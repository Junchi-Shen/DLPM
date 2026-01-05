# GenerativeLevyProcess - DLPM的生成过程包装器
# 简化版本，适配现有系统

import torch
import torch.nn as nn
from tqdm import tqdm
from .dlpm_core import DLPM, ModelMeanType, ModelVarType, LossType
from .levy_distributions import match_last_dims


def compute_loss_terms(x, y, lploss):
    """计算损失项"""
    if lploss == 2.:  # L2 loss
        tmp = nn.functional.mse_loss(x, y, reduction='none')
        tmp = torch.sqrt(tmp.mean(dim=list(range(1, len(x.shape)))))
        return tmp
    elif lploss == 1.:  # L1 loss
        tmp = nn.functional.smooth_l1_loss(x, y, beta=1, reduction='none')
        tmp = tmp.mean(dim=list(range(1, len(x.shape))))
        return tmp
    elif lploss == -1:  # squared L2 loss
        return nn.functional.mse_loss(x, y, reduction='none').mean(dim=list(range(1, len(x.shape))))
    else:
        return torch.pow(torch.linalg.norm(x - y, ord=lploss, dim=list(range(1, len(x.shape)))), 1 / lploss)


class GenerativeLevyProcess:
    """
    DLPM的训练和采样工具类
    """
    def __init__(
        self,
        alpha,
        device,
        reverse_steps,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED,
        time_spacing='linear',
        rescale_timesteps=False,
        isotropic=True,
        scale='scale_preserving',
        input_scaling=False,
    ):
        self.alpha = alpha
        self.device = device
        self.reverse_steps = reverse_steps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.time_spacing = time_spacing
        self.rescale_timesteps = rescale_timesteps
        self.isotropic = isotropic
        self.input_scaling = input_scaling

        assert (self.model_mean_type == ModelMeanType.EPSILON) and (self.model_var_type == ModelVarType.FIXED), \
            'Only epsilon prediction and fixed variance are supported'

        self.dlpm = DLPM(
            alpha,
            device,
            diffusion_steps=reverse_steps,
            time_spacing=time_spacing,
            isotropic=isotropic,
            scale=scale
        )

    def _scale_timesteps(self, t):
        """缩放时间步"""
        if self.rescale_timesteps:
            return t.float() * (1.0 / self.reverse_steps)
        return t

    def q_sample(self, x_start, t, eps=None):
        """前向扩散过程：从x_0采样x_t"""
        return self.dlpm.sample_x_t_from_xstart(x_start, t, eps)

    def p_mean_variance(self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        """计算p(x_{t-1}|x_t)的均值和方差"""
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if model_kwargs is None:
            model_kwargs = {}

        B = x.shape[0]
        assert t.shape == (B,)

        # 运行模型
        input_scaling = 1.0
        if self.input_scaling and (self.dlpm.scale == 'scale_exploding'):
            input_scaling = match_last_dims(1 / (1+self.dlpm.barsigmas[t]), x.shape)
        
        # 处理条件输入
        cond_input = model_kwargs.get('cond_input', None)
        if cond_input is not None:
            model_output = model(x * input_scaling, self._scale_timesteps(t), cond_input=cond_input)
        else:
            model_output = model(x * input_scaling, self._scale_timesteps(t))
        
        # 检查模型输出是否有NaN/Inf（在t=0时特别容易出现）
        if torch.isnan(model_output).any() or torch.isinf(model_output).any():
            t_val = t[0].item() if isinstance(t, torch.Tensor) and len(t) > 0 else (t if isinstance(t, int) else 0)
            if t_val == 0:
                # 在t=0时，模型应该输出接近0的噪声（因为x_t已经接近x_start）
                model_output = torch.where(
                    torch.isnan(model_output) | torch.isinf(model_output),
                    torch.zeros_like(model_output),
                    model_output
                )
            else:
                # 在其他时间步，使用更保守的处理
                model_output = torch.where(
                    torch.isnan(model_output) | torch.isinf(model_output),
                    torch.zeros_like(model_output),
                    model_output
                )

        if (self.model_mean_type == ModelMeanType.EPSILON) and (not clip_denoised):
            model_eps = model_output
        else:
            if self.model_mean_type == ModelMeanType.EPSILON:
                model_xstart = process_xstart(
                    self.dlpm.predict_xstart(x_t=x, t=t, eps=model_output)
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            model_eps = self.dlpm.predict_eps(x_t=x, t=t, xstart=model_xstart)

        # 计算后验均值和方差
        # 注意：这里假设批次中所有样本在同一时间步（训练时通常如此）
        # 如果批次中时间步不同，需要为每个样本单独处理
        t_val = t[0] if len(t) > 0 else t
        model_mean, model_variance = self.dlpm.anterior_mean_variance_dlpm(x, t_val, model_eps)

        assert model_mean.shape == x.shape

        return {
            'eps': model_eps,
            'mean': model_mean,
            'variance': model_variance,
        }

    def p_sample(self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        """从p(x_{t-1}|x_t)采样"""
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 1).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.sqrt(out["variance"]) * noise
        return {"sample": sample}

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=False, 
                                 denoised_fn=None, model_kwargs=None):
        """渐进式采样循环"""
        assert self.device is not None
        assert isinstance(shape, (tuple, list))

        self.dlpm.sample_A(shape, self.reverse_steps)
        self.dlpm.compute_Sigmas()

        if noise is not None:
            img = noise
        else:
            img = self.dlpm.barsigmas[-1] * self.dlpm.gen_eps.generate(size=shape)
        yield {'sample': img}

        indices = list(range(self.reverse_steps-1, 0, -1))

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            out = self.p_sample(
                model, img, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            yield out
            img = out["sample"]

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=False, 
                     denoised_fn=None, model_kwargs=None, progress=False, get_sample_history=False):
        """采样循环"""
        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total=self.reverse_steps)
        model.eval()
        x_hist = []

        with torch.inference_mode():
            final = None
            for sample in self.p_sample_loop_progressive(
                model, shape, noise=noise, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, model_kwargs=model_kwargs
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)

            if progress:
                pbar.close()
                tqdm._instances.clear()

            if get_sample_history:
                return final['sample'], torch.stack(x_hist)
            return final["sample"]

    def ddim_sample(self, model, x, t, clip_denoised=False, denoised_fn=None, 
                   model_kwargs=None, eta=0.0):
        """使用DDIM采样x_{t-1}"""
        try:
            out = self.p_mean_variance(
                model, x, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            eps = out['eps']
            
            # 检查eps是否有NaN或Inf
            if torch.isnan(eps).any() or torch.isinf(eps).any():
                t_val = t[0].item() if isinstance(t, torch.Tensor) and len(t) > 0 else (t if isinstance(t, int) else 0)
                # 只在t=0时打印警告，避免输出过多
                if t_val == 0:
                    nan_count = torch.isnan(eps).sum().item()
                    inf_count = torch.isinf(eps).sum().item()
                    total_count = eps.numel()
                    # 只在NaN/Inf比例较高时打印（避免每个样本都打印）
                    if nan_count + inf_count > total_count * 0.01:  # 超过1%才打印
                        print(f"警告: 模型输出的eps包含NaN/Inf, t={t_val}, NaN: {nan_count}/{total_count}, Inf: {inf_count}/{total_count}, 使用零值替换")
                eps = torch.where(torch.isnan(eps) | torch.isinf(eps), torch.zeros_like(eps), eps)
            
            # 限制eps的范围，防止数值爆炸
            eps = torch.clamp(eps, min=-10.0, max=10.0)
            
            model_mean, model_variance = self.dlpm.anterior_mean_variance_dlim(x, t, eps, eta=eta)
            
            # 检查并处理NaN和Inf
            if torch.isnan(model_mean).any() or torch.isinf(model_mean).any():
                t_val = t[0].item() if isinstance(t, torch.Tensor) and len(t) > 0 else (t if isinstance(t, int) else 0)
                print(f"警告: anterior_mean_variance_dlim返回的mean包含NaN/Inf, t={t_val}, 使用x作为回退")
                model_mean = torch.where(torch.isnan(model_mean) | torch.isinf(model_mean), x, model_mean)
            
            if torch.isnan(model_variance).any() or torch.isinf(model_variance).any():
                print(f"警告: anterior_mean_variance_dlim返回的variance包含NaN/Inf, 使用零值替换")
                model_variance = torch.where(torch.isnan(model_variance) | torch.isinf(model_variance), 
                                            torch.zeros_like(model_variance), model_variance)
            
            # 确定性采样（eta == 0.0）
            if eta == 0.0:
                return {"sample": model_mean}
            
            # 随机采样（eta > 0.0）
            noise = torch.randn_like(x)
            nonzero_mask = ((t != 1).float().view(-1, *([1] * (len(x.shape) - 1))))
            # 确保方差非负
            model_variance = torch.clamp(model_variance, min=0.0, max=1e6)  # 限制最大值防止溢出
            sample = model_mean + nonzero_mask * torch.sqrt(model_variance + 1e-8) * noise
            
            # 最终检查
            if torch.isnan(sample).any() or torch.isinf(sample).any():
                print(f"警告: DDIM采样结果包含NaN/Inf，使用x作为回退")
                sample = torch.where(torch.isnan(sample) | torch.isinf(sample), x, sample)
            
            return {"sample": sample}
        except Exception as e:
            print(f"错误: DDIM采样过程中发生异常: {e}")
            # 返回x作为回退
            return {"sample": x}

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=False,
                                    denoised_fn=None, model_kwargs=None, eta=0.0, sampling_timesteps=None):
        """渐进式DDIM采样循环"""
        assert self.device is not None
        assert isinstance(shape, (tuple, list))

        # 使用完整的reverse_steps来初始化A和Sigmas
        # 虽然DDIM只采样部分时间步，但需要完整的序列
        if self.dlpm.A is None or self.dlpm.Sigmas is None:
            self.dlpm.sample_A(shape, self.reverse_steps)
            self.dlpm.compute_Sigmas()
        
        # 如果指定了sampling_timesteps，创建跳步的时间序列
        if sampling_timesteps is not None and sampling_timesteps < self.reverse_steps:
            # 创建跳步的时间序列
            times = torch.linspace(self.reverse_steps - 1, 0, sampling_timesteps + 1, device=self.device)
            indices = times.int().tolist()
        else:
            indices = list(range(self.reverse_steps-1, 0, -1))

        if noise is not None:
            img = noise
        else:
            img = self.dlpm.barsigmas[-1] * self.dlpm.gen_eps.generate(size=shape)
        yield {'sample': img}

        # indices已在上面定义
        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            out = self.ddim_sample(
                model, img, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            yield out
            img = out["sample"]

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=False,
                        denoised_fn=None, model_kwargs=None, progress=False,
                        eta=0.0, get_sample_history=False, sampling_timesteps=None):
        """DDIM采样循环"""
        # 确定实际使用的采样步数
        actual_steps = sampling_timesteps if sampling_timesteps is not None else self.reverse_steps
        
        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total=actual_steps)
        
        model.eval()
        x_hist = []
        
        with torch.inference_mode():
            final = None
            for sample in self.ddim_sample_loop_progressive(
                model, shape, noise=noise, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, model_kwargs=model_kwargs, 
                eta=eta, sampling_timesteps=sampling_timesteps
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)
            
            if progress:
                pbar.close()
                tqdm._instances.clear()
            
            if get_sample_history:
                return final['sample'], torch.stack(x_hist)
            return final["sample"]

    def sample(self, models, shape, reverse_steps, time_spacing=None, initial_data=None,
              clip_denoised=False, deterministic=False, dlim_eta=1.0, print_progression=False,
              get_sample_history=False, clamp_a=None, clamp_eps=None):
        """采样接口"""
        self.dlpm.gen_a.setParams(clamp_a=clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps=clamp_eps)

        model = models['default']

        default_reverse_steps = self.reverse_steps
        default_time_spacing = self.time_spacing
        if self.reverse_steps != reverse_steps:
            assert self.rescale_timesteps, "Rescaling only works when rescale_timesteps is True"
            self.dlpm.rescale_diffusion(reverse_steps, time_spacing=time_spacing)
            self.reverse_steps = reverse_steps

        x = self.p_sample_loop(
            model,
            shape=shape,
            progress=print_progression,
            get_sample_history=get_sample_history,
            clip_denoised=clip_denoised
        )

        if self.reverse_steps != reverse_steps:
            self.dlpm.rescale_diffusion(default_reverse_steps, default_time_spacing)
            self.reverse_steps = default_reverse_steps

        return x

    def training_losses_dlpm(self, model, x_start, loss_type=None, lploss=2.0,
                            loss_monte_carlo='mean', monte_carlo_outer=1, monte_carlo_inner=1,
                            model_kwargs=None, clamp_a=None, clamp_eps=None):
        """DLPM训练损失"""
        assert self.model_mean_type == ModelMeanType.EPSILON
        
        # 如果没有指定loss_type，使用默认的EPS_LOSS
        if loss_type is None:
            loss_type = LossType.EPS_LOSS
        # 如果传入的是字符串，转换为枚举
        elif isinstance(loss_type, str):
            if loss_type.upper() == 'EPS_LOSS' or loss_type.upper() == 'EPSILON':
                loss_type = LossType.EPS_LOSS
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")
        
        assert loss_type == LossType.EPS_LOSS

        if model_kwargs is None:
            model_kwargs = {}

        self.dlpm.gen_a.setParams(clamp_a=clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps=clamp_eps)

        # 获取时间步
        t = torch.randint(1, self.reverse_steps, size=[len(x_start)]).to(self.device)

        # 蒙特卡洛估计
        total_monte_carlo = monte_carlo_outer * monte_carlo_inner
        x_start_extended = x_start.repeat(total_monte_carlo, *([1]*len(x_start.shape[1:])))
        t_extended = t.repeat(total_monte_carlo)
        outer_shape = list(x_start.shape)
        outer_shape[0] *= monte_carlo_outer
        A = self.dlpm.get_one_rv_faster_sampling(outer_shape)
        A_extended = A.repeat(monte_carlo_inner, *([1]*len(A.shape[1:])))
        z_t_extended = torch.randn_like(x_start_extended, device=self.device)

        # 获取损失元素
        x_t, eps_t = self.dlpm.get_one_rv_loss_elements(t_extended, x_start_extended, A_extended, z_t_extended)

        # 运行模型
        input_scaling = 1.0
        if self.input_scaling and (self.dlpm.scale == 'scale_exploding'):
            input_scaling = match_last_dims(1 / (1+self.dlpm.barsigmas[t_extended]), x_t.shape)
        
        cond_input = model_kwargs.get('cond_input', None)
        if cond_input is not None:
            # 扩展条件输入以匹配批次大小
            cond_input_extended = cond_input.repeat(total_monte_carlo, *([1]*len(cond_input.shape[1:])))
            model_eps = model(x_t * input_scaling, self._scale_timesteps(t_extended), cond_input=cond_input_extended)
        else:
            model_eps = model(x_t * input_scaling, self._scale_timesteps(t_extended))

        # 计算损失
        losses = compute_loss_terms(model_eps, eps_t, lploss)
        assert not torch.isnan(losses).any(), 'NaN in losses'

        if loss_monte_carlo == 'mean':
            loss = losses.mean()
        elif loss_monte_carlo == 'median':
            losses = losses.reshape(monte_carlo_outer, monte_carlo_inner, x_start.shape[0])
            losses = losses.mean(dim=1)
            losses, _ = losses.median(dim=0)
            loss = losses.mean()
        else:
            loss = losses.mean()

        return loss

    def training_losses(self, models, x_start, model_kwargs=None, **kwargs):
        """训练损失接口"""
        model = models['default']
        x_start = x_start.to(self.device)
        if model_kwargs is None:
            model_kwargs = {}

        loss = self.training_losses_dlpm(model, x_start, model_kwargs=model_kwargs, **kwargs)
        return {'loss': loss}

