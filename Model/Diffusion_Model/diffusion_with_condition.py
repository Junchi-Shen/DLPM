# diffusion.py

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from collections import namedtuple
from functools import partial
from random import random

from tqdm.auto import tqdm
from torch.amp import autocast
from einops import rearrange, reduce

# 从我们自己的文件导入
from .Utils import (
    exists,
    default,
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one
)
# 从 model.py 导入核心模型
from Model.Diffusion_Model.Unet_with_condition import Unet1D

# --- Constants ---
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        *,
        seq_length,
        timesteps = 1000,
        train_num_steps = 100000, # 新增: 总训练步数，用于动态计算预热
        warmup_ratio = 0.25,      # 新增: 预热阶段占总训练的比例
        ema_beta = 0.99,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        channels = None,
        self_condition = None,
        model: Unet1D,
        channel_first = True,
        condition_network: nn.Module = None,
       
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

        self.condition_network = condition_network 
        self.has_condition_network = exists(condition_network) 

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise, pred_x0, or pred_v'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)
        register_buffer('loss_weight', loss_weight)

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = identity

        # --- 新增：根据训练参数计算预热步数 ---
        self.train_num_steps = train_num_steps
        self.warmup_steps = int(train_num_steps * warmup_ratio)
        print(f"Loss weight annealing enabled. Warmup will be performed for {self.warmup_steps} steps.")
        self.ema_beta = ema_beta
        loss_names = ['relative_jump', 'vol_clustering', 'global_vol', 'heavy_tail', 'skewness', 'drift', 'quantile', 'spectral']
        for name in loss_names:
            register_buffer(f'ema_{name}', torch.tensor(0.0))
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, cond_input = None, clip_x_start = False, rederive_pred_noise = False, model_forward_kwargs: dict = dict()):
        """
        从模型输出计算预测的 x_start 和 pred_noise。
        处理条件输入（如果存在条件网络）。
        """
        
        # --- ** 新增: 处理条件输入 ** ---
        processed_cond_input = None
        if exists(cond_input):
            if self.has_condition_network:
                # 使用条件网络处理原始 cond_input (7维)
                processed_cond_input = self.condition_network(cond_input)
            else:
                # 如果没有条件网络，直接使用原始 cond_input (必须是 U-Net 期望的维度)
                processed_cond_input = cond_input
        # --- ** 结束新增 ** ---

        # 将处理后的条件 (processed_cond_input) 传递给 U-Net
        model_output = self.model(x, time = t, y_self_cond = x_self_cond, cond_input = processed_cond_input)
        
        # --- (后续的 x_start 和 pred_noise 推导逻辑保持不变) ---
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            # Optionally rederive noise from clipped x_start
            if clip_x_start and rederive_pred_noise:
                 pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, cond_input = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'y_self_cond': x_self_cond}
        if exists(cond_input):
            model_forward_kwargs = {**model_forward_kwargs, 'cond_input': cond_input}

        preds = self.model_predictions(x, t, x_self_cond=x_self_cond, cond_input=cond_input)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, cond_input = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, cond_input=cond_input, clip_denoised=clip_denoised, model_forward_kwargs=model_forward_kwargs)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise = False, model_forward_kwargs: dict = dict()):
        batch, device = shape[0], self.betas.device
        noise = torch.randn(shape, device=device)
        img = noise
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            cond_input = model_forward_kwargs.get('cond_input', None)
            mask = model_forward_kwargs.get('mask', None)
            
            if exists(mask):
                img_masked = img * mask
                noise_masked = noise * (1 - mask)
                img = img_masked + noise_masked
            
            img, x_start = self.p_sample(img, t, self_cond, cond_input, model_forward_kwargs = model_forward_kwargs)

        img = self.unnormalize(img)
        if exists(mask):
            img = img * mask + noise * (1 - mask)
        return (img, noise) if return_noise else img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True, model_forward_kwargs: dict = dict(), return_noise = False):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        noise = torch.randn(shape, device = device)
        img = noise
        x_start = None
        mask = model_forward_kwargs.get('mask', None)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            cond_input = model_forward_kwargs.get('cond_input', None)
            
            if exists(mask):
                img_masked = img * mask
                noise_masked = noise * (1 - mask)
                img = img_masked + noise_masked
            
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, cond_input, clip_x_start = clip_denoised, model_forward_kwargs = model_forward_kwargs)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise_pred = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise_pred

        img = self.unnormalize(img)
        if exists(mask):
            img = img * mask + noise * (1 - mask)
        return (img, noise) if return_noise else img

    @torch.no_grad()
    def sample(self, batch_size = 16, cond_input = None, mask = None, return_noise = False, model_forward_kwargs: dict = dict()):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        shape = (batch_size, channels, seq_length) if self.channel_first else (batch_size, seq_length, channels)
        
        if exists(cond_input):
            model_forward_kwargs['cond_input'] = cond_input
        if exists(mask):
            model_forward_kwargs['mask'] = mask
            
        return sample_fn(shape, return_noise=return_noise, model_forward_kwargs=model_forward_kwargs)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

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
    def _masked_quantile(self, x, mask, q: float):
        # x: (B, C, T) or (B, T) – we assume (B, C, T)
        assert 0.0 < q < 1.0
        if mask is not None and x.dim() > mask.dim():
            mask = mask.unsqueeze(1).expand_as(x)
        if mask is None:
            # quantile along last dim
            return x.nanquantile(q, dim=-1, keepdim=True)
        # flatten masked values per sample-channel
        B, C, T = x.shape
        x_flat = x.reshape(B*C, T)
        m_flat = mask.reshape(B*C, T)
        out = []
        for i in range(B*C):
            xi = x_flat[i][m_flat[i] > 0.5]
            if xi.numel() == 0:
                out.append(torch.tensor(float('nan'), device=x.device))
            else:
                out.append(xi.quantile(q))
        out = torch.stack(out, dim=0).reshape(B, C, 1)
        return out
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

    def _get_annealed_weights(self, global_step: int, device: torch.device):
        """
        (新辅助方法) 根据当前的全局训练步数，计算所有自定义损失的退火权重。
        """
        warmup_steps = self.warmup_steps
        global_scale = min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0

        base_weights = {
            'relative_jump': 0.5,
            'vol_clustering': 3.5,
            'global_vol': 8,
            'heavy_tail': 5,
            'skewness': 0.5,
            'drift': 2,
            'quantile': 3,
            'spectral': 2.0,

        }

        annealed_weights = {}
        for name, base_weight in base_weights.items():
            final_weight = base_weight * global_scale
            annealed_weights[name] = final_weight

        return annealed_weights
    def _power_spectrum(self, x, mask=None, eps=1e-8):
        # magnitude spectrum along last dim
        # return normalized power to be shape-(B,C,T) comparable
        if mask is not None and x.dim() > mask.dim():
            mask = mask.unsqueeze(1).expand_as(x)
        if mask is not None:
            # fill invalids with mean to reduce leakage
            mean, _ = self._masked_statistics(x, mask)
            x_eff = torch.where(mask > 0.5, x, mean)
        else:
            x_eff = x
        Xf = torch.fft.rfft(x_eff, dim=-1)
        P = (Xf.real**2 + Xf.imag**2).sqrt()
        P = P / (P.amax(dim=-1, keepdim=True) + eps)
        return P
    def _masked_skewness(self, x, mask, eps=1e-6):
        mean, std = self._masked_statistics(x, mask, eps)
        if mask is None:
            z = (x - mean) / (std + eps)
            s = torch.mean(z**3, dim=-1)
            return s
        
        z = (x - mean) / (std + eps)
        z_masked = z * mask
        count = mask.sum(dim=-1).clamp(min=1)
        z3_sum = (z_masked**3).sum(dim=-1)
        s = z3_sum / count
        return s


    def p_losses(self, x_start, t, noise = None, cond_input = None, mask = None, global_step=0, **model_forward_kwargs_unused): # Renamed kwargs for clarity
        b, c, n = x_start.shape; device = x_start.device; eps=1e-6

        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start = x_start, t = t, noise = noise) # Diffuse input

        # --- Self-conditioning logic (uses model_predictions internally, which is already fixed) ---
        x_self_cond = None
        if self.self_condition and random() < 0.5:
             with torch.no_grad():
                # model_predictions correctly handles condition_network now
                x_self_cond = self.model_predictions(x, t, cond_input=cond_input).pred_x_start.detach_()
        # ---

        # --- !! 核心修正：处理条件 !! ---
        processed_cond_input = None # This will be passed to self.model (U-Net)
        if exists(cond_input):
            if self.has_condition_network:
                # 使用条件网络处理原始 cond_input (7维)
                processed_cond_input = self.condition_network(cond_input)
            else:
                # 否则直接使用原始 cond_input (U-Net 必须配置为接受它)
                processed_cond_input = cond_input
        # --- !! 修正结束 !! ---

        # --- !! 核心修正：调用 U-Net !! ---
        # ** 将 *处理后* 的条件传递给 U-Net **
        # ** 将 x_self_cond 也正确传递 **
        model_out = self.model(x, time=t, y_self_cond = x_self_cond, cond_input = processed_cond_input)
        # --- !! 修正结束 !! ---

        # --- Determine target for MSE loss (Original logic) ---
        if self.objective == 'pred_noise':
            target_for_mse = noise
            # We need pred_x_start for custom losses, derive it
            pred_x_start = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == 'pred_x0':
            pred_x_start = model_out # U-Net output *is* predicted x_start
            target_for_mse = x_start
        elif self.objective == 'pred_v':
            v = model_out
            pred_x_start = self.predict_start_from_v(x, t, v)
            target_for_mse = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        # ---

        # --- Calculate MSE Loss (Original logic, ensure model_out/target match objective) ---
        mse_loss_unreduced = F.mse_loss(model_out, target_for_mse, reduction = 'none')
        if exists(mask):
            # Mask broadcasting logic might need refinement depending on mask/loss shapes
            mask_expanded = mask
            if mse_loss_unreduced.ndim > mask.ndim and mask.ndim >= 2: # Check if broadcasting needed
                 mask_expanded = mask.unsqueeze(1).expand_as(mse_loss_unreduced)
            mse_loss = (mse_loss_unreduced * mask_expanded).sum() / (mask_expanded.sum().clamp(min = 1e-5))
        else:
             mse_loss = reduce(mse_loss_unreduced, 'b ... -> b', 'mean').mean()
        # ---

        # --- !! Keep ALL original custom loss calculations !! ---
        # These use pred_x_start (derived above) and x_start
        # (relative_jump_loss, vol_clustering_loss, global_vol_loss, heavy_tail_loss,
        #  skewness_loss, quantile_loss, spectral_loss, drift_loss)
        # ... (Your complex loss calculations remain exactly the same as in the original file) ...
        pred_diff=pred_x_start[...,1:]-pred_x_start[...,:-1];target_diff=x_start[...,1:]-x_start[...,:-1];mask_diff=mask[...,1:]*mask[...,:-1] if mask is not None else None;relative_jump_loss_unreduced=F.l1_loss(pred_diff,target_diff,reduction='none');relative_jump_loss=(relative_jump_loss_unreduced*mask_diff).sum()/mask_diff.sum().clamp(min=eps) if mask_diff is not None else relative_jump_loss_unreduced.mean()
        window_size,window_stride=max(8,n//8),max(4,n//32);pred_windows=pred_x_start.unfold(dimension=-1,size=window_size,step=window_stride);target_windows=x_start.unfold(dimension=-1,size=window_size,step=window_stride);mask_windows=mask.unfold(dimension=-1,size=window_size,step=window_stride) if mask is not None else None
        if mask_windows is not None:
             pad_needed = pred_windows.shape[-2] - mask_windows.shape[-2]
             if pad_needed > 0: mask_windows = F.pad(mask_windows, (0,0,0,pad_needed), value=0)
             elif pad_needed < 0: mask_windows = mask_windows[...,:pred_windows.shape[-2],:]
        _,pred_window_vols=self._masked_statistics(pred_windows,mask_windows);_,target_window_vols=self._masked_statistics(target_windows,mask_windows);vol_clustering_loss=F.smooth_l1_loss(pred_window_vols,target_window_vols)
        _,pred_global_std=self._masked_statistics(pred_x_start,mask);_,target_global_std=self._masked_statistics(x_start,mask);global_vol_loss=F.l1_loss(pred_global_std,target_global_std)
        pred_kurt=self._masked_kurtosis(pred_x_start,mask);target_kurt=self._masked_kurtosis(x_start,mask);heavy_tail_loss=F.mse_loss(pred_kurt,target_kurt)
        pred_skew=self._masked_skewness(pred_x_start,mask);target_skew=self._masked_skewness(x_start,mask);skewness_loss=F.mse_loss(pred_skew,target_skew)
        q_low,q_high=0.01,0.99;pred_q01=self._masked_quantile(pred_x_start,mask,q_low);targ_q01=self._masked_quantile(x_start,mask,q_low);pred_q99=self._masked_quantile(pred_x_start,mask,q_high);targ_q99=self._masked_quantile(x_start,mask,q_high)
        def pinball(pred,targ,q):e=targ-pred;return torch.relu(q*e)+torch.relu((q-1)*e)
        quantile_loss=(pinball(pred_q01,targ_q01,q_low)+pinball(pred_q99,targ_q99,q_high)).nanmean()
        P_pred=self._power_spectrum(pred_x_start,mask);P_true=self._power_spectrum(x_start,mask);spectral_loss=F.smooth_l1_loss(P_pred,P_true,beta=0.25)
        if mask_diff is not None:
            pred_cumulative=(pred_diff*mask_diff).cumsum(dim=-1);target_cumulative=(target_diff*mask_diff).cumsum(dim=-1);valid_lengths=mask_diff.sum(dim=-1).long().clamp(min=1)-1;batch_indices=torch.arange(pred_cumulative.size(0),device=pred_cumulative.device)
            if valid_lengths.ndim<pred_cumulative.ndim-1:valid_lengths=valid_lengths.unsqueeze(1).expand(-1,pred_cumulative.size(1))
            pred_final=pred_cumulative.gather(-1,valid_lengths.unsqueeze(-1)).squeeze(-1);target_final=target_cumulative.gather(-1,valid_lengths.unsqueeze(-1)).squeeze(-1);drift_loss=F.mse_loss(pred_final,target_final)
        else:
            pred_cumulative=pred_diff.cumsum(dim=-1);target_cumulative=target_diff.cumsum(dim=-1);drift_loss=F.mse_loss(pred_cumulative[...,-1],target_cumulative[...,-1])
        # --- End custom losses ---

        # --- Annealed weights and EMA update (Original logic) ---
        weights = self._get_annealed_weights(global_step, device=device)
        loss_components = {'relative_jump': relative_jump_loss, 'vol_clustering': vol_clustering_loss, 'global_vol': global_vol_loss, 'heavy_tail': heavy_tail_loss, 'skewness': skewness_loss, 'drift': drift_loss, 'quantile': quantile_loss, 'spectral': spectral_loss}
        normalized_losses = {}
        for name, loss_tensor in loss_components.items():
            with torch.no_grad():
                ema_buffer = getattr(self, f'ema_{name}'); current_loss_val = loss_tensor.detach()
                if ema_buffer == 0: ema_buffer.copy_(current_loss_val)
                else: ema_buffer.copy_(ema_buffer * self.ema_beta + current_loss_val * (1 - self.ema_beta))
            normalized_loss = loss_tensor / (ema_buffer + eps)
            normalized_losses[name] = normalized_loss
        # ---

        # --- Combine losses (Original logic) ---
        total_loss = 1 * mse_loss # Base MSE loss weight
        log_str = f"Step: {global_step} | MSE: {mse_loss:.4f}"
        for name, loss_tensor in normalized_losses.items():
            weight = weights.get(name, 0.0)
            if weight > 0 and not torch.isnan(loss_tensor) and not torch.isinf(loss_tensor):
                total_loss += weight * loss_tensor
                log_str += f" | {name}(N): {loss_tensor:.3f}(w:{weight:.2f})"

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"⚠️ Warning: total_loss is NaN or Inf at step {global_step}. Components: { {k: v.item() for k, v in loss_components.items()} }. Returning only mse_loss.")
            return mse_loss # Fallback to prevent crash

        if global_step % 100 == 0: print(log_str + f" | Total: {total_loss:.4f}")
        # ---

        return total_loss

    def forward(self, img, cond_input = None, mask = None, global_step=0, *args, **kwargs):
        b, n, device, seq_length, = img.shape[0], img.shape[self.seq_index], img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        
        if exists(cond_input):
            kwargs['cond_input'] = cond_input
        if exists(mask):
            kwargs['mask'] = mask
        
        # 将 global_step 传递给 p_losses
        kwargs['global_step'] = global_step
            
        return self.p_losses(img, t, *args, **kwargs)