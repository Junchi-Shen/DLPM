# DLPM核心实现
# 从DLPM项目适配而来，简化版本

import torch
from .levy_distributions import gen_skewed_levy, gen_sas, Generator, match_last_dims


class ModelMeanType:
    """模型预测类型"""
    PREVIOUS_X = 'PREVIOUS_X'
    START_X = 'START_X'
    EPSILON = 'EPSILON'
    Z = 'Z'
    SQRT_GAMMA_EPSILON = 'SQRT_GAMMA_EPSILON'


class ModelVarType:
    """模型方差类型"""
    FIXED = 'FIXED'


class LossType:
    """损失类型"""
    LP_LOSS = 'LP_LOSS'
    MEAN_LOSS = 'MEAN_LOSS'
    EPS_LOSS = 'EPS_LOSS'
    LAMBDA_LOSS = 'LAMBDA_LOSS'
    VAR_KL = 'VAR_KL'
    VAR_LP_SUM = 'VAR_LP_SUM'


class DLPM:
    """
    Denoising Lévy Probabilistic Model 核心类
    """
    def __init__(
        self,
        alpha,
        device,
        diffusion_steps,
        time_spacing='linear',
        isotropic=True,
        clamp_a=None,
        clamp_eps=None,
        scale='scale_preserving'
    ):
        self.alpha = alpha
        self.device = device
        self.time_spacing = time_spacing
        self.isotropic = isotropic
        self.use_single_a_chain = True
        self.scale = scale

        # 生成噪声调度
        self.gammas, self.bargammas, self.sigmas, self.barsigmas = \
            (x.to(self.device) for x in self.gen_noise_schedule(diffusion_steps, scale=self.scale))
        
        self.constants = None

        # 生成器
        self.gen_a = Generator('skewed_levy', 
                               alpha=self.alpha, 
                               device=self.device,
                               isotropic=isotropic,
                               clamp_a=clamp_a)
        
        self.gen_eps = Generator('sas',
                                alpha=self.alpha, 
                                device=self.device,
                                isotropic=isotropic,
                                clamp_eps=clamp_eps)
        
        self.A = None
        self.Sigmas = None

    def get_timesteps(self, steps):
        """获取时间步"""
        if self.time_spacing == 'linear':
            timesteps = torch.tensor(range(0, steps), dtype=torch.float32)
        elif self.time_spacing == 'quadratic':
            timesteps = steps * (torch.tensor(range(0, steps), dtype=torch.float32) / steps)**2
        else:
            raise NotImplementedError(self.time_spacing)
        return timesteps

    def gen_noise_schedule(self, diffusion_steps, scale='scale_preserving'):
        """生成噪声调度"""
        if scale == 'scale_preserving':
            s = 0.008
            timesteps = self.get_timesteps(diffusion_steps)

            schedule = torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

            baralphas = schedule / schedule[0]
            betas = 1 - baralphas / torch.cat([baralphas[0:1], baralphas[0:-1]])
            alphas = 1 - betas

            # gamma的线性调度
            gammas = alphas**(1/self.alpha)
            bargammas = torch.cumprod(gammas, dim=0)

            # 保尺度调度
            sigmas = (1 - gammas**(self.alpha))**(1/self.alpha)
            barsigmas = (1 - bargammas**(self.alpha))**(1/self.alpha)
        else:
            raise NotImplementedError(f'Unknown scale: {scale}')
        
        return gammas, bargammas, sigmas, barsigmas

    def extract(self, a, t, x_shape):
        """从调度中提取对应时间步的值"""
        # 处理t可能是标量或张量的情况
        if isinstance(t, int):
            t = torch.tensor([t], device=a.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=a.device)
        elif t.numel() == 0:
            # 如果t是空张量，返回默认值
            t = torch.tensor([0], device=a.device)
        
        # 确保t是1维张量
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        b = t.shape[0] if len(t.shape) > 0 else 1
        # 确保t的值在有效范围内
        t = t.clamp(0, len(a) - 1)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def get_t_to_batch_size(self, x_t, t):
        """将t转换为批次大小"""
        if isinstance(t, int):
            return torch.full([x_t.shape[0]], t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            if t.dim() == 0:
                # 标量张量，扩展为批次
                return torch.full([x_t.shape[0]], t.item(), device=self.device, dtype=torch.long)
            elif t.numel() == 0:
                # 空张量，返回默认值
                return torch.full([x_t.shape[0]], 0, device=self.device, dtype=torch.long)
            else:
                return t.to(self.device)
        else:
            # 其他类型，尝试转换
            return torch.full([x_t.shape[0]], int(t), device=self.device, dtype=torch.long)
    
    def get_schedule_at_t(self, t, x_shape):
        """获取指定时间步的调度值"""
        # x_shape可能是张量或形状元组，需要获取批次大小
        if isinstance(x_shape, torch.Tensor):
            batch_size = x_shape.shape[0]
            actual_shape = x_shape.shape
        elif isinstance(x_shape, (tuple, list)):
            batch_size = x_shape[0]
            actual_shape = tuple(x_shape)
        else:
            batch_size = 1
            actual_shape = (1, 1, 1)
        
        # 创建一个临时张量用于get_t_to_batch_size
        if isinstance(x_shape, torch.Tensor):
            temp_tensor = x_shape
        else:
            # 创建一个临时张量
            temp_tensor = torch.empty(actual_shape, device=self.device)
        
        t_batch = self.get_t_to_batch_size(temp_tensor, t)
        
        # 确保t_batch是有效的
        if t_batch.numel() == 0:
            t_batch = torch.tensor([0], device=self.device)
        
        # 确保t_batch的值在有效范围内
        t_batch = t_batch.clamp(0, len(self.gammas) - 1)
        
        g = self.extract(self.gammas, t_batch, actual_shape)
        bg = self.extract(self.bargammas, t_batch, actual_shape)
        s = self.extract(self.sigmas, t_batch, actual_shape)
        bs = self.extract(self.barsigmas, t_batch, actual_shape)
        return g, bg, s, bs
    
    def rescale_diffusion(self, diffusion_steps, time_spacing=None):
        """重新缩放扩散步数"""
        assert isinstance(diffusion_steps, int), "Diffusion steps must be an integer"
        if time_spacing is not None:
            self.time_spacing = time_spacing
        self.gammas, self.bargammas, self.sigmas, self.barsigmas = \
            (x.to(self.device) for x in self.gen_noise_schedule(diffusion_steps))
        self.constants = None

    def predict_xstart(self, x_t, t, eps):
        """从x_t和eps预测x_start"""
        assert x_t.shape == eps.shape
        g, bg, s, bs = self.get_schedule_at_t(t, x_t.shape)
        # 添加数值稳定性：防止除零和NaN
        bg = torch.clamp(bg, min=1e-8)
        xstart = (x_t - eps*bs) / bg
        # 检查并处理NaN/Inf
        if torch.isnan(xstart).any() or torch.isinf(xstart).any():
            # 如果bg接近0，xstart应该接近x_t
            xstart = torch.where(torch.isnan(xstart) | torch.isinf(xstart), x_t, xstart)
        return xstart 

    def predict_eps(self, x_t, t, xstart):
        """从x_t和xstart预测eps"""
        g, bg, s, bs = self.get_schedule_at_t(t, x_t.shape)
        # 添加数值稳定性：防止除零和NaN
        bs = torch.clamp(bs, min=1e-8)
        eps = (x_t - xstart * bg) / bs
        # 检查并处理NaN/Inf
        if torch.isnan(eps).any() or torch.isinf(eps).any():
            # 如果bs接近0，eps应该接近0
            eps = torch.where(torch.isnan(eps) | torch.isinf(eps), torch.zeros_like(eps), eps)
        return eps
    
    def sample_x_t_from_xstart(self, xstart, t, eps=None):
        """从xstart采样x_t"""
        if eps is None:
            eps = self.gen_eps.generate(size=xstart.size())
        g, bg, s, bs = self.get_schedule_at_t(t, xstart.shape)
        x_t = bg*xstart + bs*eps
        return x_t, eps

    def sample_A(self, shape, diffusion_steps):
        """采样A_{1:T}序列"""
        self.A = torch.stack([self.gen_a.generate(size=shape) for i in range(diffusion_steps)])

    def compute_Sigmas(self):
        """计算Sigma_t序列"""
        # 获取A的形状（不包括时间步维度）
        A_shape = self.A[0].shape  # (batch_size, ...)
        
        # 初始化Sigmas列表
        # 对于t=0，使用s[0]和A[0]
        s_0 = self.sigmas[0]
        Sigmas = [s_0**2 * self.A[0]]
        
        # 对于后续时间步
        for t_idx in range(1, self.A.shape[0]):
            A_t = self.A[t_idx]
            g_t = self.gammas[t_idx]
            s_t = self.sigmas[t_idx]
            # Sigma_t = s_t^2 * A_t + g_t^2 * Sigma_{t-1}
            Sigmas.append(s_t**2 * A_t + g_t**2 * Sigmas[-1])
        
        self.Sigmas = torch.stack(Sigmas)

    def sample_x_t_from_xstart_given_Sigma(self, xstart, t, Sigma_t, z_t=None):
        """给定Sigma从xstart采样x_t"""
        g, bg, s, bs = self.get_schedule_at_t(t, xstart.shape)
        if z_t is None:
            z_t = torch.randn_like(xstart)
        x_t = bg*xstart + Sigma_t**(1/2)*z_t
        return x_t
    
    def compute_Gamma_t(self, t, Sigma_t_1, Sigma_t):
        """计算Gamma_t"""
        g, bg, s, bs = self.get_schedule_at_t(t, Sigma_t_1.shape)
        Gamma_t = 1 - (g**2 * Sigma_t_1) / Sigma_t
        return Gamma_t
    
    def compute_Sigma_tilde_t_1(self, Gamma_t, Sigma_t_1):
        """计算Sigma_tilde_{t-1}"""
        return Gamma_t * Sigma_t_1
    
    def compute_m_tilde_t_1(self, x_t, t, Gamma_t, eps_t):
        """计算m_tilde_{t-1}"""
        g, bg, s, bs = self.get_schedule_at_t(t, x_t.shape)
        m_tilde_t_1 = (x_t - bs*Gamma_t*eps_t) / g
        return m_tilde_t_1

    def anterior_mean_variance_dlim(self, x_t, t, eps, t_prev=None, eta=0.0):
        """DLIM (deterministic, DDIM-analogue) update from step t to step t_prev.

        x_t = bg_t * x0 + bs_t * eps  =>  x0_hat = (x_t - bs_t * eps) / bg_t
        x_{t_prev} = bg_{t_prev} * x0_hat + bs_{t_prev} * eps
                   = (bg_{t_prev}/bg_t) * (x_t - bs_t * eps) + bs_{t_prev} * eps

        t_prev is the schedule index of the NEXT state in the (possibly
        skipping) sampling subsequence; defaults to t-1 (adjacent step).
        """
        t_batch = self.get_t_to_batch_size(x_t, t)

        if isinstance(t, int):
            t_val = t
        elif isinstance(t, torch.Tensor):
            t_val = int(t.reshape(-1)[0].item())
        else:
            t_val = int(t)
        t_val = max(1, min(t_val, len(self.gammas) - 1))

        tp_val = t_val - 1 if t_prev is None else int(t_prev)
        tp_val = max(0, min(tp_val, t_val - 1))

        g, bg, s, bs = self.get_schedule_at_t(t_val, x_t.shape)
        g_p, bg_p, s_p, bs_p = self.get_schedule_at_t(tp_val, x_t.shape)

        ratio = bg_p / bg.clamp(min=1e-12)

        if eta == 0.0:
            sample = ratio * (x_t - bs * eps) + bs_p * eps
            if not torch.isfinite(sample).all():
                print(f"警告: DLIM采样出现NaN/Inf, t={t_val}->{tp_val}, 使用x_t回退")
                sample = torch.where(torch.isfinite(sample), sample, x_t)
            return sample, torch.zeros_like(x_t)

        # stochastic DLIM (eta > 0)
        sigma_t = eta * bs_p
        diff_alpha = torch.clamp(bs_p ** self.alpha - torch.clamp(sigma_t, min=0.0) ** self.alpha, min=0.0)
        diff_term = diff_alpha ** (1.0 / self.alpha)
        mean = ratio * (x_t - bs * eps) + diff_term * eps
        if not torch.isfinite(mean).all():
            print(f"警告: DLIM均值出现NaN/Inf, t={t_val}->{tp_val}, 使用x_t回退")
            mean = torch.where(torch.isfinite(mean), mean, x_t)

        nonzero_mask = ((t_batch != 1).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        if self.A is not None and len(self.A) > t_val and self.A[t_val].shape == x_t.shape:
            A_t = torch.clamp(self.A[t_val], min=1e-10)
            variance = nonzero_mask * sigma_t ** 2 * A_t
        else:
            variance = nonzero_mask * sigma_t ** 2
        return mean, torch.clamp(variance, min=0.0)

    def anterior_mean_variance_dlpm(self, x_t, t, eps):
        """计算DLPM的后验均值和方差"""
        # 确保t是批次张量
        t_batch = self.get_t_to_batch_size(x_t, t)
        
        # 获取时间步值（假设批次中所有样本在同一时间步，训练时通常如此）
        if isinstance(t, int):
            t_val = t
        elif isinstance(t, torch.Tensor):
            if t.numel() == 1:
                t_val = int(t.item())
            else:
                # 取第一个值（假设批次中所有样本在同一时间步）
                t_val = int(t_batch[0].item())
        else:
            t_val = int(t)
        
        # 确保t_val在有效范围内
        t_val = max(1, min(t_val, len(self.Sigmas) - 1))
        
        # 获取对应时间步的Sigma
        # self.Sigmas的形状是 [T, ...]，需要索引时间步维度
        if t_val > 0:
            Sigma_t_1_val = self.Sigmas[t_val-1]
        else:
            Sigma_t_1_val = self.Sigmas[0]
        Sigma_t_val = self.Sigmas[t_val]
        
        # 确保Sigma的形状与x_t匹配
        batch_size = x_t.shape[0]
        x_shape = x_t.shape  # [batch_size, channels, seq_length]
        
        # 简化处理：去除所有大小为1的前导维度，直到形状匹配
        # 如果Sigma是 [1, batch_size, channels, seq_length]，去除第一维得到 [batch_size, channels, seq_length]
        while len(Sigma_t_1_val.shape) > len(x_shape):
            if Sigma_t_1_val.shape[0] == 1:
                Sigma_t_1_val = Sigma_t_1_val.squeeze(0)
            else:
                # 如果第一维不是1，检查是否有batch_size维度
                if Sigma_t_1_val.shape[1] == batch_size:
                    # 形状是 [1, batch_size, ...]，去除第一维
                    Sigma_t_1_val = Sigma_t_1_val.squeeze(0)
                else:
                    # 其他情况，取第一个元素
                    Sigma_t_1_val = Sigma_t_1_val[0]
        
        # 如果形状仍然不匹配，尝试重塑
        if Sigma_t_1_val.shape != x_shape:
            # 如果第一维是batch_size，直接重塑
            if Sigma_t_1_val.shape[0] == batch_size:
                Sigma_t_1_val = Sigma_t_1_val.view(x_shape)
            else:
                # 否则，扩展到batch_size
                Sigma_t_1_val = Sigma_t_1_val.unsqueeze(0).expand(x_shape)
        
        # 对Sigma_t_val做同样的处理
        while len(Sigma_t_val.shape) > len(x_shape):
            if Sigma_t_val.shape[0] == 1:
                Sigma_t_val = Sigma_t_val.squeeze(0)
            else:
                if Sigma_t_val.shape[1] == batch_size:
                    Sigma_t_val = Sigma_t_val.squeeze(0)
                else:
                    Sigma_t_val = Sigma_t_val[0]
        
        if Sigma_t_val.shape != x_shape:
            if Sigma_t_val.shape[0] == batch_size:
                Sigma_t_val = Sigma_t_val.view(x_shape)
            else:
                Sigma_t_val = Sigma_t_val.unsqueeze(0).expand(x_shape)
        
        Gamma_t = self.compute_Gamma_t(t, Sigma_t_1_val, Sigma_t_val)
        g, bg, s, bs = self.get_schedule_at_t(t, x_t.shape)
        x_t_1 = (x_t - bs*Gamma_t*eps) / g.clamp(min=1e-8)
        Sigma_t_1 = self.compute_Sigma_tilde_t_1(Gamma_t, Sigma_t_1_val)
        return x_t_1, Sigma_t_1

    def get_one_rv_faster_sampling(self, shape):
        """快速采样单个随机变量a_t"""
        return self.gen_a.generate(size=shape)

    def compute_one_rv_Sigma_prime_t(self, t, a_t):
        """计算单个随机变量的Sigma_prime_t"""
        g, bg, s, bs = self.get_schedule_at_t(t, a_t.shape)
        Sigma_prime_t = a_t * bs**2 
        return Sigma_prime_t

    def get_one_rv_loss_elements(self, t, x_0, a_t=None, z_t=None):
        """获取单个随机变量的损失元素"""
        if a_t is None:
            a_t = self.get_one_rv_faster_sampling(x_0.shape)
        Sigma_prime_t = self.compute_one_rv_Sigma_prime_t(t, a_t)
        x_t = self.sample_x_t_from_xstart_given_Sigma(x_0, t, Sigma_prime_t, z_t=z_t)
        eps_t = self.predict_eps(x_t, t, x_0)
        return x_t, eps_t

