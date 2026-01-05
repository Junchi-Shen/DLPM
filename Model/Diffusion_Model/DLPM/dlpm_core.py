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

    def anterior_mean_variance_dlim(self, x_t, t, eps, eta=0.0):
        """计算DLIM（DDIM版本）的后验均值和方差"""
        # 确保t是批次张量
        t_batch = self.get_t_to_batch_size(x_t, t)
        
        # 获取时间步值
        if isinstance(t, int):
            t_val = t
        elif isinstance(t, torch.Tensor):
            if t.numel() == 1:
                t_val = int(t.item())
            else:
                t_val = int(t_batch[0].item())
        else:
            t_val = int(t)
        
        # 确保t_val在有效范围内
        t_val = max(1, min(t_val, len(self.gammas) - 1))
        
        # 获取当前时间步的调度值
        try:
            g, bg, s, bs = self.get_schedule_at_t(t, x_t.shape)
        except Exception as e:
            print(f"错误: 获取调度值时出错, t={t_val}, 错误: {e}")
            # 回退：直接使用索引
            t_idx = min(t_val, len(self.gammas) - 1)
            g = self.gammas[t_idx].item()
            bg = self.bargammas[t_idx].item()
            s = self.sigmas[t_idx].item()
            bs = self.barsigmas[t_idx].item()
            # 扩展到x_t的形状
            g = torch.full_like(x_t, g)
            bg = torch.full_like(x_t, bg)
            s = torch.full_like(x_t, s)
            bs = torch.full_like(x_t, bs)
        
        # 获取前一个时间步的调度值
        try:
            if t_val > 0:
                g_prev, bg_prev, s_prev, bs_prev = self.get_schedule_at_t(t_val - 1, x_t.shape)
            else:
                g_prev, bg_prev, s_prev, bs_prev = self.get_schedule_at_t(0, x_t.shape)
        except Exception as e:
            print(f"错误: 获取前一时间步调度值时出错, t={t_val}, 错误: {e}")
            # 回退
            t_prev_idx = max(0, t_val - 1)
            g_prev = self.gammas[t_prev_idx].item()
            bg_prev = self.bargammas[t_prev_idx].item()
            s_prev = self.sigmas[t_prev_idx].item()
            bs_prev = self.barsigmas[t_prev_idx].item()
            g_prev = torch.full_like(x_t, g_prev)
            bg_prev = torch.full_like(x_t, bg_prev)
            s_prev = torch.full_like(x_t, s_prev)
            bs_prev = torch.full_like(x_t, bs_prev)
        
        # 计算nonzero_mask（t != 1时添加噪声）
        nonzero_mask = ((t_batch != 1).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        # 添加数值稳定性检查
        eps = torch.clamp(eps, min=-1e6, max=1e6)  # 防止eps过大
        g = torch.clamp(g, min=1e-8)  # 防止除零
        bs = torch.clamp(bs, min=1e-8)
        bs_prev = torch.clamp(bs_prev, min=1e-8)
        
        if eta == 0.0:
            # 确定性采样（完全DDIM）
            # 公式: x_{t-1} = (x_t - bs[t]*eps) / g[t] + bs[t-1]*eps
            # 注意：这里bs和g已经是提取的值，形状匹配x_t
            sample = (x_t - bs*eps) / (g + 1e-8) + bs_prev*eps
            # 检查NaN和Inf
            if torch.isnan(sample).any() or torch.isinf(sample).any():
                print(f"警告: DDIM采样中出现NaN/Inf, t={t_val}, 使用x_t作为回退")
                sample = torch.where(torch.isnan(sample) | torch.isinf(sample), x_t, sample)
            return sample, torch.zeros_like(x_t)
        
        # 随机采样（eta > 0）
        sigma_t = eta * bs_prev
        
        # 计算均值
        sample = (x_t - bs*eps) / (g + 1e-8)
        
        # 计算第二项，需要处理数值稳定性
        bs_prev_alpha = torch.clamp(bs_prev, min=1e-8)**(self.alpha)
        sigma_t_alpha = torch.clamp(sigma_t, min=0.0)**(self.alpha)
        diff_alpha = bs_prev_alpha - sigma_t_alpha
        
        # 确保diff_alpha >= 0（数值稳定性）
        diff_alpha = torch.clamp(diff_alpha, min=1e-10)
        diff_term = diff_alpha**(1 / self.alpha)
        
        sample = sample + diff_term * eps
        mean = sample
        
        # 检查NaN和Inf
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"警告: DDIM采样均值中出现NaN/Inf, t={t_val}, eta={eta}, 使用x_t作为回退")
            mean = torch.where(torch.isnan(mean) | torch.isinf(mean), x_t, mean)
        
        # 计算方差（需要A[t]）
        if self.A is not None and len(self.A) > t_val:
            A_t = self.A[t_val]
            # 处理A_t的形状
            while len(A_t.shape) > len(x_t.shape):
                if A_t.shape[0] == 1:
                    A_t = A_t.squeeze(0)
                else:
                    A_t = A_t[0]
            # 确保形状匹配
            if A_t.shape != x_t.shape:
                if A_t.shape[0] == 1:
                    A_t = A_t.expand(x_t.shape)
                else:
                    A_t = A_t[:x_t.shape[0]]
            # 确保A_t >= 0
            A_t = torch.clamp(A_t, min=1e-10)
            variance = nonzero_mask * sigma_t**2 * A_t
        else:
            variance = nonzero_mask * sigma_t**2
        
        # 确保方差非负
        variance = torch.clamp(variance, min=0.0)
        
        return mean, variance

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
        x_t_1 = (x_t - bs*Gamma_t*eps) / g
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

