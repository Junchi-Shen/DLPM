# Lévy分布生成函数
# 从DLPM项目适配而来

import numpy as np
import torch
import scipy.stats


def match_last_dims(data, size):
    """
    Expands a 1-dimensional tensor so that its last dimensions match the target size.
    
    Args:
        data (torch.Tensor): A 1-dimensional tensor with shape [batch_size].
        size (tuple or list): The target size, where size[0] should match batch_size.
    
    Returns:
        torch.Tensor: The expanded tensor with shape `size`.
    """
    assert data.dim() == 1, f"Data must be 1-dimensional, got {data.size()}"
    
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    
    return data.expand(*size).contiguous()


def gen_skewed_levy(alpha, size, device=None, isotropic=True, clamp_a=None):
    """
    生成偏斜Lévy随机变量
    
    Args:
        alpha: 稳定分布的参数 (0 < alpha <= 2)
        size: 输出张量的形状
        device: 设备
        isotropic: 是否各向同性
        clamp_a: 裁剪值
    """
    if alpha > 2.0 or alpha <= 0.:
        raise Exception(f'Wrong value of alpha ({alpha}) for skewed levy r.v generation')
    if alpha == 2.0:
        ret = 2 * torch.ones(size)
        return ret if device is None else ret.to(device)
    
    # 生成 alpha/2, 1, 0, 2*cos(pi*alpha/4)^(2/alpha) 的稳定分布
    # Vectorized Chambers--Mallows--Stuck sampling avoids the severe scipy
    # bottleneck when coordinate-wise mixing requests thousands of draws.
    # The factor 2 matches the scale convention of the previous scipy call.
    sample_size = (size[0],) if isotropic else tuple(size)
    target_device = device if device is not None else torch.device("cpu")
    gamma = float(alpha) / 2.0
    u = (torch.rand(sample_size, device=target_device) - 0.5) * np.pi
    w = torch.empty(sample_size, device=target_device).exponential_(1.0)
    numerator = torch.sin(gamma * (u + np.pi / 2.0))
    denominator = torch.cos(u).clamp_min(1e-7).pow(1.0 / gamma)
    inner = torch.cos(u - gamma * (u + np.pi / 2.0)).clamp_min(1e-7)
    ret = 2.0 * (numerator / denominator) * (inner / w.clamp_min(1e-7)).pow((1.0 - gamma) / gamma)
    ret = ret.to(dtype=torch.float32)
    if isotropic:
        ret = match_last_dims(ret, size)
    
    if clamp_a is not None:
        ret = torch.clamp(ret, 0., clamp_a)
    
    return ret if device is None else ret.to(device)


def gen_sas(alpha, size, a=None, device=None, isotropic=True, clamp_eps=None):
    """
    生成对称alpha稳定噪声 (symmetric alpha stable noise)
    
    Args:
        alpha: 稳定分布的参数
        size: 输出张量的形状
        a: 可选的偏斜Lévy变量
        device: 设备
        isotropic: 是否各向同性
        clamp_eps: 裁剪值
    """
    if a is None:
        a = gen_skewed_levy(alpha, size, device=device, isotropic=isotropic)
    
    ret = torch.randn(size=size, device=device)
    ret = torch.sqrt(a) * ret
    
    if clamp_eps is not None:
        ret = torch.clamp(ret, -clamp_eps, clamp_eps)
    
    return ret


class Generator:
    """
    Lévy分布生成器包装类
    """
    def __init__(self, operation, device=None, **kwargs):
        self.device = device
        self.kwargs = kwargs
        self.operation = operation
        
        self.available_operations = {
            'skewed_levy': gen_skewed_levy,
            'sas': gen_sas,
        }
        
        if operation not in self.available_operations:
            raise Exception(f'Unknown operation: {operation}. Available: {list(self.available_operations.keys())}')
        
        self.generator = self.available_operations[operation]
    
    def setParams(self, **kwargs):
        """更新参数"""
        self.kwargs.update(kwargs)
    
    def generate(self, size, **kwargs):
        """生成样本"""
        tmp_kwargs = {**self.kwargs, **kwargs}
        if 'device' not in tmp_kwargs:
            tmp_kwargs['device'] = self.device
        return self.generator(size=size, **tmp_kwargs)
