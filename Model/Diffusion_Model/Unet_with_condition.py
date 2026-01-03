# model.py

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange

# 从我们自己的 utils 文件导入辅助工具
from .Utils import (
    exists,
    default,
    identity,
    Residual,
    Upsample,
    Downsample,
    PreNorm,
    RMSNorm
)


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, cond_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        # 添加条件嵌入MLP
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, dim_out * 2)
        ) if exists(cond_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, cond_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            
            # 如果有条件嵌入，将其与时间嵌入融合
            if exists(self.cond_mlp) and exists(cond_emb):
                cond_emb = self.cond_mlp(cond_emb)
                cond_emb = rearrange(cond_emb, 'b c -> b c 1')
                # 将时间和条件嵌入相加
                combined_emb = time_emb + cond_emb
            else:
                combined_emb = time_emb
                
            scale_shift = combined_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        cond_dim = None  # 添加条件维度参数
    ): 
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.cond_dim = cond_dim  # 保存条件维度
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 条件嵌入层 - 为每个分辨率层创建条件嵌入
        self.cond_mlp = None
        self.context_embs = None
        if exists(cond_dim):
            # 条件的全局嵌入
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
            # 条件嵌入直接通过ResnetBlock处理，不需要额外的context_embs

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, cond_emb_dim = time_dim if exists(cond_dim) else None, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, y, time, y_self_cond = None, cond_input = None):
        if self.self_condition:
            y_self_cond = default(y_self_cond, lambda: torch.zeros_like(y))
            y = torch.cat((y_self_cond, y), dim = 1)
        
        y = self.init_conv(y)
        r = y.clone()

        t = self.time_mlp(time)

        # 处理条件输入
        c = None
        if exists(cond_input) and exists(self.cond_mlp):
            c = self.cond_mlp(cond_input)  # (batch_size, time_dim)

        h = []

        for block1, block2, attn, downsample in self.downs:
            y = block1(y, t, c)  # 传递条件嵌入
            h.append(y)

            y = block2(y, t, c)  # 传递条件嵌入
            y = attn(y)
            h.append(y)

            y = downsample(y)

        y = self.mid_block1(y, t, c)  # 传递条件嵌入
        y = self.mid_attn(y)
        y = self.mid_block2(y, t, c)  # 传递条件嵌入

        for block1, block2, attn, upsample in self.ups:
            skip = h.pop()
            # 处理维度不匹配
            if y.shape[-1] != skip.shape[-1]:
                target_size = max(y.shape[-1], skip.shape[-1])
                if y.shape[-1] != target_size:
                    y = F.interpolate(y, size=target_size, mode='linear', align_corners=False)
                if skip.shape[-1] != target_size:
                    skip = F.interpolate(skip, size=target_size, mode='linear', align_corners=False)
            y = torch.cat((y, skip), dim = 1)
            y = block1(y, t, c)  # 传递条件嵌入

            skip = h.pop()
            # 处理维度不匹配
            if y.shape[-1] != skip.shape[-1]:
                target_size = max(y.shape[-1], skip.shape[-1])
                if y.shape[-1] != target_size:
                    y = F.interpolate(y, size=target_size, mode='linear', align_corners=False)
                if skip.shape[-1] != target_size:
                    skip = F.interpolate(skip, size=target_size, mode='linear', align_corners=False)
            y = torch.cat((y, skip), dim = 1)
            y = block2(y, t, c)  # 传递条件嵌入
            y = attn(y)

            y = upsample(y)

        # 处理与初始特征的维度不匹配
        if y.shape[-1] != r.shape[-1]:
            target_size = max(y.shape[-1], r.shape[-1])
            if y.shape[-1] != target_size:
                y = F.interpolate(y, size=target_size, mode='linear', align_corners=False)
            if r.shape[-1] != target_size:
                r = F.interpolate(r, size=target_size, mode='linear', align_corners=False)
        y = torch.cat((y, r), dim = 1)

        y = self.final_res_block(y, t, c)  # 传递条件嵌入
        return self.final_conv(y)
