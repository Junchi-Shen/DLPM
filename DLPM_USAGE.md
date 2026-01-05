# DLPM使用指南

## 概述

DLPM (Denoising Lévy Probabilistic Model) 已成功集成到现有系统中。DLPM使用Lévy稳定分布替代高斯噪声，特别适合处理重尾数据分布或类别不平衡的数据集。

## 主要特性

- **重尾噪声分布**: 使用α稳定分布（1 < α ≤ 2）替代高斯噪声
- **向后兼容**: 与现有的GaussianDiffusion1D接口兼容
- **条件支持**: 支持条件网络和条件输入
- **灵活配置**: 可通过配置文件轻松切换标准DDPM和DLPM

## 使用方法

### 1. 训练DLPM模型

在 `Config/Diffusion_config.py` 中设置：

```python
main_config = {
    # ... 其他配置 ...
    
    # DLPM参数
    'use_dlpm': True,  # 启用DLPM
    'dlpm_alpha': 1.7,  # Lévy稳定分布的alpha参数 (1 < alpha <= 2)
    'dlpm_isotropic': True,  # 是否各向同性
    'dlpm_rescale_timesteps': True,  # 是否重新缩放时间步
    'dlpm_scale': 'scale_preserving',  # 调度类型
}
```

然后运行训练：

```bash
python Pipelines/4-Run_Diffusion.py
```

### 2. 使用DLPM生成路径

在 `Config/generator_config.py` 中，已添加DLPM配置：

```python
GENERATOR_JOBS = {
    # ... 其他作业 ...
    
    "DLPM": {
        "type": "diffusion",
        "job_name": "DLPM Path Generation",
        "output_filename_base": "dlpm_generated_paths",
        "use_dlpm": True,  # 标记使用DLPM
        "dlpm_alpha": 1.7,  # DLPM的alpha参数
        
        # ... 其他配置与UNet相同 ...
    }
}
```

在 `Pipelines/6-Run_generater.py` 中添加DLPM作业：

```python
JOBS_TO_RUN = [
    {'job_name': 'DLPM', 'asset': 'CSI1000'},  # 使用DLPM
    # ... 其他作业 ...
]
```

然后运行：

```bash
python Pipelines/6-Run_generater.py
```

### 3. Alpha参数选择

- **alpha = 2.0**: 退化为标准高斯扩散（DDPM）
- **alpha = 1.7-1.9**: 推荐用于金融数据（重尾分布）
- **alpha < 1.5**: 可能过于重尾，训练不稳定

根据论文，alpha=1.7在大多数情况下表现良好。

## 文件结构

新增的DLPM相关文件：

```
Model/Diffusion_Model/
├── DLPM/
│   ├── __init__.py
│   ├── dlpm_core.py          # DLPM核心实现
│   ├── generative_levy_process.py  # 生成过程包装器
│   └── levy_distributions.py      # Lévy分布生成函数
└── diffusion_dlpm.py         # DLPM适配器（与现有系统兼容）
```

## 技术细节

### DLPM vs DDPM的主要区别

1. **噪声分布**: DLPM使用α稳定分布，DDPM使用高斯分布
2. **采样过程**: DLPM需要采样A_{1:T}序列并计算Sigmas
3. **训练损失**: DLPM使用Proposition (9)的简化损失

### 与现有系统的兼容性

- ✅ 支持条件网络（EnhancedConditionNetwork）
- ✅ 支持U-Net模型
- ✅ 兼容现有的训练器（Trainer1D）
- ✅ 兼容现有的路径生成器

## 注意事项

1. **依赖**: 需要安装`scipy`库用于生成Lévy稳定分布
2. **训练时间**: DLPM的训练时间可能与DDPM相近或略长
3. **内存**: DLPM在采样时需要存储A和Sigmas序列，内存占用可能略高
4. **数值稳定性**: 当alpha接近1时，可能出现数值不稳定，建议alpha >= 1.5

## 参考文献

- DLPM论文: [Denoising Lévy Probabilistic Models](https://arxiv.org/abs/2407.18609)
- 原始DLPM代码: `d:\Essay\DLPM\DLPM`

## 故障排除

### 问题1: 导入错误
**解决方案**: 确保所有DLPM相关文件都在正确的位置，检查`Model/Diffusion_Model/DLPM/__init__.py`

### 问题2: 训练损失为NaN
**解决方案**: 
- 检查alpha值是否在合理范围内（1.5-2.0）
- 减小学习率
- 检查数据是否已正确归一化

### 问题3: 采样速度慢
**解决方案**: 
- 减少`sampling_timesteps`
- 减少`num_paths_to_generate`
- 使用GPU加速

## 示例配置

完整的DLPM训练配置示例：

```python
# Config/Diffusion_config.py
main_config = {
    "underlying_asset": 'all',
    "model_type": 'unet',
    "use_dlpm": True,  # 启用DLPM
    "dlpm_alpha": 1.7,
    "dlpm_isotropic": True,
    "dlpm_rescale_timesteps": True,
    "dlpm_scale": 'scale_preserving',
    
    "timesteps": 1000,
    "objective": 'pred_v',
    "auto_normalize": False,
    "seq_length": 252,
    
    "train_num_steps": 18000,
    "train_batch_size": 64,
    "train_lr": 1e-6,
    
    "unet_params": {
        "dim": 64,
        "dim_mults": (1, 2, 4, 8),
        "channels": 1,
        "dropout": 0.1,
    },
    
    "use_enhanced_condition_network": True,
    "cond_net_params": {
        "output_dim": 128,
        "country_emb_dim": 64,
        "index_emb_dim": 128,
        "numerical_proj_dim": 32,
        "hidden_dim": 256
    },
}
```

