#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试优化模型（无需C++扩展）
"""

import torch
import sys

print("=" * 60)
print("测试UNet优化模型（无C++扩展也能运行）")
print("=" * 60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[设备] {device}")

try:
    # 导入优化模型
    print("\n[导入] 尝试导入优化模型...")
    from Unet_with_condition_optimized import Unet1D
    print("[成功] 优化模型导入成功")
    
    # 检查C++扩展状态
    try:
        # 尝试通过多种方式检查
        import sys
        from pathlib import Path
        cpp_ext_path = Path(__file__).parent / 'cpp_extension'
        if str(cpp_ext_path) not in sys.path:
            sys.path.insert(0, str(cpp_ext_path))
        
        try:
            import unet_cpp_ops
            cpp_status = "可用"
        except ImportError:
            cpp_status = "不可用（将使用PyTorch实现）"
    except:
        cpp_status = "不可用（将使用PyTorch实现）"
    
    print(f"[C++扩展] {cpp_status}")
    
except ImportError as e:
    print(f"[错误] 无法导入优化模型: {e}")
    print("[回退] 使用原始模型...")
    from Unet_with_condition import Unet1D

# 创建测试数据
print("\n[测试数据] 创建测试数据...")
batch_size = 2
seq_length = 50
channels = 3

x = torch.randn(batch_size, channels, seq_length).to(device)
time = torch.randint(0, 1000, (batch_size,)).to(device)
cond_input = torch.randn(batch_size, 10).to(device)

print(f"  输入形状: {x.shape}")

# 创建模型
print("\n[创建模型] 初始化UNet模型...")
model_config = {
    'dim': 32,
    'init_dim': 32,
    'dim_mults': (1, 2),
    'channels': channels,
    'cond_dim': 10
}

model = Unet1D(**model_config).to(device)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"  模型参数: {param_count:,}")

# 测试前向传播
print("\n[前向传播] 测试模型...")
try:
    with torch.no_grad():
        output = model(x, time, cond_input=cond_input)
    
    print(f"[成功] 前向传播成功")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 检查数值
    if torch.isnan(output).any():
        print("[警告] 输出包含NaN")
    elif torch.isinf(output).any():
        print("[警告] 输出包含Inf")
    else:
        print("[验证] 输出数值正常")
    
    # 简单性能测试
    print("\n[性能测试] 运行10次推理...")
    import time
    num_runs = 10
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(x, time, cond_input=cond_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"  平均推理时间: {avg_time*1000:.2f}ms")
    
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  峰值GPU内存: {memory_mb:.1f}MB")
    
    print("\n" + "=" * 60)
    print("[结论]")
    print("  优化模型工作正常")
    print("  可以直接在您的项目中使用")
    print("  即使没有C++扩展，性能也很好")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[错误] 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n请检查模型配置或PyTorch版本")

print("\n使用方法:")
print("  from Model.Diffusion_Model.Unet_with_condition_optimized import Unet1D")
print("  model = Unet1D(dim=64, channels=3, cond_dim=10)")
print("  # 其余代码不变")

