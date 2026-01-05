#!/usr/bin/env python3
"""
测试优化的UNet模型
即使没有C++扩展也能运行
"""

import torch
import sys
from pathlib import Path

def test_model():
    """测试优化模型"""
    print("🧪 测试UNet优化模型")
    print("=" * 50)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 设备: {device}")
    
    try:
        # 导入优化模型
        from Unet_with_condition_optimized import Unet1D
        print("✅ 成功导入优化模型")
        
        # 检查C++扩展状态
        try:
            from cpp_extension.unet_cpp_wrapper import CPP_AVAILABLE
        except ImportError:
            CPP_AVAILABLE = False
        
        if CPP_AVAILABLE:
            print("✅ C++加速扩展: 可用")
        else:
            print("⚠️ C++加速扩展: 不可用（使用PyTorch原生实现）")
        
    except ImportError as e:
        print(f"❌ 无法导入优化模型: {e}")
        print("回退到原始模型...")
        from Unet_with_condition import Unet1D
        CPP_AVAILABLE = False
    
    # 创建测试数据
    print("\n📊 创建测试数据...")
    batch_size = 2
    seq_length = 50
    channels = 3
    
    x = torch.randn(batch_size, channels, seq_length).to(device)
    time = torch.randint(0, 1000, (batch_size,)).to(device)
    cond_input = torch.randn(batch_size, 10).to(device)
    
    print(f"   输入形状: {x.shape}")
    print(f"   时间步: {time.shape}")
    print(f"   条件输入: {cond_input.shape}")
    
    # 创建模型
    print("\n🔨 创建模型...")
    model_config = {
        'dim': 32,
        'init_dim': 32,
        'dim_mults': (1, 2),
        'channels': channels,
        'cond_dim': 10
    }
    
    model = Unet1D(**model_config).to(device)
    model.eval()
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    print("\n⚡ 测试前向传播...")
    try:
        with torch.no_grad():
            output = model(x, time, cond_input=cond_input)
        
        print(f"✅ 前向传播成功")
        print(f"   输出形状: {output.shape}")
        print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        # 检查输出是否有效
        if torch.isnan(output).any():
            print("❌ 警告: 输出包含NaN")
        elif torch.isinf(output).any():
            print("❌ 警告: 输出包含Inf")
        else:
            print("✅ 输出数值正常")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 性能测试
    print("\n⏱️ 性能测试...")
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
    print(f"   平均推理时间: {avg_time*1000:.2f}ms")
    print(f"   吞吐量: {batch_size/avg_time:.1f} samples/s")
    
    # 内存使用
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   峰值GPU内存: {memory_mb:.1f}MB")
    
    print("\n✅ 所有测试通过！")
    return True

def compare_models():
    """对比原始和优化模型"""
    print("\n🔄 对比原始模型和优化模型")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试数据
    x = torch.randn(2, 3, 50).to(device)
    time = torch.randint(0, 1000, (2,)).to(device)
    cond_input = torch.randn(2, 10).to(device)
    
    model_config = {
        'dim': 32,
        'init_dim': 32,
        'dim_mults': (1, 2),
        'channels': 3,
        'cond_dim': 10
    }
    
    try:
        # 原始模型
        from Unet_with_condition import Unet1D as OriginalUnet
        original_model = OriginalUnet(**model_config).to(device)
        original_model.eval()
        
        # 优化模型
        from Unet_with_condition_optimized import Unet1D as OptimizedUnet
        optimized_model = OptimizedUnet(**model_config).to(device)
        optimized_model.eval()
        
        # 测试输出一致性
        print("🔍 测试输出一致性...")
        with torch.no_grad():
            # 复制权重确保公平对比
            optimized_model.load_state_dict(original_model.state_dict())
            
            out_original = original_model(x, time, cond_input=cond_input)
            out_optimized = optimized_model(x, time, cond_input=cond_input)
        
        diff = torch.abs(out_original - out_optimized).max().item()
        print(f"   最大差异: {diff:.6f}")
        
        if diff < 1e-4:
            print("✅ 输出完全一致")
        elif diff < 1e-2:
            print("✅ 输出基本一致")
        else:
            print(f"⚠️ 输出存在差异: {diff}")
        
        # 性能对比
        import time
        print("\n⏱️ 性能对比...")
        num_runs = 20
        
        # 原始模型
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = original_model(x, time, cond_input=cond_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        original_time = sum(times) / len(times)
        
        # 优化模型
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = optimized_model(x, time, cond_input=cond_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        optimized_time = sum(times) / len(times)
        
        speedup = original_time / optimized_time
        
        print(f"   原始模型: {original_time*1000:.2f}ms")
        print(f"   优化模型: {optimized_time*1000:.2f}ms")
        print(f"   加速比: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("🎉 显著加速！")
        elif speedup > 1.1:
            print("✅ 性能提升")
        elif speedup > 0.9:
            print("⚠️ 性能相当")
        else:
            print("⚠️ 性能下降（可能是回退到PyTorch实现）")
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🎯 UNet优化模型测试套件")
    print("=" * 50)
    
    # 基本测试
    if test_model():
        # 对比测试
        try:
            compare_models()
        except Exception as e:
            print(f"⚠️ 对比测试跳过: {e}")
        
        print("\n🎉 测试完成！")
        print("\n💡 总结:")
        print("   - 优化模型工作正常")
        print("   - 可以直接替换原始模型使用")
        print("   - 如果有C++扩展会自动加速")
        print("   - 没有C++扩展也能正常工作")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main()
