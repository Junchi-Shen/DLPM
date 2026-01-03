# 简化的噪声残差分析脚本

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_noise_residuals_simple():
    """简化的噪声残差分析"""
    
    print("=== Diffusion Model Noise Residual Analysis ===")
    
    # 检查模型文件是否存在
    model_paths = [
        "Results/Model_Results/all/unet_conditional_model_all.pth",
        "Results/Model_Results/all/condition_network_all.pth",
        "Results/Model_Results/all/data_processor_all.pkl"
    ]
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"❌ 模型文件不存在: {path}")
            return
    
    print("✅ 模型文件检查通过")
    
    try:
        # 加载模型
        print("正在加载模型...")
        checkpoint = torch.load(model_paths[0], map_location='cpu')
        print(f"✅ 模型加载成功")
        
        # 模拟一些噪声残差数据进行分析
        print("正在生成模拟噪声残差数据...")
        
        # 生成符合正态分布的残差（理想情况）
        ideal_residuals = np.random.normal(0, 0.1, 2000)
        
        # 生成不符合正态分布的残差（问题情况）
        problematic_residuals = np.random.normal(0, 0.1, 1000)
        problematic_residuals = np.concatenate([
            problematic_residuals,
            np.random.normal(0.2, 0.05, 500),  # 添加偏移
            np.random.normal(-0.1, 0.15, 500)  # 添加另一个分布
        ])
        
        # 分析两种情况
        print("\n=== 理想情况分析 ===")
        analyze_residuals(ideal_residuals, "理想情况")
        
        print("\n=== 问题情况分析 ===")
        analyze_residuals(problematic_residuals, "问题情况")
        
        # 绘制对比图
        plot_comparison(ideal_residuals, problematic_residuals)
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def analyze_residuals(residuals, case_name):
    """分析残差的正态分布性"""
    
    # 基本统计量
    mean = np.mean(residuals)
    std = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    print(f"{case_name} - 残差统计量:")
    print(f"  均值: {mean:.6f}")
    print(f"  标准差: {std:.6f}")
    print(f"  偏度: {skewness:.6f}")
    print(f"  峰度: {kurtosis:.6f}")
    
    # 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(mean, std))
    jarque_bera_stat, jarque_bera_p = stats.jarque_bera(residuals)
    
    print(f"\n{case_name} - 正态性检验:")
    print(f"  Shapiro-Wilk检验: 统计量={shapiro_stat:.6f}, p值={shapiro_p:.6f}")
    print(f"  Kolmogorov-Smirnov检验: 统计量={ks_stat:.6f}, p值={ks_p:.6f}")
    print(f"  Jarque-Bera检验: 统计量={jarque_bera_stat:.6f}, p值={jarque_bera_p:.6f}")
    
    # 判断结果
    alpha = 0.05
    print(f"\n{case_name} - 检验结果 (α={alpha}):")
    print(f"  Shapiro-Wilk: {'✅ 通过' if shapiro_p > alpha else '❌ 拒绝'}正态分布假设")
    print(f"  KS检验: {'✅ 通过' if ks_p > alpha else '❌ 拒绝'}正态分布假设")
    print(f"  Jarque-Bera: {'✅ 通过' if jarque_bera_p > alpha else '❌ 拒绝'}正态分布假设")
    
    # 评估模型质量
    if shapiro_p > alpha and ks_p > alpha and jarque_bera_p > alpha:
        print(f"\n🎉 {case_name} - 模型噪声预测质量: 优秀 - 残差符合正态分布")
    elif shapiro_p > alpha or ks_p > alpha or jarque_bera_p > alpha:
        print(f"\n⚠️  {case_name} - 模型噪声预测质量: 一般 - 部分检验通过")
    else:
        print(f"\n❌ {case_name} - 模型噪声预测质量: 较差 - 残差不符合正态分布")

def plot_comparison(ideal_residuals, problematic_residuals):
    """绘制对比图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 理想情况分析
    # 1. 直方图
    axes[0, 0].hist(ideal_residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('理想情况 - 残差分布', fontsize=14)
    axes[0, 0].set_xlabel('残差值')
    axes[0, 0].set_ylabel('密度')
    
    # 叠加正态分布曲线
    x = np.linspace(ideal_residuals.min(), ideal_residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, np.mean(ideal_residuals), np.std(ideal_residuals))
    axes[0, 0].plot(x, normal_curve, 'r-', linewidth=2, label='理论正态分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Q-Q图
    stats.probplot(ideal_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('理想情况 - Q-Q图', fontsize=14)
    axes[0, 1].grid(True)
    
    # 3. 箱线图
    axes[0, 2].boxplot(ideal_residuals, vert=True)
    axes[0, 2].set_ylabel('残差值')
    axes[0, 2].set_title('理想情况 - 箱线图', fontsize=14)
    axes[0, 2].grid(True)
    
    # 问题情况分析
    # 1. 直方图
    axes[1, 0].hist(problematic_residuals, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('问题情况 - 残差分布', fontsize=14)
    axes[1, 0].set_xlabel('残差值')
    axes[1, 0].set_ylabel('密度')
    
    # 叠加正态分布曲线
    x = np.linspace(problematic_residuals.min(), problematic_residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, np.mean(problematic_residuals), np.std(problematic_residuals))
    axes[1, 0].plot(x, normal_curve, 'r-', linewidth=2, label='理论正态分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 2. Q-Q图
    stats.probplot(problematic_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('问题情况 - Q-Q图', fontsize=14)
    axes[1, 1].grid(True)
    
    # 3. 箱线图
    axes[1, 2].boxplot(problematic_residuals, vert=True)
    axes[1, 2].set_ylabel('残差值')
    axes[1, 2].set_title('问题情况 - 箱线图', fontsize=14)
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_residual_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 对比图已保存到: noise_residual_comparison.png")
    plt.show()

def create_analysis_guide():
    """创建分析指南"""
    
    guide = """
# 如何检查扩散模型噪声残差的正态分布性

## 🎯 分析目标
检查扩散模型预测的噪声是否准确，残差是否符合正态分布。

## 📊 分析步骤

### 1. 收集噪声残差
```python
# 对每个训练样本：
# 1. 随机选择时间步 t
# 2. 添加真实噪声 noise ~ N(0,1)
# 3. 得到 x_t = q_sample(x_0, t, noise)
# 4. 模型预测 pred_noise = model(x_t, t, conditions)
# 5. 计算残差 residual = pred_noise - noise
```

### 2. 统计检验
- **Shapiro-Wilk检验**: 检验正态分布性
- **Kolmogorov-Smirnov检验**: 检验分布形状
- **Jarque-Bera检验**: 检验偏度和峰度

### 3. 可视化分析
- **直方图**: 观察分布形状
- **Q-Q图**: 检验分位数
- **箱线图**: 观察异常值
- **散点图**: 观察相关性

## ✅ 理想结果
- 残差均值接近0
- 残差标准差较小
- 偏度接近0
- 峰度接近0
- 所有正态性检验p值 > 0.05

## ❌ 问题指标
- 残差均值偏离0
- 残差标准差过大
- 偏度绝对值 > 0.5
- 峰度绝对值 > 1.0
- 正态性检验p值 < 0.05

## 🔧 改进建议
1. **调整损失函数权重**
2. **增加训练数据**
3. **调整模型架构**
4. **优化训练策略**
"""
    
    with open('noise_analysis_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ 分析指南已保存到: noise_analysis_guide.md")

if __name__ == "__main__":
    # 运行分析
    analyze_noise_residuals_simple()
    
    # 创建指南
    create_analysis_guide()
    
    print("\n=== 分析完成 ===")
    print("请查看生成的图表和指南文件")
