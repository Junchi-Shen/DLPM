# data_explainer_library.py
#
# 这是一个纯粹的分析和绘图库，用于分析 *输入* 的 DataFrame。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

try:
    # 尝试使用 'SimHei' (黑体)，适用于 Windows/macOS/Linux (需安装)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ Matplotlib 中文字体 'SimHei' 配置成功。")
except Exception as e:
    try:
        # 如果 SimHei 失败，尝试 'Microsoft YaHei' (微软雅黑)，通常 Windows 自带
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Matplotlib 中文字体 'Microsoft YaHei' 配置成功。")
    except Exception as e_msyh:
        print(f"⚠️ 警告：未能成功配置 Matplotlib 中文字体 ('SimHei' 或 'Microsoft YaHei')。图表中的中文可能显示为方框。")
        print(f"   错误信息: SimHei - {e}, Microsoft YaHei - {e_msyh}")
        print(f"   请确保你的系统中安装了支持中文的字体 (如 SimHei, Microsoft YaHei)，或指定其他可用字体。")

def get_basic_stats(df, name="数据集"):
    """
    生成一个基础的文本统计报告 (基于你提供的 DatasetExplainer)。
    """
    report_lines = []
    report_lines.append(f"📦 数据集名称：{name}")
    report_lines.append("=" * 60)
    report_lines.append(f"🧮 样本数量：{len(df):,}")
    report_lines.append(f"🧮 字段数量：{df.shape[1]}")
    
    report_lines.append("\n📋 字段类型、非空值与非空率：")
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Non-Null Ratio': df.notnull().mean()
    })
    report_lines.append(info_df.to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report_lines.append("\n❓ 缺失值统计：")
    if not missing.empty:
        report_lines.append(str(missing.sort_values(ascending=False)))
    else:
        report_lines.append("✅ 无缺失值")
        
    report_lines.append("\n🔍 样本示例（首行）：")
    report_lines.append(str(df.head(1).T))
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)

def plot_numeric_distributions(df, numeric_cols, output_dir: Path):
    """为所有数值列绘制直方图和KDE分布图"""
    print("  -> 正在绘制数值分布图...")
    num_cols = len(numeric_cols)
    if num_cols == 0:
        return None
        
    # 动态创建子图网格
    n_rows = int(np.ceil(num_cols / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        try:
            sns.histplot(df[col], kde=True, ax=axes[i], bins=50)
            axes[i].set_title(f'"{col}" 的分布', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        except Exception as e:
            axes[i].set_title(f'"{col}" 绘图失败: {e}', fontsize=10)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plot_path = output_dir / "numeric_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def plot_correlation_heatmap(df, numeric_cols, output_dir: Path):
    """绘制数值列的相关性热图"""
    print("  -> 正在绘制相关性热图...")
    if len(numeric_cols) < 2:
        return None
        
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0,
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("数值特征相关性热图", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path = output_dir / "correlation_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def analyze_price_series_stats(df, path_col='price_series'):
    """(可选) 对 DataFrame 内的 'price_series' 列进行深入统计"""
    print("  -> 正在分析 'price_series' (这可能需要一些时间)...")
    if path_col not in df.columns:
        return "⚠️ 缺少 'price_series' 列，跳过分析。"
        
    all_returns = []
    path_lengths = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # 尝试 eval（如果它是字符串）
            paths = df[path_col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
        except Exception:
             return "❌ 'price_series' 列包含无法解析的数据。"

    for path in paths:
        if len(path) > 1:
            all_returns.append(np.diff(np.log(path)))
            path_lengths.append(len(path))
        else:
            path_lengths.append(len(path))
    
    if not all_returns:
        return "⚠️ 'price_series' 中的路径太短，无法计算收益率。"
        
    flat_returns = np.concatenate(all_returns)
    series_returns = pd.Series(flat_returns)
    
    report_lines = []
    report_lines.append("\n📈 'price_series' 深度统计:")
    report_lines.append(f"  路径数量: {len(path_lengths):,}")
    report_lines.append(f"  平均路径长度: {np.mean(path_lengths):.1f} 天 (最小: {np.min(path_lengths)}, 最大: {np.max(path_lengths)})")
    report_lines.append(f"  总收益率数据点: {len(flat_returns):,}")
    report_lines.append("\n  **路径内日收益率统计:**")
    report_lines.append(f"  均值 (Mean): {series_returns.mean():.6f}")
    report_lines.append(f"  标准差 (Std): {series_returns.std():.6f}")
    report_lines.append(f"  年化波动率: {series_returns.std() * np.sqrt(252):.4f}")
    report_lines.append(f"  偏度 (Skew): {series_returns.skew():.4f}")
    report_lines.append(f"  峰度 (Kurtosis): {series_returns.kurtosis():.4f}")
    
    return "\n".join(report_lines)

def generate_data_markdown_report(report_path: Path, stats_text, dist_plot, corr_plot, path_stats_text):
    """将所有分析结果汇编成一个 Markdown 报告"""
    print(f"  -> 正在生成 Markdown 报告: {report_path}")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 数据集分析报告\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 基础统计摘要\n\n")
        f.write("```text\n")
        f.write(stats_text)
        f.write("\n```\n")
        
        f.write("\n## 2. 数值特征分布\n\n")
        if dist_plot:
            f.write(f"![数值分布图]({dist_plot.name})\n")
        else:
            f.write("无可绘制的数值特征。\n")
            
        f.write("\n## 3. 数值特征相关性\n\n")
        if corr_plot:
            f.write(f"![相关性热图]({corr_plot.name})\n")
        else:
            f.write("数值特征不足（<2），无法绘制热图。\n")
            
        f.write("\n## 4. 价格路径 (Price Series) 深度分析\n\n")
        f.write("```text\n")
        f.write(path_stats_text)
        f.write("\n```\n")