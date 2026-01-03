# path_explainer_library.py
#
# 这是 "模型验证" 工具箱 (函数库)。
# 它整合了 5.1 和 5.2 的所有核心功能。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, anderson_ksamp
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 1. 路径恢复 (UNet 专用)
# (来自 5.1-UnetPathExplainer.py)
# ==========================================================
def recover_price_paths_from_returns(ensemble_log_returns, start_price, data_processor):
    """从标准化对数收益率恢复原始价格路径"""
    vol_scale = data_processor.config.get('volatility_scale', 1.0)
    
    if ensemble_log_returns.ndim == 3:
        returns = ensemble_log_returns[:, 0, 1:] * vol_scale
    elif ensemble_log_returns.ndim == 2:
        returns = ensemble_log_returns * vol_scale
    else:
        raise ValueError(f"不支持的 ensemble_log_returns 形状: {ensemble_log_returns.shape}")

    log_start_prices = np.log(np.full((returns.shape[0], 1), start_price))
    cumulative_log_returns = np.cumsum(returns, axis=1)
    log_prices = np.concatenate([log_start_prices, log_start_prices + cumulative_log_returns], axis=1)
    return np.exp(log_prices)

# ==========================================================
# 2. 核心统计函数 (来自 5.1, 这是最全的版本)
# ==========================================================
def calculate_comprehensive_statistics(real_returns, generated_returns):
    """计算全面的统计指标对比"""
    real_returns = real_returns.flatten()
    gen_returns_flat = generated_returns.flatten()
    
    stats_dict = {
        'mean_real': np.mean(real_returns),
        'mean_generated': np.mean(gen_returns_flat),
        'mean_diff': abs(np.mean(real_returns) - np.mean(gen_returns_flat)),
        'std_real': np.std(real_returns),
        'std_generated': np.std(gen_returns_flat),
        'std_diff': abs(np.std(real_returns) - np.std(gen_returns_flat)),
        'skew_real': stats.skew(real_returns),
        'skew_generated': stats.skew(gen_returns_flat),
        'skew_diff': abs(stats.skew(real_returns) - stats.skew(gen_returns_flat)),
        'kurt_real': stats.kurtosis(real_returns),
        'kurt_generated': stats.kurtosis(gen_returns_flat),
        'kurt_diff': abs(stats.kurtosis(real_returns) - stats.kurtosis(gen_returns_flat)),
        'annualized_vol_real': np.std(real_returns) * np.sqrt(252),
        'annualized_vol_generated': np.std(gen_returns_flat) * np.sqrt(252),
        'vol_diff': abs(np.std(real_returns) - np.std(gen_returns_flat)) * np.sqrt(252)
    }
    
    try:
        sample_size = min(len(real_returns), len(gen_returns_flat), 5000)
        real_sample = np.random.choice(real_returns, size=sample_size, replace=False)
        gen_sample = np.random.choice(gen_returns_flat, size=sample_size, replace=False)
        
        ad_result = anderson_ksamp([real_sample, gen_sample])
        stats_dict['ad_statistic'] = ad_result.statistic
        stats_dict['ad_pvalue'] = ad_result.pvalue
        
        critical_values = ad_result.critical_values if hasattr(ad_result, 'critical_values') else []
        significance_levels = [0.25, 0.10, 0.05, 0.025, 0.01]
        rejection_level = 0.01
        for i, cv in enumerate(critical_values if len(critical_values) > 0 else []):
            if ad_result.statistic < cv:
                rejection_level = significance_levels[i] if i < len(significance_levels) else 0.25
                break
        stats_dict['ad_rejection_level'] = rejection_level
        
        ks_stat, ks_pvalue = ks_2samp(real_sample, gen_sample)
        stats_dict['ks_statistic'] = ks_stat
        stats_dict['ks_pvalue'] = ks_pvalue
        
        stats_dict['wasserstein_distance'] = wasserstein_distance(real_sample, gen_sample)
        
        stats_dict['var_1_real'] = np.percentile(real_returns, 1)
        stats_dict['var_1_generated'] = np.percentile(gen_returns_flat, 1)
        stats_dict['var_1_diff'] = abs(stats_dict['var_1_real'] - stats_dict['var_1_generated'])
        
        var_1_threshold = np.percentile(real_returns, 1)
        stats_dict['cvar_1_real'] = np.mean(real_returns[real_returns <= var_1_threshold])
        var_1_gen_threshold = np.percentile(gen_returns_flat, 1)
        stats_dict['cvar_1_generated'] = np.mean(gen_returns_flat[gen_returns_flat <= var_1_gen_threshold])
        stats_dict['cvar_1_diff'] = abs(stats_dict['cvar_1_real'] - stats_dict['cvar_1_generated'])
        
    except Exception as e:
        print(f"统计检验计算错误: {e}")
        stats_dict.update({
            'ad_statistic': np.nan, 'ad_pvalue': np.nan, 'ad_rejection_level': 0.01,
            'ks_statistic': np.nan, 'ks_pvalue': np.nan, 'wasserstein_distance': np.nan,
            'var_1_diff': np.nan, 'var_99_diff': np.nan, 'cvar_1_diff': np.nan
        })
    return stats_dict

# ==========================================================
# 3. 绘图函数 (整合 5.1 和 5.2)
# ==========================================================

def plot_qq_comparison(real_returns, generated_returns, condition_info, output_path, model_name=""):
    """绘制增强版QQ图对比 (来自 5.1)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    real_flat = real_returns.flatten()
    gen_flat = generated_returns.flatten()
    
    quantiles = np.linspace(0.01, 0.99, 99)
    real_quantiles = np.quantile(real_flat, quantiles)
    gen_quantiles = np.quantile(gen_flat, quantiles)
    
    axes[0].scatter(real_quantiles, gen_quantiles, alpha=0.6, s=20, label=f'{model_name} Quantiles')
    tail_indices = np.concatenate([np.arange(0, 5), np.arange(94, 99)])
    axes[0].scatter(real_quantiles[tail_indices], gen_quantiles[tail_indices], 
                   color='red', s=50, alpha=0.8, label='Tail Quantiles (1-5%, 95-99%)')
    
    min_val = min(real_quantiles.min(), gen_quantiles.min())
    max_val = max(real_quantiles.max(), gen_quantiles.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    axes[0].set_xlabel('Real Returns Quantiles'), axes[0].set_ylabel('Generated Returns Quantiles')
    axes[0].set_title(f'Q-Q Plot (Condition {condition_info.name})'), axes[0].legend()
    
    r_squared = np.corrcoef(real_quantiles, gen_quantiles)[0, 1] ** 2
    axes[0].text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=axes[0].transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    deviations = gen_quantiles - real_quantiles
    axes[1].plot(quantiles * 100, deviations, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].fill_between(quantiles * 100, deviations, 0, alpha=0.3)
    axes[1].axvspan(0, 5, alpha=0.2, color='red', label='Lower Tail')
    axes[1].axvspan(95, 100, alpha=0.2, color='red', label='Upper Tail')
    
    axes[1].set_xlabel('Quantile (%)'), axes[1].set_ylabel('Deviation (Generated - Real)')
    axes[1].set_title('Q-Q Deviation Plot'), axes[1].legend()
    
    plt.suptitle(f'Quantile-Quantile Analysis - Vol={condition_info["volatility"]:.3f}, Rate={condition_info["risk_free_rate"]:.3f}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight'), plt.close()
    return r_squared

def plot_enhanced_fan_chart(all_restored_prices, real_price_path, condition_info, output_path, model_name=""):
    """生成增强版扇形图 (来自 5.1)"""
    valid_length = len(real_price_path)
    prices_truncated = all_restored_prices[:, :valid_length]
    
    percentiles = {
        '0.5%': np.percentile(prices_truncated, 0.5, axis=0),
        '5%': np.percentile(prices_truncated, 5, axis=0),
        '25%': np.percentile(prices_truncated, 25, axis=0),
        '75%': np.percentile(prices_truncated, 75, axis=0),
        '95%': np.percentile(prices_truncated, 95, axis=0),
        '99.5%': np.percentile(prices_truncated, 99.5, axis=0),
        'median': np.median(prices_truncated, axis=0),
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])
    
    ax1.fill_between(range(valid_length), percentiles['0.5%'], percentiles['99.5%'], 
                     color='lightgray', alpha=0.3, label='99% CI (Tail Risk)')
    ax1.fill_between(range(valid_length), percentiles['5%'], percentiles['95%'], 
                     color='lightblue', alpha=0.4, label='90% CI')
    ax1.fill_between(range(valid_length), percentiles['25%'], percentiles['75%'], 
                     color='lightblue', alpha=0.6, label='50% CI (IQR)')
    
    ax1.plot(real_price_path, color='red', linewidth=2.5, label='Real Path', marker='o', markersize=1)
    ax1.plot(percentiles['median'], color='navy', linestyle='--', linewidth=2, label=f'Generated {model_name} Median')
    ax1.set_title(f'Enhanced Fan Chart - Condition {condition_info.name}')
    ax1.set_xlabel('Trading Days'), ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize=9), ax1.grid(True, alpha=0.3)
    
    percentile_position = [stats.percentileofscore(prices_truncated[:, t], real_price_path[t]) for t in range(valid_length)]
    
    ax2.plot(percentile_position, color='purple', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1% / 99% Tails')
    ax2.axhline(y=99, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(range(valid_length), 25, 75, alpha=0.2, color='blue', label='IQR')
    
    ax2.set_xlabel('Trading Days'), ax2.set_ylabel('Percentile of Real Path (%)')
    ax2.set_title('Real Path Position in Generated Distribution')
    ax2.set_ylim(0, 100), ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight'), plt.close()

def plot_return_distribution(real_log_returns, generated_returns, condition_info, output_path, model_name=""):
    """绘制真实与生成收益率的分布对比图 (来自 5.2)"""
    plt.figure(figsize=(12, 7))
    sns.kdeplot(real_log_returns, label='真实收益率', color='red', fill=True, linewidth=2.5)
    sns.kdeplot(generated_returns.flatten(), label=f'{model_name} 生成收益率', color='skyblue', fill=True)
    plt.title(f"收益率分布对比 (条件 {condition_info.name})")
    plt.xlabel("每日对数收益率"), plt.ylabel("密度")
    plt.legend(), plt.grid(True)
    plt.savefig(output_path), plt.close()

def plot_volatility_clustering(real_log_returns, generated_returns, condition_info, output_path, model_name=""):
    """绘制平方收益率的自相关性 (来自 5.2)"""
    max_possible_lags = len(real_log_returns) - 1
    nlags_to_compute = min(30, max_possible_lags)
    if nlags_to_compute < 1:
        return # 路径太短，无法计算
    
    real_acf = acf(real_log_returns**2, nlags=nlags_to_compute, fft=False)
    generated_acfs = [acf(generated_returns[i]**2, nlags=nlags_to_compute, fft=False) for i in range(generated_returns.shape[0])]
    mean_generated_acf = np.mean(generated_acfs, axis=0)

    plt.figure(figsize=(12, 7))
    x_axis = range(1, nlags_to_compute + 1)
    plt.bar(x_axis, real_acf[1:], alpha=0.7, label='真实路径 ACF')
    plt.plot(x_axis, mean_generated_acf[1:], 'o-', color='red', label=f'{model_name} 平均 ACF')
    plt.title(f"波动率集聚 (ACF) - 条件 {condition_info.name}")
    plt.xlabel("滞后阶数"), plt.ylabel("自相关性"), plt.legend(), plt.grid(True), plt.ylim(bottom=0)
    plt.savefig(output_path), plt.close()

# ==========================================================
# 4. 评分函数 (来自 5.1)
# ==========================================================

def calculate_model_score(all_results):
    """计算模型综合评分 (来自 5.1)"""
    scores = []
    weights = {
        'mean_alignment': 0.10,
        'volatility_alignment': 0.15,    
        'distribution_similarity': 0.20, # AD检验
        'moment_matching': 0.15,         # 偏度+峰度
        'tail_risk_matching': 0.20,
        'qq_fit': 0.20
    }
    
    for result in all_results:
        stats = result['statistics']
        condition_score = 0
        
        mean_score = max(0, 1 - min(stats['mean_diff'] / 0.01, 1)) * 100
        condition_score += weights['mean_alignment'] * mean_score
        
        vol_score = max(0, 1 - min(stats['vol_diff'] / 0.1, 1)) * 100
        condition_score += weights['volatility_alignment'] * vol_score
        
        ad_level = stats.get('ad_rejection_level', 0.01)
        if ad_level >= 0.25: ad_score = 100
        elif ad_level >= 0.10: ad_score = 80
        elif ad_level >= 0.05: ad_score = 60
        elif ad_level >= 0.025: ad_score = 40
        else: ad_score = 20
        condition_score += weights['distribution_similarity'] * ad_score
        
        skew_score = max(0, 1 - min(stats['skew_diff'] / 1.0, 1)) * 100
        kurt_score = max(0, 1 - min(stats['kurt_diff'] / 5.0, 1)) * 100
        condition_score += weights['moment_matching'] * (skew_score + kurt_score) / 2
        
        tail_scores = []
        if not np.isnan(stats.get('var_1_diff', np.nan)):
            tail_scores.append(max(0, 1 - min(stats['var_1_diff'] / 0.02, 1)) * 100)
        if not np.isnan(stats.get('var_99_diff', np.nan)):
            tail_scores.append(max(0, 1 - min(stats['var_99_diff'] / 0.02, 1)) * 100)
        if not np.isnan(stats.get('cvar_1_diff', np.nan)):
            tail_scores.append(max(0, 1 - min(stats['cvar_1_diff'] / 0.03, 1)) * 100)
        
        condition_score += weights['tail_risk_matching'] * (np.mean(tail_scores) if tail_scores else 50)
        
        qq_score = stats['qq_r_squared'] * 100
        condition_score += weights['qq_fit'] * qq_score
        
        scores.append(condition_score)
    
    overall_score = np.mean(scores)
    return {
        'overall_score': overall_score,
        'score_std': np.std(scores),
        'individual_scores': scores,
        'grade': get_model_grade(overall_score)
    }

def get_model_grade(score):
    """根据分数给出模型评级 (来自 5.1)"""
    if score >= 90: return 'A+ (Excellent)'
    elif score >= 85: return 'A (Very Good)'
    elif score >= 80: return 'A- (Good)'
    elif score >= 75: return 'B+ (Above Average)'
    elif score >= 70: return 'B (Average)'
    elif score >= 60: return 'B- (Below Average)'
    elif score >= 50: return 'C (Poor)'
    else: return 'F (Fail)'