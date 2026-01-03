# Model/Garch_Model/garch_fitter.py
#
# GARCH 模型拟合的函数库

import pandas as pd
import numpy as np
import ast
import json
import os
from arch import arch_model

def fit_and_save_garch_params(data_path, output_path, spec):
    """
    从中央数据文件加载、过滤、拟合GARCH(1,1)模型，并保存参数。
    
    Args:
        data_path (pathlib.Path): 中央 merge 文件的完整路径。
        output_path (pathlib.Path): JSON 参数文件的保存路径。
        spec (dict): 来自 garch_fitter_config.py 的作业配置。
        
    Returns:
        garch_fit: 拟合完成的 GARCH 模型结果对象。
    """
    
    asset_to_fit = spec['asset_to_fit']
    
    # --- 1. 加载和过滤数据 ---
    print(f"  [1/4] 正在加载: {data_path.name}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"  ❌ 错误: 中央数据文件未找到: {data_path}")
        raise

    print(f"  [2/4] 正在按 '{spec['filter_column']}' == '{asset_to_fit}' 过滤数据...")
    df_filtered = df[df[spec['filter_column']] == asset_to_fit].copy()
    
    if df_filtered.empty:
        raise ValueError(f"  ❌ 错误: 在中央文件中未找到资产 '{asset_to_fit}' 的数据。")
        
    # --- 2. 处理价格序列 ---
    # (复现您脚本中的逻辑：使用每个价差路径的第一个价格来构建时间序列)
    # 注意: 这种方法假设 val_df 中的行是按时间顺序排列的。
    print("  [3/4] 正在解析价格序列并计算收益率...")
    try:
        paths = df_filtered[spec['price_series_column']].apply(ast.literal_eval)
        # 使用 path[0] (即 start_price) 来构建日度价格序列
        daily_prices = paths.apply(lambda path: path[0])
    except Exception as e:
        print(f"  ❌ 错误: 解析 'price_series' 列失败: {e}")
        raise

    # 计算对数收益率
    returns = np.log(daily_prices / daily_prices.shift(1)).dropna() * 100
    if returns.empty:
        raise ValueError("  ❌ 错误: 收益率序列为空，无法拟合。")

    # --- 3. GARCH 模型拟合 ---
    dist = spec['distribution']
    print(f"  [4/4] 正在拟合 GARCH(1,1) (p=1, q=1, dist='{dist}')...")
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist=dist)
    garch_fit = garch_model.fit(disp='off')

    # --- 4. 提取并保存参数 ---
    omega = garch_fit.params['omega'] / 10000.0 # 转换为小数^2
    alpha = garch_fit.params['alpha[1]']
    beta = garch_fit.params['beta[1]']
    # 如果分布是 't'，则提取 'nu'
    nu = np.nan
    if 'nu' in garch_fit.params:
        nu = garch_fit.params['nu']

    params_dict = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "nu": nu
    }
    
    # 确保输出目录存在
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params_dict, f, indent=4)
        
    print(f"\n  ✅ 参数已保存至: {output_path}")
    
    return garch_fit, params_dict