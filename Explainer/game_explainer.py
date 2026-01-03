# Game/result_aggregator.py
# 函数库：用于汇总多个回测实验的结果

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from tqdm import tqdm

def find_backtest_results(root_report_dir, assets_to_include=None, contracts_to_include=None):
    """
    扫描报告目录，查找所有符合命名规则的回测结果文件夹和日志文件。

    Args:
        root_report_dir (Path): Option_Backtests 的根目录。
        assets_to_include (list, optional): 要包含的资产列表。None 表示全部。
        contracts_to_include (list, optional): 要包含的合约列表。None 表示全部。

    Returns:
        list: 包含每个找到的回测结果信息的字典列表。
              每个字典包含: asset, contract, q_greed, report_folder, csv_path
    """
    found_results = []
    if not root_report_dir.exists():
        print(f"⚠️ 警告: 报告根目录不存在: {root_report_dir}")
        return found_results

    print(f"🔍 正在扫描 '{root_report_dir}' 查找回测结果...")

    # 正则表达式匹配文件夹名称: <timestamp>_<contract>_Qgreed<X.Y>_backtest
    # (更健壮地处理可能的名称变化)
    folder_pattern = re.compile(r"(\d{8}_\d{6})_([a-zA-Z0-9_]+)_Qgreed(\d+\.\d+)_backtest")

    for asset_dir in root_report_dir.iterdir():
        if not asset_dir.is_dir(): continue
        asset_name = asset_dir.name
        if assets_to_include and asset_name not in assets_to_include: continue

        for contract_dir in asset_dir.iterdir():
            if not contract_dir.is_dir(): continue
            contract_name = contract_dir.name
            if contracts_to_include and contract_name not in contracts_to_include: continue

            for report_folder in contract_dir.iterdir():
                if not report_folder.is_dir(): continue

                match = folder_pattern.match(report_folder.name)
                if match:
                    timestamp, contract_from_folder, q_greed_str = match.groups()
                    # 确保文件夹名中的合约与父目录一致
                    if contract_from_folder != contract_name: continue
                    
                    try:
                        q_greed = float(q_greed_str)
                    except ValueError:
                        continue # 无法解析 Q 贪婪度

                    csv_path = report_folder / "full_trade_log.csv"
                    if csv_path.exists():
                        found_results.append({
                            "asset": asset_name,
                            "contract": contract_name,
                            "q_greed": q_greed,
                            "report_folder": report_folder,
                            "csv_path": csv_path
                        })
                    # else:
                    #     print(f"  - 警告: 在 {report_folder} 中未找到 full_trade_log.csv")

    print(f"📊 找到 {len(found_results)} 个回测结果文件。")
    return found_results

def calculate_summary_metrics(df_trades):
    """
    从单个回测的 trade log DataFrame 计算汇总指标。
    """
    num_trades = len(df_trades)
    if num_trades == 0:
        return {
            "total_trades": 0, "total_pnl": 0, "avg_pnl": 0,
            "win_rate": np.nan, "sharpe_ratio": np.nan,
            "pnl_std": 0
        }

    pnl_col = 'P模型盈亏' #
    
    total_pnl = df_trades[pnl_col].sum()
    avg_pnl = df_trades[pnl_col].mean()
    win_rate = (df_trades[pnl_col] > 0).mean()
    pnl_std = df_trades[pnl_col].std()
    # 简单的年化夏普 (假设每日交易，可能需要根据实际调整)
    sharpe = (avg_pnl / pnl_std) * np.sqrt(252) if pnl_std > 1e-9 else 0 

    return {
        "total_trades": num_trades,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "pnl_std": pnl_std
    }

def aggregate_results(found_results):
    """
    加载所有找到的 CSV 文件，计算汇总指标，并进行聚合。

    Args:
        found_results (list): find_backtest_results 返回的列表。

    Returns:
        tuple: (all_summaries_df, per_contract_summary, overall_summary)
            - all_summaries_df: 包含每个实验汇总指标的 DataFrame。
            - per_contract_summary: 按合约和 Q 贪婪度聚合的 DataFrame。
            - overall_summary: 所有实验的总体聚合结果 DataFrame。
    """
    all_summaries = []

    print(f"🔄 正在处理 {len(found_results)} 个结果文件...")
    for result_info in tqdm(found_results, desc="处理结果"):
        try:
            df_log = pd.read_csv(result_info["csv_path"])
            df_trades = df_log[df_log['交易类型'] != 'No Trade'].copy() # 只分析实际发生的交易

            summary_metrics = calculate_summary_metrics(df_trades)

            # 添加标识信息
            summary_metrics['asset'] = result_info['asset']
            summary_metrics['contract'] = result_info['contract']
            summary_metrics['q_greed'] = result_info['q_greed']
            
            all_summaries.append(summary_metrics)

        except Exception as e:
            print(f"  ⚠️ 警告: 处理文件 {result_info['csv_path']} 时出错: {e}")
            continue
            
    if not all_summaries:
        print("❌ 未能成功处理任何结果文件。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_summaries_df = pd.DataFrame(all_summaries)
    print(f"✅ 已计算 {len(all_summaries_df)} 个实验的汇总指标。")

    # --- 执行聚合 ---
    print("📊 正在执行聚合...")

    # 1. 按期权品种汇总 (分析敏感性)
    #    对每个合约，按 Q 贪婪度分组，计算指标的平均值
    per_contract_summary = all_summaries_df.groupby(['contract', 'q_greed']).agg(
        avg_total_pnl=('total_pnl', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        avg_sharpe=('sharpe_ratio', 'mean'),
        avg_trades=('total_trades', 'mean'),
        num_assets_tested=('asset', 'nunique') # 记录这个组合在多少资产上测试过
    ).reset_index()
    print("  - 已完成按期权品种和贪婪度的聚合。")

    # 2. 按模型整体汇总 (评估 P vs Q)
    #    计算所有实验的总体平均指标
    overall_summary_data = {
        'metric': [
            'Overall Average Total PnL',
            'Overall Average Win Rate',
            'Overall Average Sharpe Ratio',
            'Overall Average Trades per Run'
        ],
        'value': [
            all_summaries_df['total_pnl'].mean(),
            all_summaries_df['win_rate'].mean(),
            all_summaries_df['sharpe_ratio'].mean(),
            all_summaries_df['total_trades'].mean()
        ],
        'count': [ # 记录总共有多少个实验点
            len(all_summaries_df),
            len(all_summaries_df),
            len(all_summaries_df),
            len(all_summaries_df)
        ]
    }
    overall_summary = pd.DataFrame(overall_summary_data)
    print("  - 已完成模型整体表现的聚合。")

    return all_summaries_df, per_contract_summary, overall_summary

def save_aggregated_reports(output_dir, all_df, contract_df, overall_df):
    """将聚合结果保存到 CSV 文件。"""
    os.makedirs(output_dir, exist_ok=True)

    all_path = output_dir / "all_individual_run_summaries.csv"
    contract_path = output_dir / "summary_by_contract_and_q_greed.csv"
    overall_path = output_dir / "overall_model_performance_summary.csv"

    try:
        all_df.to_csv(all_path, index=False, encoding='utf-8-sig')
        contract_df.to_csv(contract_path, index=False, encoding='utf-8-sig')
        overall_df.to_csv(overall_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 聚合报告已保存至: {output_dir}")
        print(f"  - 详细汇总: {all_path.name}")
        print(f"  - 按合约汇总: {contract_path.name}")
        print(f"  - 总体汇总: {overall_path.name}")
    except Exception as e:
        print(f"\n❌ 保存聚合报告失败: {e}")