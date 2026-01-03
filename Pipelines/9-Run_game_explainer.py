# Pipelines/aggregate_backtest_results.py
#
# 一键聚合所有期权回测实验的结果。

import sys
from pathlib import Path
from datetime import datetime

# --- 1. 项目路径设置 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

try:
    import Project_Path as pp
    from Explainer.game_explainer import find_backtest_results, aggregate_results, save_aggregated_reports
except ImportError as e:
    print(f"❌ 启动器错误：导入失败: {e}")
    print("  请确保 Project_Path.py 和 Game/result_aggregator.py 文件存在。")
    sys.exit(1)

if __name__ == '__main__':

    # --- !! 1. 在这里配置聚合范围 (可选) !! ---

    # A. 指定要包含的资产 (None 表示包含所有找到的资产)
    ASSETS_TO_AGGREGATE = ['CSI1000']
    # ASSETS_TO_AGGREGATE = None # 包含所有资产

    # B. 指定要包含的合约 (None 表示包含所有找到的合约)
    CONTRACTS_TO_AGGREGATE = ['my_snowball_A',
        'my_accumulator',
        'vanilla_call', 
        "standard_lookback",
        "standard_asian"]

    # CONTRACTS_TO_AGGREGATE = None # 包含所有合约
    
    # ----------------------------------------------

    print("--- 回测结果聚合启动器 ---")

    try:
        # --- 2. 确定报告根目录 ---
        # (与 Backtester 逻辑一致)
        base_report_dir = getattr(pp, "Report_Results_DIR")
        option_report_subfolder = "Option_Backtests"
        root_report_dir = base_report_dir / option_report_subfolder

        # --- 3. 查找所有结果文件 ---
        found_results = find_backtest_results(
            root_report_dir, 
            assets_to_include=ASSETS_TO_AGGREGATE, 
            contracts_to_include=CONTRACTS_TO_AGGREGATE
        )

        if not found_results:
            print("⏹️ 未找到任何符合条件的回测结果，聚合结束。")
            sys.exit(0)

        # --- 4. 执行聚合 ---
        all_summaries_df, per_contract_summary, overall_summary = aggregate_results(found_results)

        # --- 5. 保存报告 ---
        if not all_summaries_df.empty:
            # 创建带时间戳的输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 将聚合报告保存在 Option_Backtests 根目录下
            output_dir = root_report_dir / f"_AGGREGATED_{timestamp}" 
            
            save_aggregated_reports(output_dir, all_summaries_df, per_contract_summary, overall_summary)

        print("\n🎉 聚合任务完成。")

    except Exception as e:
        print(f"\n❌ 聚合过程中发生错误: {e}")
        import traceback
        traceback.print_exc()