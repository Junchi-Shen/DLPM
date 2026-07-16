# Pipelines/run_game.py
# (已修改为支持多任务和多“Q贪婪度”(Spread)测试)

import sys
import json
import argparse
import traceback
from pathlib import Path
import itertools
import copy

import numpy as np

current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

# 导入 "设定集" 和 "引擎"
import Config.option_config as option_config
from Game.backtest_engine import Backtester


def summarize_run(backtester, spec):
    """从回测结果 DataFrame 提取机器可读汇总"""
    df = backtester.results_df
    trades = df[df['交易类型'] != 'No Trade']
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'cum_pnl': 0.0, 'win_rate': None, 'sharpe': None,
                'longs': 0, 'shorts': 0}
    pnl = trades['P模型盈亏']
    scale = 1_000_000 if spec.get('report_style') == 'notional' else 1.0
    std = pnl.std()
    return {
        'trades': int(n),
        'cum_pnl': float(pnl.sum() * scale),
        'avg_pnl': float(pnl.mean() * scale),
        'win_rate': float((pnl > 0).mean()),
        'sharpe': float(pnl.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0,
        'longs': int((trades['交易类型'] == 'P_Buy').sum()),
        'shorts': int((trades['交易类型'] == 'P_Sell').sum()),
    }


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--p_base', default='dlpm_generated_paths')
    cli.add_argument('--q_base', default='garch_paths_fitted')
    cli.add_argument('--p_label', default='dlpm')
    cli.add_argument('--out_json', default=None)
    args = cli.parse_args()
    
    # --- !! 1. 在这里定义所有要运行的回测任务 !! ---
    
    # A. 定义要回测的标的资产列表:
    TARGET_ASSETS = [
        'CSI1000',
    ]
    
    # B. 定义要在 *每个* 资产上运行的合约列表:
    CONTRACTS_TO_RUN = [
        'my_snowball_A',
        'my_accumulator',
        'vanilla_call', 
        "standard_lookback",
        "standard_asian"

    ]
    
    # --- !! C. 新增：定义要测试的 Q 模型贪婪度列表 (百分比 Spread) !! ---
    #    (0.0 = 0%, 0.1 = 10%, etc.)
    Q_GREEDINESS_LEVELS_TO_TEST = [0.0, 0.1, 0.2, 0.3, 0.4] 
    # ---------------------------------------------------------
    
    # --- !! D. 新增：定义 P 模型的固定交易成本 (相对阈值) !! ---
    P_FIXED_TRADE_COST_THRESHOLD = 0.10 # 论文口径：相对价差超过10%才交易
    # ---------------------------------------------------------

    # --- 2. 通用模型配置 (所有任务共享) ---
    MODEL_CONFIG = {
            "P_model_type": 'dlpm',
            "P_paths_filename_base": args.p_base,
            "processor_source_folder": "all",
            'processor_type_subfolder': 'Diffusion_Model_DLPM',
            "Q_model_type": 'gbm' if 'gbm' in args.q_base else 'garch',
            "Q_paths_filename_base": args.q_base,
        }
    # ----------------------------------------------------

    # --- 3. 通用回测参数 (所有任务共享) ---
    BACKTEST_PARAMS = {
        "maturity_col_name": "actual_trading_days",
        "start_price_col": "start_price",
        "real_path_col": "price_series"
    }
    # ----------------------------------------------------

    # --- 4. 生成所有任务组合 (资产 x 合约 x Q贪婪度) ---
    # (修改) 使用 Q_GREEDINESS_LEVELS_TO_TEST
    all_tasks = list(itertools.product(TARGET_ASSETS, CONTRACTS_TO_RUN, Q_GREEDINESS_LEVELS_TO_TEST)) 
    
    if not all_tasks:
        print("⚠️ 警告: 任务列表为空。")
        sys.exit(0)
        
    print(f"--- 启动器 (多任务+多Q贪婪度模式): 将执行 {len(all_tasks)} 个回测任务 ---")
    print(f"--- 资产: {TARGET_ASSETS}")
    print(f"--- 合约: {CONTRACTS_TO_RUN}")
    # (修改) 打印 Q 贪婪度
    print(f"--- Q贪婪度 (百分比 Spread): {Q_GREEDINESS_LEVELS_TO_TEST}") 
    print(f"--- P交易成本 (固定相对阈值): {P_FIXED_TRADE_COST_THRESHOLD:.1%}")

    # --- 5. 循环执行所有任务 ---
    failed_tasks = []
    all_summaries = {}
    # (修改循环变量)
    for asset, contract_name, q_greed_level in all_tasks:
        
        # (修改) task_id 包含 q_greed_level
        task_id = f"{contract_name}_Qgreed{q_greed_level:.1f}/{asset}" 
        
        print(f"\n==========================================================")
        print(f"🏁 开始执行任务: {task_id}")
        print(f"==========================================================")
        
        try:
            # 5.1 获取原始合约参数
            original_contract_spec = option_config.CONTRACT_SPECS[contract_name]
            
            # --- !! 5.2 关键: 复制并修改合约参数以应用当前 Q 贪婪度 和 固定 P 成本 !! ---
            current_contract_spec = copy.deepcopy(original_contract_spec) 
            
            # 强制使用百分比价差风格 (因为测试的是百分比)
            current_contract_spec['spread_style'] = 'percentage'  #
            # 强制使用相对阈值风格 (因为 P 成本是百分比)
            current_contract_spec['trade_threshold_style'] = 'relative' #
            if contract_name == 'my_snowball_A':
                # --- 雪球：利率套利模式 ---
                # 缩放 Spread 量级：原本 0.4 的贪婪度对利率太夸张，缩放到 0.04 (4% 的利差点差)
                total_spread = q_greed_level * 0.05
                current_contract_spec['spread_value'] = total_spread
                current_contract_spec['trade_threshold_style']='absolute'
                current_contract_spec['spread_style'] ='absolute'
                half_spread = total_spread / 2
                current_contract_spec['trade_threshold_value'] = half_spread * 1.2
            
            elif contract_name == 'my_accumulator':
                # --- 累加器：比例套利模式 ---
                # 累加器的估值通常较小，也需要较低的 Spread 和门槛
                current_contract_spec['spread_value'] = q_greed_level * 0.5
                current_contract_spec['trade_threshold_value'] = 0.01  # 1% 的比例差异即交易
            
            else:
                # --- 其他期权 (Vanilla/Asian/Lookback) ---
                # 保持原始的高波动套利参数
                current_contract_spec['spread_value'] = q_greed_level
                current_contract_spec['trade_threshold_value'] = P_FIXED_TRADE_COST_THRESHOLD

            # 5.3 构建当前任务的完整配置
            current_config = {
                **MODEL_CONFIG, 
                "underlying_asset": asset, 
                "contract_name": contract_name, 
                # (修改) 将 q_greed_level 传入 config
                "q_greed_level": q_greed_level 
            }
        
            # 5.4 运行回测 (传入修改后的 contract_spec)
            backtester = Backtester(current_config, current_contract_spec)
            backtester.run(**BACKTEST_PARAMS)
            all_summaries[f'{contract_name}|{q_greed_level:.1f}'] = summarize_run(
                backtester, current_contract_spec)

            print(f"✅ 任务 '{task_id}' 执行完毕。")
            
        # ... (后续的 except 错误处理逻辑保持不变, 只需更新 task_id) ...
        except FileNotFoundError as e:
            print(f"\n❌ 任务 '{task_id}' 失败：找不到必需的文件。")
            print(f"  - 详情: {e}")
            failed_tasks.append(task_id)
        except KeyError as e:
            if str(e) in option_config.CONTRACT_SPECS:
                 print(f"\n❌ 任务 '{task_id}' 失败：在 option_config.py 中未找到合约 '{e}'。")
            elif str(e) in BACKTEST_PARAMS.values():
                 print(f"\n❌ 任务 '{task_id}' 失败：在验证数据 (val_df) 中找不到列 '{e}'。")
            else:
                 print(f"\n❌ 任务 '{task_id}' 失败：配置或代码中缺少键 '{e}'。")
            failed_tasks.append(task_id)
        except Exception as e:
            print(f"\n❌ 任务 '{task_id}' 发生未知错误: {e}")
            traceback.print_exc()
            failed_tasks.append(task_id)


    # --- 6. 最终总结 ---
    # (保持不变)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'p_label': args.p_label, 'p_base': args.p_base,
                       'q_base': args.q_base, 'summaries': all_summaries}, f,
                      indent=2, ensure_ascii=False)
        print(f"📊 汇总已写入 {out_path}")

    print(f"\n==========================================================")
    print(f"✅ 所有 {len(all_tasks)} 个回测任务执行完毕。")
    if failed_tasks:
        print(f"❌ 失败的任务 ({len(failed_tasks)}):")
        for task in failed_tasks:
            print(f"   - {task}")
    else:
        print(f"🎉 所有任务均已成功完成。")
    print(f"==========================================================")