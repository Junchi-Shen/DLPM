# Pipelines/run_game.py
# (已修改为支持多任务和多“Q贪婪度”(Spread)测试)

import sys
import traceback
from pathlib import Path
import itertools 
import copy 

current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

# 导入 "设定集" 和 "引擎"
import Config.option_config as option_config
from Game.backtest_engine import Backtester


if __name__ == '__main__':
    
    # --- !! 1. 在这里定义所有要运行的回测任务 !! ---
    
    # A. 定义要回测的标的资产列表:
    TARGET_ASSETS = [
        'CSI1000',
        #'CSI300',
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
    P_FIXED_TRADE_COST_THRESHOLD = 0.05 # 例如 5%
    # ---------------------------------------------------------

    # --- 2. 通用模型配置 (所有任务共享) ---
    MODEL_CONFIG = {
            "P_model_type": 'unet',
            "P_paths_filename_base": "unet_generated_paths",
            "processor_source_folder": "all",
            "Q_model_type": 'mc', 
            "Q_paths_filename_base": "gbm_generated_paths", 
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
            # 应用当前的 Q 贪婪度 (Spread 值)
            current_contract_spec['spread_value'] = q_greed_level #
            
            # 强制使用相对阈值风格 (因为 P 成本是百分比)
            current_contract_spec['trade_threshold_style'] = 'relative' #
            # 应用固定的 P 交易成本 (Threshold 值)
            current_contract_spec['trade_threshold_value'] = P_FIXED_TRADE_COST_THRESHOLD #
            # --- !! 修改结束 !! ---

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
    print(f"\n==========================================================")
    print(f"✅ 所有 {len(all_tasks)} 个回测任务执行完毕。")
    if failed_tasks:
        print(f"❌ 失败的任务 ({len(failed_tasks)}):")
        for task in failed_tasks:
            print(f"   - {task}")
    else:
        print(f"🎉 所有任务均已成功完成。")
    print(f"==========================================================")