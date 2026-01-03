# Game/backtest_engine.py
# (已大幅修改以匹配 Generator/Explainer 的路径和数据加载逻辑)

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import ast # 需要 ast 来解析 real_path

# --- 环境设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# --- 路径导入 (已修正) ---
try:
    current_file_dir = Path(__file__).parent.resolve()
    project_root = current_file_dir.parent
    sys.path.append(str(project_root))
    # 导入正确的目录变量
    import Project_Path as pp 
    # (确保 DataProcessor 类可以被导入，以便 joblib 加载)
    from Data.Input_preparation import DataProcessor 
except (ImportError, NameError) as e:
    print(f"❌ 严重错误: 未能导入 Project_Path 或 DataProcessor: {e}")
    sys.exit(1)


# ==============================================================================
# 通用回测器引擎 (THE ENGINE)
# ==============================================================================
class Backtester:
    """
    一个通用的、由配置驱动的对抗式回测引擎。
    (已更新以匹配 Generator/Explainer 的路径和数据加载逻辑)
    """
    def __init__(self, config, contract_spec):
        
        self.config = config
        self.spec = contract_spec 
        
        self.contract_name = self.config['contract_name']
        self.asset_name = self.config['underlying_asset'] # e.g., 'CSI1000'
        
        print(f"🚀 正在初始化回测引擎 (合约: {self.contract_name}, 资产: {self.asset_name})...")
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # --- 路径和数据占位符 ---
        self.p_paths_file_path = None
        self.q_paths_file_path = None
        self.processor_path = None
        self.val_file_path = None # 指向中央 merge 文件
        self.report_dir = None
        self.images_dir = None
        
        self.p_model_paths = None # 将存储 P 模型 *价格* 路径
        self.q_model_paths = None # 将存储 Q 模型价格路径
        self.data_processor = None
        self.val_df = None # 将存储 *过滤后* 的验证数据
        
        # --- 执行设置和加载 ---
        self._setup_paths()
        self._load_data() # (包含过滤逻辑)
        
        print("✅ 初始化完成。回测器已准备就绪。")

    def _setup_paths(self):
        """根据配置(包含合约名称)定义所有文件和目录路径。
           (已修改，使用正确的目录和文件夹逻辑)"""
        
        # --- 1. 推断 P 和 Q 模型数据所在的文件夹 ---
        # 假设 UNet ('unet') 数据在 'all'，MC ('mc') 数据在具体资产文件夹下
        p_model_data_folder = 'CSI1000' if self.config.get('P_model_type') == 'unet' else self.asset_name
        q_model_data_folder = 'CSI1000' if self.config.get('Q_model_type') == 'unet' else self.asset_name
        
        # --- 2. 获取根目录 ---
        paths_root_dir = getattr(pp, "Path_Generator_Results_DIR")
        model_root_dir = getattr(pp, "Model_Results_DIR")
        # (假设验证数据来自 Testing_DATA_DIR)
        data_root_dir = getattr(pp, "Testing_DATA_DIR")
        base_report_dir = getattr(pp, "Report_Results_DIR")
        option_report_subfolder = "Option_Backtests"
        report_root_dir = base_report_dir / option_report_subfolder
        
        # --- 3. 构建输入文件路径 ---
        self.p_paths_base_dir = paths_root_dir / p_model_data_folder
        self.p_paths_filename_base = self.config['P_paths_filename_base']
        self.q_paths_base_dir = paths_root_dir / q_model_data_folder
        self.q_paths_filename_base = self.config['Q_paths_filename_base']
        
        # Processor 总是在 P 模型对应的文件夹下
        # (假设 processor 文件名固定或从 config 读取)
        processor_filename = "data_processor_all.pkl" # 或者 self.config.get('processor_filename', 'data_processor_all.pkl')
        processor_folder = self.config.get('processor_source_folder', p_model_data_folder)
        self.processor_path = model_root_dir / processor_folder / processor_filename
        
        # 验证数据指向中央 merge 文件
        # (假设中央文件名固定)
        central_data_file = "testing_data_merged.csv" 
        self.val_file_path = data_root_dir / central_data_file
        
        # --- 4. 构建输出路径 (逻辑不变) ---
        q_greed_level = self.config.get("q_greed_level", "NA") #
        q_greed_str = f"Qgreed{q_greed_level:.1f}" # 格式化为 "Qgreed0.1"
        report_folder_name = f"{self.timestamp}_{self.contract_name}_{q_greed_str}_backtest"
        self.report_dir = report_root_dir / self.asset_name / self.contract_name / report_folder_name
        self.images_dir = self.report_dir / "images"
        
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"📂 报告将保存至: {self.report_dir}")
        print(f"  - P 模型路径将从: {self.p_paths_file_path}")
        print(f"  - Q 模型路径将从: {self.q_paths_file_path}")
        print(f"  - 处理器将从: {self.processor_path}")
        print(f"  - 验证数据将从: {self.val_file_path} (并按 '{self.asset_name}' 过滤)")
    def _find_and_load_latest_npy(self, base_dir, pattern, exclude_keywords=None):
        """
        在指定目录搜索匹配模式的 .npy 文件，排除特定关键词，并加载最新的一个。
        (逻辑借鉴自 Explainer)
        """
        if not base_dir.exists():
            raise FileNotFoundError(f"搜索目录未找到: {base_dir}")
            
        try:
            all_matching_files = list(base_dir.glob(pattern))
        except Exception as e:
            raise IOError(f"搜索文件 '{pattern}' 时出错于 '{base_dir}': {e}")

        if not all_matching_files:
            raise FileNotFoundError(f"自动检测失败：在 '{base_dir}' 中未找到匹配 '{pattern}' 的文件。")

        # 过滤掉包含排除关键词的文件
        valid_files = all_matching_files
        if exclude_keywords:
            valid_files = [
                f for f in all_matching_files
                if not any(keyword in f.name for keyword in exclude_keywords)
            ]

        if not valid_files:
             raise FileNotFoundError(f"自动检测失败：在 '{base_dir}' 中未找到 *有效* 的文件 (已排除 {exclude_keywords})。模式: '{pattern}'")

        # 按修改时间排序，找到最新的
        valid_files_sorted = sorted(valid_files, key=os.path.getmtime, reverse=True)
        latest_file_path = valid_files_sorted[0]
        
        if len(valid_files_sorted) > 1:
            print(f"    ⚠️ 警告: 找到 {len(valid_files_sorted)} 个有效文件。将自动使用最新的一个: {latest_file_path.name}")
            
        # 加载 numpy 文件
        try:
            loaded_data = np.load(latest_file_path)
            return loaded_data, latest_file_path
        except Exception as e:
            raise IOError(f"加载文件 '{latest_file_path}' 失败: {e}")
        
    def _load_data(self):
        """加载所有必需的数据文件，并过滤验证集。
           (已修改，包含过滤逻辑和 P 模型路径恢复)"""
        print("\n🔄 正在加载数据文件...")
        
        # --- 1. 自动搜索并加载 P 模型路径文件 ---
        p_pattern = f"{self.p_paths_filename_base}_*_samples.npy"
        print(f"  - 正在 '{self.p_paths_base_dir}' 中搜索 P 模型文件 (模式: '{p_pattern}')...")
        p_model_output_raw, self.p_paths_file_path = self._find_and_load_latest_npy(self.p_paths_base_dir, p_pattern)
        print(f"  - ✅ P 模型文件已加载: {self.p_paths_file_path.name} (原始形状: {p_model_output_raw.shape})")

        # --- 2. 自动搜索并加载 Q 模型路径文件 ---
        q_pattern = f"{self.q_paths_filename_base}_*_samples.npy"
        print(f"  - 正在 '{self.q_paths_base_dir}' 中搜索 Q 模型文件 (模式: '{q_pattern}')...")
        # (注意：Q 模型加载后直接赋值给 self.q_model_paths)
        self.q_model_paths, self.q_paths_file_path = self._find_and_load_latest_npy(self.q_paths_base_dir, q_pattern, exclude_keywords=['_mask', '_sigma2'])
        print(f"  - ✅ Q 模型文件已加载: {self.q_paths_file_path.name} (原始形状: {self.q_model_paths.shape})")

        # --- 3. 加载处理器 (如果 P 模型需要) ---
        if self.config.get('P_model_type') == 'unet':
             # (路径构建已在 _setup_paths 中完成)
             if not self.processor_path.exists():
                 raise FileNotFoundError(f"处理器文件未找到: {self.processor_path}")
             self.data_processor = joblib.load(self.processor_path)
             print(f"  - 数据处理器 (data_processor) 已加载。")
        else:
             print(f"  - P 模型非 UNet，无需加载处理器。")

        # 4. 加载并过滤验证数据 (val_df)
        print(f"  - 正在加载中央验证文件: {self.val_file_path.name}")
        full_val_df = pd.read_csv(self.val_file_path)
        
        # (假设过滤列固定或从 config 读取)
        filter_column = 'asset_underlying' 
        print(f"  - 正在按 '{filter_column}' == '{self.asset_name}' 过滤验证数据...")
        self.val_df = full_val_df[full_val_df[filter_column] == self.asset_name].copy()
        if self.val_df.empty:
            raise ValueError(f"在 '{self.val_file_path}' 中找不到资产 '{self.asset_name}' 的数据。")
        print(f"  - 验证数据集过滤完成，得到 {len(self.val_df)} 条记录。")

        # 4. 准备 P 模型 *价格* 路径
        #    (如果 P 是 UNet，需要恢复价格；否则假设已经是价格)
        num_conditions_expected = len(self.val_df)
        
        if self.config.get('P_model_type') == 'unet':
            print(f"  - 正在恢复 P 模型 (UNet) 的价格路径...")
            if p_model_output_raw.shape[0] != num_conditions_expected:
                 raise ValueError(f"P 模型输出形状 ({p_model_output_raw.shape}) 与过滤后的条件数量 ({num_conditions_expected}) 不匹配！")
            
            p_prices_list = []
            for i in range(num_conditions_expected):
                start_price = self.val_df.iloc[i][self.config.get('start_price_col', 'start_price')]
                # 注意：传递索引 i 对应的对数收益率
                p_prices_list.append(self._recover_single_condition_prices(p_model_output_raw[i], start_price))
            self.p_model_paths = np.array(p_prices_list) # 形状 (N_cond, N_sim, SeqLen+1)
            print(f"  - P 模型价格路径已恢复。形状: {self.p_model_paths.shape}")
        else:
             self.p_model_paths = p_model_output_raw 
             print(f"  - P 模型非 UNet，直接使用加载的路径。形状: {self.p_model_paths.shape}")

        # 5. 验证 Q 模型形状
        #   (MC 模型保存为 (N_cond * N_sim, 1, SeqLen+1))
        expected_q_rows = num_conditions_expected * (self.q_model_paths.shape[0] // num_conditions_expected) # 计算 N_sim
        if self.q_model_paths.shape[0] != expected_q_rows:
             print(f"  ⚠️ 警告: Q 模型路径文件行数 ({self.q_model_paths.shape[0]}) 与预期 ({expected_q_rows}) 不完全匹配。请检查生成过程。")
        print(f"  - Q 模型价格路径已加载。原始形状: {self.q_model_paths.shape}")

    def _recover_single_condition_prices(self, single_condition_log_returns, start_price):
        """为单个市场环境恢复P模型的价格路径 (UNet专用)。
           (逻辑与 Explainer 基本一致)"""
        # 确保输入是 (N_sim, SeqLen)
        if single_condition_log_returns.ndim == 3 and single_condition_log_returns.shape[1] == 1:
             log_returns_squeezed = np.squeeze(single_condition_log_returns, axis=1)
        # 假设 UNet 输出已经是 (N_sim, SeqLen)
        elif single_condition_log_returns.ndim == 2: 
             log_returns_squeezed = single_condition_log_returns
        else:
             raise ValueError(f"无法处理的 UNet 输出形状: {single_condition_log_returns.shape}")

        # (假设 vol_scale 在 processor.config 中，否则设为 1.0)
        vol_scale = getattr(self.data_processor, 'config', {}).get('volatility_scale', 1.0)
        
        real_returns = log_returns_squeezed * vol_scale
        
        log_start_prices = np.log(np.full((real_returns.shape[0], 1), start_price))
        cumulative_log_returns = np.cumsum(real_returns, axis=1)
        
        # 拼接 t=0 的价格
        log_prices = np.concatenate([log_start_prices, log_start_prices + cumulative_log_returns], axis=1)
        
        return np.exp(log_prices) # 返回 (N_sim, SeqLen+1)

    def run(self, maturity_col_name, start_price_col, real_path_col):
        """
        执行主对抗式回测循环。
        (已修改：正确处理 Q 模型路径的索引)
        """
        print(f"\n🏁 开始执行回测... (合约: {self.contract_name})")
        num_environments = len(self.val_df) # 使用过滤后的 df 长度
        print(f"  - 将在 {num_environments} 个市场环境下进行测试。")
        
        # --- 1. 从 "参数集" 获取所有配置 (逻辑不变) ---
        payoff_func = self.spec['payoff_function']
        payoff_base_arg_name = self.spec['payoff_base_arg'] 
        payoff_extra_params = self.spec.get('payoff_params', {})
        
        pricing_style = self.spec['pricing_style']
        spread_style = self.spec['spread_style']
        spread_value = self.spec['spread_value']
        threshold_style = self.spec['trade_threshold_style']
        threshold_value = self.spec['trade_threshold_value']
        
        trade_log = []

        # --- !! Q 模型路径处理 !! ---
        # 计算 Q 模型每个条件有多少个 simulation
        if num_environments == 0:
             print("  ⚠️ 警告: 没有有效的市场环境可供回测。")
             return # 提前退出
             
        q_n_sim = self.q_model_paths.shape[0] // num_environments
        if q_n_sim * num_environments != self.q_model_paths.shape[0]:
             print(f"  ⚠️ 警告: Q 模型文件行数 ({self.q_model_paths.shape[0]}) 不是条件数 ({num_environments}) 的整数倍。")
             # 可以选择截断或报错，这里选择继续并打印警告

        for i in tqdm(range(num_environments), desc="回测进度"):
            params = self.val_df.iloc[i]
            
            # 2. 提取通用参数 (逻辑不变)
            T_days = int(params[maturity_col_name])
            T_years = T_days / 252.0 # T in years
            r = params['risk_free_rate']
            start_price = params[start_price_col]
            
            # 安全地解析 real_path
            try:
                real_path_list = ast.literal_eval(params[real_path_col])
                real_path = np.array(real_path_list)
            except (ValueError, SyntaxError, TypeError):
                 print(f"  ⚠️ 警告: 无法解析环境 {i} 的真实路径，跳过此环境。")
                 continue # 跳过这个环境

            # --- !! 获取 P 和 Q 路径 (已修改) !! ---
            # P 模型路径已经是 (N_cond, N_sim, SeqLen+1)，直接索引
            p_paths_env = self.p_model_paths[i] # 形状 (N_sim, SeqLen+1)
            
            # Q 模型路径需要计算切片
            q_start_row = i * q_n_sim
            q_end_row = (i + 1) * q_n_sim
            q_paths_env_raw = self.q_model_paths[q_start_row:q_end_row] # 形状 (N_sim, 1, SeqLen+1)
            q_paths_env = q_paths_env_raw.squeeze(axis=1) # 形状 (N_sim, SeqLen+1)
            # --- !! 修改结束 !! ---

            # 3. 动态构建 Payoff 函数的参数 (逻辑不变)
            base_arg_val = 0
            # (修改: 使用 start_price 作为 strike 的默认值，如果 payoff 需要 strike)
            if payoff_base_arg_name == 'strike':
                base_arg_val = params.get('strike', start_price) # 优先用 val_df 中的 strike，否则用 start_price
            elif payoff_base_arg_name == 'start_price':
                base_arg_val = start_price 
            
            base_payoff_args = {
                'maturity_steps': T_days, # 使用天数
                payoff_base_arg_name: base_arg_val
            }
            full_payoff_args = {**base_payoff_args, **payoff_extra_params}

            # 4. 计算 Payoff (使用修正后的路径)
            try:
                # 确保 payoff 函数接收的是 (N_sim, SeqLen+1) 或 (SeqLen+1)
                q_payoffs = payoff_func(paths=q_paths_env, **full_payoff_args)
                p_payoffs = payoff_func(paths=p_paths_env, **full_payoff_args)
                # 真实 payoff 输入需要是 (1, SeqLen+1) 或 (SeqLen+1)
                actual_payoff_arr = payoff_func(paths=real_path, **full_payoff_args)
                actual_payoff = actual_payoff_arr[0] if isinstance(actual_payoff_arr, np.ndarray) else actual_payoff_arr

                q_expected_payoff = np.mean(q_payoffs)
                p_expected_payoff = np.mean(p_payoffs)

            except Exception as payoff_err:
                 print(f"  ⚠️ 警告: 环境 {i} 计算 Payoff 时出错: {payoff_err}。跳过此环境。")
                 continue # 跳过这个环境

            # 5. 计算价格 (逻辑不变)
            price_q, value_p = 0, 0
            if pricing_style == 'rate':
                price_q = q_expected_payoff
                value_p = p_expected_payoff
            else: # 'discounted'
                price_q = q_expected_payoff * np.exp(-r * T_years) # 使用年化 T
                value_p = p_expected_payoff * np.exp(-r * T_years)

            # 6. 计算价差 (逻辑不变)
            price_q_ask, price_q_bid = 0, 0
            if spread_style == 'absolute':
                price_q_ask = price_q + (spread_value / 2)
                price_q_bid = price_q - (spread_value / 2)
            else: # 'percentage'
                # (修正：百分比价差应基于公允价 price_q)
                spread_amount = price_q * spread_value / 2 
                price_q_ask = price_q + spread_amount
                price_q_bid = price_q - spread_amount
                # 确保价格不为负
                price_q_ask = max(0, price_q_ask)
                price_q_bid = max(0, price_q_bid)


            # 7. 交易逻辑 (逻辑不变)
            pnl, trade_type, trade_price = 0, "No Trade", np.nan
            
            # --- 修正: 确保比较的是 *折现后* 或 *未折现* 的 payoff/price ---
            actual_value_for_trade = 0
            if pricing_style == 'rate':
                actual_value_for_trade = actual_payoff # payoff 本身
            else: # 'discounted'
                actual_value_for_trade = actual_payoff * np.exp(-r * T_years) # 折现后的 payoff
            # --- 修正结束 ---

            if threshold_style == 'absolute':
                 # P 觉得价值(value_p) > Q 的卖价(price_q_ask) + 阈值
                 if (value_p - price_q_ask) > threshold_value:
                     trade_type, trade_price = "P_Buy", price_q_ask
                     # 盈利 = 实际价值 - 买入成本
                     pnl = actual_value_for_trade - trade_price 
                 # Q 的买价(price_q_bid) > P 觉得价值(value_p) + 阈值
                 elif (price_q_bid - value_p) > threshold_value:
                     trade_type, trade_price = "P_Sell", price_q_bid
                     # 盈利 = 卖出收入 - 实际价值
                     pnl = trade_price - actual_value_for_trade
            else: # 'relative'
                 # P 买入条件: P估值比Q卖价高出超过阈值比例
                 if price_q_ask > 1e-9 and (value_p - price_q_ask) / price_q_ask > threshold_value:
                     trade_type, trade_price = "P_Buy", price_q_ask
                     pnl = actual_value_for_trade - trade_price
                 # P 卖出条件: Q买价比P估值高出超过阈值比例
                 elif price_q_bid > 1e-9 and (price_q_bid - value_p) / price_q_bid > threshold_value:
                     trade_type, trade_price = "P_Sell", price_q_bid
                     pnl = trade_price - actual_value_for_trade

            trade_log.append({
                "环境ID": i, 
                "Q模型公允价": price_q, 
                "P模型估值": value_p,
                "Q模型卖价": price_q_ask,
                "Q模型买价": price_q_bid,
                "实际Payoff": actual_payoff,
                # (新增) 实际折现价值 (用于比较)
                "实际折现价值": actual_value_for_trade, 
                "交易类型": trade_type, 
                "交易价格": trade_price,
                # (重命名) 模型盈亏 (P模型的视角)
                "P模型盈亏": pnl 
            })
            
        self.results_df = pd.DataFrame(trade_log)
        self._generate_report()

    # --- _generate_report 和 _plot_cumulative_pnl (与之前版本基本一致) ---
    # --- 只需要确保它们使用更新后的列名 ('P模型盈亏') ---
    
    def _generate_report(self):
        """分析并保存回测结果 (由 'report_style' 驱动)。"""
        print("\n📝 正在生成最终报告...")
        log_path = self.report_dir / "full_trade_log.csv"
        self.results_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        print(f"  - 完整交易日志已保存至: {log_path}")
        
        # 使用 'P模型盈亏' 列
        trades_only_df = self.results_df[self.results_df['交易类型'] != 'No Trade'].copy()
        num_trades = len(trades_only_df)
        pnl_col_name = 'P模型盈亏' # 使用新列名
        
        summary = f"================ 最终业绩摘要 ({self.contract_name.upper()}) ================\n"
        
        if num_trades > 0:
            if self.spec['report_style'] == 'notional':
                notional = 1_000_000 
                # 计算金额列
                pnl_amount_col = f'{pnl_col_name}_金额'
                trades_only_df[pnl_amount_col] = trades_only_df[pnl_col_name] * notional
                
                total_pnl = trades_only_df[pnl_amount_col].sum()
                avg_pnl = trades_only_df[pnl_amount_col].mean()
                win_rate = (trades_only_df[pnl_amount_col] > 0).mean()
                # Sharpe 基于原始 PnL (rate or discounted value)
                pnl_std_rate = trades_only_df[pnl_col_name].std() 
                sharpe = (trades_only_df[pnl_col_name].mean() / pnl_std_rate) * np.sqrt(252) if pnl_std_rate > 1e-9 else 0
                
                summary += f"总交易次数: {num_trades}\n"
                summary += f"累计盈亏 (PnL, 假设百万本金): {total_pnl:,.2f}\n"
                summary += f"平均每笔交易盈亏 (假设百万本金): {avg_pnl:,.2f}\n"
                summary += f"胜率 (Win Rate): {win_rate:.2%}\n"
                summary += f"年化夏普率 (基于原始值): {sharpe:.2f}\n" # 澄清基于什么计算
                # 分组分析也用金额列
                grouped_analysis = trades_only_df.groupby('交易类型')[pnl_amount_col].agg(['sum', 'mean', 'count'])
                summary += f"\n按交易类型分析 (假设百万本金):\n{grouped_analysis}\n"
            
            else: # 'pnl'
                total_pnl = trades_only_df[pnl_col_name].sum()
                avg_pnl = trades_only_df[pnl_col_name].mean()
                win_rate = (trades_only_df[pnl_col_name] > 0).mean()
                pnl_std = trades_only_df[pnl_col_name].std()
                sharpe = (avg_pnl / pnl_std) * np.sqrt(252) if pnl_std > 1e-9 else 0
                
                summary += f"总交易次数: {num_trades}\n"
                summary += f"累计盈亏 (PnL): {total_pnl:,.2f}\n"
                summary += f"平均每笔交易盈亏: {avg_pnl:,.2f}\n"
                summary += f"胜率 (Win Rate): {win_rate:.2%}\n"
                summary += f"年化夏普率 (Annualized Sharpe Ratio): {sharpe:.2f}\n"
                # 分组分析用原始 PnL 列
                summary += f"\n按交易类型分析:\n{trades_only_df.groupby('交易类型')[pnl_col_name].agg(['sum', 'mean', 'count'])}\n"

            self._plot_cumulative_pnl(trades_only_df)
        else:
            summary += "在整个回测期间没有发生任何交易。\n"
            
        summary += "================================================================"
        print(summary)
        summary_path = self.report_dir / "performance_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"  - 业绩摘要已保存至: {summary_path}")

    def _plot_cumulative_pnl(self, trades_df):
        """绘制并保存累计盈亏曲线图 (使用修正后的列名)。"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        y_col, y_label = '', ''
        pnl_col_name = 'P模型盈亏' # 使用新列名
        
        if self.spec['report_style'] == 'notional':
            pnl_amount_col = f'{pnl_col_name}_金额'
            # 确保金额列已计算
            if pnl_amount_col not in trades_df.columns:
                 notional = 1_000_000 
                 trades_df[pnl_amount_col] = trades_df[pnl_col_name] * notional
                 
            cum_pnl_col = f'累计{pnl_amount_col}'
            trades_df[cum_pnl_col] = trades_df[pnl_amount_col].cumsum()
            y_col = cum_pnl_col
            y_label = '累计盈利与亏损 (假设百万本金)'
        else: # 'pnl'
            cum_pnl_col = f'累计{pnl_col_name}'
            # 确保 PnL 列存在
            if pnl_col_name in trades_df.columns:
                trades_df[cum_pnl_col] = trades_df[pnl_col_name].cumsum()
            else:
                 # 如果没有交易，创建一个全零列以避免绘图错误
                 trades_df[cum_pnl_col] = 0 
                 
            y_col = cum_pnl_col
            y_label = '累计盈利与亏损'

        # 绘图前确保 DataFrame 非空且包含 x 列
        if not trades_df.empty and '环境ID' in trades_df.columns and y_col in trades_df.columns:
            trades_df.plot(x='环境ID', y=y_col, ax=ax, legend=None,
                           title=f'P模型 vs Q模型 ({self.contract_name.upper()}): 累计盈亏曲线 ({self.asset_name})')
            ax.set_xlabel('市场环境ID (时间顺序)', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        else:
             # 如果没有交易，显示一个空图或提示
             ax.set_title(f'P模型 vs Q模型 ({self.contract_name.upper()}): 无交易发生')
             ax.text(0.5, 0.5, '没有交易发生', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             
        plt.tight_layout()
        plot_path = self.images_dir / "cumulative_pnl.png"
        plt.savefig(plot_path, dpi=300)
        print(f"  - 累计盈亏曲线图已保存至: {plot_path}")
        plt.close(fig)