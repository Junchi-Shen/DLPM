# path_explainer_engine.py
#
# 这是一个通用的 "模型验证" 引擎。
# 它可以分析 UNet, GBM, GARCH 等任何模型的输出。

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import ast

# --- 导入自定义模块 ---
import Explainer.path_explainer_library as lib # 我们的 "函数库"

# --- 路径导入 ---
try:
    current_file_dir = Path(__file__).parent.resolve()
    project_root = current_file_dir.parent
    sys.path.append(str(project_root))
    import Project_Path as pp
    from Data.Input_preparation import DataProcessor # 确保能导入
except (ImportError, NameError) as e:
    print(f"❌ 严重错误: 未能导入 Project_Path 或 DataProcessor: {e}")
    sys.exit(1)

class PathExplainerEngine:
    """
    一个通用的、由配置驱动的路径验证引擎。
    它整合了 5.1 和 5.2 的功能。
    """
    def __init__(self, asset_name, job_spec):
        self.asset_name = asset_name
        self.spec = job_spec
        self.job_name_safe = self.spec['display_name'].replace(' ', '_').replace('(', '').replace(')', '')
        self.model_type = self.spec['model_type']
        
        print(f"🚀 正在初始化路径验证引擎 (模型: {self.spec['display_name']})...")
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._setup_paths()
        
        # 引擎组件
        self.data_processor = None
        self.generated_paths = None
        self.val_df = None
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("colorblind")

    def _setup_paths(self):
        """创建所有输出目录"""
        report_name = f"{self.timestamp}_{self.job_name_safe}_validation_report"
        self.report_dir = getattr(pp, "Report_Results_DIR") /"Path_Generator_Report"/ self.asset_name / self.job_name_safe / report_name
        self.images_dir = self.report_dir / "images"
        
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"📂 报告将保存至: {self.report_dir}")
        print(f"📊 图表将保存至: {self.images_dir}")

    def load_data(self):
        """
        根据作业规范 (job_spec) 加载所有必需的数据。
        (已更新，支持 data_asset_folder 和路径文件自动检测)
        """
        print("\n--- 步骤1: 加载分析所需文件 ---")

        # 确定数据文件的真实存放位置
        data_folder = self.spec.get('data_asset_folder', self.asset_name)
        if data_folder != self.asset_name:
            print(f"ℹ️  正在从 '{data_folder}' 文件夹加载数据，用于分析 '{self.asset_name}' 资产。")

        # 1. 自动检测并加载路径文件
        paths_dir_key = self.spec['paths_dir_key']
        paths_dir = getattr(pp, paths_dir_key, None)
        if paths_dir is None:
             raise AttributeError(f"Project_Path.py 缺少 '{paths_dir_key}' 变量")

        target_dir = paths_dir / data_folder
        if not target_dir.exists():
            raise FileNotFoundError(f"数据目录未找到: {target_dir}")

        # 从配置中获取基础名
        base_name = self.spec.get('paths_filename_base')
        if not base_name:
            raise ValueError(f"作业 '{self.job_name_safe}' 的配置中缺少 'paths_filename_base' 键")

        # 定义搜索模式
        pattern = f"{base_name}_*_samples.npy"
        print(f"ℹ️  正在 '{target_dir}' 中自动搜索最新路径文件，模式: '{pattern}'...")

        # 搜索所有匹配文件，并按修改时间排序（最新的在最前面）
        try:
            matching_files = sorted(
                target_dir.glob(pattern), 
                key=os.path.getmtime, 
                reverse=True
            )
        except Exception as e:
            raise IOError(f"搜索文件时出错: {e}")

        if not matching_files:
            raise FileNotFoundError(f"自动检测失败：在 '{target_dir}' 中未找到匹配 '{pattern}' 的路径文件。")

        paths_file = matching_files[0] # 获取最新的那个文件

        if len(matching_files) > 1:
            print(f"   ⚠️ 警告: 找到 {len(matching_files)} 个匹配文件。将自动使用最新的一个: {paths_file.name}")

        self.generated_paths = np.load(paths_file)
        print(f"✅ 已加载路径: {paths_file} (形状: {self.generated_paths.shape})")

        # --- 2. 加载验证集 (已修改为从中央 merge 文件过滤) ---
        print("   🔄 正在加载 *中央* 验证数据 (testing_data_merged.csv) 作为基准...")
        
        # 假设基准数据都在 Testing_DATA_DIR
        val_dir_key = "Testing_DATA_DIR" 
        val_base_dir = getattr(pp, val_dir_key, None)
        if val_base_dir is None: raise AttributeError(f"Project_Path.py 缺少 '{val_dir_key}'")

        # 假设中央文件名 (如果你的文件名不同，请在此处修改)
        central_file_name = 'testing_data_merged.csv'
        val_file_path = val_base_dir / central_file_name
        
        if not val_file_path.exists():
            # 备用：万一文件名是 val_df.csv
            val_file_path_fallback = val_base_dir / 'val_df.csv'
            if val_file_path_fallback.exists():
                val_file_path = val_file_path_fallback
            else:
                 raise FileNotFoundError(f"中央 merge 文件未找到: {val_file_path} 或 {val_file_path_fallback}")

        try:
            full_val_df = pd.read_csv(val_file_path)
            print(f"   ✅ 中央验证文件加载成功: {val_file_path}")

            # 关键: *过滤* DataFrame 以获取当前资产的子集
            asset_to_filter = self.asset_name # e.g., 'CSI1000'
            
            # !! 关键假设 !! 
            # 假设用于过滤的列名是 'asset_name'
            # 如果你的列名不同 (例如 'index')，请在此处修改
            filter_column_name = 'asset_underlying' 
            
            print(f"   🔄 正在过滤 '{asset_to_filter}' 的子集 (基于列 '{filter_column_name}')...")
            if filter_column_name not in full_val_df.columns:
                 raise KeyError(f"中央 merge 文件 '{val_file_path}' 中缺少用于过滤的列: '{filter_column_name}'")

            self.val_df = full_val_df[full_val_df[filter_column_name] == asset_to_filter].copy()

            if self.val_df.empty:
                raise ValueError(f"在 '{val_file_path}' 中找不到资产 '{asset_to_filter}' 的任何数据。")
            
            print(f"✅ 验证子集加载成功。将使用 {len(self.val_df)} 个 '{asset_to_filter}' 市场条件。")
        
        except Exception as e:
            print(f"❌ 加载或处理中央 merge 文件时出错: {e}")
            raise

        # 3. (条件) 加载 DataProcessor (逻辑更新以匹配路径)
        if self.model_type == 'unet':
            proc_dir_key = self.spec['processor_dir_key']
            proc_dir = getattr(pp, proc_dir_key, None)
            if proc_dir is None:
                raise AttributeError(f"Project_Path.py 缺少 '{proc_dir_key}' 变量")

            processor_folder = self.spec.get('processor_source_folder', data_folder)
            processor_type_subfolder = self.spec.get('processor_type_subfolder', None)
            if processor_type_subfolder:
                proc_file_path = proc_dir / processor_type_subfolder / processor_folder / self.spec['processor_filename']
            else:
                proc_file_path = proc_dir / processor_folder / self.spec['processor_filename']

            if not proc_file_path.exists():
                raise FileNotFoundError(f"DataProcessor 未找到: {proc_file_path}")

            self.data_processor = joblib.load(proc_file_path)
            print(f"✅ 已加载 DataProcessor: {proc_file_path}")
        else:
            print("ℹ️  MC 模型分析无需加载 DataProcessor。")

    def run_analysis(self, indices_to_analyze):
        """
        对选定的索引执行循环分析。
        """
        print(f"\n--- 步骤2: 开始分析 {len(indices_to_analyze)} 个市场条件 ---")
        all_results = []
        
        num_conditions = min(len(indices_to_analyze), self.generated_paths.shape[0], len(self.val_df))
        
        for idx in tqdm(indices_to_analyze, desc="分析进度", total=num_conditions):
            if idx >= num_conditions:
                print(f"警告: 索引 {idx} 超出数据范围，停止。")
                break
                
            try:
                result = self._analyze_single_condition(idx)
                all_results.append(result)
            except Exception as e:
                print(f"❌ 条件 {idx} 分析失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✅ 已完成 {len(all_results)} 个条件的分析。")
        return all_results

    def _analyze_single_condition(self, idx):
        """
        对单个条件执行全套分析 (调用函数库)。
        这是 UNet (5.1) 和 MC (5.2) 逻辑的融合点。
        """
        condition_info = self.val_df.iloc[idx]
        real_price_path = np.array(ast.literal_eval(condition_info['price_series']))
        all_restored_prices = None # 最终需要的形状: (N_sim, SeqLen)
        
        if self.model_type == 'unet':
            # UNet 输出是对数收益率，形状: (N_sim, Channels=1, SeqLen)
            ensemble_log_returns = self.generated_paths[idx]
            # (确保 recover 函数返回的是 (N_sim, SeqLen) 或进行相应调整)
            all_restored_prices = lib.recover_price_paths_from_returns(
                ensemble_log_returns, 
                condition_info['start_price'], 
                self.data_processor
            )
            # 如果 recover 函数返回 (N_sim, 1, SeqLen)，也需要 squeeze
            if all_restored_prices.ndim == 3 and all_restored_prices.shape[1] == 1:
                all_restored_prices = all_restored_prices.squeeze(axis=1)

        else: # 'mc' (GBM/GARCH)
            # MC 保存的文件形状是 (N_cond * N_sim, 1, SeqLen)
            
            # --- !! 修改这里的逻辑 !! ---
            # 1. 计算每个条件有多少个 simulation (N_sim)
            #    (假设所有条件都有相同数量的 simulation)
            num_conditions_in_file = len(self.val_df) # 或者从文件名解析？更安全的是用 val_df 长度
            if self.generated_paths.shape[0] % num_conditions_in_file != 0:
                raise ValueError("加载的 MC 路径文件总行数无法被条件数量整除，形状可能不匹配。")
            n_sim = self.generated_paths.shape[0] // num_conditions_in_file
            
            # 2. 计算当前条件 idx 对应的切片范围
            start_row = idx * n_sim
            end_row = (idx + 1) * n_sim
            
            # 3. 提取属于该条件的所有 simulation，形状 (N_sim, 1, SeqLen)
            ensemble_prices_for_idx = self.generated_paths[start_row:end_row]
            
            # 4. 现在 squeeze(axis=1) 可以正常工作了，得到 (N_sim, SeqLen)
            all_restored_prices = ensemble_prices_for_idx.squeeze(axis=1)
            # --- !! 修改结束 !! ---

        if all_restored_prices is None or all_restored_prices.ndim != 2:
             raise ValueError(f"条件 {idx}: 未能正确准备价格路径，最终形状为 {all_restored_prices.shape if all_restored_prices is not None else 'None'}")
        
        # --- 2. 通用步骤：计算收益率 ---
        valid_length = len(real_price_path)
        real_log_returns = np.diff(np.log(real_price_path))
        
        valid_prices = all_restored_prices[:, :valid_length]
        valid_prices = valid_prices[~np.isnan(valid_prices).any(axis=1)] # 移除 GARCH 的 NaN
        if len(valid_prices) == 0:
            raise ValueError(f"条件 {idx} 没有有效的生成路径（可能全为NaN）。")
            
        generated_returns = np.diff(np.log(valid_prices), axis=1)
        
        # --- 3. 通用步骤：调用函数库 (使用 5.1 的高级标准) ---
        
        # 使用 5.1 的高级统计
        stats_results = lib.calculate_comprehensive_statistics(real_log_returns, generated_returns)
        
        # 绘图
        plot_paths = {}
        model_name = self.spec['display_name']
        
        plot_paths['fan_chart'] = self.images_dir / f"fan_chart_cond_{idx}.png"
        lib.plot_enhanced_fan_chart(valid_prices, real_price_path, condition_info, plot_paths['fan_chart'], model_name)
        
        plot_paths['qq_plot'] = self.images_dir / f"qq_plot_cond_{idx}.png"
        qq_r_squared = lib.plot_qq_comparison(real_log_returns, generated_returns, condition_info, plot_paths['qq_plot'], model_name)
        stats_results['qq_r_squared'] = qq_r_squared

        # 使用 5.2 的补充图
        plot_paths['distribution'] = self.images_dir / f"dist_plot_cond_{idx}.png"
        lib.plot_return_distribution(real_log_returns, generated_returns, condition_info, plot_paths['distribution'], model_name)
        
        plot_paths['vol_clustering'] = self.images_dir / f"vol_cluster_cond_{idx}.png"
        lib.plot_volatility_clustering(real_log_returns, generated_returns, condition_info, plot_paths['vol_clustering'], model_name)

        return {
            'condition_idx': idx,
            'condition_info': condition_info.to_dict(),
            'statistics': stats_results,
            'plots': {k: str(v) for k, v in plot_paths.items()} # 存储为字符串路径
        }

    def generate_report(self, all_results):
        """
        计算最终评分并保存所有报告 (CSV, Markdown)。
        (逻辑来自 5.1)
        """
        if not all_results:
            print("❌ 没有分析结果，无法生成报告。")
            return
            
        print(f"\n--- 步骤3: 计算模型综合评分 ---")
        # 使用 5.1 的高级评分系统
        model_scores = lib.calculate_model_score(all_results)
        
        print(f"   模型综合评分: {model_scores['overall_score']:.2f}/100")
        print(f"   评分标准差: {model_scores['score_std']:.2f}")
        print(f"   模型等级: {model_scores['grade']}")

        print(f"\n--- 步骤4: 生成评估报告 ---")
        self._save_results_to_csv(all_results, model_scores)
    
    def _save_results_to_csv(self, all_results, model_scores):
        """私有辅助函数：保存 CSV (来自 5.1)"""
        
        detailed_stats = [
            {'condition_idx': r['condition_idx'], **r['statistics']} 
            for r in all_results
        ]
        detailed_df = pd.DataFrame(detailed_stats)
        detailed_path = self.report_dir / 'detailed_statistics.csv'
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        
        # 汇总 (来自 5.1)
        summary_df = pd.DataFrame(columns=['metric', 'mean', 'std', 'min', 'max'])
        metrics_mapping = {
            'Mean Difference': 'mean_diff', 'Volatility Difference': 'vol_diff',
            'Skewness Difference': 'skew_diff', 'Kurtosis Difference': 'kurt_diff',
            'AD Statistic': 'ad_statistic', 'AD Rejection Level (%)': 'ad_rejection_level',
            'KS Statistic': 'ks_statistic', 'KS P-value': 'ks_pvalue',
            'Wasserstein Distance': 'wasserstein_distance', 'QQ R-squared': 'qq_r_squared',
            'VaR 1% Difference': 'var_1_diff', 'VaR 99% Difference': 'var_99_diff',
            'CVaR 1% Difference': 'cvar_1_diff'
        }
        
        summary_data = []
        for metric_name, col_name in metrics_mapping.items():
            if col_name in detailed_df.columns:
                values = detailed_df[col_name].dropna()
                if col_name == 'ad_rejection_level': values *= 100
                summary_data.append({
                    'metric': metric_name,
                    'mean': values.mean(), 'std': values.std(),
                    'min': values.min(), 'max': values.max()
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.report_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        # 评分 (来自 5.1)
        score_report = {
            'overall_score': [model_scores['overall_score']],
            'score_std': [model_scores['score_std']],
            'grade': [model_scores['grade']],
            'total_conditions_analyzed': [len(all_results)],
            'analysis_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'model_type': [self.spec['display_name']],
            'underlying_asset': [self.asset_name]
        }
        score_df = pd.DataFrame(score_report)
        score_path = self.report_dir / 'model_evaluation_report.csv'
        score_df.to_csv(score_path, index=False, encoding='utf-8-sig')
        
        print(f"   ✅ CSV 报告已保存至: {self.report_dir}")
        self._generate_markdown_report(all_results, summary_df, model_scores)

    def _generate_markdown_report(self, all_results, summary_df, model_scores):
        """私有辅助函数：生成 Markdown 报告 (来自 5.1 & 5.2)"""
        report_path = self.report_dir / 'Model_Validation_Report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 模型验证报告: {self.spec['display_name']}\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**标的资产**: {self.asset_name}\n\n")
            
            f.write(f"## 1. 模型综合评分\n\n")
            f.write(f"| 指标 | 结果 |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **总分 (Overall Score)** | **{model_scores['overall_score']:.2f} / 100** |\n")
            f.write(f"| **评级 (Grade)** | **{model_scores['grade']}** |\n")
            f.write(f"| 评分标准差 (Score Std.) | {model_scores['score_std']:.2f} |\n")
            f.write(f"| 分析的条件总数 | {len(all_results)} |\n\n")

            f.write(f"## 2. 核心指标汇总 (所有条件平均)\n\n")
            f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
            
            f.write(f"\n\n## 3. 分场景详细分析 (抽样)\n\n")
            f.write(f"*仅显示前 10 个分析的场景。*\n\n")

            for result in all_results[:10]: # 最多显示前10个
                idx = result['condition_idx']
                stats_df = pd.DataFrame([result['statistics']]).T.reset_index()
                stats_df.columns = ['Metric', 'Value']
                
                f.write(f"\n---\n\n### 场景 (Condition) {idx}\n\n")
                condition_df = pd.Series(result['condition_info']).to_frame().T
                f.write(f"**初始条件:**\n")
                f.write(condition_df[['start_price', 'volatility', 'risk_free_rate', 'actual_trading_days']].to_markdown(index=False))
                f.write(f"\n\n**统计指纹 (Statistics):**\n")
                f.write(stats_df.to_markdown(index=False, floatfmt=".4f"))
                f.write(f"\n\n**图表:**\n\n")
                
                # 动态生成相对路径
                img_path = Path(result['plots']['fan_chart']).relative_to(self.report_dir)
                f.write(f"![Fan Chart {idx}]({img_path.as_posix()})\n")
                img_path = Path(result['plots']['qq_plot']).relative_to(self.report_dir)
                f.write(f"![QQ Plot {idx}]({img_path.as_posix()})\n")
                img_path = Path(result['plots']['distribution']).relative_to(self.report_dir)
                f.write(f"![Distribution {idx}]({img_path.as_posix()})\n")
                img_path = Path(result['plots']['vol_clustering']).relative_to(self.report_dir)
                f.write(f"![Volatility Clustering {idx}]({img_path.as_posix()})\n")

        print(f"   ✅ Markdown 报告已保存至: {report_path}")