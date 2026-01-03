# data_explainer_engine.py
#
# 这是一个通用的数据分析引擎。

import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import datetime

current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))



# --- 导入自定义模块 ---
from . import data_explainer_library as lib

# --- 路径导入 ---
try:
    current_file_dir = Path(__file__).parent.resolve()
    project_root = current_file_dir.parent
    sys.path.append(str(project_root))
    import Project_Path as pp
except (ImportError, NameError) as e:
    print(f"❌ 严重错误: 未能导入 Project_Path: {e}")
    sys.exit(1)

class DataExplainerEngine:
    """
    一个通用的、由配置驱动的数据集分析引擎。
    """
    def __init__(self, asset_name, job_spec):
        self.asset_name = asset_name
        self.spec = job_spec
        self.job_name_safe = self.spec['job_name'].replace(' ', '_').replace('(', '').replace(')', '')
        
        print(f"🚀 正在初始化数据分析引擎 (作业: {self.spec['job_name']})...")
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._setup_paths()
        
        self.df = None

    def _setup_paths(self):
        """创建所有输出目录 - 在 Report_Results_DIR 下添加 dataset_report 子目录"""
        report_name = f"{self.timestamp}_{self.job_name_safe}_report"
        
        # 1. 获取通用的报告根目录
        base_report_dir = getattr(pp, "Report_Results_DIR", None) 
        if base_report_dir is None:
             raise AttributeError("Project_Path.py 中未定义 'Report_Results_DIR'。请检查您的 Project_Path.py 文件。")
        
        # --- 核心修改 ---
        # 2. 在通用目录下先进入 'dataset_report' 子目录，然后再按资产和时间戳创建
        #    例如：.../Results/Report_Results/dataset_report/CSI1000/20251021.../
        self.report_dir = base_report_dir / "dataset_report" / self.asset_name / report_name 
        # --- 修改结束 ---
        
        self.images_dir = self.report_dir / "images" # 图片仍然放在特定报告文件夹内
        
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"📂 报告将保存至: {self.report_dir}")
    
    def load_data(self):
        """根据作业规范 (job_spec) 加载数据集，并根据 asset_name 选择正确的文件名。"""
        print("\n--- 步骤1: 加载数据文件 ---")
        
        # 1. 获取基础输入目录 (例如 Trainning_DATA_DIR)
        base_input_dir = getattr(pp, self.spec['input_dir_key'])
        
        # --- 核心修改：根据 asset_name 和 config 选择文件名 ---
        filename = None
        input_dir = base_input_dir # 默认使用基础目录
        
        if self.asset_name.lower() == 'all':
            filename = self.spec.get('input_filename_all') # 获取 'all' 的文件名
            if filename:
                 print(f"   检测到 'all' 资产，尝试加载合并文件: {filename}")
            else:
                 raise ValueError(f"❌ 错误: 作业 '{self.job_name_safe}' 没有为 'all' 资产配置 input_filename_all。")
        else:
            filename = self.spec.get('input_filename_asset') # 获取特定资产的文件名
            if filename:
                input_dir = base_input_dir / self.asset_name # 特定资产需要在子目录查找
                print(f"   加载特定资产 '{self.asset_name}' 的数据: {filename} 从 {input_dir}")
            else:
                 raise ValueError(f"❌ 错误: 作业 '{self.job_name_safe}' 没有为特定资产配置 input_filename_asset。")
                 
        input_file = input_dir / filename
        # --- 修改结束 ---

        if not input_file.exists():
            error_msg = f"数据集文件未找到: {input_file}\n"
            error_msg += f"   请检查 Project_Path.py 中的 '{self.spec['input_dir_key']}' 设置，"
            if self.asset_name.lower() != 'all' and filename == self.spec.get('input_filename_asset'):
                error_msg += f"以及是否存在 '{self.asset_name}' 子目录，"
            error_msg += f"并确保文件 '{filename}' 存在。"
            raise FileNotFoundError(error_msg)
        
        # 加载逻辑保持不变
        if str(input_file).endswith('.csv'):
            self.df = pd.read_csv(input_file)
        elif str(input_file).endswith('.pkl'):
            self.df = pd.read_pickle(input_file)
        else:
            raise ValueError(f"不支持的文件格式: {input_file}")
            
        print(f"✅ 已加载数据集: {input_file} (形状: {self.df.shape})")

    def run_analysis(self):
        """
        执行完整的分析流程：统计、绘图、报告。
        """
        print(f"\n--- 步骤2: 开始分析 {self.spec['job_name']} ---")
        if self.df is None:
            print("❌ 数据未加载。请先调用 .load_data()")
            return

        # 1. 识别列类型
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        # 排除掉常见的ID或日期戳，它们不适合绘制直方图
        cols_to_exclude = ['id', 'ID', 'Id', 'index', 'timestamp']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]

        # 2. 生成基础统计文本
        stats_text = lib.get_basic_stats(self.df, self.spec['job_name'])

        # 3. 绘制核心图表
        dist_plot = lib.plot_numeric_distributions(self.df, numeric_cols, self.images_dir)
        corr_plot = lib.plot_correlation_heatmap(self.df, numeric_cols, self.images_dir)

        # 4. (可选) 运行耗时的 price_series 分析
        path_stats_text = ""
        if self.spec.get('run_price_series_analysis', False):
            path_stats_text = lib.analyze_price_series_stats(self.df)
        else:
            path_stats_text = "ℹ️ 已跳过 'price_series' 深度分析（按配置）。"

        # 5. 汇编最终报告
        print("\n--- 步骤3: 汇编最终报告 ---")
        report_path = self.report_dir / "Data_Analysis_Report.md"
        
        # 使图表路径相对于报告文件
        dist_plot_rel = Path(dist_plot).relative_to(self.report_dir) if dist_plot else None
        corr_plot_rel = Path(corr_plot).relative_to(self.report_dir) if corr_plot else None
        
        lib.generate_data_markdown_report(
            report_path, 
            stats_text, 
            dist_plot_rel, 
            corr_plot_rel, 
            path_stats_text
        )
        
        print(f"🎉 作业 {self.spec['job_name']} 已成功完成！")