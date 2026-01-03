
# Config/garch_fitter_config.py
#
# GARCH 模型拟合任务的“模板”配置文件
# (已修正：所有拟合只使用训练数据)

# 唯一模板: 永远使用 *训练数据* 进行拟合
GARCH_FIT_TEMPLATE = {
    # 1. GARCH(1,1) 的分布假设 ('Normal' 或 't')
    "distribution": "t", #

    # 2. 数据源 (永远是训练集)
    "data_source_dir_key": "Trainning_DATA_DIR", # 修正：只使用训练集
    "central_data_file": "trainning_data_merged.csv", # 修正：只使用训练集
    
    # 3. 过滤设置
    "filter_column": "asset_underlying", #
    "price_series_column": "price_series", #

    # 4. 输出设置
    "output_dir_key": "Model_Results_DIR", #
    "output_subfolder": "Garch_Fit_Results", #
    "output_filename": "garch_params.json" #
}