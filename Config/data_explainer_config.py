
# Config/data_explainer_config.py
# (已修改以显式区分 'all' 和 'asset' 文件名)

DATA_JOBS = {
    
    "explain_train_data": {
        "job_name": "训练集", # 保持名称通用
        "input_dir_key": "Trainning_DATA_DIR", 
        # --- 新增/修改 ---
        "input_filename_all": "trainning_data_merged.csv", # 用于 TARGET_ASSET = 'all'
        "input_filename_asset": "train_df.csv",         # 用于 TARGET_ASSET = 'CSI1000' 等
        # -----------
        "run_price_series_analysis": True 
    },
    
    # 验证集通常是按资产划分的，但也可能有一个合并的版本？
    # 我们假设验证集只有按资产的文件 val_df.csv
    "explain_val_data": {
        "job_name": "验证集",
        "input_dir_key": "Testing_DATA_DIR", 
        # --- 新增/修改 ---
        "input_filename_all": None,                     # 假设没有合并的验证文件
        "input_filename_asset": "val_df.csv",         
        # -----------
        "run_price_series_analysis": True
    },

    # 测试集似乎主要是合并后的文件
    "explain_test_data": { 
        "job_name": "测试集 (Merged)",
        "input_dir_key": "Testing_DATA_DIR", 
        # --- 新增/修改 ---
        "input_filename_all": "testing_data_merged.csv", 
        "input_filename_asset":"val_df.csv",
        # -----------
        "run_price_series_analysis": False 
    }
}