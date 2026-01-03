
# Config/path_explainer_config.py (修正版)

PATH_JOBS = {
    
    "validate_unet": { 
        "model_type": "unet", 
        "display_name": "UNet (P-Model)",
        "paths_dir_key": "Path_Generator_Results_DIR", 
        
        # 修正: "pathss" -> "paths"
        "paths_filename_base": "unet_generated_paths", # <-- 修正 (应与 generator_config.py 一致)
        
        "data_asset_folder":  "CSI1000", 
        "processor_source_folder": "all",
        "processor_dir_key": "Model_Results_DIR", 
        "processor_filename": "data_processor_all.pkl" 
    },
    
    "validate_gbm": { 
        "model_type": "mc",
        "display_name": "GBM (Q-Model)",
        "paths_dir_key": "Path_Generator_Results_DIR",
        
        # 修正: "paaths" -> "paths"
        "paths_filename_base": "gbm_generated_paths", # <-- 修正 (应与 generator_config.py 一致)
        
        "data_asset_folder": "CSI1000", 
        "processor_dir_key": None,
        "processor_filename": None
    },
    
    "validate_garch": { 
        "model_type": "mc",
        "display_name": "GARCH (Q-Model)",
        "paths_dir_key": "Path_Generator_Results_DIR",
        
        # 修正: "fittted" -> "fitted"
        "paths_filename_base": "garch_paths_fitted", # <-- 修正 (应与 generator_config.py 一致)
        
        "data_asset_folder": "CSI1000", 
        "processor_dir_key": None,
        "processor_filename": None
    }
}