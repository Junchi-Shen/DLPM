import Generator.path_simulators as ps

GENERATOR_JOBS = {

    # --- Q-Model 作业 ---
    "GBM": { # <--- 名称不含数量
        "type": "mc",
        "simulator_function": ps.simulate_gbm,
        "job_name": "GBM Path Generation",
        "output_filename_base": "gbm_generated_paths", # <-- 基础名称
        "output_dir": "Path_Generator_Results_DIR",
        "params": {
            "n_simulations": 1024, # <-- 保留默认值
            "n_steps_total": 252
        },
        "save_extra_outputs": False
    },

    "GARCH": { # <--- 名称不含数量
        "type": "mc",
        "simulator_function": ps.simulate_garch,
        "job_name": "GARCH Path Generation (fitted)",
        "output_filename_base": "garch_paths_fitted", # <-- 基础名称
        "output_dir": "Path_Generator_Results_DIR",
        "params": {
            "n_simulations": 1024, # <-- 保留默认值
            "n_steps_total": 252,
            "innov_dist": "t",
            "seed": 42
        },
        "garch_params_filename": "garch_params.json",
        "garch_params_dir_key": "Model_Results_DIR",
        "garch_params_subfolder": "Garch_Fit_Results",
        "save_extra_outputs": False
    },

    # --- P-Model 作业 (Diffusion) ---
    "UNet": { # <--- 名称不含数量 (假设这是 'all' 资产模型)
        "type": "diffusion",
        "job_name": "UNet Diffusion Path Generation (all assets model)",
        "output_filename_base": "unet_generated_paths", # <-- 基础名称
        "output_dir": "Path_Generator_Results_DIR",
        "model_source_folder": "all",

        "model_loader_params": {
            "model_dir_root_key": "Model_Results_DIR",
            "processor_filename": "data_processor_all.pkl",
            "model_filename": "unet_conditional_model_all.pth",
            "condition_network_filename": "condition_network_all.pth"
        },
        "unet_config": { # 确保参数匹配
            "model_type": 'unet',
            "model_params": {"dim": 64, "dim_mults": (1, 2, 4, 8), "channels": 1,"dropout": 0.1},
        },
        "cond_net_config": { # 确保参数匹配
             "output_dim": 128, "country_emb_dim": 64, "index_emb_dim": 128,
             "numerical_proj_dim": 32, "hidden_dim": 256
        },
        "diffusion_config": { # 确保参数匹配
            "seq_length": 252, "timesteps": 1000, "sampling_timesteps": 200,
            "objective": 'pred_v', "auto_normalize": False, "ddim_sampling_eta": 0.75,  # 改为0.5增加随机性（0.0是完全确定性，会导致路径相同）
        },
        "generation_params": {
            "runner_function": ps.run_diffusion_mega_batch,
            "num_paths_to_generate": 100, # <-- 保留默认值
            "generation_batch_size": 32768,
        },
        "save_extra_outputs": False
    },
    
    # --- P-Model 作业 (DLPM) ---
    "DLPM": { # <--- DLPM模型
        "type": "diffusion",
        "job_name": "DLPM Path Generation (all assets model)",
        "output_filename_base": "dlpm_generated_paths", # <-- 基础名称
        "output_dir": "Path_Generator_Results_DIR",
        "model_source_folder": "all",
        "use_dlpm": True,  # <-- 标记使用DLPM
        "dlpm_alpha": 1.7,  # <-- DLPM的alpha参数 (1 < alpha <= 2)

        "model_loader_params": {
            "model_dir_root_key": "Model_Results_DIR",
            "processor_filename": "data_processor_all.pkl",
            "model_filename": "unet_conditional_model_all.pth",
            "condition_network_filename": "condition_network_all.pth"
        },
        "unet_config": { # 确保参数匹配
            "model_type": 'unet',
            "model_params": {"dim": 64, "dim_mults": (1, 2, 4, 8), "channels": 1,"dropout": 0.1},
        },
        "cond_net_config": { # 确保参数匹配
             "output_dim": 128, "country_emb_dim": 64, "index_emb_dim": 128,
             "numerical_proj_dim": 32, "hidden_dim": 256
        },
        "diffusion_config": { # 确保参数匹配 - 使用DDIM（DLIM方法）
            "seq_length": 252, "timesteps": 1000, "sampling_timesteps": 200,  # 使用100步DDIM采样（DLIM方法）
            "objective": 'pred_v', "auto_normalize": False, "ddim_sampling_eta": 0.75,  # eta=0.5增加随机性（0.0是完全确定性，会导致路径相同）
        },
        "generation_params": {
            "runner_function": ps.run_diffusion_mega_batch,
            "num_paths_to_generate": 100, # <-- 保留默认值
            "generation_batch_size": 32768,
        },
        "save_extra_outputs": False
    },
    # 可以添加特定资产的 UNet 作业，例如:
    # "UNet_CSI1000": { ... }
}