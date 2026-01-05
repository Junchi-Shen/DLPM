# Config/Diffusion_config.py

main_config = {
    "underlying_asset": 'all',
    "model_type": 'unet',
    "volatility_scale": 0.09,
    "input_sequence_length": 252, # Matches diffusion_params 'seq_length'
    "base_trading_days": 252,

    # Diffusion process parameters
    'timesteps': 1000,
    'objective': 'pred_v',       # Example objective
    'auto_normalize': False,     # Example setting
    'seq_length': 252,           # Should match input_sequence_length
    
    # DLPM parameters (可选，如果使用DLPM)
    'use_dlpm': True,  # 是否使用DLPM而不是标准DDPM
    'dlpm_alpha': 1.7,  # DLPM的alpha参数 (1 < alpha <= 2, alpha=2时退化为高斯)
    'dlpm_isotropic': True,  # DLPM是否各向同性
    'dlpm_rescale_timesteps': True,  # DLPM是否重新缩放时间步
    'dlpm_scale': 'scale_preserving',  # DLPM调度类型

    # Training parameters
    'train_num_steps': 18000,
    'warmup_ratio': 0.25,
    'train_batch_size': 64,      # Example batch size
    'train_lr': 1e-6,            # Example learning rate
    'ema_decay': 0.995,          # Example EMA decay
    'amp': True,                 # Example mixed precision setting
    # 'trainer_params': {...}, # Optionally group trainer params here

    # --- U-Net Specific Parameters ---
    "unet_params": {
        "dim": 64,                 # <<<--- **ADD THIS LINE** (Choose 8, 64, or another suitable value)
        "dim_mults": (1, 2, 4, 8), # Parameters for Unet1D
        "channels": 1,             # Parameters for Unet1D
        "dropout": 0.1,            # Parameters for Unet1D
        # "learned_variance": False, # Add other Unet1D params here if needed
    },
    # ---

    # --- (Optional but Recommended) Condition Network Parameters ---
    "use_enhanced_condition_network": True, # Control whether to use it
    "cond_net_params": {
        "output_dim": 128,          # Must match Unet1D's expected cond_dim if used
        "country_emb_dim": 64,
        "index_emb_dim": 128,
        "numerical_proj_dim": 32,
        "hidden_dim": 256
    },
    # ---

    # --- (Optional) Trainer Parameters (Alternative grouping) ---
    # "trainer_params": {
    #     "train_batch_size": 64,
    #     "train_lr": 1e-6,
    #     "gradient_accumulate_every": 1,
    #     "ema_decay": 0.995,
    #     "amp": True,
    #     "save_and_sample_every": 1000
    # },
}