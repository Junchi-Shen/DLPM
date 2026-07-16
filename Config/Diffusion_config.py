# Config/Diffusion_config.py
# DDPM (Gaussian diffusion) 配置 — 供 ddpm_complex / ddpm_simple 两种损失模式共用

main_config = {
    "underlying_asset": 'all',
    "country": ['China'],
    "model_type": 'unet',
    "volatility_scale": 0.09,
    "input_sequence_length": 252, # Matches diffusion_params 'seq_length'
    "base_trading_days": 252,

    # Diffusion process parameters
    'timesteps': 500,             # 与 DLPM 保持一致，保证公平比较
    'objective': 'pred_x0',
    'auto_normalize': False,
    'seq_length': 252,

    # 损失模式: 'complex' = MSE + 金融统计正则项(波动聚集/厚尾/漂移/分位数/频谱等)
    #           'simple'  = 仅标准去噪MSE (DDPM原始目标)
    # 由 Pipelines/4-Run_Diffusion.py 的 --variant 参数覆盖
    'loss_mode': 'complex',

    'use_dlpm': False,

    # Training parameters
    'train_num_steps': 4000,
    'warmup_ratio': 0.15,
    'train_batch_size': 64,
    'train_lr': 1e-4,
    'ema_decay': 0.995,
    'amp': False,

    # --- U-Net Specific Parameters ---
    "unet_params": {
        "dim": 64,
        "dim_mults": (1, 2, 4, 8),
        "channels": 1,
        "dropout": 0.1,
    },

    # --- Condition Network Parameters ---
    "use_enhanced_condition_network": True,
    "cond_net_params": {
        "output_dim": 128,          # Must match Unet1D's expected cond_dim if used
        "country_emb_dim": 64,
        "index_emb_dim": 128,
        "numerical_proj_dim": 32,
        "hidden_dim": 256
    },
}
