# Config/Diffusion_config_DLPM.py
# DLPM (Denoising Lévy Probabilistic Model, arXiv:2407.18609) 配置
# 使用论文标准损失：A条件化的 eps 预测 + Lp 目标

main_config = {
    "underlying_asset": 'all',
    "country": ['China'],
    "model_type": 'unet',
    "volatility_scale": 0.09,
    "input_sequence_length": 252, # Matches diffusion_params 'seq_length'
    "base_trading_days": 252,

    # Diffusion process parameters
    'timesteps': 500,
    'auto_normalize': False,
    'seq_length': 252,

    # DLPM parameters
    'use_dlpm': True,
    'dlpm_alpha': 1.8,            # 稳定指数 α ∈ (1, 2]; α=2 退化为高斯DDPM
    'dlpm_isotropic': True,
    'dlpm_scale': 'scale_preserving',

    # 论文标准损失 (Sec. 3, arXiv:2407.18609)
    'dlpm_lploss': 2.0,           # Lp 目标; p=2 -> 每样本 L2 范数(不平方), 对 α>1 有限
    'dlpm_monte_carlo_outer': 1,  # a_t 的蒙特卡洛外层样本数
    'dlpm_monte_carlo_inner': 1,  # 内层 z 样本数
    'dlpm_loss_monte_carlo': 'mean',  # 'mean' | 'median' (median-of-means)
    'dlpm_clamp_a': 100.0,        # 论文建议的数值稳定性截断 (仅截 a_t 的极端抽样)
    'dlpm_clamp_eps': None,

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
