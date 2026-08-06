# Hybrid DLPM configuration for path-aware financial generation.
#
# This keeps the current DLPM architecture but expands the condition vector
# with strictly past market context. Financial regularizers are active in the
# hybrid loss; the base DLPM variant remains available as a clean ablation.

from .Diffusion_config_DLPM import main_config as _base

main_config = dict(_base)

main_config.update({
    "use_history_context": True,
    "condition_price_transform": "log",
    "train_data_filename": "trainning_data_merged_state_context.csv",
    "test_data_filename": "testing_data_merged_state_context.csv",
    "experiment_tag": "hybrid_state_context_v1",
    "dlpm_isotropic": False,
    "dlpm_alpha": 1.9,
    "dlpm_lploss": 1.0,

    # Use a conservative setting closer to the old local prototype for the
    # first CSI1000 validation run. Full-data training should be started only
    # after path diagnostics pass on this smaller validation.
    "train_num_steps": 12000,
    "train_lr": 1e-5,
    "timesteps": 1000,

    "history_feature_columns": [
        "hist_returns_60", "hist_returns_252", "hist_vol_path_60",
        "hist_trend_20", "hist_trend_60", "hist_trend_252",
        "hist_max_drawdown_60", "hist_max_drawdown_252",
        "hist_rv_20", "hist_rv_60", "hist_rv_252",
        "hist_downside_vol_60", "hist_last_return",
        "hist_valid_60", "hist_valid_252",
        "hist_trend_gap_20_60", "hist_trend_gap_60_252",
        "hist_momentum_accel_20", "hist_current_drawdown_60",
        "hist_drawdown_recovery_60", "hist_vol_ratio_20_60",
        "hist_vol_ratio_60_252", "hist_recent_reversal_10",
    ],
    "history_feature_lengths": {
        "hist_returns_60": 60,
        "hist_returns_252": 252,
        "hist_vol_path_60": 60,
    },

    # Hybrid loss should act as a path-shape stabilizer, not as the primary
    # denoising objective. Use a small scale and warm-up to avoid early spikes.
    "use_financial_regularizers": True,
    "financial_loss_scale": 0.07,
    "financial_loss_warmup_steps": 1000,
    "loss_diagnostics_every": 100,
    "financial_regularizer_clip": 2.5,
    "financial_return_clip": 1.25,
    "sample_x0_clip": 1.35,
    "financial_loss_weights": {
        "pointwise_x0": 2.5,
        "return_range": 5.0,
        "terminal_drift": 0.7,
        "cumulative_return": 0.7,
        "global_vol": 3.0,
        "mean_abs_return": 5.0,
        "vol_clustering": 0.5,
        "volatility_clustering_acf": 0.35,
        "return_autocorr_1": 0.20,
        "return_autocorr_5": 0.10,
        "leverage_effect": 0.10,
        "tail_quantile": 0.1,
        "abs_tail_quantile": 4.5,
        "tail_iqr_ratio": 0.05,
        "tail_exceedance_0.5": 1.0,
        "tail_exceedance_1": 0.6,
        "tail_exceedance_2": 0.2,
        "drawdown": 0.35,
        "path_log_quantile": 0.50,
        "path_abs_quantile": 0.75,
        "path_terminal_quantile": 0.90,
        "path_drift": 0.35,
    },
})

main_config["unet_params"] = dict(main_config.get("unet_params", {}))
main_config["unet_params"].update({
    "output_scaling": 2.0,
})

main_config["cond_net_params"] = dict(main_config.get("cond_net_params", {}))
main_config["cond_net_params"].update({
    "numerical_input_dim": 397,
    "numerical_proj_dim": 128,
    "hidden_dim": 384,
    "temporal_history_encoder": True,
    "history_encoder_dim": 32,
    "history_feature_layout": {
        "returns_60": [5, 60],
        "returns_252": [65, 252],
        "vol_path_60": [317, 60],
    },
})
