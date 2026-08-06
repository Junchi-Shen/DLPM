import Game.option_payoffs as op


CONTRACT_SPECS = {
    "vanilla_call": {
        "payoff_function": op.calculate_european_payoff,
        "payoff_base_arg": "strike",
        "payoff_params": {},
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.40,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.03,
        "report_style": "pnl",
    },

    "standard_asian": {
        "payoff_function": op.calculate_asian_payoff,
        "payoff_base_arg": "strike",
        "payoff_params": {},
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.40,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.03,
        "report_style": "pnl",
    },

    "standard_lookback": {
        "payoff_function": op.calculate_lookback_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {},
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.40,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.03,
        "report_style": "pnl",
    },

    "my_accumulator": {
        "payoff_function": op.calculate_accumulator_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "strike_pct": 0.85,
            "ko_pct": 1.05,
            "obs_freq_days": 1,
            "leverage_below_strike": 2.0,
            "normalize_by_schedule": True,
        },
        "pricing_style": "rate",
        "spread_style": "percentage",
        "spread_value": 0.40,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.03,
        "report_style": "notional",
    },

    "my_snowball_A": {
        "payoff_function": op.calculate_snowball_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "ko_pct": 1.05,
            "ki_pct": 0.8,
            "coupon_rate": 0.15,
            "obs_freq_days": 5,
        },
        "pricing_style": "rate",
        "spread_style": "absolute",
        "spread_value": 0.02,
        "trade_threshold_style": "absolute",
        "trade_threshold_value": 0.005,
        "report_style": "notional",
    },

    "down_in_put": {
        "payoff_function": op.calculate_down_in_put_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "strike_pct": 1.0,
            "barrier_pct": 0.8,
        },
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.20,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.05,
        "report_style": "pnl",
    },

    "up_out_call": {
        "payoff_function": op.calculate_up_out_call_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "strike_pct": 1.0,
            "barrier_pct": 1.2,
        },
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.20,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.05,
        "report_style": "pnl",
    },

    "double_barrier_call": {
        "payoff_function": op.calculate_double_barrier_call_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "strike_pct": 1.0,
            "lower_barrier_pct": 0.8,
            "upper_barrier_pct": 1.2,
            "rebate": 0.0,
        },
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.20,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.05,
        "report_style": "pnl",
    },

    "parisian_down_in_put": {
        "payoff_function": op.calculate_parisian_down_in_put_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "strike_pct": 1.0,
            "barrier_pct": 0.85,
            "window_days": 10,
        },
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.20,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.05,
        "report_style": "pnl",
    },

    "monthly_cliquet": {
        "payoff_function": op.calculate_cliquet_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "obs_freq_days": 21,
            "local_floor": -0.03,
            "local_cap": 0.03,
            "global_floor": 0.0,
        },
        "pricing_style": "rate",
        "spread_style": "percentage",
        "spread_value": 0.20,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.02,
        "report_style": "notional",
    },

    "phoenix_note": {
        "payoff_function": op.calculate_phoenix_note_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "ko_pct": 1.0,
            "coupon_barrier_pct": 0.85,
            "ki_pct": 0.75,
            "coupon_rate": 0.012,
            "obs_freq_days": 21,
            "memory_coupon": True,
        },
        "pricing_style": "rate",
        "spread_style": "absolute",
        "spread_value": 0.01,
        "trade_threshold_style": "absolute",
        "trade_threshold_value": 0.003,
        "report_style": "notional",
    },
}
