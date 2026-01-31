# option_config.py
# 这是你的 "产品参数集"
# 在这里定义你所有想回测的期权合约

import Game.option_payoffs as op

CONTRACT_SPECS = {
    
    "vanilla_call": {
        "payoff_function": op.calculate_european_payoff,
        "payoff_base_arg": "strike", # Payoff函数需要 'strike' 还是 'start_price'
        "payoff_params": {},       # Payoff函数需要的额外固定参数
        "pricing_style": "discounted", # 'discounted' (折现) 或 'rate' (收益率)
        "spread_style": "percentage",  # 'percentage' (百分比) 或 'absolute' (绝对值)
        "spread_value": 0.40,
        "trade_threshold_style": "relative", # 'relative' (相对) 或 'absolute' (绝对)
        "trade_threshold_value": 0.03,
        "report_style": "pnl" # 'pnl' (原始盈亏) 或 'notional' (名义本金)
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
        "report_style": "pnl"
    },

    "standard_lookback": {
        "payoff_function": op.calculate_lookback_payoff,
        "payoff_base_arg": "start_price", # 虽然函数不用它，但保持一致性
        "payoff_params": {},
        "pricing_style": "discounted",
        "spread_style": "percentage",
        "spread_value": 0.40,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.03,
        "report_style": "pnl"
    },
    
    "my_accumulator": {
        "payoff_function": op.calculate_accumulator_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "strike_pct": 0.8,
            "ko_pct": 1.5
        },
        "pricing_style": "rate",
        "spread_style": "percentage",
        "spread_value": 0.40,
        "trade_threshold_style": "relative",
        "trade_threshold_value": 0.03,
        "report_style": "pnl"
    },

    "my_snowball_A": {
        "payoff_function": op.calculate_snowball_payoff,
        "payoff_base_arg": "start_price",
        "payoff_params": {
            "ko_pct": 1.05,
            "ki_pct": 0.8,
            "coupon_rate": 0.15,
            "obs_freq_days": 5
        },
        "pricing_style": "rate",
        "spread_style": "absolute",
        "spread_value": 0.02,
        "trade_threshold_style": "absolute",
        "trade_threshold_value": 0.005,
        "report_style": "notional"
    }

    # 未来：想测一个新的雪球？
    # 在这里复制粘贴 "my_snowball_A"，改名为 "my_snowball_B"
    # 然后只修改 "payoff_params" 里的数字即可。
}