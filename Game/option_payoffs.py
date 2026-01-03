# option_payoffs.py
# 这是一个纯粹的数学计算库，不依赖任何外部类。

import numpy as np

def standardize_paths(paths):
    """将不同来源的路径统一为 (n_paths, n_steps) 的2D数组"""
    if paths.ndim == 3:
        return paths.squeeze(axis=1)
    elif paths.ndim == 1:
        return paths.reshape(1, -1)
    return paths

def get_relevant_paths(paths, maturity_steps):
    """
    安全地从路径中截取从 t=0 到 t=T (maturity_steps) 的部分。
    一个 T=20 天的期权，路径有 21 个点 (索引 0 到 20)。
    """
    paths_2d = standardize_paths(paths)
    # 截取 [0, 1, ..., maturity_steps]，总共 maturity_steps + 1 个点
    return paths_2d[:, :maturity_steps + 1]

# --- 所有Payoff函数 ---

def calculate_european_payoff(paths, maturity_steps, strike):
    """欧式看涨期权: max(S_T - K, 0)"""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    final_prices = relevant_paths[:, -1] # S_T
    payoffs = np.maximum(final_prices - strike, 0)
    return payoffs

def calculate_asian_payoff(paths, maturity_steps, strike):
    """亚式看涨期权: max(S_avg - K, 0)"""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    average_prices = np.mean(relevant_paths, axis=1) # S_avg
    payoffs = np.maximum(average_prices - strike, 0)
    return payoffs

def calculate_lookback_payoff(paths, maturity_steps, **kwargs):
    """回望看涨期权: max(S_T - S_min, 0)"""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    min_prices = np.min(relevant_paths, axis=1)
    terminal_prices = relevant_paths[:, -1]
    payoffs = np.maximum(terminal_prices - min_prices, 0)
    return payoffs

def calculate_accumulator_payoff(paths, maturity_steps, start_price, strike_pct, ko_pct):
    """累计收益期权 (Accumulator)"""
    strike_price = start_price * strike_pct
    knock_out_barrier = start_price * ko_pct
    
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    num_paths = relevant_paths.shape[0]
    total_payoffs = np.zeros(num_paths)

    for i in range(num_paths):
        path = relevant_paths[i]
        accumulated_pnl = 0
        
        for t in range(1, len(path)): # 从 t=1 (第二天) 开始
            current_price = path[t]
            
            if current_price >= knock_out_barrier:
                break # 敲出，终止

            if current_price >= strike_price:
                accumulated_pnl += (current_price - strike_price)
            else:
                accumulated_pnl += 2 * (current_price - strike_price)
        
        total_payoffs[i] = accumulated_pnl
        
    return total_payoffs

def calculate_snowball_payoff(paths, maturity_steps, start_price, ko_pct, ki_pct, coupon_rate, obs_freq_days):
    """雪球期权 (Snowball) - 离散观察"""
    knock_out_barrier = start_price * ko_pct
    knock_in_barrier = start_price * ki_pct
    
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    num_paths = relevant_paths.shape[0]
    final_payoffs = np.zeros(num_paths)

    for i in range(num_paths):
        path = relevant_paths[i]
        is_knocked_in = False
        payoff = 0
        
        for t in range(1, len(path)): # 从 t=1 (第二天) 开始
            current_price = path[t]
            
            if not is_knocked_in and current_price <= knock_in_barrier:
                is_knocked_in = True
            
            is_observation_day = (t % obs_freq_days == 0) or (t == len(path) - 1)
            
            if is_observation_day:
                if current_price >= knock_out_barrier:
                    payoff = coupon_rate * (t / 252.0)
                    break 
            
            if t == len(path) - 1: # 循环正常结束
                if not is_knocked_in:
                    payoff = coupon_rate
                else:
                    terminal_price = path[-1]
                    if terminal_price >= start_price:
                        payoff = 0
                    else:
                        payoff = (terminal_price / start_price) - 1
        
        final_payoffs[i] = payoff
        
    return final_payoffs