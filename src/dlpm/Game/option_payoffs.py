import numpy as np


def standardize_paths(paths):
    """Convert paths from common shapes to a 2D array: (n_paths, n_steps)."""
    paths = np.asarray(paths)
    if paths.ndim == 3:
        return paths.squeeze(axis=1)
    if paths.ndim == 1:
        return paths.reshape(1, -1)
    return paths


def get_relevant_paths(paths, maturity_steps):
    """Return points from t=0 to t=T, inclusive."""
    paths_2d = standardize_paths(paths)
    return paths_2d[:, :maturity_steps + 1]


def calculate_european_payoff(paths, maturity_steps, strike):
    """European call: max(S_T - K, 0)."""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    final_prices = relevant_paths[:, -1]
    return np.maximum(final_prices - strike, 0.0)


def calculate_asian_payoff(paths, maturity_steps, strike):
    """Arithmetic Asian call: max(avg(S) - K, 0)."""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    average_prices = np.mean(relevant_paths, axis=1)
    return np.maximum(average_prices - strike, 0.0)


def calculate_lookback_payoff(paths, maturity_steps, **kwargs):
    """Floating-strike lookback call: max(S_T - min(S), 0)."""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    min_prices = np.min(relevant_paths, axis=1)
    terminal_prices = relevant_paths[:, -1]
    return np.maximum(terminal_prices - min_prices, 0.0)


def calculate_accumulator_payoff(paths, maturity_steps, start_price, strike_pct,
                                 ko_pct, obs_freq_days=1,
                                 leverage_below_strike=2.0,
                                 normalize_by_schedule=True):
    """Accumulator-style note return with discounted strike and KO termination.

    At each observation date the holder accumulates exposure at a discounted
    strike. If spot is below strike, the downside fixing is leveraged. The
    payoff is normalized by the scheduled number of fixings so it is comparable
    to a return on accumulation notional rather than an unbounded path sum.
    """
    strike_price = start_price * strike_pct
    knock_out_barrier = start_price * ko_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    num_paths, num_steps = relevant_paths.shape
    total_payoffs = np.zeros(num_paths)

    obs_points = list(range(int(obs_freq_days), num_steps, int(obs_freq_days)))
    if not obs_points or obs_points[-1] != num_steps - 1:
        obs_points.append(num_steps - 1)
    schedule_count = max(len(obs_points), 1)

    for i in range(num_paths):
        path = relevant_paths[i]
        accumulated_pnl_pct = 0.0
        for t in obs_points:
            current_price = path[t]
            if current_price >= knock_out_barrier:
                break

            pnl_pct = (current_price - strike_price) / start_price
            multiplier = 1.0 if current_price >= strike_price else leverage_below_strike
            accumulated_pnl_pct += multiplier * pnl_pct

        if normalize_by_schedule:
            accumulated_pnl_pct /= schedule_count
        total_payoffs[i] = accumulated_pnl_pct
    return total_payoffs


def calculate_snowball_payoff(paths, maturity_steps, start_price, ko_pct, ki_pct,
                              coupon_rate, obs_freq_days):
    """Stylized snowball/autocall note with discrete observations."""
    knock_out_barrier = start_price * ko_pct
    knock_in_barrier = start_price * ki_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    num_paths = relevant_paths.shape[0]
    final_payoffs = np.zeros(num_paths)

    for i in range(num_paths):
        path = relevant_paths[i]
        is_knocked_in = False
        payoff = 0.0

        for t in range(1, len(path)):
            current_price = path[t]
            if not is_knocked_in and current_price <= knock_in_barrier:
                is_knocked_in = True

            is_observation_day = (t % obs_freq_days == 0) or (t == len(path) - 1)
            if is_observation_day and current_price >= knock_out_barrier:
                payoff = coupon_rate * (t / 252.0)
                break

            if t == len(path) - 1:
                if not is_knocked_in:
                    payoff = coupon_rate * (t / 252.0)
                elif path[-1] >= start_price:
                    payoff = 0.0
                else:
                    payoff = (path[-1] / start_price) - 1.0

        final_payoffs[i] = payoff
    return final_payoffs


def calculate_down_in_put_payoff(paths, maturity_steps, strike_pct, barrier_pct,
                                 start_price):
    """Down-and-in put: activates if the path touches the downside barrier."""
    strike = start_price * strike_pct
    barrier = start_price * barrier_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    terminal_prices = relevant_paths[:, -1]
    knocked_in = np.min(relevant_paths, axis=1) <= barrier
    return np.where(knocked_in, np.maximum(strike - terminal_prices, 0.0), 0.0)


def calculate_up_out_call_payoff(paths, maturity_steps, strike_pct, barrier_pct,
                                 start_price):
    """Up-and-out call: pays a call payoff only if the upside barrier is never hit."""
    strike = start_price * strike_pct
    barrier = start_price * barrier_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    terminal_prices = relevant_paths[:, -1]
    knocked_out = np.max(relevant_paths, axis=1) >= barrier
    return np.where(knocked_out, 0.0, np.maximum(terminal_prices - strike, 0.0))


def calculate_double_barrier_call_payoff(paths, maturity_steps, strike_pct,
                                         lower_barrier_pct, upper_barrier_pct,
                                         start_price, rebate=0.0):
    """Interval-survival call with no rebate by default."""
    strike = start_price * strike_pct
    lower_barrier = start_price * lower_barrier_pct
    upper_barrier = start_price * upper_barrier_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    terminal_prices = relevant_paths[:, -1]
    touched_lower = np.min(relevant_paths, axis=1) <= lower_barrier
    touched_upper = np.max(relevant_paths, axis=1) >= upper_barrier
    survived = ~(touched_lower | touched_upper)
    vanilla_payoff = np.maximum(terminal_prices - strike, 0.0)
    return np.where(survived, vanilla_payoff, rebate)


def _max_consecutive_true(mask):
    """Return the longest consecutive run of True values for each row."""
    longest = np.zeros(mask.shape[0], dtype=int)
    current = np.zeros(mask.shape[0], dtype=int)
    for col in range(mask.shape[1]):
        current = np.where(mask[:, col], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def calculate_parisian_down_in_put_payoff(paths, maturity_steps, strike_pct,
                                          barrier_pct, window_days,
                                          start_price):
    """Parisian down-and-in put requiring N consecutive days below barrier."""
    strike = start_price * strike_pct
    barrier = start_price * barrier_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    terminal_prices = relevant_paths[:, -1]
    below_barrier = relevant_paths <= barrier
    longest_breach = _max_consecutive_true(below_barrier)
    activated = longest_breach >= int(window_days)
    return np.where(activated, np.maximum(strike - terminal_prices, 0.0), 0.0)


def calculate_cliquet_payoff(paths, maturity_steps, start_price, obs_freq_days=21,
                             local_floor=-0.03, local_cap=0.03,
                             global_floor=0.0):
    """Cliquet-style payoff: sum capped/floored period returns as a rate."""
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    num_paths, num_steps = relevant_paths.shape
    payoffs = np.zeros(num_paths)
    obs_points = list(range(0, num_steps, obs_freq_days))
    if obs_points[-1] != num_steps - 1:
        obs_points.append(num_steps - 1)

    for i in range(num_paths):
        path = relevant_paths[i]
        total = 0.0
        for left, right in zip(obs_points[:-1], obs_points[1:]):
            if path[left] <= 0:
                continue
            period_return = path[right] / path[left] - 1.0
            total += np.clip(period_return, local_floor, local_cap)
        payoffs[i] = max(total, global_floor)
    return payoffs


def calculate_phoenix_note_payoff(paths, maturity_steps, start_price, ko_pct=1.0,
                                  coupon_barrier_pct=0.75, ki_pct=0.65,
                                  coupon_rate=0.012, obs_freq_days=21,
                                  memory_coupon=True):
    """Phoenix note excess return with memory coupon, autocall, and terminal KI loss."""
    ko_barrier = start_price * ko_pct
    coupon_barrier = start_price * coupon_barrier_pct
    ki_barrier = start_price * ki_pct
    relevant_paths = get_relevant_paths(paths, maturity_steps)
    num_paths, num_steps = relevant_paths.shape
    payoffs = np.zeros(num_paths)

    obs_points = list(range(obs_freq_days, num_steps, obs_freq_days))
    if not obs_points or obs_points[-1] != num_steps - 1:
        obs_points.append(num_steps - 1)

    for i in range(num_paths):
        path = relevant_paths[i]
        knocked_in = np.min(path) <= ki_barrier
        accrued_coupon = 0.0
        missed_coupon_periods = 0
        autocalled = False

        for t in obs_points:
            if path[t] >= coupon_barrier:
                if memory_coupon:
                    accrued_coupon += coupon_rate * (missed_coupon_periods + 1)
                    missed_coupon_periods = 0
                else:
                    accrued_coupon += coupon_rate
            else:
                missed_coupon_periods += 1

            if path[t] >= ko_barrier:
                autocalled = True
                break

        if autocalled or not knocked_in:
            payoffs[i] = accrued_coupon
        else:
            terminal_return = path[-1] / start_price - 1.0
            payoffs[i] = accrued_coupon + min(terminal_return, 0.0)
    return payoffs
