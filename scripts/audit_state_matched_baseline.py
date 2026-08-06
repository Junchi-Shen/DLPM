"""State-matched nearest-neighbour empirical baseline on frozen chronological data."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("hist_rv_60", "hist_trend_60", "hist_current_drawdown_60")


def calendar_block_inference(frame, block_days, replications=2000, seed=20260802):
    dates = pd.to_datetime(frame["date"])
    origin = dates.min()
    block_id = ((dates - origin).dt.days // block_days).astype(int)
    block_means = frame.assign(_block=block_id).groupby("_block")["difference"].mean().to_numpy()
    rng = np.random.default_rng(seed + block_days)
    boot = np.mean(
        rng.choice(block_means, size=(replications, len(block_means)), replace=True), axis=1
    )
    observed = float(frame["difference"].mean())
    centered = boot - float(boot.mean())
    return {
        "block_days": block_days,
        "effective_blocks": int(len(block_means)),
        "mean_difference_dlpm_minus_baseline": observed,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_value_two_sided": float(np.mean(np.abs(centered) >= abs(observed))),
    }


def crps(samples, observed):
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(np.mean(np.abs(x - observed)) - np.dot(weights, x) / (n * n))


def terminal_return(row):
    path = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)
    return float(path[-1] / float(row.start_price) - 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--dlpm-window-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neighbors", type=int, default=40)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False).sort_values(
        ["asset_underlying", "start_date"]
    ).reset_index(drop=True)
    dlpm = pd.read_csv(args.dlpm_window_csv).sort_values("row_id").reset_index(drop=True)
    if len(dlpm) != len(test):
        raise ValueError("DLPM score table does not match test rows")
    train["terminal_return"] = train.apply(terminal_return, axis=1)
    records = []
    for (asset, tenor), test_group in test.groupby(
        ["asset_underlying", "contract_calendar_days"], sort=True
    ):
        pool = train[
            (train.asset_underlying == asset)
            & (train.contract_calendar_days == tenor)
        ].copy()
        if len(pool) < args.neighbors:
            pool = train[train.asset_underlying == asset].copy()
        mean = pool.loc[:, FEATURES].mean().to_numpy(dtype=float)
        std = pool.loc[:, FEATURES].std(ddof=1).replace(0, 1).to_numpy(dtype=float)
        pool_x = (pool.loc[:, FEATURES].to_numpy(dtype=float) - mean) / std
        test_x = (test_group.loc[:, FEATURES].to_numpy(dtype=float) - mean) / std
        pool_returns = pool.terminal_return.to_numpy(dtype=float)
        for local, (row_id, row) in enumerate(test_group.iterrows()):
            distance = np.sum((pool_x - test_x[local]) ** 2, axis=1)
            nearest = np.argpartition(distance, args.neighbors - 1)[: args.neighbors]
            observed = terminal_return(row)
            records.append({
                "row_id": int(row_id),
                "asset": asset,
                "country": row.country,
                "date": row.start_date,
                "tenor": int(tenor),
                "hist_trend_60": float(row.hist_trend_60),
                "hist_current_drawdown_60": float(row.hist_current_drawdown_60),
                "dlpm_crps": float(dlpm.loc[row_id, "dlpm_crps"]),
                "state_matched_crps": crps(pool_returns[nearest], observed),
            })
        print(f"completed {asset} tenor {tenor}: {len(test_group)}", flush=True)
    result = pd.DataFrame(records).sort_values("row_id")
    result["difference"] = result.dlpm_crps - result.state_matched_crps
    result["trend_rank"] = result.groupby("asset")["hist_trend_60"].rank(pct=True)
    result["drawdown_rank"] = result.groupby("asset")["hist_current_drawdown_60"].rank(pct=True)
    result["trend_state"] = pd.cut(
        result.trend_rank, [0, 1/3, 2/3, 1], labels=["low", "middle", "high"], include_lowest=True
    ).astype(str)
    result["drawdown_state"] = pd.cut(
        result.drawdown_rank, [0, 1/3, 2/3, 1], labels=["deep", "middle", "shallow"], include_lowest=True
    ).astype(str)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "state_matched_crps_by_window.csv", index=False)
    summary = {
        "protocol": "40 nearest training windows within the same index and contractual tenor, using standardized pre-window 60-day volatility, trend, and current drawdown.",
        "overall": {
            "windows": len(result),
            "dlpm_crps": float(result.dlpm_crps.mean()),
            "state_matched_crps": float(result.state_matched_crps.mean()),
        },
        "groups": {},
        "calendar_block_inference": [
            calendar_block_inference(result, days) for days in (20, 60, 90, 120, 150)
        ],
    }
    for dimension in ("trend_state", "drawdown_state", "asset", "tenor", "country"):
        summary["groups"][dimension] = {}
        for group, part in result.groupby(dimension, sort=True):
            summary["groups"][dimension][str(group)] = {
                "windows": len(part),
                "dlpm_crps": float(part.dlpm_crps.mean()),
                "state_matched_crps": float(part.state_matched_crps.mean()),
                "dlpm_minus_baseline": float(part.difference.mean()),
            }
    (args.output / "state_matched_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

