"""Paired calendar-block inference for DLPM minus historical-bootstrap CRPS."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def empirical_crps(samples: np.ndarray, observed: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    first = np.mean(np.abs(x - observed))
    weights = 2 * np.arange(1, n + 1) - n - 1
    pair_term = 2.0 * np.dot(weights, x) / (n * n)
    return float(first - 0.5 * pair_term)


def block_inference(part: pd.DataFrame, block_days: int, replications: int, rng) -> dict:
    origin = part.date.min().normalize()
    working = part.copy()
    working["block_id"] = ((working.date.dt.normalize() - origin).dt.days // block_days).astype(int)
    blocks = [group.difference.to_numpy() for _, group in working.groupby("block_id", sort=True)]
    draws = np.empty(replications)
    centered_blocks = [values - part.difference.mean() for values in blocks]
    null_draws = np.empty(replications)
    for b in range(replications):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        draws[b] = np.concatenate([blocks[j] for j in chosen]).mean()
        null_draws[b] = np.concatenate([centered_blocks[j] for j in chosen]).mean()
    estimate = float(part.difference.mean())
    p_value = float(np.mean(np.abs(null_draws) >= abs(estimate)))
    return {
        "windows": int(len(part)),
        "block_days": block_days,
        "effective_blocks": len(blocks),
        "mean_difference_dlpm_minus_bootstrap": estimate,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "p_value_two_sided": p_value,
    }


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min((len(values) - rank) * float(values.iloc[position]), 1.0)
        running = max(running, candidate)
        adjusted[position] = running
    return pd.Series(adjusted, index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--dlpm-archive", required=True, type=Path)
    parser.add_argument("--bootstrap-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.09)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    rows = pd.read_csv(args.test_csv, low_memory=False).sort_values(
        ["asset_underlying", "start_date"]
    ).reset_index(drop=True)
    rows["date"] = pd.to_datetime(rows.start_date)
    dlpm = np.load(args.dlpm_archive, mmap_mode="r")
    bootstrap = np.load(args.bootstrap_archive, mmap_mode="r")
    if dlpm.shape[0] != len(rows) or bootstrap.shape[0] != len(rows):
        raise ValueError("Archive row count mismatch")

    records = []
    for i, row in rows.iterrows():
        maturity = int(row.actual_trading_days)
        spot = float(row.start_price)
        realized = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)
        observed = float(realized[-1] / spot - 1.0)
        daily = np.asarray(dlpm[i, :, 1 : 1 + maturity], dtype=float) * args.scale
        dlpm_terminal = np.exp(daily.sum(axis=1)) - 1.0
        boot_paths = np.asarray(bootstrap[i], dtype=float)
        boot_terminal = boot_paths[:, min(maturity, boot_paths.shape[1] - 1)] / spot - 1.0
        records.append({
            "row_id": i,
            "asset": row.asset_underlying,
            "date": row.date,
            "tenor": int(row.contract_calendar_days),
            "country": row.country,
            "trend_60": float(row.hist_trend_60),
            "drawdown_60": float(row.hist_current_drawdown_60),
            "dlpm_crps": empirical_crps(dlpm_terminal, observed),
            "bootstrap_crps": empirical_crps(boot_terminal, observed),
        })
    frame = pd.DataFrame(records)
    frame["difference"] = frame.dlpm_crps - frame.bootstrap_crps
    frame["trend_rank"] = frame.groupby("asset")["trend_60"].rank(pct=True)
    frame["drawdown_rank"] = frame.groupby("asset")["drawdown_60"].rank(pct=True)
    frame["trend_state"] = pd.cut(
        frame.trend_rank, [0, 1 / 3, 2 / 3, 1], labels=["low", "middle", "high"],
        include_lowest=True,
    ).astype(str)
    frame["drawdown_state"] = pd.cut(
        frame.drawdown_rank, [0, 1 / 3, 2 / 3, 1], labels=["deep", "middle", "shallow"],
        include_lowest=True,
    ).astype(str)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "paired_crps_by_window.csv", index=False)

    rng = np.random.default_rng(args.seed)
    block_lengths = (20, 60, 90, 120, 150)
    results = []
    for block in block_lengths:
        results.append(block_inference(frame, block, args.replications, rng))

    group_records = []
    dimensions = {
        "trend_state": "trend_state",
        "drawdown_state": "drawdown_state",
        "index": "asset",
        "tenor": "tenor",
        "country": "country",
    }
    for dimension, column in dimensions.items():
        for group, part in frame.groupby(column, sort=True):
            for block in (20, 60):
                item = block_inference(part, block, args.replications, rng)
                item.update({"dimension": dimension, "group": str(group)})
                group_records.append(item)
    grouped = pd.DataFrame(group_records)
    state_mask = grouped.dimension.isin(["trend_state", "drawdown_state"])
    for block in (20, 60):
        mask = state_mask & (grouped.block_days == block)
        grouped.loc[mask, "p_value_holm_state_slices"] = holm_adjust(
            grouped.loc[mask, "p_value_two_sided"]
        )
    grouped.to_csv(args.output / "paired_crps_group_inference.csv", index=False)
    payload = {
        "interpretation": "Negative DLPM-minus-bootstrap CRPS favors DLPM.",
        "replications": args.replications,
        "joint_calendar_blocks": True,
        "results": results,
        "group_inference_file": "paired_crps_group_inference.csv",
        "multiple_testing": "Holm correction across six pre-specified trend/drawdown slices, separately by block length.",
    }
    (args.output / "paired_crps_calendar_block.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

