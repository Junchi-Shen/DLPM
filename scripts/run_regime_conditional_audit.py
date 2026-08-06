"""Regime-stratified held-out CRPS audit for the frozen DLPM and bootstrap."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp


def crps_ensemble(values: np.ndarray, observation: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    count = len(values)
    pairwise_mean = ((2 * np.arange(1, count + 1) - count - 1) * values).sum() / count**2
    return float(np.abs(values - observation).mean() - pairwise_mean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "Results" / "Regime_Conditional_Audit" / "regime_conditional_summary.json"))
    parser.add_argument("--return-scale", type=float, default=0.09)
    args = parser.parse_args()
    rows = (
        pd.read_csv(pp.Testing_DATA_DIR / "clean_test_history_context.csv", low_memory=False)
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    dlpm = np.load(ROOT / "Results" / "Path_Archives" / "frozen_history_dlpm_seed20260730.npy", mmap_mode="r")
    bootstrap = np.load(ROOT / "Results" / "Path_Archives" / "unconditional_block_bootstrap.npy", mmap_mode="r")
    records = []
    for index, row in rows.iterrows():
        maturity, spot = int(row.actual_trading_days), float(row.start_price)
        dlpm_terminal = np.exp(np.asarray(dlpm[index, :, 1 : maturity + 1], dtype=float).sum(axis=1) * args.return_scale) - 1.0
        bootstrap_terminal = np.asarray(bootstrap[index, :, : maturity + 1], dtype=float)[:, -1] / spot - 1.0
        realized = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)
        terminal = float(realized[min(maturity, len(realized) - 1)] / spot - 1.0)
        records.append({
            "dlpm_crps": crps_ensemble(dlpm_terminal, terminal),
            "bootstrap_crps": crps_ensemble(bootstrap_terminal, terminal),
            "asset": row.asset_underlying,
            "trend_60": float(row.hist_trend_60),
            "drawdown_60": float(row.hist_current_drawdown_60),
        })
    frame = pd.DataFrame(records)
    frame["trend_rank"] = frame.groupby("asset")["trend_60"].rank(pct=True)
    frame["drawdown_rank"] = frame.groupby("asset")["drawdown_60"].rank(pct=True)
    states = {
        "low_60d_trend": frame["trend_rank"] <= 1 / 3,
        "middle_60d_trend": (frame["trend_rank"] > 1 / 3) & (frame["trend_rank"] < 2 / 3),
        "high_60d_trend": frame["trend_rank"] >= 2 / 3,
        "deep_current_drawdown": frame["drawdown_rank"] <= 1 / 3,
        "middle_current_drawdown": (frame["drawdown_rank"] > 1 / 3) & (frame["drawdown_rank"] < 2 / 3),
        "shallow_current_drawdown": frame["drawdown_rank"] >= 2 / 3,
    }
    summary = {"protocol": "Per-index terciles of pre-window 60-day state; terminal CRPS on frozen held-out paths.", "states": {}}
    for name, mask in states.items():
        result = frame.loc[mask, ["dlpm_crps", "bootstrap_crps"]].mean()
        summary["states"][name] = {
            "windows": int(mask.sum()),
            "dlpm_terminal_crps": float(result.dlpm_crps),
            "bootstrap_terminal_crps": float(result.bootstrap_crps),
            "relative_dlpm_improvement_pct": float((result.bootstrap_crps - result.dlpm_crps) / result.bootstrap_crps * 100.0),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

