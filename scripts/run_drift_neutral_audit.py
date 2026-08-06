"""Drift-neutral P-side audit for the frozen conditional DLPM.

For each test window, this audit applies a deterministic time-linear log-price
shift to every generated P path so that its cross-path mean terminal price is
S0 exp(rT).  The adjustment locks the initial spot and preserves path-relative
variation, volatility clustering, drawdowns, and tail ordering.  It therefore
removes the window-level P-versus-Q terminal-drift wedge without replacing the
conditional path generator with a different model.

The calculation is resumable at 64-window checkpoints and is intentionally
stored separately from the unadjusted economic attribution experiment.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
from run_economic_attribution import (  # noqa: E402
    CONTRACTS,
    RELATIVE_SIGNAL_THRESHOLD,
    SPREADS,
    append_records,
    atomic_json,
    bootstrap,
    load_processor,
    load_q_paths,
    model_registry,
    payoffs,
    raw_to_prices,
    summarize_trade_log,
)
import Project_Path as pp  # noqa: E402


def align_terminal_mean(paths: np.ndarray, start: float, rate: float, maturity: int):
    """Align E[P][S_T] to S0 exp(rT) with a log shift proportional to time."""
    paths = np.asarray(paths, dtype=float)
    terminal_mean = float(np.mean(paths[:, -1]))
    target = float(start * np.exp(rate * maturity / 252.0))
    if not np.isfinite(terminal_mean) or terminal_mean <= 0.0:
        raise ValueError(f"Invalid generated terminal mean: {terminal_mean}")
    shift = float(np.log(target / terminal_mean))
    time_fraction = np.linspace(0.0, 1.0, paths.shape[1], dtype=float)
    adjusted = paths * np.exp(shift * time_fraction)[None, :]
    return adjusted, {
        "terminal_mean_before": terminal_mean,
        "terminal_target": target,
        "terminal_mean_after": float(np.mean(adjusted[:, -1])),
        "log_drift_shift": shift,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "suite_config.json"))
    parser.add_argument(
        "--evaluation-state", default=str(ROOT / "artifacts" / "state" / "evaluation_state.json")
    )
    parser.add_argument(
        "--state", default=str(ROOT / "artifacts" / "state" / "drift_neutral_state.json")
    )
    parser.add_argument(
        "--model-id", default="frozen_history_dlpm_seed20260730"
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "reports" / "drift_neutral"),
    )
    parser.add_argument("--rows-csv", type=Path, default=Path(pp.Testing_DATA_DIR) / "clean_test_history_context.csv")
    parser.add_argument("--source-test-csv", type=Path, required=True)
    parser.add_argument("--rnq-root", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(args.evaluation_state).read_text(encoding="utf-8"))
    if evaluation.get("status") != "completed":
        raise RuntimeError("Path evaluation suite is not complete")

    registry = {item["id"]: item for item in model_registry(config, evaluation)}
    if args.model_id not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown model id {args.model_id}; available: {available}")
    model = registry[args.model_id]
    if model["archive_type"] != "raw":
        raise ValueError("Drift-neutral audit expects raw diffusion outputs")

    rows = (
        pd.read_csv(args.rows_csv, low_memory=False)
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    rows["date"] = pd.to_datetime(rows["start_date"])
    local_indices = rows.groupby("asset_underlying", sort=True).cumcount().to_numpy()
    q_root = args.rnq_root.resolve() if args.rnq_root else (ROOT / config["frozen_rnq_root"]).resolve()
    q_paths = load_q_paths(q_root, rows, args.source_test_csv)
    archive = np.load(model["archive"], mmap_mode="r")
    processor = load_processor(Path(model["processor"]))
    scale = float(processor.config.get("volatility_scale", 1.0))

    output = Path(args.output_root) / args.model_id
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "trade_log.csv"
    alignment_path = output / "alignment_log.csv"
    state_path = Path(args.state)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"status": "pending", "model_id": args.model_id, "completed_rows": 0}
    )
    if state.get("model_id") != args.model_id:
        raise ValueError("State belongs to a different model")
    completed = int(state.get("completed_rows", 0))
    if completed > len(rows):
        raise ValueError("Invalid completed_rows in drift-neutral state")
    if state.get("status") == "completed":
        return

    trade_buffer: list[dict] = []
    alignment_buffer: list[dict] = []
    state["status"] = "running"
    atomic_json(state_path, state)
    for index in range(completed, len(rows)):
        row = rows.iloc[index]
        maturity = int(row.actual_trading_days)
        start = float(row.start_price)
        rate = float(row.risk_free_rate)
        p_prices = raw_to_prices(archive[index], row, scale)
        p_prices, alignment = align_terminal_mean(p_prices, start, rate, maturity)
        asset = str(row.asset_underlying)
        q_prices = np.asarray(
            q_paths[asset][local_indices[index], :, : maturity + 1], dtype=float
        ).squeeze()
        actual_prices = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)
        discount = np.exp(-rate * maturity / 252.0)
        alignment_buffer.append(
            {
                "row_id": index,
                "asset": asset,
                "date": row.date.date().isoformat(),
                "maturity_days": maturity,
                **alignment,
            }
        )
        for contract in CONTRACTS:
            p_value = float(payoffs(p_prices, contract, start, maturity).mean()) * discount
            q_value = float(payoffs(q_prices, contract, start, maturity).mean()) * discount
            actual = float(payoffs(actual_prices[None, :], contract, start, maturity)[0]) * discount
            for spread in SPREADS:
                ask = max(q_value * (1.0 + spread / 2.0), 0.0)
                bid = max(q_value * (1.0 - spread / 2.0), 0.0)
                trade_type, pnl = "No Trade", 0.0
                if ask > 1e-9 and (p_value - ask) / ask > RELATIVE_SIGNAL_THRESHOLD:
                    trade_type, pnl = "P_Buy", actual - ask
                elif bid > 1e-9 and (bid - p_value) / bid > RELATIVE_SIGNAL_THRESHOLD:
                    trade_type, pnl = "P_Sell", bid - actual
                trade_buffer.append(
                    {
                        "model": args.model_id,
                        "row_id": index,
                        "asset": asset,
                        "date": row.date.date().isoformat(),
                        "contract": contract,
                        "spread": spread,
                        "p_value": p_value,
                        "q_value": q_value,
                        "actual_value": actual,
                        "trade_type": trade_type,
                        "pnl": pnl,
                        "pnl_over_spot": pnl / start,
                    }
                )
        done = index + 1
        if done % 64 == 0 or done == len(rows):
            append_records(log_path, trade_buffer, write_header=(completed == 0))
            append_records(alignment_path, alignment_buffer, write_header=(completed == 0))
            trade_buffer, alignment_buffer = [], []
            completed = done
            state["completed_rows"] = done
            atomic_json(state_path, state)
            print(f"drift-neutral {args.model_id}: {done}/{len(rows)}", flush=True)

    summary = summarize_trade_log(log_path)
    alignment_frame = pd.read_csv(alignment_path)
    alignment_summary = {
        "adjustment": "time-linear log-price shift with initial spot locked",
        "target": "cross-path mean terminal price equals S0 exp(rT) in every window",
        "mean_log_drift_shift": float(alignment_frame["log_drift_shift"].mean()),
        "median_log_drift_shift": float(alignment_frame["log_drift_shift"].median()),
        "mean_terminal_relative_error": float(
            np.mean(
                np.abs(
                    alignment_frame["terminal_mean_after"]
                    / alignment_frame["terminal_target"]
                    - 1.0
                )
            )
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "alignment_summary.json", alignment_summary)
    frame = pd.read_csv(log_path, parse_dates=["date"])
    bootstrap_results = {
        "protocol": "joint calendar-block bootstrap; P&L normalized by initial spot",
        "windows_are_independent": False,
        "repetitions": int(config["evaluation"]["joint_calendar_bootstrap_repetitions"]),
        "block_diagnostics": {},
        "results": {},
    }
    for block in config["evaluation"]["joint_calendar_block_lengths"]:
        origin = frame["date"].min()
        observed_blocks = int(
            ((frame["date"] - origin).dt.days // int(block)).nunique()
        )
        bootstrap_results["block_diagnostics"][str(block)] = {
            "calendar_block_days": int(block),
            "observed_nonempty_blocks": observed_blocks,
            "interpretation": (
                "Long-dependence sensitivity check with limited effective blocks."
                if int(block) >= 252
                else (
                    "Primary dependence-robust inference block length."
                    if int(block) <= 60
                    else "Medium-horizon sensitivity check with limited effective blocks."
                )
            ),
        }
        bootstrap_results["results"][str(block)] = bootstrap(
            frame, int(block), bootstrap_results["repetitions"], 20260730
        )
    atomic_json(output / "joint_calendar_bootstrap.json", bootstrap_results)
    state["status"] = "completed"
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()

