"""Initial RN-Q delta-hedge audit for the frozen P--Q payoff experiment.

The audit leaves the existing quote and execution rule untouched.  For each
executed trade at the selected dealer spread it estimates the time-zero
RN-Q delta with a central finite difference, holds the opposite index delta
until the contract horizon, and reports the discounted self-financing P&L.
The calculation is resumable and writes into a separate result directory.
"""

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
import Project_Path as pp  # noqa: E402
from run_economic_attribution import (  # noqa: E402
    atomic_json,
    load_q_paths,
    payoffs,
)


MODEL_ID = "frozen_history_dlpm_seed20260730"
CONTRACTS = ("vanilla_call", "asian_call")


def q_value_with_spot(
    q_paths: np.ndarray,
    contract: str,
    spot: float,
    strike: float,
    maturity: int,
    discount: float,
) -> float:
    """Price a fixed-strike contract after scaling the RN-Q price paths."""
    base_spot = float(q_paths[0, 0])
    if not np.isfinite(base_spot) or base_spot <= 0.0:
        raise ValueError(f"Invalid RN-Q initial spot: {base_spot}")
    scaled = q_paths[:, : maturity + 1] * (spot / base_spot)
    return float(payoffs(scaled, contract, strike, maturity).mean() * discount)


def central_delta(
    q_paths: np.ndarray,
    contract: str,
    start: float,
    maturity: int,
    discount: float,
    epsilon: float,
) -> tuple[float, float]:
    """Return finite-difference RN-Q delta and the unperturbed RN-Q price."""
    up = q_value_with_spot(
        q_paths, contract, start * (1.0 + epsilon), start, maturity, discount
    )
    down = q_value_with_spot(
        q_paths, contract, start * (1.0 - epsilon), start, maturity, discount
    )
    base = q_value_with_spot(q_paths, contract, start, start, maturity, discount)
    return float((up - down) / (2.0 * epsilon * start)), base


def summarize(frame: pd.DataFrame) -> dict:
    result = {
        "protocol": {
            "hedge": "one initial RN-Q central-difference delta, held to maturity",
            "hedge_pnl": "discounted self-financing stock-leg P&L",
            "q_delta_epsilon": float(frame["epsilon"].iloc[0]),
            "selection": (
                "P--Q trade decisions recomputed before hedging from the frozen "
                "aligned valuations"
            ),
            "signal_threshold_mode": str(frame["signal_threshold_mode"].iloc[0]),
            "signal_threshold": float(frame["signal_threshold"].iloc[0]),
        },
        "contracts": {},
    }
    for contract, group in frame.groupby("contract", sort=True):
        active = group[group["trade_type"] != "No Trade"]
        buys = active[active["trade_type"] == "P_Buy"]
        sells = active[active["trade_type"] == "P_Sell"]
        result["contracts"][contract] = {
            "windows": int(len(group)),
            "trades": int(len(active)),
            "trade_rate": float(len(active) / len(group)),
            "buy_share_of_active": float(len(buys) / len(active)) if len(active) else None,
            "sell_share_of_active": float(len(sells) / len(active)) if len(active) else None,
            "unhedged_pnl_over_spot": float(group["unhedged_pnl_over_spot"].mean()),
            "hedge_pnl_over_spot": float(group["hedge_pnl_over_spot"].mean()),
            "hedged_pnl_over_spot": float(group["hedged_pnl_over_spot"].mean()),
            "hedged_win_rate_active": float((active["hedged_pnl_over_spot"] > 0.0).mean()) if len(active) else None,
            "unhedged_buy_contribution": float(buys["unhedged_pnl_over_spot"].sum() / len(group)),
            "unhedged_sell_contribution": float(sells["unhedged_pnl_over_spot"].sum() / len(group)),
            "hedged_buy_contribution": float(buys["hedged_pnl_over_spot"].sum() / len(group)),
            "hedged_sell_contribution": float(sells["hedged_pnl_over_spot"].sum() / len(group)),
            "mean_delta_active": float(active["delta"].mean()) if len(active) else None,
            "median_delta_active": float(active["delta"].median()) if len(active) else None,
            "max_q_value_reproduction_error": float(group["q_value_error"].abs().max()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "suite_config.json"))
    parser.add_argument(
        "--rnq-root",
        default=None,
        help="Optional RN-Q path-archive root; overrides frozen_rnq_root in the suite config.",
    )
    parser.add_argument("--rows-csv", type=Path, default=Path(pp.Testing_DATA_DIR) / "clean_test_history_context.csv")
    parser.add_argument("--source-test-csv", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--spread", type=float, default=0.05)
    parser.add_argument(
        "--signal-threshold-mode",
        choices=("spot", "relative_quote"),
        default="spot",
        help="Normalize the valuation gap by initial spot or by the side quote.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.01,
        help="Pre-specified execution threshold (default: 1%% of initial spot).",
    )
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--checkpoint-rows", type=int, default=64)
    parser.add_argument(
        "--trade-root",
        default=str(ROOT / "artifacts" / "reports" / "economic_attribution"),
        help="Aligned economic-attribution result root supplying the frozen trades.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "reports" / "initial_delta_hedge"),
    )
    args = parser.parse_args()
    if args.epsilon <= 0.0 or args.epsilon >= 0.05:
        raise ValueError("epsilon must be positive and below 0.05")
    if args.signal_threshold < 0.0:
        raise ValueError("signal threshold must be non-negative")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = (
        pd.read_csv(args.rows_csv, low_memory=False)
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    rows["date"] = pd.to_datetime(rows["start_date"])
    local_indices = rows.groupby("asset_underlying", sort=True).cumcount().to_numpy()
    q_root = (
        Path(args.rnq_root).resolve()
        if args.rnq_root
        else (ROOT / config["frozen_rnq_root"]).resolve()
    )
    q_by_asset = load_q_paths(q_root, rows, args.source_test_csv)

    trade_log = pd.read_csv(Path(args.trade_root) / args.model_id / "trade_log.csv")
    selected = trade_log[
        np.isclose(trade_log["spread"].to_numpy(dtype=float), args.spread)
        & trade_log["contract"].isin(CONTRACTS)
    ].copy()
    if len(selected) != len(rows) * len(CONTRACTS):
        raise ValueError("Expected one selected trade record per window and contract")
    selected = selected.set_index(["row_id", "contract"], verify_integrity=True)

    mode_tag = "spot" if args.signal_threshold_mode == "spot" else "quote"
    threshold_tag = f"{args.signal_threshold:.4f}".replace(".", "p")
    out_dir = Path(args.output_root) / args.model_id / f"{mode_tag}_{threshold_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "hedged_trade_log.csv"
    state_path = out_dir / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"status": "pending", "completed_rows": 0, "model_id": args.model_id}
    )
    if state.get("model_id") != args.model_id:
        raise ValueError("The existing state belongs to another model")
    completed = int(state.get("completed_rows", 0))
    if completed > len(rows):
        raise ValueError("Invalid completed_rows in state")
    if state.get("status") == "completed":
        print(f"Completed results already present: {out_dir}")
        return

    state["status"] = "running"
    atomic_json(state_path, state)
    records: list[dict] = []
    for index in range(completed, len(rows)):
        row = rows.iloc[index]
        start = float(row.start_price)
        rate = float(row.risk_free_rate)
        maturity = int(row.actual_trading_days)
        discount = float(np.exp(-rate * maturity / 252.0))
        actual = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)
        if len(actual) <= maturity:
            raise ValueError(f"Actual path is short for row {index}")
        q_paths = np.asarray(
            q_by_asset[str(row.asset_underlying)][local_indices[index], :, : maturity + 1],
            dtype=float,
        ).squeeze()
        if q_paths.ndim != 2:
            raise ValueError(f"Unexpected Q path shape for row {index}: {q_paths.shape}")
        discounted_forward_move = discount * float(actual[maturity]) - start

        for contract in CONTRACTS:
            original = selected.loc[(index, contract)]
            delta, q_reproduced = central_delta(
                q_paths, contract, start, maturity, discount, args.epsilon
            )
            q_value = float(original.q_value)
            p_value = float(original.p_value)
            actual_value = float(original.actual_value)
            ask = max(q_value * (1.0 + args.spread / 2.0), 0.0)
            bid = max(q_value * (1.0 - args.spread / 2.0), 0.0)
            buy_denominator = start if args.signal_threshold_mode == "spot" else ask
            sell_denominator = start if args.signal_threshold_mode == "spot" else bid
            if buy_denominator > 1e-9 and (p_value - ask) / buy_denominator > args.signal_threshold:
                trade_type = "P_Buy"
                unhedged = actual_value - ask
            elif sell_denominator > 1e-9 and (bid - p_value) / sell_denominator > args.signal_threshold:
                trade_type = "P_Sell"
                unhedged = bid - actual_value
            else:
                trade_type = "No Trade"
                unhedged = 0.0
            if trade_type == "P_Buy":
                hedge_units = -delta
            elif trade_type == "P_Sell":
                hedge_units = delta
            else:
                hedge_units = 0.0
            hedge_pnl = hedge_units * discounted_forward_move
            records.append(
                {
                    "row_id": index,
                    "asset": str(row.asset_underlying),
                    "date": row.date.date().isoformat(),
                    "contract": contract,
                    "spread": args.spread,
                    "trade_type": trade_type,
                    "start_price": start,
                    "maturity_days": maturity,
                    "epsilon": args.epsilon,
                    "signal_threshold_mode": args.signal_threshold_mode,
                    "signal_threshold": args.signal_threshold,
                    "q_value_logged": float(original.q_value),
                    "q_value_reproduced": q_reproduced,
                    "q_value_error": q_reproduced - float(original.q_value),
                    "delta": delta,
                    "hedge_units": hedge_units,
                    "discounted_forward_move": discounted_forward_move,
                    "unhedged_pnl": unhedged,
                    "hedge_pnl": hedge_pnl,
                    "hedged_pnl": unhedged + hedge_pnl,
                    "unhedged_pnl_over_spot": unhedged / start,
                    "hedge_pnl_over_spot": hedge_pnl / start,
                    "hedged_pnl_over_spot": (unhedged + hedge_pnl) / start,
                }
            )

        done = index + 1
        if done % args.checkpoint_rows == 0 or done == len(rows):
            pd.DataFrame(records).to_csv(
                output_path,
                mode="w" if completed == 0 else "a",
                header=completed == 0,
                index=False,
            )
            records = []
            completed = done
            state["completed_rows"] = completed
            state["status"] = "completed" if completed == len(rows) else "running"
            atomic_json(state_path, state)
            print(f"initial-delta hedge: {completed}/{len(rows)}", flush=True)

    frame = pd.read_csv(output_path)
    summary = summarize(frame)
    atomic_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

