"""Recompute P--Q execution diagnostics from an aligned trade log.

The log records P value, RN-Q value, realized payoff, and the chronological
row ID for every window. This utility changes only the pre-specified relative
signal threshold; it neither regenerates paths nor selects a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",")})
    if not thresholds or thresholds[0] < 0:
        raise ValueError("thresholds must be a non-empty list of non-negative values")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-log", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="0.03,0.05,0.10,0.15,0.20,0.30",
        help="Comma-separated relative side-quote thresholds.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("relative_quote", "spot"),
        default="relative_quote",
        help="Normalize the valuation gap by the side quote or by initial spot.",
    )
    args = parser.parse_args()

    windows = (
        pd.read_csv(args.test_csv, low_memory=False)
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    spot_by_row = windows["start_price"].astype(float).to_numpy()
    frame = pd.read_csv(args.trade_log)
    if frame["row_id"].max() >= len(spot_by_row) or frame["row_id"].min() < 0:
        raise ValueError("trade-log row IDs do not match the supplied chronological test CSV")
    frame["spot"] = spot_by_row[frame["row_id"].to_numpy(dtype=int)]
    thresholds = parse_thresholds(args.thresholds)

    records: list[dict] = []
    for threshold in thresholds:
        ask = frame["q_value"] * (1.0 + frame["spread"] / 2.0)
        bid = frame["q_value"] * (1.0 - frame["spread"] / 2.0)
        denominator_buy = ask if args.threshold_mode == "relative_quote" else frame["spot"]
        denominator_sell = bid if args.threshold_mode == "relative_quote" else frame["spot"]
        buy = (denominator_buy > 1e-9) & ((frame["p_value"] - ask) / denominator_buy > threshold)
        sell = (denominator_sell > 1e-9) & ((bid - frame["p_value"]) / denominator_sell > threshold)
        pnl = np.where(buy, frame["actual_value"] - ask, 0.0)
        pnl = np.where(sell, bid - frame["actual_value"], pnl)
        active = buy | sell
        working = frame.assign(
            active=active,
            direction=np.where(buy, "P_buy", np.where(sell, "P_sell", "no_trade")),
            pnl_over_spot=pnl / frame["spot"],
        )
        for (spread, contract), group in working.groupby(["spread", "contract"], sort=True):
            records.append(
                {
                    "relative_signal_threshold": threshold,
                    "spread": float(spread),
                    "contract": str(contract),
                    "windows": int(len(group)),
                    "trade_rate": float(group["active"].mean()),
                    "buy_share_of_active": float((group["direction"] == "P_buy").sum() / max(group["active"].sum(), 1)),
                    "unconditional_pnl_over_spot": float(group["pnl_over_spot"].mean()),
                    "active_pnl_over_spot": float(group.loc[group["active"], "pnl_over_spot"].mean()) if group["active"].any() else 0.0,
                }
            )

    payload = {
        "protocol": "fixed aligned P/Q valuations; threshold-only recomputation",
        "threshold_mode": args.threshold_mode,
        "threshold_definition": (
            "buy if (V_P-ask)/ask > tau; sell if (bid-V_P)/bid > tau"
            if args.threshold_mode == "relative_quote"
            else "buy if (V_P-ask)/S0 > tau; sell if (bid-V_P)/S0 > tau"
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

