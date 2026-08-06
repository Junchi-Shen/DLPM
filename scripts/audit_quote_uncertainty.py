"""Audit quote-selection uncertainty from frozen 40-path P and 256-path Q archives."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACTS = ("vanilla_call", "asian_call", "lookback_call")


def payoffs(paths: np.ndarray, contract: str, strike: float) -> np.ndarray:
    if contract == "vanilla_call":
        return np.maximum(paths[:, -1] - strike, 0.0)
    if contract == "asian_call":
        return np.maximum(paths.mean(axis=1) - strike, 0.0)
    if contract == "lookback_call":
        return np.maximum(paths[:, -1] - paths.min(axis=1), 0.0)
    raise ValueError(contract)


def raw_to_prices(raw: np.ndarray, row, scale: float) -> np.ndarray:
    maturity = int(row.actual_trading_days)
    daily = np.asarray(raw[:, 1 : 1 + maturity], dtype=float) * scale
    return float(row.start_price) * np.exp(
        np.concatenate([np.zeros((len(raw), 1)), np.cumsum(daily, axis=1)], axis=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--q-source-csv", required=True, type=Path)
    parser.add_argument("--p-archive", required=True, type=Path)
    parser.add_argument("--q-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.09)
    parser.add_argument("--spread", type=float, default=0.05)
    parser.add_argument("--taus", default="0.0025,0.005,0.0075,0.01,0.0125,0.015,0.02")
    parser.add_argument("--ks", default="0,1,1.96")
    args = parser.parse_args()

    rows = pd.read_csv(args.test_csv, low_memory=False).sort_values(
        ["asset_underlying", "start_date"]
    ).reset_index(drop=True)
    source = pd.read_csv(args.q_source_csv, low_memory=False)
    keys = ["asset_underlying", "start_date", "start_price", "actual_trading_days", "risk_free_rate"]
    source["_source_row"] = source.groupby("asset_underlying", sort=False).cumcount()
    aligned = rows.reset_index(names="row_id").merge(
        source[keys + ["_source_row"]], on=keys, how="left", validate="one_to_one", sort=False
    ).sort_values("row_id")
    if aligned["_source_row"].isna().any():
        raise ValueError("Q source alignment failed")

    p_raw = np.load(args.p_archive, mmap_mode="r")
    if p_raw.shape[:2] != (len(rows), 40):
        raise ValueError(f"Expected ({len(rows)}, 40, tokens), got {p_raw.shape}")

    records: list[dict] = []
    for asset, group in rows.groupby("asset_underlying", sort=True):
        q_file = sorted((args.q_root / str(asset)).glob("garch_risk_neutral_paths_*_samples.npy"))[-1]
        q_raw = np.load(q_file, mmap_mode="r")
        source_count = int((source.asset_underlying == asset).sum())
        q_blocks = q_raw.reshape(source_count, -1, q_raw.shape[-1])
        if q_blocks.shape[1] != 256:
            raise ValueError(f"Expected 256 Q paths for {asset}, got {q_blocks.shape}")
        asset_rows = group.index.to_numpy(dtype=int)
        q_positions = aligned.loc[asset_rows, "_source_row"].to_numpy(dtype=int)
        for row_id, q_position in zip(asset_rows, q_positions):
            row = rows.iloc[row_id]
            maturity = int(row.actual_trading_days)
            spot = float(row.start_price)
            discount = np.exp(-float(row.risk_free_rate) * maturity / 252.0)
            p_prices = raw_to_prices(p_raw[row_id], row, args.scale)
            q_prices = np.asarray(q_blocks[q_position, :, : maturity + 1], dtype=float).squeeze()
            actual_prices = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)[None, :]
            if not np.allclose(q_prices[:, 0], spot, rtol=2e-6, atol=1e-2):
                raise ValueError(f"Q spot mismatch at row {row_id}")
            for contract in CONTRACTS:
                p_samples = payoffs(p_prices, contract, spot) * discount
                q_samples = payoffs(q_prices, contract, spot) * discount
                actual = float(payoffs(actual_prices, contract, spot)[0] * discount)
                records.append({
                    "row_id": row_id,
                    "asset": asset,
                    "date": row.start_date,
                    "contract": contract,
                    "spot": spot,
                    "p_value": float(p_samples.mean()),
                    "q_value": float(q_samples.mean()),
                    "p_se": float(p_samples.std(ddof=1) / np.sqrt(len(p_samples))),
                    "q_se": float(q_samples.std(ddof=1) / np.sqrt(len(q_samples))),
                    "actual_value": actual,
                })
        print(f"completed {asset}: {len(group)} windows", flush=True)

    frame = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "quote_mc_uncertainty_by_window.csv", index=False)

    taus = [float(x) for x in args.taus.split(",")]
    ks = [float(x) for x in args.ks.split(",")]
    scans: list[dict] = []
    for tau in taus:
        for k in ks:
            for contract, part in frame.groupby("contract", sort=True):
                ask = part.q_value * (1.0 + args.spread / 2.0)
                bid = part.q_value * (1.0 - args.spread / 2.0)
                uncertainty = k * np.sqrt(part.p_se**2 + part.q_se**2)
                hurdle = tau * part.spot + uncertainty
                buy = part.p_value - ask > hurdle
                sell = bid - part.p_value > hurdle
                active = buy | sell
                pnl = np.where(buy, part.actual_value - ask, np.where(sell, bid - part.actual_value, 0.0))
                scans.append({
                    "tau_over_spot": tau,
                    "k": k,
                    "spread": args.spread,
                    "contract": contract,
                    "windows": int(len(part)),
                    "trade_rate": float(active.mean()),
                    "buy_share_active": float(buy.sum() / max(active.sum(), 1)),
                    "unconditional_pnl_over_spot": float(np.mean(pnl / part.spot)),
                    "active_win_rate": float(np.mean(pnl[active] > 0)) if active.any() else None,
                    "mean_p_se_over_spot": float(np.mean(part.p_se / part.spot)),
                    "mean_q_se_over_spot": float(np.mean(part.q_se / part.spot)),
                    "mean_joint_se_over_spot": float(np.mean(np.sqrt(part.p_se**2 + part.q_se**2) / part.spot)),
                })
    scan = pd.DataFrame(scans)
    scan.to_csv(args.output / "quote_mc_threshold_sensitivity.csv", index=False)
    payload = {
        "protocol": {
            "p_paths": 40,
            "q_paths": 256,
            "spread": args.spread,
            "rule": "abs side-quote gap > tau*S0 + k*sqrt(SE_P^2+SE_Q^2)",
            "same_frozen_paths": True,
        },
        "records": scan.to_dict(orient="records"),
    }
    (args.output / "quote_mc_threshold_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

