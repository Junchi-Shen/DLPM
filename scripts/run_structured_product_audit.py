"""Aligned P--Q stress audit for the accumulator and snowball scenarios.

The P archive follows the clean chronological test table, whereas RN-Q was
archived in the source-table order.  This script joins them by the full window
key before evaluating products.  It deliberately reports payoff and event
diagnostics, rather than mixing structured-note returns with vanilla P&L.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp

KEYS = [
    "asset_underlying",
    "start_date",
    "start_price",
    "actual_trading_days",
    "risk_free_rate",
]
ACCUMULATOR = {"strike_pct": 0.85, "ko_pct": 1.05, "leverage": 2.0}
SNOWBALL = {"ko_pct": 1.05, "ki_pct": 0.80, "coupon_rate": 0.15, "obs_days": 5}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def raw_to_prices(raw: np.ndarray, row, scale: float) -> np.ndarray:
    days = int(row.actual_trading_days)
    returns = np.asarray(raw[:, 1 : days + 1], dtype=float) * scale
    return float(row.start_price) * np.exp(
        np.concatenate([np.zeros((len(returns), 1)), np.cumsum(returns, axis=1)], axis=1)
    )


def accumulator(paths: np.ndarray, start: float, maturity: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prices = np.asarray(paths, dtype=float)[:, : maturity + 1]
    obs = np.arange(1, prices.shape[1], dtype=int)
    observed = prices[:, obs]
    strike = start * ACCUMULATOR["strike_pct"]
    hit_ko = observed >= start * ACCUMULATOR["ko_pct"]
    live = np.cumsum(hit_ko, axis=1) == 0  # First KO fixing itself does not accrue.
    below = observed < strike
    quantity = np.where(below, ACCUMULATOR["leverage"], 1.0)
    payoff = (live * quantity * (observed - strike) / start).sum(axis=1) / max(len(obs), 1)
    return payoff, {
        "ko": hit_ko.any(axis=1),
        "leveraged_fixing_share": (live * below).sum(axis=1) / np.maximum(live.sum(axis=1), 1),
        "loss": payoff < 0.0,
    }


def snowball(paths: np.ndarray, start: float, maturity: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prices = np.asarray(paths, dtype=float)[:, : maturity + 1]
    n_paths, steps = prices.shape
    times = np.arange(1, steps, dtype=int)
    obs_mask = ((times % SNOWBALL["obs_days"]) == 0) | (times == steps - 1)
    observation_prices = prices[:, 1:][:, obs_mask]
    observation_times = times[obs_mask]
    ko_mask = observation_prices >= start * SNOWBALL["ko_pct"]
    knocked_out = ko_mask.any(axis=1)
    first_ko = np.where(knocked_out, ko_mask.argmax(axis=1), 0)
    ko_time = observation_times[first_ko]
    knocked_in = (prices[:, 1:] <= start * SNOWBALL["ki_pct"]).any(axis=1)
    terminal = prices[:, -1]
    payoff = np.zeros(n_paths, dtype=float)
    payoff[knocked_out] = SNOWBALL["coupon_rate"] * ko_time[knocked_out] / 252.0
    survives = ~knocked_out
    payoff[survives & ~knocked_in] = SNOWBALL["coupon_rate"] * maturity / 252.0
    payoff[survives & knocked_in & (terminal < start)] = terminal[survives & knocked_in & (terminal < start)] / start - 1.0
    return payoff, {
        "ko": knocked_out,
        "ki": knocked_in,
        "loss": payoff < 0.0,
        "coupon": payoff > 0.0,
    }


def expected_shortfall(values: np.ndarray, level: float = 0.05) -> float:
    cutoff = np.quantile(values, level)
    return float(values[values <= cutoff].mean())


def summarize(values: list[np.ndarray], events: dict[str, list[np.ndarray]]) -> dict:
    pooled = np.concatenate(values)
    result = {
        "mean_payoff": float(pooled.mean()),
        "median_payoff": float(np.median(pooled)),
        "p05_payoff": float(np.quantile(pooled, 0.05)),
        "p95_payoff": float(np.quantile(pooled, 0.95)),
        "expected_shortfall_5pct": expected_shortfall(pooled),
        "loss_rate": float((pooled < 0.0).mean()),
    }
    for name, values_ in events.items():
        result[f"{name}_rate"] = float(np.concatenate(values_).mean())
    return result


def q_blocks_for_asset(q_root: Path, source: pd.DataFrame, asset: str, positions: np.ndarray) -> np.ndarray:
    files = sorted((q_root / asset).glob("garch_risk_neutral_paths_*_samples.npy"), key=lambda item: item.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No RN-Q path archive for {asset}")
    raw = np.load(files[-1], mmap_mode="r")
    source_count = int((source["asset_underlying"] == asset).sum())
    if raw.shape[0] % source_count:
        raise ValueError(f"RN-Q path count is incompatible with source rows for {asset}")
    return raw.reshape(source_count, -1, raw.shape[-1])[positions, :, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "artifacts" / "reports" / "structured_product_audit"))
    parser.add_argument("--model-id", default="frozen_history_dlpm_seed20260730")
    parser.add_argument("--return-scale", type=float, default=0.09, help="Frozen processor volatility scale.")
    parser.add_argument("--p-archive", type=Path, default=ROOT / "artifacts" / "path_archives" / "dlpm_40paths.npy")
    parser.add_argument("--rows-csv", type=Path, default=Path(pp.Testing_DATA_DIR) / "clean_test_history_context.csv")
    parser.add_argument("--source-test-csv", type=Path, required=True)
    parser.add_argument("--rnq-root", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads((ROOT / "suite_config.json").read_text(encoding="utf-8"))
    archive = np.load(args.p_archive, mmap_mode="r")
    scale = float(args.return_scale)
    rows = pd.read_csv(args.rows_csv, low_memory=False).sort_values(["asset_underlying", "start_date"]).reset_index(drop=True)
    source = pd.read_csv(args.source_test_csv, low_memory=False)
    source["_source_row"] = source.groupby("asset_underlying", sort=False).cumcount()
    aligned = rows.reset_index(names="_row").merge(source[KEYS + ["_source_row"]], on=KEYS, how="left", validate="one_to_one", sort=False).sort_values("_row")
    if aligned["_source_row"].isna().any():
        raise ValueError("Unable to uniquely align a structured-product window to RN-Q")
    q_root = args.rnq_root.resolve() if args.rnq_root else (ROOT / config["frozen_rnq_root"]).resolve()
    output = Path(args.output_root) / args.model_id
    output.mkdir(parents=True, exist_ok=True)
    state_file = output / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"completed_assets": []}
    all_records: list[dict] = []

    for asset, group in rows.groupby("asset_underlying", sort=True):
        asset = str(asset)
        asset_file = output / f"{asset}_windows.csv"
        if asset in state["completed_assets"]:
            all_records.extend(pd.read_csv(asset_file).to_dict("records"))
            continue
        indices = group.index.to_numpy(dtype=int)
        positions = aligned.loc[aligned["asset_underlying"] == asset, "_source_row"].to_numpy(dtype=int)
        q_paths = q_blocks_for_asset(q_root, source, asset, positions)
        if not np.allclose(q_paths[:, 0, 0], group["start_price"].to_numpy(float), rtol=2e-6, atol=1e-2):
            raise ValueError(f"RN-Q initial price alignment failed for {asset}")
        records: list[dict] = []
        payload: dict[str, list[np.ndarray]] = {}
        for local, (index, row) in enumerate(group.iterrows()):
            maturity, start = int(row.actual_trading_days), float(row.start_price)
            p_paths = raw_to_prices(archive[index], row, scale)
            q_prices = q_paths[local, :, : maturity + 1]
            actual = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)[None, : maturity + 1]
            for contract, function in (("accumulator", accumulator), ("snowball", snowball)):
                p_payoff, p_events = function(p_paths, start, maturity)
                q_payoff, q_events = function(q_prices, start, maturity)
                actual_payoff, actual_events = function(actual, start, maturity)
                for prefix, values, events in (("p", p_payoff, p_events), ("q", q_payoff, q_events), ("actual", actual_payoff, actual_events)):
                    payload.setdefault(f"{prefix}_{contract}_payoff", []).append(np.asarray(values, dtype=np.float32))
                    for name, flag in events.items():
                        payload.setdefault(f"{prefix}_{contract}_{name}", []).append(np.asarray(flag, dtype=np.float32))
                record = {
                    "asset": asset, "row_id": int(index), "date": str(row.start_date), "contract": contract,
                    "p_mean": float(p_payoff.mean()), "q_mean": float(q_payoff.mean()), "actual": float(actual_payoff[0]),
                    "p_q_gap": float(p_payoff.mean() - q_payoff.mean()),
                    "p_es5": expected_shortfall(p_payoff), "q_es5": expected_shortfall(q_payoff),
                    "p_8_mean": float(function(p_paths[:8], start, maturity)[0].mean()),
                    "p_16_mean": float(function(p_paths[:16], start, maturity)[0].mean()),
                    "p_40_mean": float(p_payoff.mean()),
                }
                for prefix, events in (("p", p_events), ("q", q_events), ("actual", actual_events)):
                    for name, flag in events.items():
                        record[f"{prefix}_{name}"] = float(np.mean(flag))
                records.append(record)
        pd.DataFrame(records).to_csv(asset_file, index=False)
        np.savez_compressed(
            output / f"{asset}_payoffs.npz",
            **{name: np.concatenate(values) for name, values in payload.items()},
        )
        all_records.extend(records)
        state["completed_assets"].append(asset)
        atomic_json(state_file, state)
        print(f"completed {asset}: {len(records)} windows", flush=True)

    frame = pd.DataFrame(all_records)
    frame.to_csv(output / "all_windows.csv", index=False)
    pooled: dict[str, list[np.ndarray]] = {}
    for payload_file in output.glob("*_payoffs.npz"):
        with np.load(payload_file) as payload:
            for name in payload.files:
                pooled.setdefault(name, []).append(payload[name])
    p_path_count = int(archive.shape[1])
    subsets = [value for value in (8, 16, 32) if value < p_path_count]
    summary = {"protocol": {"p_paths": p_path_count, "q_paths": 256, "alignment": "unique window key", "p_stability_subsets": subsets}, "contracts": {}}
    for contract, group in frame.groupby("contract", sort=True):
        def values(prefix: str) -> np.ndarray:
            return np.concatenate(pooled[f"{prefix}_{contract}_payoff"])
        def events(prefix: str) -> dict[str, list[np.ndarray]]:
            result = {}
            for name in ("ko", "ki", "coupon", "leveraged_fixing_share", "loss"):
                key = f"{prefix}_{contract}_{name}"
                if key in pooled:
                    result[name] = pooled[key]
            return result
        entry = {
            "windows": int(len(group)),
            "p": summarize([values("p")], events("p")),
            "q": summarize([values("q")], events("q")),
            "realized": summarize([values("actual")], events("actual")),
            "mean_p_q_gap": float(group["p_q_gap"].mean()),
            "p_window_es5": expected_shortfall(group["p_mean"].to_numpy()),
            "q_window_es5": expected_shortfall(group["q_mean"].to_numpy()),
            "p_path_count_stability": {
                "mean_abs_8_vs_40": float(np.mean(np.abs(group["p_8_mean"] - group["p_40_mean"]))),
                "mean_abs_16_vs_40": float(np.mean(np.abs(group["p_16_mean"] - group["p_40_mean"]))),
            },
        }
        summary["contracts"][contract] = entry
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()

