from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp


CONTRACTS = ("vanilla_call", "asian_call", "lookback_call")
SPREADS = (0.05, 0.10, 0.20)
RELATIVE_SIGNAL_THRESHOLD = 0.03


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Windows can briefly lock the target while a monitor or sync client reads
    # it.  The state file is resumability infrastructure, so retry instead of
    # turning a transient lock into a failed multi-hour experiment.
    for attempt in range(25):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 24:
                raise
            time.sleep(0.2)


def load_processor(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def model_registry(config: dict, evaluation: dict) -> list[dict]:
    primary = (ROOT / config["frozen_primary_model"]).resolve()
    registry = []
    for task in evaluation["tasks"]:
        if task["kind"] == "test":
            model = task["model"]
            registry.append(
                {
                    "id": model["id"],
                    "archive": str(
                        ROOT / "Results" / "Path_Archives" / f"{model['id']}.npy"
                    ),
                    "archive_type": "raw",
                    "processor": str(Path(model["run_dir"]) / "data_processor.pkl"),
                }
            )
        elif task["kind"] == "statistical":
            registry.append(
                {
                    "id": task["baseline"],
                    "archive": str(
                        ROOT
                        / "Results"
                        / "Path_Archives"
                        / f"{task['baseline']}.npy"
                    ),
                    "archive_type": "prices",
                }
            )
        elif task["kind"] == "neural_sde":
            registry.append(
                {
                    "id": task["id"],
                    "archive": str(
                        ROOT
                        / "Results"
                        / "External_Baselines"
                        / task["id"]
                        / "raw_test_paths.npy"
                    ),
                    "archive_type": "raw",
                    "processor": str(primary / "data_processor.pkl"),
                }
            )
        elif task["kind"] == "chronos":
            registry.append(
                {
                    "id": task["id"],
                    "archive": str(
                        ROOT
                        / "Results"
                        / "External_Baselines"
                        / task["id"]
                        / "price_paths.npy"
                    ),
                    "archive_type": "prices",
                }
            )
    seen = set()
    return [item for item in registry if not (item["id"] in seen or seen.add(item["id"]))]


def load_q_paths(
    q_root: Path, rows: pd.DataFrame, source_csv: Path | None = None
) -> dict[str, np.ndarray]:
    """Load RN-Q paths in the exact row order of ``rows``.

    The RN-Q archives were generated from the source testing table, while the
    clean history-aware table is chronologically re-sorted.  Array position is
    therefore not a valid join key.  We map each clean row back to its unique
    source-table position before selecting a Q-path block.
    """
    if source_csv is None:
        raise ValueError(
            "source_csv is required; pass the RN-Q generation table explicitly"
        )
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing RN-Q source test table: {source_csv}")
    keys = [
        "asset_underlying",
        "start_date",
        "start_price",
        "actual_trading_days",
        "risk_free_rate",
    ]
    source = pd.read_csv(source_csv, low_memory=False)
    source["_source_row"] = source.groupby("asset_underlying", sort=False).cumcount()
    aligned = (
        rows.reset_index(names="_clean_row")
        .merge(source[keys + ["_source_row"]], on=keys, how="left", validate="one_to_one", sort=False)
        .sort_values("_clean_row")
    )
    if aligned["_source_row"].isna().any():
        raise ValueError("Some clean test windows cannot be aligned to RN-Q source rows")
    output = {}
    for asset, group in rows.groupby("asset_underlying", sort=True):
        files = sorted(
            (q_root / str(asset)).glob(
                "garch_risk_neutral_paths_*_samples.npy"
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if not files:
            raise FileNotFoundError(f"No RN-Q archive for {asset}")
        raw = np.load(files[-1], mmap_mode="r")
        source_count = int((source["asset_underlying"] == asset).sum())
        if raw.shape[0] % source_count:
            raise ValueError(f"RN-Q row mismatch for {asset}: {raw.shape}")
        blocks = raw.reshape(source_count, -1, raw.shape[-1])
        positions = aligned.loc[
            aligned["asset_underlying"] == asset, "_source_row"
        ].to_numpy(dtype=int)
        selected = blocks[positions]
        expected_spot = group["start_price"].to_numpy(dtype=float)
        observed_spot = selected[:, 0, 0]
        if not np.allclose(observed_spot, expected_spot, rtol=2e-6, atol=1e-2):
            raise ValueError(f"RN-Q initial-price alignment failed for {asset}")
        output[str(asset)] = selected
    return output


def raw_to_prices(raw: np.ndarray, row, scale: float) -> np.ndarray:
    n_returns = int(row.actual_trading_days)
    daily = np.asarray(raw[:, 1 : 1 + n_returns], dtype=float) * scale
    return float(row.start_price) * np.exp(
        np.concatenate(
            [np.zeros((daily.shape[0], 1)), np.cumsum(daily, axis=1)], axis=1
        )
    )


def payoffs(paths: np.ndarray, contract: str, strike: float, maturity: int):
    relevant = np.asarray(paths, dtype=float)[:, : maturity + 1]
    if contract == "vanilla_call":
        return np.maximum(relevant[:, -1] - strike, 0.0)
    if contract == "asian_call":
        return np.maximum(relevant.mean(axis=1) - strike, 0.0)
    if contract == "lookback_call":
        return np.maximum(relevant[:, -1] - relevant.min(axis=1), 0.0)
    raise ValueError(contract)


def append_records(path: Path, records: list[dict], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def summarize_trade_log(path: Path) -> dict:
    frame = pd.read_csv(path)
    results = {}
    grouped = frame.groupby(["spread", "contract"], sort=True)
    for (spread, contract), group in grouped:
        active = group[group["trade_type"] != "No Trade"]
        buys = group[group["trade_type"] == "P_Buy"]
        sells = group[group["trade_type"] == "P_Sell"]
        key = f"g{spread:.2f}_{contract}"
        results[key] = {
            "rows": int(len(group)),
            "trades": int(len(active)),
            "trade_rate": float(len(active) / len(group)),
            "buy_trades": int(len(buys)),
            "sell_trades": int(len(sells)),
            "buy_share_of_active": float(len(buys) / len(active))
            if len(active)
            else None,
            "sell_share_of_active": float(len(sells) / len(active))
            if len(active)
            else None,
            "mean_pnl_over_spot_unconditional": float(group["pnl_over_spot"].mean()),
            "mean_pnl_over_spot_active": float(active["pnl_over_spot"].mean())
            if len(active)
            else 0.0,
            "median_pnl_over_spot_active": float(active["pnl_over_spot"].median())
            if len(active)
            else 0.0,
            "win_rate_active": float((active["pnl_over_spot"] > 0).mean())
            if len(active)
            else None,
            "pnl_q05_active": float(active["pnl_over_spot"].quantile(0.05))
            if len(active)
            else None,
            "pnl_q95_active": float(active["pnl_over_spot"].quantile(0.95))
            if len(active)
            else None,
            "mean_pnl_over_spot_buy": float(buys["pnl_over_spot"].mean())
            if len(buys)
            else None,
            "mean_pnl_over_spot_sell": float(sells["pnl_over_spot"].mean())
            if len(sells)
            else None,
        }
    return results


def bootstrap(frame: pd.DataFrame, block_days: int, reps: int, seed: int) -> dict:
    origin = frame["date"].min()
    frame = frame.copy()
    frame["block"] = ((frame["date"] - origin).dt.days // block_days).astype(int)
    blocks = np.sort(frame["block"].unique())
    block_position = {block: index for index, block in enumerate(blocks)}
    rng = np.random.default_rng(seed + block_days)
    sampled = rng.integers(0, len(blocks), size=(reps, len(blocks)))
    weights = np.zeros((reps, len(blocks)), dtype=np.int16)
    for row in range(reps):
        weights[row] = np.bincount(sampled[row], minlength=len(blocks))
    results = {}
    for (spread, contract), group in frame.groupby(["spread", "contract"], sort=True):
        sums = np.zeros(len(blocks), dtype=float)
        counts = np.zeros(len(blocks), dtype=float)
        active_sums = np.zeros(len(blocks), dtype=float)
        active_counts = np.zeros(len(blocks), dtype=float)
        for block, block_group in group.groupby("block"):
            index = block_position[block]
            values = block_group["pnl_over_spot"].to_numpy(dtype=float)
            active = block_group["trade_type"].to_numpy() != "No Trade"
            sums[index] = values.sum()
            counts[index] = len(values)
            active_sums[index] = values[active].sum()
            active_counts[index] = active.sum()
        unconditional = (weights @ sums) / np.maximum(weights @ counts, 1.0)
        active_mean = (weights @ active_sums) / np.maximum(
            weights @ active_counts, 1.0
        )
        results[f"g{spread:.2f}_{contract}"] = {
            "unconditional_mean": float(unconditional.mean()),
            "unconditional_q025": float(np.quantile(unconditional, 0.025)),
            "unconditional_q975": float(np.quantile(unconditional, 0.975)),
            "active_mean": float(active_mean.mean()),
            "active_q025": float(np.quantile(active_mean, 0.025)),
            "active_q975": float(np.quantile(active_mean, 0.975)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "suite_config.json"))
    parser.add_argument(
        "--evaluation-state", default=str(ROOT / "artifacts" / "state" / "evaluation_state.json")
    )
    parser.add_argument(
        "--state", default=str(ROOT / "artifacts" / "state" / "economic_state.json")
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional single model identifier for a focused, resumable rerun.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "reports" / "economic_attribution"),
        help="Output directory; use a new root when auditing an earlier run.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=RELATIVE_SIGNAL_THRESHOLD,
        help="Relative P-versus-side-quote gap required to execute (default: 0.03).",
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=Path(pp.Testing_DATA_DIR) / "clean_test_history_context.csv",
        help="Chronological history-aware held-out window table.",
    )
    parser.add_argument(
        "--source-test-csv",
        type=Path,
        default=None,
        help="Source test table used when the RN-Q archive was generated.",
    )
    parser.add_argument(
        "--rnq-root",
        type=Path,
        default=None,
        help="RN-Q archive root; overrides frozen_rnq_root in the suite config.",
    )
    args = parser.parse_args()
    if args.signal_threshold < 0:
        raise ValueError("--signal-threshold must be non-negative")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(args.evaluation_state).read_text(encoding="utf-8"))
    if evaluation.get("status") != "completed":
        raise RuntimeError("Path evaluation suite is not complete")
    registry = model_registry(config, evaluation)
    if args.model_id:
        registry = [item for item in registry if item["id"] == args.model_id]
        if not registry:
            raise KeyError(f"Unknown model id: {args.model_id}")
    rows = (
        pd.read_csv(
            args.rows_csv, low_memory=False
        )
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    rows["date"] = pd.to_datetime(rows["start_date"])
    local_indices = rows.groupby("asset_underlying", sort=True).cumcount().to_numpy()
    q_root = (
        args.rnq_root.resolve()
        if args.rnq_root
        else (ROOT / config["frozen_rnq_root"]).resolve()
    )
    q_paths = load_q_paths(q_root, rows, args.source_test_csv)
    state_path = Path(args.state)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"status": "running", "models": {}}
    )

    for model_info in registry:
        model_id = model_info["id"]
        output = Path(args.output_root) / model_id
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "trade_log.csv"
        model_state = state["models"].setdefault(
            model_id, {"completed_rows": 0, "status": "pending"}
        )
        if model_state.get("status") == "completed":
            continue
        completed = int(model_state.get("completed_rows", 0))
        archive = np.load(model_info["archive"], mmap_mode="r")
        processor = (
            load_processor(Path(model_info["processor"]))
            if model_info["archive_type"] == "raw"
            else None
        )
        scale = (
            float(processor.config.get("volatility_scale", 1.0))
            if processor is not None
            else 1.0
        )
        buffer = []
        model_state["status"] = "running"
        atomic_json(state_path, state)
        for index in range(completed, len(rows)):
            row = rows.iloc[index]
            maturity = int(row.actual_trading_days)
            start = float(row.start_price)
            if model_info["archive_type"] == "raw":
                p_prices = raw_to_prices(archive[index], row, scale)
            else:
                p_prices = np.asarray(archive[index, :, : maturity + 1], dtype=float)
            asset = str(row.asset_underlying)
            q_prices = np.asarray(
                q_paths[asset][local_indices[index], :, : maturity + 1], dtype=float
            ).squeeze()
            actual_prices = np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)
            discount = np.exp(-float(row.risk_free_rate) * maturity / 252.0)
            for contract in CONTRACTS:
                p_value = float(payoffs(p_prices, contract, start, maturity).mean()) * discount
                q_value = float(payoffs(q_prices, contract, start, maturity).mean()) * discount
                actual = float(
                    payoffs(actual_prices[None, :], contract, start, maturity)[0]
                ) * discount
                for spread in SPREADS:
                    ask = max(q_value * (1.0 + spread / 2.0), 0.0)
                    bid = max(q_value * (1.0 - spread / 2.0), 0.0)
                    trade_type = "No Trade"
                    pnl = 0.0
                    if ask > 1e-9 and (p_value - ask) / ask > args.signal_threshold:
                        trade_type = "P_Buy"
                        pnl = actual - ask
                    elif bid > 1e-9 and (bid - p_value) / bid > args.signal_threshold:
                        trade_type = "P_Sell"
                        pnl = bid - actual
                    buffer.append(
                        {
                            "model": model_id,
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
                append_records(log_path, buffer, write_header=(completed == 0))
                buffer = []
                completed = done
                model_state["completed_rows"] = done
                model_state["status"] = (
                    "completed" if done == len(rows) else "running"
                )
                atomic_json(state_path, state)
                print(f"{model_id}: {done}/{len(rows)}", flush=True)
        summary = summarize_trade_log(log_path)
        summary["relative_signal_threshold"] = float(args.signal_threshold)
        atomic_json(output / "summary.json", summary)
        frame = pd.read_csv(log_path, parse_dates=["date"])
        bootstrap_results = {
            "protocol": (
                "joint calendar-block bootstrap; all assets share sampled blocks; "
                "P&L normalized by initial spot"
            ),
        "windows_are_independent": False,
        "repetitions": int(
            config["evaluation"]["joint_calendar_bootstrap_repetitions"]
        ),
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
            frame,
                int(block),
                bootstrap_results["repetitions"],
                20260730,
            )
        atomic_json(output / "joint_calendar_bootstrap.json", bootstrap_results)
        model_state["status"] = "completed"
        atomic_json(state_path, state)
    state["status"] = "completed"
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()

