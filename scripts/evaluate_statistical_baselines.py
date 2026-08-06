from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from arch import arch_model


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp
from evaluate_path_models import (
    pooled_terminal_metrics,
    score_paths,
)


BASELINES = (
    "unconditional_block_bootstrap",
    "historical_drift_gbm",
    "physical_student_t_garch",
    "momentum_trend_rule",
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_array(value) -> np.ndarray:
    return np.asarray(ast.literal_eval(str(value)), dtype=float)


def training_return_pools(train: pd.DataFrame) -> dict[str, np.ndarray]:
    pools = {}
    for asset, group in train.groupby("asset_underlying", sort=True):
        group = group.sort_values("start_date").reset_index(drop=True)
        # Each retained history block is roughly one trading year apart. This
        # avoids treating heavily overlapping windows as independent returns.
        blocks = [
            parse_array(value)
            for value in group.iloc[::252]["hist_returns_252"]
        ]
        pool = np.concatenate([block[np.isfinite(block)] for block in blocks])
        pools[str(asset)] = pool.astype(float)
    return pools


def fit_garch_models(pools: dict[str, np.ndarray]) -> dict[str, dict]:
    fitted = {}
    for asset, returns in pools.items():
        scaled = returns * 100.0
        result = arch_model(
            scaled,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="StudentsT",
            rescale=False,
        ).fit(disp="off", show_warning=False)
        params = result.params
        fitted[asset] = {
            "mu": float(params.get("mu", 0.0)) / 100.0,
            "omega": float(params["omega"]) / 10000.0,
            "alpha": float(params["alpha[1]"]),
            "beta": float(params["beta[1]"]),
            "nu": float(params["nu"]),
        }
    return fitted


def padded_horizon(rows: pd.DataFrame) -> int:
    # actual_trading_days counts returns; a price path has one extra S_0 point.
    return int(rows["actual_trading_days"].max()) + 1


def simulate_row(
    baseline: str,
    row,
    pool: np.ndarray,
    garch: dict | None,
    paths: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    steps = int(row.actual_trading_days)
    horizon = steps + 1
    start = float(row.start_price)
    returns = np.zeros((paths, width - 1), dtype=np.float32)
    asset_mu = float(np.mean(pool))
    asset_sigma = float(np.std(pool, ddof=1))
    history = parse_array(row.hist_returns_252)
    history = history[np.isfinite(history)]

    if baseline == "unconditional_block_bootstrap":
        block = 10
        generated = np.empty((paths, steps), dtype=float)
        for path_id in range(paths):
            cursor = 0
            while cursor < steps:
                start_id = int(rng.integers(0, max(len(pool) - block + 1, 1)))
                sample = pool[start_id : start_id + block]
                count = min(len(sample), steps - cursor)
                generated[path_id, cursor : cursor + count] = sample[:count]
                cursor += count
        returns[:, :steps] = generated
    elif baseline == "historical_drift_gbm":
        returns[:, :steps] = rng.normal(asset_mu, asset_sigma, size=(paths, steps))
    elif baseline == "momentum_trend_rule":
        trend60 = float(row.hist_trend_60) / 60.0
        recent_mu = float(np.mean(history[-60:])) if history.size else asset_mu
        drift = np.clip(0.50 * asset_mu + 0.25 * trend60 + 0.25 * recent_mu, -0.003, 0.003)
        recent_sigma = float(np.std(history[-60:], ddof=1)) if history.size >= 3 else asset_sigma
        sigma = np.clip(recent_sigma, 0.35 * asset_sigma, 2.5 * asset_sigma)
        returns[:, :steps] = rng.normal(drift, sigma, size=(paths, steps))
    elif baseline == "physical_student_t_garch":
        assert garch is not None
        mu = garch["mu"]
        omega = garch["omega"]
        alpha = garch["alpha"]
        beta = garch["beta"]
        nu = max(garch["nu"], 2.05)
        unconditional = omega / max(1.0 - alpha - beta, 1e-4)
        initial_var = (
            float(np.var(history[-60:], ddof=1))
            if history.size >= 3
            else unconditional
        )
        variance = np.full(paths, max(initial_var, 1e-10), dtype=float)
        previous_residual = np.full(
            paths,
            float(history[-1] - mu) if history.size else 0.0,
            dtype=float,
        )
        scale = np.sqrt(nu / (nu - 2.0))
        for step in range(steps):
            variance = omega + alpha * previous_residual**2 + beta * variance
            innovation = rng.standard_t(nu, size=paths) / scale
            residual = np.sqrt(np.maximum(variance, 1e-12)) * innovation
            returns[:, step] = mu + residual
            previous_residual = residual
    else:
        raise ValueError(baseline)

    prices = start * np.exp(
        np.concatenate(
            [np.zeros((paths, 1)), np.cumsum(returns, axis=1)], axis=1
        )
    )
    if horizon < width:
        prices[:, horizon:] = prices[:, horizon - 1 : horizon]
    return prices.astype(np.float32)


def summarize(rows: pd.DataFrame, generated: np.ndarray, paths: int) -> dict:
    path_list = [generated[index] for index in range(len(rows))]
    scores = score_paths(rows, path_list, paths)
    summary = scores.mean(numeric_only=True).to_dict()
    summary.update(pooled_terminal_metrics(rows, path_list))
    scores = scores.copy()
    scores["asset_underlying"] = rows["asset_underlying"].astype(str).to_numpy()
    summary["per_asset"] = {
        str(asset): {
            **{
                key: float(value)
                for key, value in group.mean(numeric_only=True).to_dict().items()
            },
            "windows": int(len(group)),
        }
        for asset, group in scores.groupby("asset_underlying", sort=True)
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-every", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    train = pd.read_csv(
        pp.Trainning_DATA_DIR / "clean_train_history_context.csv", low_memory=False
    )
    rows = (
        pd.read_csv(
            pp.Testing_DATA_DIR / "clean_test_history_context.csv", low_memory=False
        )
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    if args.max_rows is not None:
        rows = rows.head(int(args.max_rows)).reset_index(drop=True)
    pools = training_return_pools(train)
    garch_models = (
        fit_garch_models(pools)
        if args.baseline == "physical_student_t_garch"
        else {}
    )
    state_path = Path(args.state_file)
    archive_path = Path(args.archive)
    output_path = Path(args.output)
    width = padded_horizon(rows)
    completed = 0
    generated_parts = []
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            completed = int(state["completed_rows"])
            if completed:
                generated_parts.append(np.asarray(state["generated"], dtype=np.float32))
    for index in range(completed, len(rows)):
        row = rows.iloc[index]
        asset = str(row["asset_underlying"])
        # Per-row seeds make a resumed run bitwise identical to an
        # uninterrupted run without serializing generator internals.
        row_rng = np.random.default_rng(args.seed + 1009 * index)
        generated_parts.append(
            simulate_row(
                args.baseline,
                row,
                pools[asset],
                garch_models.get(asset),
                args.paths,
                width,
                row_rng,
            )[None, ...]
        )
        done = index + 1
        if done % max(args.state_every, 1) == 0 or done == len(rows):
            generated = np.concatenate(generated_parts, axis=0)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state_path.with_suffix(state_path.suffix + ".tmp")
            np.savez(
                temporary,
                completed_rows=np.int64(done),
                generated=generated,
            )
            actual = Path(str(temporary) + ".npz")
            if actual.exists():
                temporary = actual
            os.replace(temporary, state_path)
            generated_parts = [generated]
            print(f"{args.baseline}: {done}/{len(rows)}", flush=True)

    generated = np.concatenate(generated_parts, axis=0)
    summary = summarize(rows, generated, args.paths)
    summary.update(
        {
            "baseline": args.baseline,
            "seed": args.seed,
            "rows": int(len(rows)),
            "paths_per_row": args.paths,
            "training_return_pool_method": "252-row-spaced history blocks",
        }
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with temporary_archive.open("wb") as handle:
        np.save(handle, generated, allow_pickle=False)
    os.replace(temporary_archive, archive_path)
    atomic_json(output_path, summary)
    state_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

