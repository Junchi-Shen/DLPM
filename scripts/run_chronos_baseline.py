from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp
from evaluate_path_models import pooled_terminal_metrics, score_paths


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def context_prices(row) -> np.ndarray:
    returns = np.asarray(ast.literal_eval(str(row.hist_returns_252)), dtype=float)
    returns = returns[np.isfinite(returns)]
    levels = np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    levels *= float(row.start_price) / max(levels[-1], 1e-12)
    return levels[-252:].astype(np.float32)


def summarize(rows, generated, paths):
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
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    try:
        from chronos import ChronosPipeline
    except ImportError as exc:
        raise SystemExit(
            "Chronos is an optional baseline dependency. Install chronos-forecasting "
            "before running this command."
        ) from exc
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    partial_path = run_dir / "price_paths.partial.npy"
    archive_path = run_dir / "price_paths.npy"
    metrics_path = run_dir / "test_path_metrics.json"
    rows = (
        pd.read_csv(
            pp.Testing_DATA_DIR / "clean_test_history_context.csv", low_memory=False
        )
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    if args.max_rows is not None:
        rows = rows.head(int(args.max_rows)).reset_index(drop=True)
    width = int(rows["actual_trading_days"].max()) + 1
    shape = (len(rows), args.paths, width)
    completed = 0
    if archive_path.exists() and metrics_path.exists():
        return
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed_rows"])
        generated = np.lib.format.open_memmap(partial_path, mode="r+")
        if tuple(generated.shape) != shape:
            raise ValueError(f"Chronos partial shape mismatch: {generated.shape}")
    else:
        generated = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=shape
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        args.model_dir,
        device_map=device,
        torch_dtype=torch.float32,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for start in range(completed, len(rows), max(args.batch_rows, 1)):
        stop = min(start + max(args.batch_rows, 1), len(rows))
        torch.manual_seed(args.seed + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + start)
        batch = rows.iloc[start:stop]
        contexts = [
            torch.tensor(context_prices(row), dtype=torch.float32)
            for row in batch.itertuples(index=False)
        ]
        horizon = int(batch["actual_trading_days"].max())
        with torch.inference_mode():
            forecast = pipeline.predict(
                inputs=contexts,
                prediction_length=horizon,
                num_samples=args.paths,
                limit_prediction_length=False,
            )
        forecast = np.asarray(forecast.cpu(), dtype=np.float32)
        for offset, row in enumerate(batch.itertuples(index=False)):
            maturity = int(row.actual_trading_days)
            path = np.concatenate(
                [
                    np.full((args.paths, 1), float(row.start_price), dtype=np.float32),
                    np.maximum(forecast[offset, :, :maturity], 1e-6),
                ],
                axis=1,
            )
            generated[start + offset, :, :] = path[:, -1:]
            generated[start + offset, :, : maturity + 1] = path
        generated.flush()
        atomic_json(
            progress_path,
            {
                "completed_rows": stop,
                "rows": len(rows),
                "paths": args.paths,
                "width": width,
                "model_dir": str(Path(args.model_dir).resolve()),
            },
        )
        print(f"chronos: {stop}/{len(rows)}", flush=True)
    generated.flush()
    del generated
    os.replace(partial_path, archive_path)
    progress_path.unlink(missing_ok=True)
    generated = np.load(archive_path, mmap_mode="r")
    summary = summarize(rows, generated, args.paths)
    summary.update(
        {
            "baseline": "chronos_t5_small_zero_shot",
            "seed": args.seed,
            "rows": int(len(rows)),
            "paths_per_row": args.paths,
            "context": "252 reconstructed historical price levels",
        }
    )
    atomic_json(metrics_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

