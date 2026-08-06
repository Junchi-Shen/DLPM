"""Audit realized tail thickness from frozen, forecast-aligned path archives.

The primary sample uses the first future log return at each forecast origin.
This avoids counting the same realized return repeatedly through overlapping
multi-day test windows. Model archives contribute one first-step return per
Monte Carlo path and forecast origin.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def excess_kurtosis(values: np.ndarray) -> float:
    """Unbiased Fisher excess kurtosis without an optional SciPy dependency."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 4:
        return np.nan
    centred = values - np.mean(values)
    m2 = float(np.mean(centred**2))
    if m2 <= 0.0:
        return np.nan
    g2 = float(np.mean(centred**4) / (m2 * m2) - 3.0)
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0))


def realized_first_returns(rows: pd.DataFrame) -> np.ndarray:
    values = []
    for series in rows["price_series"]:
        prices = np.asarray(ast.literal_eval(str(series)), dtype=float)
        if prices.size >= 2 and np.all(np.isfinite(prices[:2])) and np.all(prices[:2] > 0):
            values.append(float(np.log(prices[1] / prices[0])))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def raw_first_returns(archive: np.ndarray, scale: float) -> np.ndarray:
    if archive.ndim != 3 or archive.shape[2] < 2:
        raise ValueError(f"Expected [windows, paths, tokens>=2], received {archive.shape}")
    return np.asarray(archive[:, :, 1], dtype=float) * float(scale)


def price_first_returns(archive: np.ndarray) -> np.ndarray:
    if archive.ndim != 3 or archive.shape[2] < 2:
        raise ValueError(f"Expected [windows, paths, prices>=2], received {archive.shape}")
    first = np.asarray(archive[:, :, :2], dtype=float)
    valid = np.all(np.isfinite(first), axis=2) & np.all(first > 0, axis=2)
    result = np.full(first.shape[:2], np.nan, dtype=float)
    result[valid] = np.log(first[:, :, 1][valid] / first[:, :, 0][valid])
    return result


def hill_alpha(values: np.ndarray, side: str, fraction: float) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    centre = float(np.median(values))
    magnitudes = values - centre if side == "right" else centre - values
    magnitudes = np.sort(magnitudes[magnitudes > 0.0])[::-1]
    k = max(5, int(np.floor(fraction * values.size)))
    k = min(k, magnitudes.size - 1)
    if k < 5:
        return np.nan, int(max(k, 0))
    threshold = float(magnitudes[k])
    if threshold <= 0.0:
        return np.nan, int(k)
    logs = np.log(np.maximum(magnitudes[:k] / threshold, 1.0))
    denominator = float(np.sum(logs))
    return (float(k / denominator) if denominator > 0 else np.nan), int(k)


def summarize(values: np.ndarray, realized_q01: float, realized_q99: float) -> dict:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    q01, q05, q25, q50, q75, q95, q99 = np.quantile(
        values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    iqr = max(float(q75 - q25), 1e-12)
    left_5, left_5_k = hill_alpha(values, "left", 0.05)
    right_5, right_5_k = hill_alpha(values, "right", 0.05)
    left_1, left_1_k = hill_alpha(values, "left", 0.01)
    right_1, right_1_k = hill_alpha(values, "right", 0.01)
    return {
        "observations": int(values.size),
        "hill_left_5pct": left_5,
        "hill_right_5pct": right_5,
        "hill_left_1pct": left_1,
        "hill_right_1pct": right_1,
        "hill_k_left_5pct": left_5_k,
        "hill_k_right_5pct": right_5_k,
        "hill_k_left_1pct": left_1_k,
        "hill_k_right_1pct": right_1_k,
        "excess_kurtosis": excess_kurtosis(values),
        "left_tail_iqr_ratio": float((q50 - q01) / iqr),
        "right_tail_iqr_ratio": float((q99 - q50) / iqr),
        "q01": float(q01),
        "q05": float(q05),
        "q95": float(q95),
        "q99": float(q99),
        "realized_left_1pct_exceedance": float(np.mean(values < realized_q01)),
        "realized_right_1pct_exceedance": float(np.mean(values > realized_q99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--dlpm-archive", required=True, type=Path)
    parser.add_argument("--gaussian-archive", required=True, type=Path)
    parser.add_argument("--bootstrap-archive", required=True, type=Path)
    parser.add_argument("--gbm-archive", required=True, type=Path)
    parser.add_argument("--return-scale", type=float, default=0.09)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    rows = (
        pd.read_csv(args.test_csv, low_memory=False)
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    realized = realized_first_returns(rows)
    realized_q01, realized_q99 = np.nanquantile(realized, [0.01, 0.99])

    archives = {
        "Realized": realized[:, None],
        "History-aware DLPM": raw_first_returns(
            np.load(args.dlpm_archive, mmap_mode="r"), args.return_scale
        ),
        "Gaussian DDPM": raw_first_returns(
            np.load(args.gaussian_archive, mmap_mode="r"), args.return_scale
        ),
        "Historical block bootstrap": price_first_returns(
            np.load(args.bootstrap_archive, mmap_mode="r")
        ),
        "Historical-drift GBM": price_first_returns(
            np.load(args.gbm_archive, mmap_mode="r")
        ),
    }
    for name, array in archives.items():
        if array.shape[0] != len(rows):
            raise ValueError(f"{name}: {array.shape[0]} rows, expected {len(rows)}")

    overall = []
    per_index = []
    assets = rows["asset_underlying"].astype(str).to_numpy()
    for name, array in archives.items():
        metrics = summarize(array, float(realized_q01), float(realized_q99))
        overall.append({"source": name, **metrics})
        for asset in sorted(np.unique(assets)):
            mask = assets == asset
            actual_asset = realized[mask]
            asset_q01, asset_q99 = np.nanquantile(actual_asset, [0.01, 0.99])
            per_index.append({
                "source": name,
                "asset_underlying": asset,
                **summarize(array[mask], float(asset_q01), float(asset_q99)),
            })

    payload = {
        "protocol": {
            "sample": "first future log return at each chronological forecast origin",
            "reason": "avoids repeated realized returns induced by overlapping multi-day windows",
            "windows": int(len(rows)),
            "p_paths": int(archives["History-aware DLPM"].shape[1]),
            "baseline_paths": int(archives["Historical block bootstrap"].shape[1]),
            "hill_definition": "tail magnitudes around the source median; k=floor(fraction*n)",
            "interpretation": "finite-sample effective tails under the deployed numerical caps, not an unbounded stable-law claim",
        },
        "overall": overall,
        "per_index": per_index,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(overall).to_csv(args.output_csv, index=False)
    print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    main()
