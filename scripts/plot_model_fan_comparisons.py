"""Create same-window fan charts for every archived final model family.

The plot intentionally normalizes all paths by the shared initial price and
uses the same mechanically selected held-out window within each market. It is
therefore a visual model comparison, not a cherry-picked performance exhibit.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp

ARCHIVES = ROOT / "Results" / "Path_Archives"
EXTERNAL = ROOT / "Results" / "External_Baselines"
OUT = ROOT / "Results" / "Final_Attribution" / "Fan_Comparisons"
SCALE = 0.09

MODELS = [
    ("History-aware DLPM", ARCHIVES / "frozen_history_dlpm_seed20260730.npy", "raw"),
    ("Contract-only DLPM", ARCHIVES / "contract_only_dlpm_seed20260730_formal.npy", "raw"),
    ("Gaussian DDPM (simple)", ARCHIVES / "history_gaussian_simple_seed20260730_formal.npy", "raw"),
    ("Gaussian DDPM (finance)", ARCHIVES / "history_gaussian_finance_seed20260730_formal.npy", "raw"),
    ("Historical block bootstrap", ARCHIVES / "unconditional_block_bootstrap.npy", "price"),
    ("Historical-drift GBM", ARCHIVES / "historical_drift_gbm.npy", "price"),
    ("Physical Student-t GARCH", ARCHIVES / "physical_student_t_garch.npy", "price"),
    ("Momentum/trend rule", ARCHIVES / "momentum_trend_rule.npy", "price"),
    ("Conditional Neural SDE", EXTERNAL / "conditional_neural_sde_seed20260730" / "raw_test_paths.npy", "raw"),
    ("Chronos-T5 zero-shot", EXTERNAL / "chronos_t5_small_zero_shot" / "price_paths.npy", "price"),
]


def realized(row) -> np.ndarray:
    return np.asarray(ast.literal_eval(str(row.price_series)), dtype=float)


def recover(raw: np.ndarray, row, kind: str) -> np.ndarray:
    steps = int(row.actual_trading_days)
    horizon = steps + 1
    if kind == "price":
        return np.asarray(raw[:, :horizon], dtype=float)
    tokens = np.asarray(raw[:, :], dtype=float)
    daily = tokens[:, 1:horizon] * SCALE
    return float(row.start_price) * np.exp(
        np.concatenate([np.zeros((tokens.shape[0], 1)), np.cumsum(daily, axis=1)], axis=1)
    )


def choose_row(rows: pd.DataFrame, asset: str, archive: np.ndarray) -> int:
    candidates = np.flatnonzero(rows.asset_underlying.astype(str).to_numpy() == asset)
    errors = []
    for index in candidates:
        row = rows.iloc[index]
        paths = recover(archive[index], row, "raw")
        truth = realized(row)
        terminal = min(paths.shape[1], truth.size) - 1
        errors.append(abs(np.median(paths[:, terminal] / paths[:, 0] - 1.0) - (truth[terminal] / truth[0] - 1.0)))
    rank = np.argsort(errors)[len(errors) // 2]
    return int(candidates[rank])


def plot_asset(rows: pd.DataFrame, asset: str, selected: int, arrays: dict[str, np.ndarray]) -> None:
    row = rows.iloc[selected]
    truth = realized(row)
    horizon = min(int(row.actual_trading_days) + 1, truth.size)
    fig, axes = plt.subplots(5, 2, figsize=(14, 19), sharex=True, sharey=True)
    for ax, (label, _, kind) in zip(axes.flat, MODELS):
        paths = recover(arrays[label][selected], row, kind)[:, :horizon] / float(row.start_price)
        real = truth[:horizon] / float(row.start_price)
        q05, q25, q50, q75, q95 = np.quantile(paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        days = np.arange(horizon)
        ax.fill_between(days, q05, q95, color="#9ecae1", alpha=0.42, linewidth=0)
        ax.fill_between(days, q25, q75, color="#3182bd", alpha=0.26, linewidth=0)
        ax.plot(days, q50, color="#08519c", linewidth=1.55, label="Generated median")
        ax.plot(days, real, color="#222222", linewidth=1.35, label="Realized")
        ax.axhline(1.0, color="#9a9a9a", linewidth=0.65, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.22, linewidth=0.5)
    for ax in axes[-1, :]: ax.set_xlabel("Trading day")
    for ax in axes[:, 0]: ax.set_ylabel("Index level / S0")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        f"Same-window fan comparison: {asset} | start {row.start_date} | mechanically selected median-DLPM-error window",
        y=0.998, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / f"fan_comparison_{asset.replace(' ', '_')}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mechanically selected, same-window fan comparisons from archived paths."
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=Path(pp.Testing_DATA_DIR) / "clean_test_history_context.csv",
        help="Processed held-out rows used to construct the frozen archives.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.rows_csv, low_memory=False)
    rows = rows.sort_values(["asset_underlying", "start_date"]).reset_index(drop=True)
    arrays = {label: np.load(path, mmap_mode="r") for label, path, _ in MODELS}
    chosen = []
    primary = arrays["History-aware DLPM"]
    for asset in ("CSI1000", "NASDAQ"):
        index = choose_row(rows, asset, primary)
        plot_asset(rows, asset, index, arrays)
        chosen.append({"asset": asset, "row_index": index, "start_date": str(rows.iloc[index].start_date)})
    pd.DataFrame(chosen).to_csv(OUT / "selection_manifest.csv", index=False)
    print(pd.DataFrame(chosen).to_string(index=False))


if __name__ == "__main__":
    main()

