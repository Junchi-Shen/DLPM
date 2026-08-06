"""Calibration--sharpness audit for an existing frozen path archive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))

import Project_Path as pp
from evaluate_path_models import condition_rows, load_processor, real_prices, recover_paths


ALPHA = 0.10


def interval_score(y, lower, upper, alpha=ALPHA):
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return (
        upper - lower
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def tercile(series: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    percentile = series.groupby(level=0).rank(method="average", pct=True)
    return pd.cut(percentile, [-np.inf, 1 / 3, 2 / 3, np.inf], labels=labels).astype(str)


def summarize(frame: pd.DataFrame, dimension: str, group_column: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(group_column, sort=False, dropna=False):
        rows.append({
            "dimension": dimension,
            "group": str(group),
            "windows": int(len(part)),
            "terminal_coverage_90": float(part.terminal_covered.mean()),
            "terminal_coverage_error": float(abs(part.terminal_covered.mean() - 0.90)),
            "terminal_width_90_over_s0": float(part.terminal_width.mean()),
            "terminal_interval_score_over_s0": float(part.terminal_interval_score.mean()),
            "pointwise_coverage_90": float(part.pointwise_coverage.mean()),
            "pointwise_coverage_error": float(abs(part.pointwise_coverage.mean() - 0.90)),
            "pointwise_width_90_over_s0": float(part.pointwise_width.mean()),
            "pointwise_interval_score_over_s0": float(part.pointwise_interval_score.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = pd.read_csv(
        pp.Testing_DATA_DIR / "clean_test_history_context.csv", low_memory=False
    ).sort_values(["asset_underlying", "start_date"]).reset_index(drop=True)
    raw = np.load(args.archive, allow_pickle=False)
    if raw.ndim != 3 or raw.shape[0] != len(rows):
        raise ValueError(f"Expected [{len(rows)}, paths, tokens], received {raw.shape}")

    processor = load_processor(args.run_dir)
    _, _, masks = condition_rows(processor, rows)
    generated = recover_paths(
        raw, rows, masks.numpy(), float(processor.config.get("volatility_scale", 1.0))
    )

    records = []
    for row, paths in zip(rows.itertuples(index=False), generated):
        realized = real_prices(row)
        horizon = min(len(realized), paths.shape[1])
        realized = realized[:horizon]
        paths = paths[:, :horizon]
        start = max(float(realized[0]), 1e-12)
        lower, upper = np.quantile(paths, [0.05, 0.95], axis=0)
        point_scores = interval_score(realized, lower, upper) / start

        realized_terminal = realized[-1] / start - 1.0
        generated_terminal = paths[:, -1] / paths[:, 0] - 1.0
        terminal_lower, terminal_upper = np.quantile(generated_terminal, [0.05, 0.95])
        terminal_score = interval_score(realized_terminal, terminal_lower, terminal_upper)
        records.append({
            "asset_underlying": str(row.asset_underlying),
            "start_date": str(row.start_date),
            "tenor_days": int(row.contract_calendar_days),
            "hist_trend_60": float(row.hist_trend_60),
            "hist_current_drawdown_60": float(row.hist_current_drawdown_60),
            "hist_rv_60": float(row.hist_rv_60),
            "terminal_covered": float(terminal_lower <= realized_terminal <= terminal_upper),
            "terminal_width": float(terminal_upper - terminal_lower),
            "terminal_interval_score": float(terminal_score),
            "pointwise_coverage": float(np.mean((realized >= lower) & (realized <= upper))),
            "pointwise_width": float(np.mean((upper - lower) / start)),
            "pointwise_interval_score": float(np.mean(point_scores)),
        })

    detail = pd.DataFrame(records)
    indexed = detail.set_index("asset_underlying", drop=False)
    detail["trend_state"] = tercile(
        indexed.hist_trend_60, ("Low trend", "Middle trend", "High trend")
    ).to_numpy()
    detail["drawdown_state"] = tercile(
        indexed.hist_current_drawdown_60,
        ("Deep drawdown", "Middle drawdown", "Shallow drawdown"),
    ).to_numpy()
    detail["volatility_state"] = tercile(
        indexed.hist_rv_60, ("Low volatility", "Middle volatility", "High volatility")
    ).to_numpy()
    detail["overall"] = "All windows"

    summaries = [
        summarize(detail, "Overall", "overall"),
        summarize(detail, "Index", "asset_underlying"),
        summarize(detail, "Tenor", "tenor_days"),
        summarize(detail, "Trend state", "trend_state"),
        summarize(detail, "Drawdown state", "drawdown_state"),
        summarize(detail, "Volatility state", "volatility_state"),
    ]
    summary = pd.concat(summaries, ignore_index=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "window_calibration_sharpness.csv", index=False)
    summary.to_csv(output / "group_calibration_sharpness.csv", index=False)
    payload = {
        "archive": str(Path(args.archive).resolve()),
        "rows": int(len(rows)),
        "paths_per_row": int(raw.shape[1]),
        "interval": "central 90% (5th--95th percentiles)",
        "units": "width and interval score are normalized by initial spot",
        "groups": summary.to_dict(orient="records"),
    }
    (output / "calibration_sharpness.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    panels = [
        ("Index", "By index"),
        ("Tenor", "By contractual tenor"),
        ("Volatility state", "By pre-window volatility state"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for axis, (dimension, title) in zip(axes, panels):
        part = summary[summary.dimension == dimension].reset_index(drop=True)
        for index, item in part.iterrows():
            axis.scatter(
                item.pointwise_width_90_over_s0,
                item.pointwise_coverage_90,
                s=50,
                color=colors[index % len(colors)],
                label=item.group,
            )
        axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Mean 90% fan width / $S_0$")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Empirical pointwise coverage")
    fig.suptitle("Calibration--sharpness audit: narrower is sharper; 0.90 is calibrated")
    fig.tight_layout()
    fig.savefig(output / "calibration_sharpness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

