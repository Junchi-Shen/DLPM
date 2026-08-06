# -*- coding: utf-8 -*-
"""Build publication figures for the final DLPM paper from frozen outputs."""
from __future__ import annotations

import ast
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DLPM_PAPER_FIGURE_DIR", ROOT / "paper" / "figures" / "final"))
OUT.mkdir(parents=True, exist_ok=True)

DATA = Path(os.environ.get(
    "DLPM_PAPER_TEST_DATA",
    ROOT / "data" / "processed" / "Testing_Dataset" / "clean_test_history_context.csv",
))
P_ROOT = Path(os.environ.get("DLPM_P_PATH_ROOT", ROOT / "artifacts" / "path_archives" / "p"))
Q_ROOT = Path(os.environ.get("DLPM_Q_PATH_ROOT", ROOT / "artifacts" / "path_archives" / "rnq"))
QUALITY = Path(os.environ.get(
    "DLPM_PATH_QUALITY_CSV", ROOT / "artifacts" / "reports" / "path_quality_summary.csv"
))

ASSETS = [
    "CSI1000",
    "CSI300",
    "CSI500",
    "SSE_Composite",
    "SP500",
    "NASDAQ",
    "Dow_Jones",
    "Russell_2000",
]
LABELS = {
    "CSI1000": "CSI 1000",
    "CSI300": "CSI 300",
    "CSI500": "CSI 500",
    "SSE_Composite": "SSE Composite",
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ",
    "Dow_Jones": "Dow Jones",
    "Russell_2000": "Russell 2000",
}
CONTRACTS = ["vanilla_call", "standard_asian", "standard_lookback"]
CONTRACT_LABELS = {
    "vanilla_call": "Vanilla",
    "standard_asian": "Asian",
    "standard_lookback": "Lookback",
    "my_accumulator": "Accumulator",
    "my_snowball_A": "Snowball",
}
BLUE = "#2368A2"
LIGHT_BLUE = "#8EC1DA"
ORANGE = "#D8832F"
INK = "#222222"
GREEN = "#2A7F62"
RED = "#B64B4B"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def parse_path(value: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(value), dtype=float)


def p_paths(asset: str) -> np.ndarray:
    candidates = sorted((P_ROOT / asset).glob("*_generated_paths*_samples.npy"))
    if not candidates:
        raise FileNotFoundError(f"No P-path archive found for {asset} below {P_ROOT}")
    return np.load(candidates[0], mmap_mode="r")


def q_paths(asset: str, rows: int) -> np.ndarray:
    files = list((Q_ROOT / asset).glob("garch_risk_neutral_paths_*_samples.npy"))
    raw = np.load(files[0], mmap_mode="r")
    return raw.reshape(rows, 256, raw.shape[-1])


def representative_index(frame: pd.DataFrame, paths: np.ndarray) -> int:
    errors = []
    for i, value in enumerate(frame["price_series"]):
        realized = parse_path(value)
        horizon = min(len(realized), paths.shape[-1])
        terminal = np.median(paths[i, :, horizon - 1])
        errors.append(abs(terminal / realized[horizon - 1] - 1.0))
    order = np.argsort(np.asarray(errors))
    return int(order[len(order) // 2])


def build_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.57, 0.16, 0.27, "Pre-start state", "60/252-day returns\ntrend, drawdown,\nvolatility path"),
        (0.22, 0.57, 0.16, 0.27, "Condition encoder", "397 numerical inputs\n+ country/index\nembeddings"),
        (0.42, 0.57, 0.16, 0.27, "Conditional DLPM", r"$\alpha$-stable diffusion" "\nfinance-aware loss\nEMA checkpoint"),
        (0.62, 0.57, 0.16, 0.27, "P distribution", "64 conditional paths\nDLIM-50 sampler\npath diagnostics"),
        (0.82, 0.57, 0.16, 0.27, "Payoff beliefs", "Vanilla, Asian,\nLookback, Accumulator,\nSnowball"),
        (0.42, 0.10, 0.16, 0.24, "Historical fit", "Deduplicated returns\nthrough validation"),
        (0.62, 0.10, 0.16, 0.24, "RN-Q benchmark", "Student-t GARCH\nunit-variance cap\nmartingale correction"),
        (0.82, 0.10, 0.16, 0.24, "Economic audit", "Bid/ask spread\nrealized P&L\ncalendar bootstrap"),
    ]
    for x, y, w, h, title, body in boxes:
        color = LIGHT_BLUE if y > 0.5 else "#F2C38B"
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=color,
            edgecolor="#35556E" if y > 0.5 else "#8F5A23",
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", weight="bold")
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=8.6)
    arrows = [
        ((0.18, 0.705), (0.22, 0.705)),
        ((0.38, 0.705), (0.42, 0.705)),
        ((0.58, 0.705), (0.62, 0.705)),
        ((0.78, 0.705), (0.82, 0.705)),
        ((0.50, 0.34), (0.68, 0.57)),
        ((0.58, 0.22), (0.62, 0.22)),
        ((0.78, 0.22), (0.82, 0.22)),
        ((0.90, 0.57), (0.90, 0.34)),
    ]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#53606A"))
    ax.text(
        0.02,
        0.93,
        "History-aware P generator and independently constructed RN-Q benchmark",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    fig.savefig(OUT / "01_system_workflow.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_quality() -> None:
    q = pd.read_csv(QUALITY)
    q = q[q["asset"] != "ALL"].copy()
    q["label"] = q["asset"].map(LABELS)
    q["terminal_bias_pp"] = 100 * q["terminal_median_error"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    x = np.arange(len(q))
    colors = [BLUE if v >= 0 else ORANGE for v in q["terminal_bias_pp"]]
    axes[0].bar(x, q["terminal_bias_pp"], color=colors)
    axes[0].axhline(0, color=INK, linewidth=0.8)
    axes[0].set_title("Terminal median bias")
    axes[0].set_ylabel("Percentage points")
    axes[1].bar(x, q["generated_vol_ratio"], color=GREEN)
    axes[1].axhline(1, color=INK, linewidth=1, linestyle="--")
    axes[1].set_title("Generated / realized volatility")
    axes[2].bar(x, q["terminal_crps"], color="#6C77A8")
    axes[2].set_title("Terminal CRPS")
    axes[2].set_ylabel("Lower is better")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(q["label"], rotation=38, ha="right", fontsize=8)
    fig.savefig(OUT / "02_path_quality_by_index.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_p_fans() -> None:
    data = pd.read_csv(DATA, low_memory=False)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), constrained_layout=True)
    chosen = {}
    for ax, asset in zip(axes.flat, ASSETS):
        frame = data[data["asset_underlying"] == asset].reset_index(drop=True)
        paths = p_paths(asset)
        idx = representative_index(frame, paths)
        chosen[asset] = idx
        realized = parse_path(frame.loc[idx, "price_series"])
        horizon = min(len(realized), paths.shape[-1])
        samples = np.asarray(paths[idx, :, :horizon])
        q05, q25, q50, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        t = np.arange(horizon)
        ax.fill_between(t, q05, q95, color=LIGHT_BLUE, alpha=0.38, label="5-95%")
        ax.fill_between(t, q25, q75, color=BLUE, alpha=0.23, label="25-75%")
        ax.plot(t, q50, color=BLUE, linewidth=1.5, label="DLPM median")
        ax.plot(t, realized[:horizon], color=INK, linewidth=1.35, label="Realized")
        ax.set_title(LABELS[asset])
        ax.set_xlabel("Trading day")
        ax.set_ylabel("Index level")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Representative conditional DLPM fans (median-ranked terminal error)", y=1.03, fontsize=13)
    fig.savefig(OUT / "03_p_only_fans_all_indices.png", dpi=260, bbox_inches="tight")
    (OUT / "03_p_only_fans_selection.json").write_text(
        json.dumps(chosen, indent=2), encoding="utf-8"
    )
    plt.close(fig)


def build_pq_fans() -> None:
    data = pd.read_csv(DATA, low_memory=False)
    fig, axes = plt.subplots(4, 2, figsize=(11.8, 13.0), constrained_layout=True)
    for row, asset in enumerate(ASSETS):
        frame = data[data["asset_underlying"] == asset].reset_index(drop=True)
        pp = p_paths(asset)
        idx = representative_index(frame, pp)
        realized = parse_path(frame.loc[idx, "price_series"])
        horizon = min(len(realized), pp.shape[-1])
        qq = q_paths(asset, len(frame))
        p = np.asarray(pp[idx, :, :horizon])
        q = np.asarray(qq[idx, :, :horizon])
        ax = axes[row // 2 * 2 + row % 2, 0] if False else None
        # Each row in the figure is one asset pair; two assets are stacked per block.
        block = row
        if block >= axes.shape[0]:
            break
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(14.0, 11.5), constrained_layout=True)
    for k, asset in enumerate(ASSETS):
        rr, cc = divmod(k, 2)
        frame = data[data["asset_underlying"] == asset].reset_index(drop=True)
        pp = p_paths(asset)
        idx = representative_index(frame, pp)
        realized = parse_path(frame.loc[idx, "price_series"])
        horizon = min(len(realized), pp.shape[-1])
        p = np.asarray(pp[idx, :, :horizon])
        q = np.asarray(q_paths(asset, len(frame))[idx, :, :horizon])
        for ax, samples, color, model in [
            (axes[rr, 2 * cc], p, BLUE, "DLPM P"),
            (axes[rr, 2 * cc + 1], q, ORANGE, "RN-Q"),
        ]:
            lo, med, hi = np.quantile(samples, [0.05, 0.5, 0.95], axis=0)
            t = np.arange(horizon)
            ax.fill_between(t, lo, hi, color=color, alpha=0.24)
            ax.plot(t, med, color=color, linewidth=1.35)
            ax.plot(t, realized[:horizon], color=INK, linewidth=1.15)
            ax.set_title(f"{LABELS[asset]}: {model}", fontsize=9.5)
            ax.set_xlabel("Day", fontsize=8)
            ax.set_ylabel("Index", fontsize=8)
    fig.suptitle("DLPM P beliefs and martingale-corrected RN-Q benchmark", fontsize=13)
    fig.savefig(OUT / "04_p_vs_rnq_fans.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_game(label: str) -> tuple[dict, dict]:
    base = json.loads(
        (ROOT / "Results" / "Final_Test" / f"formal_universal_v1_pq_{label}.json").read_text(
            encoding="utf-8"
        )
    )
    snow = json.loads(
        (
            ROOT
            / "Results"
            / "Final_Test"
            / f"formal_universal_v1_pq_{label}_snowfix.json"
        ).read_text(encoding="utf-8")
    )
    return base, snow


def economic_summary() -> pd.DataFrame:
    data = pd.read_csv(DATA, low_memory=False)
    rows = []
    for label, g in [("g005", 0.05), ("g010", 0.10), ("g020", 0.20)]:
        game, snow = load_game(label)
        all_rows = [r for r in game["results"] if r["contract"] != "my_snowball_A"]
        all_rows.extend(snow["results"])
        for row in all_rows:
            asset = row["asset"]
            contract = row["contract"]
            frame = data[data["asset_underlying"] == asset].reset_index(drop=True)
            log = pd.read_csv(Path(row["trade_log"]) / "full_trade_log.csv", encoding="utf-8-sig")
            active = log["äº¤æ˜“ç±»åž‹"].astype(str) != "No Trade"
            pnl = pd.to_numeric(log["Pæ¨¡åž‹ç›ˆäº"], errors="coerce").fillna(0.0)
            if contract in CONTRACTS:
                values = (pnl[active].to_numpy() / frame.loc[active, "start_price"].to_numpy())
            else:
                values = pnl[active].to_numpy()
            rows.append(
                {
                    "g": g,
                    "asset": asset,
                    "contract": contract,
                    "trades": int(active.sum()),
                    "rows": len(log),
                    "mean_active": float(np.mean(values)) if len(values) else 0.0,
                    "mean_unconditional": float(np.mean(pnl.to_numpy())),
                    "win_rate": float(np.mean(values > 0)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_economics() -> None:
    s = economic_summary()
    smooth = s[s["contract"].isin(CONTRACTS)]
    aggregate = (
        smooth.groupby(["g", "contract"])
        .apply(
            lambda x: pd.Series(
                {
                    "mean": np.average(x["mean_active"], weights=x["trades"]),
                    "win": np.average(x["win_rate"], weights=x["trades"]),
                    "trade_rate": x["trades"].sum() / x["rows"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    width = 0.23
    gs = sorted(aggregate["g"].unique())
    x = np.arange(3)
    for j, g in enumerate(gs):
        sub = aggregate[aggregate["g"] == g].set_index("contract").loc[CONTRACTS]
        axes[0].bar(x + (j - 1) * width, 100 * sub["mean"], width, label=f"g={g:.2f}")
        axes[1].bar(x + (j - 1) * width, sub["win"], width)
        axes[2].bar(x + (j - 1) * width, sub["trade_rate"], width)
    axes[0].set_title("Active-trade P&L / initial spot")
    axes[0].set_ylabel("Percent")
    axes[1].set_title("Active-trade win rate")
    axes[1].axhline(0.5, color=INK, linestyle="--", linewidth=0.8)
    axes[2].set_title("Execution rate")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([CONTRACT_LABELS[c] for c in CONTRACTS])
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.55, 1.23))
    fig.savefig(OUT / "05_economic_sensitivity.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    heat = smooth[smooth["g"] == 0.05].pivot(
        index="asset", columns="contract", values="mean_active"
    ).loc[ASSETS, CONTRACTS]
    fig, ax = plt.subplots(figsize=(7.3, 4.5), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(100 * heat.to_numpy())))
    im = ax.imshow(100 * heat.to_numpy(), cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax))
    ax.set_xticks(np.arange(3), [CONTRACT_LABELS[c] for c in CONTRACTS])
    ax.set_yticks(np.arange(8), [LABELS[a] for a in ASSETS])
    for i in range(8):
        for j in range(3):
            ax.text(j, i, f"{100 * heat.iloc[i, j]:.2f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean active P&L / initial spot")
    ax.set_title("Cross-market economic transfer at g=0.05")
    fig.savefig(OUT / "06_cross_asset_heatmap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_bootstrap() -> None:
    path = (
        ROOT
        / "Results"
        / "Final_Test"
        / "formal_universal_v1_pq_g005_joint_calendar_bootstrap.json"
    )
    b = json.loads(path.read_text(encoding="utf-8"))["results"]["60"]
    rows = []
    for c in CONTRACTS:
        d = b[c]
        rows.append((CONTRACT_LABELS[c], d["mean"], d["q025"], d["q975"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    y = np.arange(len(rows))
    mean = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", color=BLUE, capsize=4)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Unconditional window P&L (index points)")
    ax.set_title("Joint calendar-block bootstrap, g=0.05, 60-day blocks")
    ax.invert_yaxis()
    fig.savefig(OUT / "07_joint_bootstrap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_structured() -> None:
    base, snow = load_game("g005")
    rows = [r for r in base["results"] if r["contract"] == "my_accumulator"]
    rows.extend(snow["results"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for ax, contract in zip(axes, ["my_accumulator", "my_snowball_A"]):
        values = []
        for row in rows:
            if row["contract"] != contract:
                continue
            log = pd.read_csv(Path(row["trade_log"]) / "full_trade_log.csv", encoding="utf-8-sig")
            active = log["äº¤æ˜“ç±»åž‹"].astype(str) != "No Trade"
            values.extend(pd.to_numeric(log.loc[active, "Pæ¨¡åž‹ç›ˆäº"], errors="coerce").dropna())
        ax.hist(values, bins=55, density=True, color=BLUE if contract == "my_accumulator" else ORANGE, alpha=0.72)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(CONTRACT_LABELS[contract])
        ax.set_xlabel("Active-trade P&L in notional-rate units")
        ax.set_ylabel("Density")
    fig.suptitle("Structured-product payoff diagnostics at g=0.05")
    fig.savefig(OUT / "08_structured_pnl_distributions.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        choices=("all", "workflow", "quality", "fans"),
        default="all",
        help="Select which frozen-output figure group to rebuild.",
    )
    args = parser.parse_args()
    style()
    if args.figures in ("all", "workflow"):
        build_workflow()
    if args.figures in ("all", "quality"):
        build_quality()
    if args.figures in ("all", "fans"):
        build_p_fans()
    print(f"Saved paper Figures 1--3 to {OUT}")


if __name__ == "__main__":
    main()

