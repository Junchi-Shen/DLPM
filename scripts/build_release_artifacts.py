"""Build the paper manifest, generated LaTeX rows, and compact figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
GENERATED = ROOT / "paper" / "generated"
FIGURES = ROOT / "paper" / "figures" / "final"


def read_json(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float, decimals: int = 2) -> str:
    return f"{100.0 * value:.{decimals}f}\\%"


def pp(value: float, decimals: int = 2) -> str:
    return f"{100.0 * value:.{decimals}f}pp"


def write_rows(name: str, rows: list[str]) -> None:
    (GENERATED / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sources = {
        "path_quality": "path_quality_frozen_ema12000.json",
        "economic": "economic_40path_summary.json",
        "delta_hedge": "delta_hedge_40path_summary.json",
        "structured": "structured_product_40path_summary.json",
        "calibration": "calibration_sharpness_40paths.json",
        "effective_tails": "effective_tail_behavior.json",
        "quote_uncertainty": "quote_mc_threshold_sensitivity.csv",
    }
    path_quality = read_json(sources["path_quality"])
    economic = read_json(sources["economic"])
    hedged = read_json(sources["delta_hedge"])
    structured = read_json(sources["structured"])
    calibration = read_json(sources["calibration"])
    effective_tails = read_json(sources["effective_tails"])

    if structured["protocol"]["p_paths"] != 40:
        raise ValueError("Structured-product source is not the frozen 40-path audit")
    for contract in ("vanilla_call", "asian_call", "lookback_call"):
        if economic[f"g0.05_{contract}"]["rows"] != 6824:
            raise ValueError(f"Unexpected row count for {contract}")
    if calibration["paths_per_row"] != 40 or calibration["rows"] != 6824:
        raise ValueError("Calibration source is not the frozen 40-path audit")

    model_labels = {
        "history_aware_dlpm": "History-aware DLPM",
        "contract_only_dlpm": "Contract-only DLPM",
        "gaussian_ddpm_simple": "Gaussian DDPM, simple loss",
        "gaussian_ddpm_finance_aware": "Gaussian DDPM, finance-aware",
        "historical_block_bootstrap": "Historical block bootstrap",
        "historical_drift_gbm": "Historical-drift GBM",
        "conditional_neural_sde": "Conditional neural SDE",
        "chronos_t5_zero_shot": "Chronos-T5 zero-shot",
    }
    path_rows = []
    for key, label in model_labels.items():
        values = path_quality["models"][key]
        path_rows.append(
            f"{label} & {values['terminal_crps']:.4f} & {values['energy_score']:.4f} & "
            f"{values['coverage_90']:.3f} & {values['volatility_ratio']:.3f} & "
            f"{values['tail_error']:.4f} \\\\"
        )
    write_rows("path_quality_rows.tex", path_rows)

    tail_rows = []
    for item in effective_tails["overall"]:
        exceedance = (
            f"{100.0 * item['realized_left_1pct_exceedance']:.2f} / "
            f"{100.0 * item['realized_right_1pct_exceedance']:.2f}"
        )
        tail_rows.append(
            f"{item['source']} & "
            f"{item['hill_left_5pct']:.2f} / {item['hill_right_5pct']:.2f} & "
            f"{item['hill_left_1pct']:.2f} / {item['hill_right_1pct']:.2f} & "
            f"{item['excess_kurtosis']:.2f} & "
            f"{item['left_tail_iqr_ratio']:.2f} / {item['right_tail_iqr_ratio']:.2f} & "
            f"{exceedance} \\\\"
        )
    write_rows("effective_tail_rows.tex", tail_rows)

    contract_labels = {
        "vanilla_call": "Vanilla call",
        "asian_call": "Asian call",
        "lookback_call": "Floating-strike lookback call",
    }
    main_rows = []
    spread_rows = []
    for contract, label in contract_labels.items():
        base = economic[f"g0.05_{contract}"]
        buy_contribution = (
            base["buy_trades"] / base["rows"] * base["mean_pnl_over_spot_buy"]
        )
        sell_contribution = (
            base["sell_trades"] / base["rows"] * base["mean_pnl_over_spot_sell"]
        )
        main_rows.append(
            f"{label} & {pct(base['mean_pnl_over_spot_unconditional'])} & "
            f"{pct(base['trade_rate'])} & {pct(base['win_rate_active'])} & "
            f"{pct(base['buy_share_of_active'])} & {pct(buy_contribution)} & "
            f"{pct(base['sell_share_of_active'])} & {pct(sell_contribution)} \\\\"
        )
        values = []
        for spread in ("0.05", "0.10", "0.20"):
            item = economic[f"g{spread}_{contract}"]
            values.extend(
                [pct(item["mean_pnl_over_spot_unconditional"]), pct(item["trade_rate"])]
            )
        spread_rows.append(
            f"{label.replace(' call', '')} & " + " & ".join(values) + " \\\\"
        )
    write_rows("main_pnl_rows.tex", main_rows)
    write_rows("spread_rows.tex", spread_rows)

    hedge_rows = []
    for contract in ("vanilla_call", "asian_call"):
        item = hedged["contracts"][contract]
        hedge_rows.append(
            f"{contract_labels[contract]} & {pct(item['trade_rate'])} & "
            f"{pct(item['unhedged_pnl_over_spot'])} & {pct(item['hedged_pnl_over_spot'])} & "
            f"{pct(item['hedged_win_rate_active'])} \\\\"
        )
    write_rows("hedged_rows.tex", hedge_rows)

    accumulator = structured["contracts"]["accumulator"]
    snowball = structured["contracts"]["snowball"]
    structured_rows = [
        "Accumulator & "
        + " & ".join(
            [
                pct(accumulator["p"]["mean_payoff"]),
                pct(accumulator["q"]["mean_payoff"]),
                pct(accumulator["realized"]["mean_payoff"]),
                pp(accumulator["mean_p_q_gap"]),
                " / ".join(
                    pct(accumulator[side]["ko_rate"])
                    for side in ("p", "q", "realized")
                ),
                "--",
            ]
        )
        + " \\\\",
        "Snowball & "
        + " & ".join(
            [
                pct(snowball["p"]["mean_payoff"]),
                pct(snowball["q"]["mean_payoff"]),
                pct(snowball["realized"]["mean_payoff"]),
                pp(snowball["mean_p_q_gap"]),
                " / ".join(
                    pct(snowball[side]["ko_rate"])
                    for side in ("p", "q", "realized")
                ),
                " / ".join(
                    pct(snowball[side]["coupon_rate"])
                    for side in ("p", "q", "realized")
                ),
            ]
        )
        + " \\\\",
    ]
    write_rows("structured_rows.tex", structured_rows)
    macro_payloads = {
        "PathQualityRows": path_rows,
        "EffectiveTailRows": tail_rows,
        "MainPnlRows": main_rows,
        "HedgedRows": hedge_rows,
        "SpreadRows": spread_rows,
        "StructuredRows": structured_rows,
    }
    macro_text = []
    for name, rows in macro_payloads.items():
        macro_text.append(f"\\newcommand{{\\{name}}}{{%")
        macro_text.extend(rows)
        macro_text.append("}")
    (GENERATED / "results_macros.tex").write_text(
        "\n".join(macro_text) + "\n", encoding="utf-8"
    )

    groups = pd.DataFrame(calibration["groups"])
    panels = [
        ("Index", "By index"),
        ("Tenor", "By contractual tenor"),
        ("Volatility state", "By pre-window volatility state"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    colors = plt.cm.tab10.colors
    plotted = groups[groups.dimension.isin([item[0] for item in panels])]
    global_width_max = float(plotted.pointwise_width_90_over_s0.max())
    for axis, (dimension, title) in zip(axes, panels):
        part = groups[groups.dimension == dimension].reset_index(drop=True)
        for index, row in part.iterrows():
            axis.scatter(
                row.pointwise_width_90_over_s0,
                row.pointwise_coverage_90,
                s=50,
                color=colors[index % len(colors)],
                label=row.group,
            )
        axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Mean 90% fan width / $S_0$")
        axis.set_xlim(0.0, 1.08 * global_width_max)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Empirical pointwise coverage")
    fig.suptitle("Calibration--sharpness audit: narrower is sharper; 0.90 is calibrated")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_calibration_sharpness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "frozen_protocol": {
            "windows": 6824,
            "p_paths": 40,
            "q_paths": 256,
            "sampling_steps": 50,
            "primary_spread": 0.05,
            "materiality_threshold_over_spot": 0.01,
        },
        "sources": {
            key: {"file": filename, "sha256": sha256(DATA / filename)}
            for key, filename in sources.items()
        },
        "headline": {
            "terminal_crps": path_quality["models"]["history_aware_dlpm"]["terminal_crps"],
            "energy_score": path_quality["models"]["history_aware_dlpm"]["energy_score"],
            "coverage_90": path_quality["models"]["history_aware_dlpm"]["coverage_90"],
            "volatility_ratio": path_quality["models"]["history_aware_dlpm"]["volatility_ratio"],
            "vanilla_unconditional_pnl_over_spot": economic["g0.05_vanilla_call"]["mean_pnl_over_spot_unconditional"],
            "asian_unconditional_pnl_over_spot": economic["g0.05_asian_call"]["mean_pnl_over_spot_unconditional"],
            "lookback_unconditional_pnl_over_spot": economic["g0.05_lookback_call"]["mean_pnl_over_spot_unconditional"],
            "vanilla_delta_hedged_pnl_over_spot": hedged["contracts"]["vanilla_call"]["hedged_pnl_over_spot"],
            "asian_delta_hedged_pnl_over_spot": hedged["contracts"]["asian_call"]["hedged_pnl_over_spot"],
        },
    }
    (DATA / "authoritative_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

