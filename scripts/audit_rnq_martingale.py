# -*- coding: utf-8 -*-
"""Audit whether the current Student-t GARCH benchmark is discounted-martingale.

This is deliberately a diagnostic, not a risk-neutralization procedure.  The
current simulator uses risk-free drift plus a Gaussian-style log correction;
with Student-t innovations that correction is not theoretically sufficient.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp
from Generator.path_simulators import simulate_garch, simulate_garch_risk_neutral
def load_fitter():
    import importlib.util
    path = Path(__file__).with_name("fit_garch_q_paths.py")
    spec = importlib.util.spec_from_file_location("garch_fit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_garch_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=50)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--risk-neutral", action="store_true")
    parser.add_argument("--innovation-cap", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=Path(
        "Results/Path_Generator_Results/audited_v2_final/garch_discounted_martingale_audit.json"
    ))
    args = parser.parse_args()

    train_df = pd.read_csv(Path(pp.Trainning_DATA_DIR) / "trainning_data_merged.csv")
    test_df = pd.read_csv(Path(pp.Testing_DATA_DIR) / "testing_data_merged.csv")
    fit_garch_params = load_fitter()
    results = []

    for asset_idx, asset in enumerate(sorted(test_df["asset_underlying"].unique())):
        test_asset = test_df[test_df["asset_underlying"] == asset].head(args.max_windows)
        params_path = Path(pp.Model_Results_DIR) / "Garch_Fit_Results" / asset / "garch_params.json"
        params = fit_garch_params(train_df, asset, params_path)
        ratios = []
        for i, row in test_asset.reset_index(drop=True).iterrows():
            simulator = simulate_garch_risk_neutral if args.risk_neutral else simulate_garch
            sim_kwargs = dict(
                S0=float(row["start_price"]),
                r=float(row["risk_free_rate"]),
                dividend_yield=float(row.get("dividend_yield", 0.0)),
                initial_vol_ann=float(row["volatility"]),
                T_days=int(row["actual_trading_days"]),
                n_simulations=args.n_sim,
                n_steps_total=int(row["actual_trading_days"]) + 1,
                garch_params=params,
                innov_dist="t",
                seed=args.seed + asset_idx * 100000 + i,
            )
            if args.risk_neutral:
                sim_kwargs.pop("innov_dist")
                sim_kwargs["innovation_cap"] = args.innovation_cap
            paths, _ = simulator(**sim_kwargs)
            t = int(row["actual_trading_days"])
            carry_daily = (
                float(row["risk_free_rate"])
                - float(row.get("dividend_yield", 0.0))
            ) / 252.0
            discounted = (
                paths[:, 0, t]
                * np.exp(-carry_daily * t)
                / float(row["start_price"])
            )
            ratios.append(float(np.mean(discounted)))
        results.append({
            "asset": asset,
            "windows": len(ratios),
            "mean_discounted_terminal_ratio": float(np.mean(ratios)),
            "median_discounted_terminal_ratio": float(np.median(ratios)),
            "mean_martingale_error": float(np.mean(np.asarray(ratios) - 1.0)),
            "alpha_plus_beta": float(params["alpha"] + params["beta"]),
            "student_t_nu": float(params["nu"]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "interpretation": (
            "Discrete carry-adjusted martingale Q audit; dividend_yield defaults to zero."
            if args.risk_neutral else
            "Historical benchmark diagnostic; no risk-neutral transform is applied."
        ),
        "assets": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved martingale audit to {args.output}")


if __name__ == "__main__":
    main()


