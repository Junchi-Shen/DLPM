# -*- coding: utf-8 -*-
"""Regenerate the RN-Q side with unit-variance post-truncation innovations."""
import json
import importlib.util
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp
from Generator.path_simulators import (
    simulate_garch_risk_neutral_batch,
    truncated_standardized_t_variance,
)

ASSETS = ["CSI300", "CSI500", "CSI1000", "SSE_Composite",
          "SP500", "NASDAQ", "Dow_Jones", "Russell_2000"]
DEFAULT_TAG = "rnq_unitvar_cap8"


def solve_raw_cap_for_unit_variance(nu: float, final_cap: float = 8.0) -> float:
    """Choose raw truncation so restandardization leaves final cap at final_cap."""
    def ratio(raw_cap):
        v = truncated_standardized_t_variance(nu, raw_cap)
        return raw_cap / np.sqrt(max(v, 1e-12)) - final_cap
    hi = 8.0
    while ratio(hi) < 0.0:
        hi *= 1.5
    return float(brentq(ratio, 1e-3, hi))


def main():
    parser = argparse.ArgumentParser(
        description="Generate unit-variance, martingale-corrected Student-t GARCH RN-Q paths."
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Output directory under path archives.")
    parser.add_argument("--stamp", default="rnq_unitvar_cap8", help="Filename tag for generated arrays.")
    parser.add_argument("--n-sim", type=int, default=256, help="RN-Q paths per held-out window.")
    parser.add_argument("--window-chunk", type=int, default=96, help="Windows processed per resumable chunk.")
    parser.add_argument("--final-cap", type=float, default=8.0, help="Innovation cap after restandardization.")
    args = parser.parse_args()

    train = pd.read_csv(Path(pp.Trainning_DATA_DIR) / "trainning_data_merged.csv")
    test = pd.read_csv(Path(pp.Testing_DATA_DIR) / "testing_data_merged.csv")
    target_root = Path(pp.Path_Generator_Results_DIR) / args.tag
    target_root.mkdir(parents=True, exist_ok=True)
    fit_spec = importlib.util.spec_from_file_location(
        "garch_fit_deduplicated", Path(__file__).with_name("fit_garch_q_paths.py")
    )
    fit_module = importlib.util.module_from_spec(fit_spec)
    fit_spec.loader.exec_module(fit_module)
    summary = {"tag": args.tag, "q_model": "restandardized truncated Student-t GARCH RN-Q",
               "final_standardized_cap": args.final_cap, "post_truncation_restandardized": True,
               "garch_variance_units": "one-trading-day log-return variance",
               "dividend_yield_assumption": 0.0,
               "n_sim_q": args.n_sim, "assets": {}}
    for asset_idx, asset in enumerate(ASSETS):
        asset_df = test[test["asset_underlying"] == asset].reset_index(drop=True)
        dst = target_root / asset
        dst.mkdir(parents=True, exist_ok=True)
        params = fit_module.fit_garch_params(
            train, asset, dst / "garch_params_deduplicated.json"
        )
        raw_cap = solve_raw_cap_for_unit_variance(float(params["nu"]), args.final_cap)
        q_path = dst / f"garch_risk_neutral_paths_{args.stamp}_samples.npy"
        n_windows = len(asset_df)
        expected_shape = (n_windows * args.n_sim, 1, 253)
        if q_path.exists() and tuple(np.load(q_path, mmap_mode="r").shape) == expected_shape:
            print(f"{asset}: Q already present; reusing saved file", flush=True)
        else:
            if q_path.exists():
                q_path.unlink()
            out = np.lib.format.open_memmap(
                q_path, mode="w+", dtype=np.float32, shape=expected_shape
            )
            S0 = asset_df["start_price"].to_numpy(float)
            rates = asset_df["risk_free_rate"].to_numpy(float)
            vols = asset_df["volatility"].to_numpy(float)
            maturities = asset_df["actual_trading_days"].to_numpy(int)
            for start in range(0, n_windows, args.window_chunk):
                stop = min(start + args.window_chunk, n_windows)
                paths = simulate_garch_risk_neutral_batch(
                    S0=S0[start:stop], r=rates[start:stop],
                    initial_vol_ann=vols[start:stop],
                    T_days=maturities[start:stop], n_simulations=args.n_sim,
                    n_steps_total=253, garch_params=params,
                    innovation_cap=raw_cap, quadrature_order=64,
                    mgf_sigma_max=2.5, mgf_grid_size=513,
                    seed=4242 + asset_idx * 100000 + start,
                    restandardize_innovation=True,
                )
                out[start * args.n_sim:stop * args.n_sim] = paths.astype(np.float32)
                out.flush()
            del out
            print(f"{asset}: Q saved in chunks", flush=True)
        v = truncated_standardized_t_variance(float(params.get("nu", np.nan)), raw_cap)
        (dst / f"garch_risk_neutral_meta_{args.stamp}.json").write_text(
            json.dumps({"asset": asset, "q_model": summary["q_model"],
                        "n_windows": len(asset_df), "n_sim_q": args.n_sim,
                        "raw_innovation_cap": raw_cap,
                        "final_standardized_cap": args.final_cap,
                        "post_truncation_restandardized": True,
                        "pre_restandardization_variance": v,
                        "effective_innovation_variance": 1.0,
                        "garch_variance_units": "one-trading-day log-return variance",
                        "dividend_yield_assumption": 0.0,
                        "params": params}, indent=2), encoding="utf-8"
        )
        summary["assets"][asset] = {"n_windows": len(asset_df),
                                    "q_path": str(q_path),
                                    "pre_restandardization_variance": v,
                                    "raw_innovation_cap": raw_cap,
                                    "final_standardized_cap": args.final_cap}
        print(f"{asset}: Q saved; raw cap={raw_cap:.4f}, pre-restandardization variance={v:.6f}", flush=True)
    (target_root / f"rnq_restandardized_summary_{args.stamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()


