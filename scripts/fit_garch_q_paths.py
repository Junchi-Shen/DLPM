import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src" / "dlpm"))

import Project_Path as pp  # noqa: E402
from Generator.path_simulators import simulate_garch  # noqa: E402


DEFAULT_ASSETS = [
    "CSI300",
    "CSI500",
    "CSI1000",
    "SSE_Composite",
    "SP500",
    "NASDAQ",
    "Dow_Jones",
    "Russell_2000",
]


def fit_garch_params(train_df: pd.DataFrame, asset: str, out_path: Path) -> dict:
    asset_df = train_df[train_df["asset_underlying"] == asset].copy()
    if asset_df.empty:
        raise ValueError(f"No training rows for {asset}")

    # GARCH fitting requires chronological observations. Do not rely on the
    # incidental row order of a CSV export.
    asset_df["start_date"] = pd.to_datetime(asset_df["start_date"])
    asset_df = (
        asset_df.sort_values("start_date")
        .drop_duplicates(subset=["start_date"], keep="first")
        .reset_index(drop=True)
    )

    # Use the same convention as Model/Garch_Model/Garch_fitter.py:
    # overlapping windows are ordered by start date, and path[0] is the daily start price.
    daily_prices = asset_df["price_series"].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
    returns_pct = np.log(daily_prices / daily_prices.shift(1)).dropna() * 100.0
    if returns_pct.empty:
        raise ValueError(f"Empty return series for {asset}")

    model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="t")
    fit = model.fit(disp="off")
    params = {
        "omega": float(fit.params["omega"]) / 10000.0,
        "alpha": float(fit.params["alpha[1]"]),
        "beta": float(fit.params["beta[1]"]),
        "nu": float(fit.params["nu"]) if "nu" in fit.params else np.nan,
        "n_train_returns": int(len(returns_pct)),
        "loglikelihood": float(fit.loglikelihood),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, allow_nan=True)
    return params


def generate_asset_paths(test_df: pd.DataFrame, asset: str, params: dict, out_dir: Path,
                         timestamp: str, n_sim: int, n_steps_total: int, seed: int) -> Path:
    # Preserve the testing CSV order: P paths and realized windows use this
    # exact order, so re-sorting Q paths here would silently misalign P/Q.
    asset_df = test_df[test_df["asset_underlying"] == asset].copy()
    if asset_df.empty:
        raise ValueError(f"No testing rows for {asset}")
    asset_df = asset_df.reset_index(drop=True)

    chunks = []
    for i, row in asset_df.iterrows():
        paths, _ = simulate_garch(
            S0=float(row["start_price"]),
            r=float(row["risk_free_rate"]),
            initial_vol_ann=float(row["volatility"]),
            T_days=int(row["actual_trading_days"]),
            n_simulations=n_sim,
            n_steps_total=n_steps_total,
            garch_params=params,
            innov_dist="t",
            seed=seed + i,
        )
        chunks.append(paths.astype(np.float32))

    out = np.concatenate(chunks, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"garch_paths_fitted_{timestamp}_samples.npy"
    np.save(out_path, out)

    meta = {
        "timestamp": timestamp,
        "asset": asset,
        "q_model": "Student-t GARCH(1,1)",
        "n_windows": int(len(asset_df)),
        "n_sim_q": int(n_sim),
        "n_steps_total": int(n_steps_total),
        "params": params,
    }
    with (out_dir / f"garch_paths_meta_{timestamp}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, allow_nan=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_tag", default="global_all_v2")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--n_sim_q", type=int, default=256)
    parser.add_argument("--n_steps_total", type=int, default=253)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    train_path = Path(pp.Trainning_DATA_DIR) / "trainning_data_merged.csv"
    test_path = Path(pp.Testing_DATA_DIR) / "testing_data_merged.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = []

    for asset_idx, asset in enumerate(args.assets):
        print(f"\n=== {asset}: fitting Student-t GARCH(1,1) and generating Q paths ===", flush=True)
        param_path = Path(pp.Model_Results_DIR) / "Garch_Fit_Results" / asset / "garch_params.json"
        params = fit_garch_params(train_df, asset, param_path)
        print(
            f"params: omega={params['omega']:.3e}, alpha={params['alpha']:.4f}, "
            f"beta={params['beta']:.4f}, alpha+beta={params['alpha'] + params['beta']:.4f}, "
            f"nu={params['nu']:.2f}",
            flush=True,
        )
        out_dir = Path(pp.Path_Generator_Results_DIR) / args.experiment_tag / asset
        out_path = generate_asset_paths(
            test_df=test_df,
            asset=asset,
            params=params,
            out_dir=out_dir,
            timestamp=timestamp,
            n_sim=args.n_sim_q,
            n_steps_total=args.n_steps_total,
            seed=args.seed + asset_idx * 100000,
        )
        print(f"saved: {out_path}", flush=True)
        summary.append({
            "asset": asset,
            "path": str(out_path),
            "alpha_plus_beta": params["alpha"] + params["beta"],
            "nu": params["nu"],
        })

    summary_path = Path(pp.Path_Generator_Results_DIR) / args.experiment_tag / f"garch_q_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()

