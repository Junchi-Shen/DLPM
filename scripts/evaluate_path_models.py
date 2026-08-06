# -*- coding: utf-8 -*-
"""Unified path-only evaluation for DLPM and Gaussian diffusion baselines.

No payoff, P-Q result, or 2024+ test row enters the score. The script records
the sampler resolution explicitly because a checkpoint can look different
under full ancestral sampling and accelerated sampling. Generation state is
saved atomically so a power interruption can resume at the next row batch.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp
from Model.Diffusion_Model.Unet_with_condition import Unet1D
from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork
from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_processor(run_dir):
    path = Path(run_dir) / "data_processor.pkl"
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def load_model(
    run_dir,
    checkpoint,
    device,
    sampling_steps,
    sampling_eta=None,
    clamp_a=None,
    isotropic=None,
    x0_clip=None,
):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    cond_params = dict(cfg["cond_net_params"])
    cond_net = EnhancedConditionNetwork(
        num_countries=max(cfg["data_info"]["num_countries"] + 5, 20),
        num_indices=max(cfg["data_info"]["num_indices"] + 10, 100),
        **cond_params,
    ).to(device)
    unet_params = dict(cfg["unet_params"])
    if isinstance(unet_params.get("dim_mults"), list):
        unet_params["dim_mults"] = tuple(unet_params["dim_mults"])
    model = Unet1D(cond_dim=cond_params.get("output_dim", 128), **unet_params).to(device)
    suffix = "" if checkpoint is None else f"-{int(checkpoint)}"
    model_path = run_dir / "checkpoints" / f"unet_conditional_model_ema{suffix}.pth"
    cond_path = run_dir / "checkpoints" / f"condition_network_ema{suffix}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    cond_net.load_state_dict(torch.load(cond_path, map_location=device, weights_only=True))
    model.eval()
    cond_net.eval()
    blocked = {"variant", "experiment_tag", "output_folder", "data_info", "unet_params", "cond_net_params"}
    if sampling_eta is not None:
        cfg["ddim_sampling_eta"] = float(sampling_eta)
    if clamp_a is not None:
        cfg["dlpm_clamp_a"] = float(clamp_a)
    if isotropic is not None:
        cfg["dlpm_isotropic"] = bool(isotropic)
    if x0_clip is not None:
        cfg["sample_x0_clip"] = float(x0_clip)
    variant = str(cfg.get("variant", "dlpm"))
    if variant.startswith("ddpm"):
        cfg["sampling_timesteps"] = int(sampling_steps)
        diffusion = GaussianDiffusion1D(
            model=model,
            condition_network=cond_net,
            **{k: v for k, v in cfg.items() if k not in blocked},
        ).to(device)
    else:
        diffusion = DLPMDiffusion1D(
            model=model,
            condition_network=cond_net,
            alpha=cfg["dlpm_alpha"],
            **{k: v for k, v in cfg.items() if k not in blocked},
        ).to(device)
    diffusion.eval()
    return diffusion, cfg


def condition_rows(processor, rows):
    transformed = processor.transform_price_sequence(processor.process_price_data(rows.copy()))
    info = processor.create_condition_tensors(transformed, fit_scaler=False)
    y = np.asarray(transformed["transformed_sequence"].tolist(), dtype=np.float32).reshape(len(rows), 1, -1)
    mask = np.asarray(transformed["validity_mask"].tolist(), dtype=np.float32).reshape(len(rows), 1, -1)
    return torch.from_numpy(info["conditions"]), torch.from_numpy(y), torch.from_numpy(mask)


def real_prices(row):
    value = getattr(row, "price_series") if hasattr(row, "price_series") else row["price_series"]
    return np.asarray(ast.literal_eval(str(value)), dtype=float)


def recover_paths(raw, rows, masks, scale):
    output = []
    for i, row in enumerate(rows.itertuples(index=False)):
        start = float(getattr(row, "start_price"))
        horizon = int(masks[i, 0].sum())
        tokens = raw[i, :, :]
        n_returns = max(min(horizon - 1, tokens.shape[-1] - 1), 0)
        daily = tokens[:, 1:1 + n_returns] * float(scale)
        output.append(start * np.exp(np.concatenate([
            np.zeros((daily.shape[0], 1)), np.cumsum(daily, axis=1)
        ], axis=1)))
    return output


def acf1(x):
    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.std(x) < 1e-12:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def ensemble_crps(samples, observation):
    """Empirical CRPS without constructing a full pairwise matrix."""
    values = np.sort(np.asarray(samples, dtype=float).reshape(-1))
    n = values.size
    if n == 0:
        return np.nan
    first = float(np.mean(np.abs(values - float(observation))))
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    pair_term = float(np.sum(weights * values) / (n * n))
    return first - pair_term


def ensemble_energy_score(paths, realized):
    """Energy score for a multivariate path ensemble."""
    samples = np.asarray(paths, dtype=float)
    target = np.asarray(realized, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return np.nan
    first = float(np.linalg.norm(samples - target[None, :], axis=1).mean())
    pairwise = samples[:, None, :] - samples[None, :, :]
    second = 0.5 * float(np.linalg.norm(pairwise, axis=2).mean())
    return first - second


def maximum_drawdown(path):
    path = np.asarray(path, dtype=float)
    running_peak = np.maximum.accumulate(path)
    return float(np.min(path / np.maximum(running_peak, 1e-12) - 1.0))


def score_paths(rows, generated, n_paths):
    records = []
    for row, paths in zip(rows.itertuples(index=False), generated):
        real = real_prices(row)
        horizon = min(len(real), paths.shape[1])
        real = real[:horizon]
        paths = paths[:, :horizon]
        q05, q15, q25, q50, q75, q85, q95 = np.quantile(
            paths, [0.05, 0.15, 0.25, 0.50, 0.75, 0.85, 0.95], axis=0
        )
        real_terminal = real[-1] / real[0] - 1.0
        gen_terminal = paths[:, -1] / paths[:, 0] - 1.0
        terminal_quantiles = np.quantile(
            gen_terminal, [0.05, 0.15, 0.25, 0.50, 0.75, 0.85, 0.95]
        )
        real_r = np.diff(np.log(np.maximum(real, 1e-12)))
        gen_r = np.diff(np.log(np.maximum(paths, 1e-12)), axis=1)
        gen_vol = np.std(gen_r, axis=1, ddof=1)
        real_quantiles = np.quantile(real_r, [0.05, 0.95])
        generated_quantiles = np.quantile(gen_r.reshape(-1), [0.05, 0.95])
        real_drawdown = maximum_drawdown(real)
        generated_drawdowns = np.asarray(
            [maximum_drawdown(path) for path in paths], dtype=float
        )
        normalized_paths = paths / max(float(real[0]), 1e-12)
        normalized_real = real / max(float(real[0]), 1e-12)
        records.append({
            "terminal_coverage_50": float(
                terminal_quantiles[2] <= real_terminal <= terminal_quantiles[4]
            ),
            "terminal_coverage_70": float(
                terminal_quantiles[1] <= real_terminal <= terminal_quantiles[5]
            ),
            "terminal_coverage_90": float(
                terminal_quantiles[0] <= real_terminal <= terminal_quantiles[6]
            ),
            "terminal_median_abs_error": float(abs(np.median(gen_terminal) - real_terminal)),
            "terminal_median_bias": float(np.median(gen_terminal) - real_terminal),
            "terminal_crps": float(ensemble_crps(gen_terminal, real_terminal)),
            "terminal_width_50": float(terminal_quantiles[4] - terminal_quantiles[2]),
            "terminal_width_70": float(terminal_quantiles[5] - terminal_quantiles[1]),
            "terminal_width_90": float(terminal_quantiles[6] - terminal_quantiles[0]),
            "path_coverage_50": float(np.mean((real >= q25) & (real <= q75))),
            "path_coverage_70": float(np.mean((real >= q15) & (real <= q85))),
            "path_coverage_90": float(np.mean((real >= q05) & (real <= q95))),
            "path_width_50": float(np.mean((q75 - q25) / max(float(real[0]), 1e-12))),
            "path_width_70": float(np.mean((q85 - q15) / max(float(real[0]), 1e-12))),
            "path_width_90": float(np.mean((q95 - q05) / max(float(real[0]), 1e-12))),
            "path_energy_score": float(
                ensemble_energy_score(normalized_paths, normalized_real)
            ),
            "generated_vol_ratio": float(np.mean(gen_vol) / max(np.std(real_r, ddof=1), 1e-12)),
            "volatility_abs_error": float(abs(np.mean(gen_vol) - np.std(real_r, ddof=1))),
            "abs_acf_error": float(abs(np.mean([acf1(np.abs(x)) for x in gen_r]) - acf1(np.abs(real_r)))),
            "terminal_tail_rate": float(np.mean(np.abs(gen_terminal) > 0.50)),
            "tail_quantile_error": float(np.mean(np.abs(generated_quantiles - real_quantiles))),
            "drawdown_error": float(abs(np.median(generated_drawdowns) - real_drawdown)),
        })
    return pd.DataFrame(records)


def pooled_terminal_metrics(rows, generated):
    realized = []
    simulated = []
    for row, paths in zip(rows.itertuples(index=False), generated):
        real = real_prices(row)
        horizon = min(len(real), paths.shape[1])
        realized.append(float(real[horizon - 1] / real[0] - 1.0))
        simulated.extend((paths[:, horizon - 1] / paths[:, 0] - 1.0).tolist())
    realized = np.asarray(realized, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    quantiles = np.linspace(0.01, 0.99, 99)
    real_q = np.quantile(realized, quantiles)
    simulated_q = np.quantile(simulated, quantiles)
    denominator = float(np.sum((real_q - real_q.mean()) ** 2))
    qq_r2 = 1.0 - float(np.sum((real_q - simulated_q) ** 2)) / max(denominator, 1e-12)
    ks = ks_2samp(realized, simulated, alternative="two-sided", method="auto")
    return {
        "terminal_ks_statistic": float(ks.statistic),
        "terminal_ks_pvalue": float(ks.pvalue),
        "terminal_wasserstein": float(wasserstein_distance(realized, simulated)),
        "terminal_qq_r2": float(qq_r2),
        "realized_terminal_mean": float(realized.mean()),
        "generated_terminal_mean": float(simulated.mean()),
        "terminal_drift_error": float(simulated.mean() - realized.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[4000, 8000, 12000, 16000])
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--row-selection", choices=["head", "evenly_spaced"],
                    default="head",
                    help="How to select a small per-asset validation subset.")
    ap.add_argument("--paths", type=int, default=32)
    ap.add_argument("--batch-rows", type=int, default=8,
                    help="Number of windows evaluated per generation batch.")
    ap.add_argument("--sampler", choices=["dlim", "native_skip", "ancestral"], default="dlim",
                    help="Sampler protocol to audit; ancestral requires sampling_steps=1000.")
    ap.add_argument("--eta", type=float, default=None,
                    help="Override stochastic DLIM eta; 0 is deterministic and >0 uses A_t in the reverse variance.")
    ap.add_argument("--clamp-a", type=float, default=None,
                    help="Override the positive-stable auxiliary-scale cap for sampler audits.")
    ap.add_argument("--isotropic", action="store_true",
                    help="Override the sampler to use one positive-stable scale per path.")
    ap.add_argument("--x0-clip", type=float, default=None,
                    help="Override the standardized x0/token clip for sampler audits.")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--countries", nargs="+", default=None,
                    help="Optional country filter applied before row sampling.")
    ap.add_argument("--output", default=None,
                    help="Optional result JSON path outside the checkpoint directory.")
    ap.add_argument("--state-file", default=None,
                    help="NPZ checkpoint for resumable path generation. Use one file for a single checkpoint.")
    ap.add_argument("--state-every", type=int, default=8,
                    help="Save a resumable state after this many generation batches.")
    ap.add_argument("--archive-raw", default=None,
                    help="Optional .npy archive of transformed model samples for downstream payoff diagnostics.")
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    run_dir = Path(args.run_dir)
    processor = load_processor(run_dir)
    data_dir = pp.Validation_DATA_DIR if args.split == "validation" else pp.Testing_DATA_DIR
    data_path = data_dir / ("clean_validation_history_context.csv" if args.split == "validation" else "clean_test_history_context.csv")
    all_rows = pd.read_csv(data_path, low_memory=False).sort_values(["asset_underlying", "start_date"])
    if args.countries:
        all_rows = all_rows[all_rows["country"].isin(args.countries)].copy()
        if all_rows.empty:
            raise ValueError(f"No {args.split} rows for countries={args.countries}")
    assets = all_rows["asset_underlying"].drop_duplicates().tolist()
    if args.rows >= len(all_rows):
        rows = all_rows.reset_index(drop=True)
    else:
        per_asset = max(args.rows // max(len(assets), 1), 1)
        if args.row_selection == "evenly_spaced":
            selected = []
            for _, group in all_rows.groupby("asset_underlying", sort=True):
                count = min(per_asset, len(group))
                positions = np.linspace(0, len(group) - 1, count, dtype=int)
                selected.append(group.iloc[positions])
            rows = pd.concat(selected, ignore_index=True)
        else:
            rows = (all_rows.groupby("asset_underlying", group_keys=False)
                    .head(per_asset)
                    .reset_index(drop=True))
    cond, _, mask = condition_rows(processor, rows)
    masks = mask.numpy()
    scale = float(processor.config.get("volatility_scale", 1.0))
    eta_tag = "default" if args.eta is None else f"eta{args.eta:g}"
    cap_tag = "default" if args.clamp_a is None else f"cap{args.clamp_a:g}"
    clip_tag = "default" if args.x0_clip is None else f"x0clip{args.x0_clip:g}"
    iso_tag = "iso" if args.isotropic else "coord"
    out = Path(args.output) if args.output else run_dir / f"{args.split}_checkpoint_scores_{args.sampler}_{args.sampling_steps}step_{eta_tag}_{cap_tag}_{clip_tag}_{iso_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for checkpoint in args.checkpoints:
        diffusion, cfg = load_model(
            run_dir,
            checkpoint,
            device,
            sampling_steps=args.sampling_steps,
            sampling_eta=args.eta,
            clamp_a=args.clamp_a,
            isotropic=True if args.isotropic else None,
            x0_clip=args.x0_clip,
        )
        raw_parts = []
        state_path = None
        completed_rows = 0
        if args.state_file:
            base_state = Path(args.state_file)
            if len(args.checkpoints) == 1:
                state_path = base_state
            else:
                state_path = base_state.with_name(f"{base_state.stem}_checkpoint{checkpoint}{base_state.suffix}")
            if state_path.exists():
                with np.load(state_path, allow_pickle=False) as state:
                    saved_checkpoint = int(state["checkpoint"])
                    completed_rows = int(state["completed_rows"])
                    saved_raw = np.asarray(state["raw"], dtype=np.float32) if completed_rows else None
                if saved_checkpoint != int(checkpoint):
                    raise ValueError(f"State checkpoint {saved_checkpoint} does not match requested {checkpoint}")
                if completed_rows < 0 or completed_rows > len(rows):
                    raise ValueError(f"Invalid completed_rows in state: {completed_rows}")
                if completed_rows:
                    raw_parts.append(saved_raw)
                print(f"Resuming checkpoint {checkpoint} from row {completed_rows}/{len(rows)} using {state_path}")
        with torch.no_grad():
            batch_rows = max(int(args.batch_rows), 1)
            batches_since_save = 0
            for start in range(completed_rows, len(rows), batch_rows):
                stop = min(start + batch_rows, len(rows))
                batch_seed = int(
                    (args.seed + int(checkpoint) * 1000003 + start) % (2**31 - 1)
                )
                np.random.seed(batch_seed)
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(batch_seed)
                cb = cond[start:stop].repeat_interleave(args.paths, dim=0).to(device)
                mb = mask[start:stop].repeat_interleave(args.paths, dim=0).to(device)
                if str(cfg.get("variant", "dlpm")).startswith("ddpm"):
                    raw_tensor = diffusion.sample(
                        batch_size=cb.shape[0], cond_input=cb, mask=mb
                    )
                else:
                    raw_tensor = diffusion.sample(
                        batch_size=cb.shape[0], cond_input=cb, mask=mb,
                        sampling_timesteps=args.sampling_steps,
                        sampler=args.sampler,
                    )
                raw = raw_tensor.detach().cpu().numpy()[:, 0, :]
                raw_parts.append(raw.reshape(stop - start, args.paths, -1))
                batches_since_save += 1
                if state_path is not None and (batches_since_save >= max(int(args.state_every), 1) or stop == len(rows)):
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_so_far = np.concatenate(raw_parts, axis=0)
                    tmp_path = state_path.with_name(state_path.name + ".tmp")
                    np.savez(tmp_path, raw=raw_so_far, completed_rows=np.int64(stop), checkpoint=np.int64(checkpoint))
                    actual_tmp = Path(str(tmp_path) + ".npz") if not tmp_path.name.endswith(".npz") else tmp_path
                    os.replace(actual_tmp, state_path)
                    batches_since_save = 0
        raw = np.concatenate(raw_parts, axis=0)
        generated = recover_paths(raw, rows, masks, scale)
        scores = score_paths(rows, generated, args.paths)
        summary = scores.mean(numeric_only=True).to_dict()
        summary.update(pooled_terminal_metrics(rows, generated))
        # Pooled means are useful for a headline number but can hide an
        # index-specific failure. Keep the same path metrics by asset so the
        # paper can report dispersion without re-running generation.
        per_asset = {}
        asset_col = "asset_underlying" if "asset_underlying" in rows.columns else None
        if asset_col is not None:
            scores = scores.copy()
            scores[asset_col] = rows[asset_col].astype(str).to_numpy()
            for asset, group in scores.groupby(asset_col, sort=True):
                per_asset[str(asset)] = {
                    key: float(value)
                    for key, value in group.mean(numeric_only=True).to_dict().items()
                }
                per_asset[str(asset)]["windows"] = int(len(group))
        # A path-only auxiliary score used only on validation.  It penalizes
        # interval miscoverage, volatility-scale misspecification, and
        # dependence error; payoff P&L never enters this score.
        coverage_error = np.mean([
            abs(float(summary["path_coverage_50"]) - 0.50),
            abs(float(summary["path_coverage_70"]) - 0.70),
            abs(float(summary["path_coverage_90"]) - 0.90),
            abs(float(summary["terminal_coverage_50"]) - 0.50),
            abs(float(summary["terminal_coverage_70"]) - 0.70),
            abs(float(summary["terminal_coverage_90"]) - 0.90),
        ])
        summary["coverage_calibration_error"] = float(coverage_error)
        summary["path_calibration_score"] = (
            float(coverage_error)
            + 0.50 * float(summary["terminal_crps"])
            + 0.25 * float(summary["path_energy_score"])
            + 0.50 * abs(float(np.log(max(summary["generated_vol_ratio"], 1e-8))))
            + 0.25 * abs(float(summary["abs_acf_error"]))
            + 0.10 * float(summary["terminal_median_abs_error"])
        )
        summary.update({"checkpoint": checkpoint, "sampling_steps": args.sampling_steps, "sampling_eta": cfg.get("ddim_sampling_eta", 0.0), "sampler": args.sampler, "clamp_a": cfg.get("dlpm_clamp_a"), "x0_clip": cfg.get("sample_x0_clip"), "requested_rows": args.rows, "rows": len(rows), "row_selection": args.row_selection, "paths_per_row": args.paths, "variant": cfg.get("variant"), "alpha": cfg.get("dlpm_alpha"), "isotropic": cfg.get("dlpm_isotropic"), "countries": args.countries or ["all"]})
        summary["per_asset"] = per_asset
        all_scores.append(summary)
        if args.archive_raw:
            archive_path = Path(args.archive_raw)
            if len(args.checkpoints) > 1:
                archive_path = archive_path.with_name(
                    f"{archive_path.stem}_checkpoint{checkpoint}{archive_path.suffix}"
                )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_archive = archive_path.with_name(archive_path.name + ".tmp")
            with tmp_archive.open("wb") as handle:
                np.save(handle, raw.astype(np.float32, copy=False), allow_pickle=False)
            os.replace(tmp_archive, archive_path)
        if state_path is not None and state_path.exists():
            state_path.unlink()
        del diffusion
        if device == "cuda":
            torch.cuda.empty_cache()
    out.write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
    print(json.dumps(all_scores, indent=2))
    print(f"Saved validation-only scores to {out}")


if __name__ == "__main__":
    main()

