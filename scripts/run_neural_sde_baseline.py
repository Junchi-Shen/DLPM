from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
import Project_Path as pp
from evaluate_path_models import (
    condition_rows,
    pooled_terminal_metrics,
    recover_paths,
    score_paths,
)


class ConditionalNeuralSDE(nn.Module):
    def __init__(self, condition_dim: int, hidden: int = 128, embedding: int = 128):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, embedding),
        )
        self.gru = nn.GRU(embedding + 2, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def nll(self, target, mask, condition):
        returns = target[:, 0, :]
        valid = mask[:, 0, :]
        previous = torch.cat(
            [torch.zeros_like(returns[:, :1]), returns[:, :-1]], dim=1
        )
        fraction = torch.linspace(
            0.0, 1.0, returns.shape[1], device=returns.device
        )[None, :, None].expand(returns.shape[0], -1, -1)
        embedding = self.condition(condition)[:, None, :].expand(
            -1, returns.shape[1], -1
        )
        output, _ = self.gru(
            torch.cat([previous[:, :, None], fraction, embedding], dim=2)
        )
        parameters = self.head(output)
        mean = parameters[:, :, 0]
        log_sigma = parameters[:, :, 1].clamp(-6.0, 2.0)
        nll = 0.5 * (
            ((returns - mean) / log_sigma.exp()) ** 2
            + 2.0 * log_sigma
            + np.log(2.0 * np.pi)
        )
        effective = valid.clone()
        effective[:, 0] = 0.0
        return (nll * effective).sum() / effective.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(self, condition, mask):
        batch, length = mask.shape[0], mask.shape[-1]
        embedding = self.condition(condition)
        hidden = None
        previous = torch.zeros(batch, device=condition.device)
        samples = torch.zeros(batch, 1, length, device=condition.device)
        for step in range(1, length):
            fraction = torch.full(
                (batch, 1), step / length, device=condition.device
            )
            recurrent_input = torch.cat(
                [previous[:, None], fraction, embedding], dim=1
            )[:, None, :]
            output, hidden = self.gru(recurrent_input, hidden)
            parameters = self.head(output[:, 0, :])
            mean = parameters[:, 0]
            sigma = parameters[:, 1].clamp(-6.0, 2.0).exp()
            value = (mean + sigma * torch.randn_like(mean)) * mask[:, 0, step]
            samples[:, 0, step] = value
            previous = value
        return samples


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_processor(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def summarize(rows, generated, paths):
    scores = score_paths(rows, generated, paths)
    summary = scores.mean(numeric_only=True).to_dict()
    summary.update(pooled_terminal_metrics(rows, generated))
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processor", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor = load_processor(Path(args.processor))
    train_rows = pd.read_csv(
        pp.Trainning_DATA_DIR / "clean_train_history_context.csv", low_memory=False
    )
    test_rows = (
        pd.read_csv(
            pp.Testing_DATA_DIR / "clean_test_history_context.csv", low_memory=False
        )
        .sort_values(["asset_underlying", "start_date"])
        .reset_index(drop=True)
    )
    train_condition, train_target, train_mask = condition_rows(processor, train_rows)
    test_condition, _, test_mask = condition_rows(processor, test_rows)
    condition_dim = int(train_condition.shape[1])
    model = ConditionalNeuralSDE(condition_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    start_step = 0
    checkpoint_files = sorted(
        checkpoints.glob("neural_sde-*.pt"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )
    if checkpoint_files:
        saved = torch.load(checkpoint_files[-1], map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_step = int(saved["step"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    atomic_json(
        manifest_path,
        {
            "status": "running",
            "seed": args.seed,
            "last_completed_step": start_step,
            "target_steps": args.steps,
            "device": str(device),
        },
    )
    model.train()
    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(0, len(train_rows), (args.batch_size,))
        condition = train_condition[indices].to(device)
        target = train_target[indices].to(device)
        mask = train_mask[indices].to(device)
        loss = model.nll(target, mask, condition)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 25 == 0:
            print(f"step={step} nll={float(loss):.6f}", flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            temporary = checkpoints / f"neural_sde-{step}.pt.tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                    "numpy_rng": np.random.get_state(),
                },
                temporary,
            )
            os.replace(temporary, checkpoints / f"neural_sde-{step}.pt")
            atomic_json(
                manifest_path,
                {
                    "status": "training_completed" if step == args.steps else "running",
                    "seed": args.seed,
                    "last_completed_step": step,
                    "target_steps": args.steps,
                    "elapsed_seconds": time.time() - started,
                    "last_loss": float(loss),
                },
            )

    generation_state = run_dir / "generation_state.npz"
    raw_parts = []
    completed_rows = 0
    if generation_state.exists():
        with np.load(generation_state, allow_pickle=False) as saved:
            completed_rows = int(saved["completed_rows"])
            if completed_rows:
                raw_parts.append(np.asarray(saved["raw"], dtype=np.float32))
    model.eval()
    batch_rows = 8
    with torch.no_grad():
        for start in range(completed_rows, len(test_rows), batch_rows):
            stop = min(start + batch_rows, len(test_rows))
            batch_seed = int((args.seed + 1000003 * start) % (2**31 - 1))
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            condition = test_condition[start:stop].repeat_interleave(
                args.paths, dim=0
            ).to(device)
            mask = test_mask[start:stop].repeat_interleave(args.paths, dim=0).to(device)
            raw = (
                model.sample(condition, mask)
                .cpu()
                .numpy()[:, 0, :]
                .reshape(stop - start, args.paths, -1)
            )
            raw_parts.append(raw.astype(np.float32))
            if (stop // batch_rows) % 4 == 0 or stop == len(test_rows):
                combined = np.concatenate(raw_parts, axis=0)
                temporary = generation_state.with_suffix(".npz.tmp")
                np.savez(
                    temporary,
                    completed_rows=np.int64(stop),
                    raw=combined,
                )
                actual = Path(str(temporary) + ".npz")
                if actual.exists():
                    temporary = actual
                os.replace(temporary, generation_state)
                raw_parts = [combined]
                print(f"generation={stop}/{len(test_rows)}", flush=True)

    raw = np.concatenate(raw_parts, axis=0)
    masks = test_mask.numpy()
    scale = float(processor.config.get("volatility_scale", 1.0))
    generated = recover_paths(raw, test_rows, masks, scale)
    summary = summarize(test_rows, generated, args.paths)
    summary.update(
        {
            "baseline": "conditional_neural_sde",
            "seed": args.seed,
            "training_steps": args.steps,
            "rows": int(len(test_rows)),
            "paths_per_row": args.paths,
        }
    )
    archive = run_dir / "raw_test_paths.npy"
    temporary_archive = archive.with_suffix(".npy.tmp")
    with temporary_archive.open("wb") as handle:
        np.save(handle, raw, allow_pickle=False)
    os.replace(temporary_archive, archive)
    atomic_json(run_dir / "test_path_metrics.json", summary)
    generation_state.unlink(missing_ok=True)
    atomic_json(
        manifest_path,
        {
            "status": "completed",
            "seed": args.seed,
            "last_completed_step": args.steps,
            "target_steps": args.steps,
            "test_metrics": str(run_dir / "test_path_metrics.json"),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

