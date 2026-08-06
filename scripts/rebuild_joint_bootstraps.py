"""Rebuild joint calendar-block bootstrap files from completed trade logs.

The economic logs are the authoritative records.  This utility deliberately
does not regenerate paths or trades; it only restores the dependence-robust
interval artifacts for every completed model in a uniform format.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))
from run_economic_attribution import atomic_json, bootstrap


def main() -> None:
    config = json.loads((ROOT / "suite_config.json").read_text(encoding="utf-8"))
    repetitions = int(config["evaluation"]["joint_calendar_bootstrap_repetitions"])
    block_lengths = [int(value) for value in config["evaluation"]["joint_calendar_block_lengths"]]
    economic_root = ROOT / "Results" / "Economic_Attribution"
    completed = 0
    for log_path in sorted(economic_root.glob("*/trade_log.csv")):
        frame = pd.read_csv(log_path, parse_dates=["date"])
        origin = frame["date"].min()
        diagnostics = {}
        results = {}
        for block in block_lengths:
            observed = int(((frame["date"] - origin).dt.days // block).nunique())
            diagnostics[str(block)] = {
                "calendar_block_days": block,
                "observed_nonempty_blocks": observed,
                "interpretation": (
                    "Long-dependence sensitivity check with limited effective blocks."
                    if block >= 180
                    else (
                        "Primary dependence-robust inference block length."
                        if block <= 60
                        else "Medium-horizon sensitivity check with limited effective blocks."
                    )
                ),
            }
            results[str(block)] = bootstrap(frame, block, repetitions, 20260730)
        payload = {
            "protocol": "joint calendar-block bootstrap; all assets share sampled blocks; P&L normalized by initial spot",
            "windows_are_independent": False,
            "repetitions": repetitions,
            "block_diagnostics": diagnostics,
            "results": results,
        }
        atomic_json(log_path.parent / "joint_calendar_bootstrap.json", payload)
        completed += 1
        print(f"bootstrap rebuilt: {log_path.parent.name}", flush=True)
    print(f"rebuilt {completed} bootstrap files", flush=True)


if __name__ == "__main__":
    main()

