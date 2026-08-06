"""Build chronological train/validation/test windows with purge and embargo.

The input is a processed window table containing ``start_date`` and
``end_date``. A row is assigned to an earlier split only when its realized
target ends at least ``embargo_calendar_days`` before the next split begins.
This is the exact rule described in the manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_windows(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    embargo_calendar_days: int,
) -> dict[str, pd.DataFrame]:
    required = {"start_date", "end_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = frame.copy()
    data["start_date"] = pd.to_datetime(data["start_date"], errors="raise")
    data["end_date"] = pd.to_datetime(data["end_date"], errors="raise")
    if (data.end_date < data.start_date).any():
        raise ValueError("Found a target end before its window start")

    buffer = pd.Timedelta(days=embargo_calendar_days)
    train = data[data.end_date <= validation_start - buffer]
    validation = data[
        (data.start_date >= validation_start)
        & (data.end_date <= test_start - buffer)
    ]
    test = data[data.start_date >= test_start]

    for name, part in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if part.empty:
            raise ValueError(f"The {name} split is empty")

    if train.end_date.max() > validation.start_date.min() - buffer:
        raise AssertionError("Train/validation purge or embargo failed")
    if validation.end_date.max() > test.start_date.min() - buffer:
        raise AssertionError("Validation/test purge or embargo failed")
    return {"train": train, "validation": validation, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-start", default="2023-01-03")
    parser.add_argument("--test-start", default="2024-01-02")
    parser.add_argument("--embargo-calendar-days", type=int, default=20)
    args = parser.parse_args()

    source = pd.read_csv(args.input, low_memory=False)
    splits = split_windows(
        source,
        pd.Timestamp(args.validation_start),
        pd.Timestamp(args.test_start),
        args.embargo_calendar_days,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha256": sha256(args.input),
        "validation_start": args.validation_start,
        "test_start": args.test_start,
        "embargo_calendar_days": args.embargo_calendar_days,
        "purge_rule": "end_date <= next_split_start - embargo",
        "splits": {},
    }
    for name, part in splits.items():
        destination = args.output_dir / f"{name}.csv"
        part.to_csv(destination, index=False, date_format="%Y-%m-%d")
        manifest["splits"][name] = {
            "rows": int(len(part)),
            "earliest_start": str(part.start_date.min().date()),
            "latest_start": str(part.start_date.max().date()),
            "latest_target_end": str(part.end_date.max().date()),
            "sha256": sha256(destination),
        }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()


