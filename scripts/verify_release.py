"""Verify the compact release without requiring proprietary large artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DATA = PAPER / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(
        (DATA / "authoritative_results_manifest.json").read_text(encoding="utf-8")
    )
    protocol = manifest["frozen_protocol"]
    assert protocol == {
        "windows": 6824,
        "p_paths": 40,
        "q_paths": 256,
        "sampling_steps": 50,
        "primary_spread": 0.05,
        "materiality_threshold_over_spot": 0.01,
    }
    for source in manifest["sources"].values():
        path = DATA / source["file"]
        assert path.is_file(), path
        assert sha256(path) == source["sha256"], path

    archive_manifest = json.loads(
        (ROOT / "artifacts" / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8-sig")
    )
    assert archive_manifest["protocol"]["p_paths_per_window"] == 40
    assert archive_manifest["protocol"]["q_paths_per_window"] == 256
    assert len(archive_manifest["rnq_archive"]["files"]) == 24

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    for relative in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        assert (PAPER / relative).is_file(), relative
    for relative in re.findall(r"\\input\{([^}]+)\}", tex):
        candidate = PAPER / relative
        if candidate.suffix != ".tex":
            candidate = candidate.with_suffix(".tex")
        assert candidate.is_file(), relative
    for macro in (
        "\\PathQualityRows",
        "\\MainPnlRows",
        "\\HedgedRows",
        "\\SpreadRows",
        "\\StructuredRows",
    ):
        assert macro in tex

    forbidden = ("C:\\Users\\", "Documents\\Codex", "p_paths\": 32")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".png", ".zip"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            assert token not in text, f"{token!r} leaked in {path}"
    assert not list(ROOT.rglob("*.pyc"))
    print("Release verification passed")


if __name__ == "__main__":
    main()
