# Conditional Deep Levy Models for Exotic Derivatives

This directory is the clean 40-path paper-release package for the aligned final audit.
It intentionally contains only the manuscript, the five figures used by the
manuscript, compact numerical summaries, and the figure-building script.
Exploratory notebooks, legacy Q archives, and superseded product experiments
are excluded.

## Contents

- `main.tex`: manuscript source.
- `main.pdf`: checked local build for review.
- `figures/final/`: every figure referenced by `main.tex`.
- `data/`: compact JSON summaries underlying the aligned economic tables,
  bootstrap intervals, drift-neutral counterfactual, initial-delta hedge, the
  aligned Accumulator/Snowball scenario audit, and the regime-conditional
  DLPM versus bootstrap comparison.
- `scripts/build_aligned_economic_figures.py`: regenerates Figure 4 from the
  JSON summaries.
- `scripts/run_structured_product_audit.py`: the resumable unique-key-aligned
  Accumulator/Snowball audit used for the structured-product table.
 - The regime-conditional audit script reproduces the within-index,
  pre-window terminal-CRPS comparison.

## Reproducibility scope

The release reports a frozen history-aware DLPM evaluated on the chronological
held-out panel. The reported P archive contains 40 protocol-matched paths per
window and RN-Q contains 256 paths per window. P and RN-Q archives are joined by asset, start date, spot,
tenor, and rate before payoff analysis; the paper does not use positional row
alignment. All reported P&L is unconditional across 6,824 windows and
normalized by initial spot.

The paper distinguishes a raw P--Q payoff diagnostic from two counterfactuals:
terminal-drift neutralization and a one-time initial RN-Q delta hedge. These
are diagnostic controls, not claims of trading alpha or a market-implied option
pricing surface.

## Build

Compile `main.tex` twice with a standard LaTeX installation:

```text
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses only standard packages: `amsmath`, `booktabs`,
`graphicx`, `natbib`, and `hyperref`.
