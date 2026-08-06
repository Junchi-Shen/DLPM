# Release manifest

## Scope

This repository is the curated implementation accompanying the final
conditional DLPM manuscript.  It contains only the code paths and compact
reported outputs used by the final analysis.

## Included workflow

1. Build chronological train/validation/test tables with target-end purging
   and embargo using `scripts/build_chronological_splits.py`, then construct
   historical condition vectors with `src/dlpm/Data/`.
2. Train the conditional DLPM or its reported Gaussian ablations with
   `scripts/run_clean_split_training.py`.
3. Generate and score conditional paths with `scripts/evaluate_path_models.py`
   and `scripts/plot_model_fan_comparisons.py`; run the reported Neural SDE
   and optional Chronos external baselines with their corresponding scripts.
4. Fit and simulate the restandardized, martingale-corrected Student-t GARCH
   RN-Q benchmark using `scripts/fit_garch_q_paths.py`,
   `scripts/generate_rnq_paths.py`, and `scripts/audit_rnq_martingale.py`.
5. Run the aligned P-Q, drift, hedge, Monte Carlo, regime, and structured-note
   diagnostics under `scripts/`.
6. Generate the authoritative compact manifest, table macros, and
   calibration--sharpness figure with
   `scripts/build_release_artifacts.py`, then compile `paper/main.tex`.

The paper-figure mapping and construction scripts are documented in
`PAPER_REPRODUCTION.md`.

## Deliberately excluded

- Raw and processed market data.
- Checkpoints, fitted preprocessors, RN-Q archives, and generated path arrays.
- Logs, caches, interrupted-run state, and all superseded exploratory branches.
- Unreported product prototypes and plotting drafts.

The exclusions are intentional: they prevent a public code release from
silently publishing large generated data artifacts or presenting exploratory
results as part of the final study.  Local artifact locations and expected
directory names are documented in `data/README.md` and `artifacts/README.md`.
