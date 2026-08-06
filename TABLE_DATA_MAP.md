# Paper table and result provenance

This map corresponds to the current 40-path manuscript only. Superseded path
counts and exploratory outputs are intentionally excluded.

| Paper result | Compact source | Reproduction entry point |
| --- | --- | --- |
| Chronological sample construction | processed split manifests | `scripts/build_chronological_splits.py` |
| Held-out path quality and calibration | `paper/data/path_quality_frozen_ema12000.json` and calibration summaries | `scripts/build_release_artifacts.py` |
| Effective realized tail behaviour | `paper/data/effective_tail_behavior.json` | `scripts/audit_effective_tail_behavior.py` |
| Gaussian, GARCH, GBM, momentum, neural-SDE, Chronos baselines | common chronological split | baseline scripts listed in `BASELINES.md` |
| Smooth-option definitions and P--Q audit | aligned 40-path P and 256-path RN-Q archives | `scripts/run_economic_attribution.py` |
| Monte Carlo quote-error and threshold audit | `paper/data/quote_mc_threshold_sensitivity.csv` | `scripts/audit_quote_uncertainty.py` |
| Accumulator and Snowball scenario audit | `paper/data/structured_product_40path_summary.json` | `scripts/run_structured_product_audit.py` |
| Joint calendar-block P&L intervals | aligned per-window trade log | `scripts/run_economic_attribution.py` |
| Unconditional-bootstrap paired CRPS inference | `paper/data/paired_crps_calendar_block.json` and `paired_crps_group_inference.csv` | `scripts/audit_paired_crps.py` |
| State-matched empirical baseline | `paper/data/state_matched_summary.json` | `scripts/audit_state_matched_baseline.py` |
| Final hyperparameters and archive hashes | `FINAL_RUN_MANIFEST.md` | frozen JSON configuration and release assets |

Raw market data, checkpoints, and full generated arrays are not Git blobs.
Their provider rules, expected locations, and SHA-256 identities are documented
in `data/README.md`, `artifacts/README.md`, and `FINAL_RUN_MANIFEST.md`.
