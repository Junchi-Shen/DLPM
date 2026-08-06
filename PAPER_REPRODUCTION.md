# Paper reproduction map

The rendered paper is included in `paper/main.pdf`; its source is
`paper/main.tex`.  The five manuscript figures and the tables have the
following construction paths.

| Paper item | Source data or archive | Reproduction code |
| --- | --- | --- |
| Figure 1, workflow | No data dependency | `scripts/build_paper_path_figures.py` |
| Figure 2, path quality | Held-out path-quality summary | `scripts/build_paper_path_figures.py` |
| Figure 3, P-only fans | Held-out test table plus P-path archive | `scripts/build_paper_path_figures.py` |
| Figure 4, aligned economic audit | Compact JSON data tracked in `paper/data/` | Frozen release figure; its numerical sources are hashed in the authoritative manifest |
| Figure 5, calibration--sharpness | `paper/data/calibration_sharpness_40paths.json` | `scripts/build_release_artifacts.py` |
| Effective-tail table | Frozen DLPM, Gaussian, bootstrap, and GBM archives plus the chronological test table | `scripts/audit_effective_tail_behavior.py` |
| Path-quality and economic tables | Hashed compact sources under `paper/data/` | `scripts/build_release_artifacts.py` |
| Baseline tables | Common-split baseline outputs | `scripts/evaluate_statistical_baselines.py` and external-baseline scripts |
| P-Q, drift, hedge and MC diagnostics | Aligned 40-path P and 256-path RN-Q archives | the named `run_*_audit.py` scripts under `scripts/` |
| Quote MC-SE and threshold sensitivity | Aligned payoff samples | `scripts/audit_quote_uncertainty.py` |
| Paired CRPS and state-slice inference | Held-out DLPM and empirical archives | `scripts/audit_paired_crps.py` |
| State-matched conditional control | Frozen train/test split | `scripts/audit_state_matched_baseline.py` |
| Structured-product tables | Aligned P/Q archives and held-out data | `scripts/run_structured_product_audit.py` |

The first three figure builders read the local large artifacts specified in
`artifacts/README.md`; all relevant input locations can be overridden with
the `DLPM_PAPER_*` environment variables at the top of the script.  Figure 4
is distributed as the frozen rendered audit panel. Figure 5 and every
headline table are rebuilt from compact data committed with the paper.

Large generated arrays, checkpoints, and licensed raw market data are
excluded from Git. Their exact hashes and schemas are published; the code
that creates or consumes them is included. Full numerical regeneration
therefore requires obtaining the matching release assets and market data,
whereas the paper tables, Figure 5, and LaTeX document rebuild directly from
the tracked compact sources.
