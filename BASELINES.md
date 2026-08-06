# Baseline code map

Every baseline reported in the manuscript has a corresponding code path in
this repository.  All path-quality baselines use the same chronological split
and metric functions in `scripts/evaluate_path_models.py`.

| Baseline | Implementation | How it is run |
| --- | --- | --- |
| Gaussian DDPM, simple loss | `GaussianDiffusion1D` | `scripts/run_clean_split_training.py --variant ddpm_simple` |
| Gaussian DDPM, finance-aware loss | `GaussianDiffusion1D` plus the configured regularizers | `scripts/run_clean_split_training.py --variant ddpm_complex` |
| Conditional DLPM | `DLPMDiffusion1D` | `scripts/run_clean_split_training.py --variant dlpm` or `dlpm_hybrid` |
| Unconditional historical block bootstrap | `scripts/evaluate_statistical_baselines.py` | statistical baseline evaluation |
| Historical-drift GBM | `scripts/evaluate_statistical_baselines.py` | statistical baseline evaluation |
| Physical Student-t GARCH(1,1) | `scripts/evaluate_statistical_baselines.py` | implemented statistical control; not part of the reported manuscript table |
| Momentum/trend rule | `scripts/evaluate_statistical_baselines.py` | implemented statistical control; not part of the reported manuscript table |
| Conditional Neural SDE | `scripts/run_neural_sde_baseline.py` | standalone external-baseline run |
| Chronos zero-shot | `scripts/run_chronos_baseline.py` | optional dependency and upstream model required |
| Student-t GARCH RN-Q benchmark | `src/dlpm/Generator/path_simulators.py` and `scripts/generate_rnq_paths.py` | generate then audit with `scripts/audit_rnq_martingale.py` |

The RN-Q process is a benchmark used only for the P-Q payoff diagnostic; it is
not a path-quality baseline for the P-side generator.  The audit scripts use
the composite-key join documented in `artifacts/README.md`.
