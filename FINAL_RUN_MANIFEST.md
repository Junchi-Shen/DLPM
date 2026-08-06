# Frozen final run and paper-data manifest

## Frozen model

The reported final conditional DLPM is the **EMA-12000** checkpoint selected
on the validation period from the
`formal_universal_v1_full_seed20260730_16000` training run.  It is not the
terminal EMA-16000 snapshot.

| Field | Frozen value |
| --- | --- |
| Training seed | 20260730 |
| Training steps | 16,000 |
| Sequence length | 252 trading days |
| Diffusion steps | 1,000 |
| DLPM alpha | 1.9 |
| DLPM loss exponent | 1.0 |
| Learning rate | 1e-5 |
| Batch size | 32 |
| EMA decay | 0.995 |
| U-Net widths | 64, 128, 256, 512 |
| U-Net dropout | 0.1 |
| Raw / fused condition dimension | 397 numerical entries plus two categorical IDs (399 raw slots); country/index embeddings are fused to 128 dimensions |
| History inputs | 60-day returns, 252-day returns, 60-day volatility path and engineered state features |
| Financial-loss scale / warm-up | 0.07 / 1,000 steps |
| Financial-loss component weights | Explicitly frozen in `formal_universal_regime_v1.json` |
| Stable-noise mixing | Coordinate-wise (`dlpm_isotropic=false`), not path-isotropic |
| Regime weights | asset-normalized, tempered inverse-frequency power 0.4, cap 1.5 |

The exact resolved parameter file is
`src/dlpm/Config/formal_universal_regime_v1.json`.  It is the Git-tracked
source of record for the reported run rather than a mutable default config.

## Checkpoint release asset

The code repository intentionally excludes the large binary artifact.  Store
the following three files together in a GitHub Release, Zenodo record, or
private archival store. The companion inference archive prepared for this
release is `DLPM_Final_EMA12000_Checkpoint_20260802.zip`. The exact
optimizer-state archive is `DLPM_Final_EMA12000_Resume_State_20260802.zip`.

| Artifact | SHA-256 | Purpose |
| --- | --- | --- |
| `unet_conditional_model_ema-12000.pth` | `0D4DFBB8F836FCEA3E6B65683600D4BCFD3947AE4C699C0302B1825327C070F8` | Validation-selected EMA U-Net weights |
| `condition_network_ema-12000.pth` | `C6938EB3B9D271CEF8A7075B389810629F6242F0DFE05A50AD5705B807A107A3` | Validation-selected EMA condition-network weights |
| `data_processor.pkl` | `8896B8DFA13FE3CFFD3EC725BD9EBE04BF6120B1C0AD5D22148A2335E7654AC5` | Fitted feature transforms and category mappings |

The checkpoint is not usable without the matching `data_processor.pkl`.
For exact optimizer-state resumption, use `model-12000.pt` from
`DLPM_Final_EMA12000_Resume_State_20260802.zip`. It includes optimizer,
scheduler, RNG, and training-state information and is not required for frozen
generation or paper evaluation.

## Tables and compact data

The paper does not depend on opaque spreadsheets.  The compact JSON inputs
for reported economic and robustness tables are tracked under `paper/data/`:

| File | Used for |
| --- | --- |
| `economic_spot1_delta_hedge_summary.json` | Main spot-normalized P-Q, Delta-hedge, and joint bootstrap summary |
| `execution_threshold_spot_scale_sensitivity.json` | Spot-normalized materiality and dealer-spread sensitivity |
| `execution_threshold_spot_scale_drift_neutral.json` | Terminal-drift-neutral diagnostic under the same spot-normalized threshold |
| `regime_conditional_summary.json` | Conditional regime comparison |
| `structured_product_40path_summary.json` | Accumulator and Snowball stress-test summary |
| `path_quality_frozen_ema12000.json` | Held-out path-quality and baseline comparison table |
| `effective_tail_behavior.json` | Forecast-origin effective-tail audit under the deployed numerical caps |
| `quote_mc_threshold_sensitivity.csv` | Per-contract Monte Carlo-SE and materiality-threshold sensitivity |
| `paired_crps_group_inference.csv` | Joint calendar-block inference for unconditional resampling comparisons |
| `state_matched_summary.json` | Same-index, same-tenor state-matched empirical baseline and block intervals |

## Frozen evaluation archives

Large path arrays are release assets rather than Git blobs. Their exact
identities are fixed by SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Chronological held-out test CSV | `68B964021FC1D9F899364E41EAA871A072F19E1AA8A670FAA68DC3D64685B8D8` |
| RN-Q source-order test CSV | `C93B4CB9D4BC8F72792A51D7481A302189C37258918913248721D93ABD1441B4` |
| Frozen 40-path P archive | `7F44D27E0711F8F4AC39E113A150A426EDF2312D1E6639874F8BF0FD7A44A974` |
| Unconditional block-bootstrap archive | `80C3EE618C49725386A430CADBDA4A1FD366259F2AC5E4291F22CB0EB9FC37B4` |

The economic audit uses 40 P paths and 256 RN-Q paths per held-out window.
All archive joins use the unique asset/date/spot/tenor/rate key.
Per-file hashes for the 24 frozen RN-Q arrays and metadata files are recorded
in `artifacts/ARCHIVE_MANIFEST.json`; array position is never used as a
cross-table identity.

The numerical path-quality table is reproduced from the held-out evaluation
script and is reported in `paper/main.tex`; its input paths and figure mapping
are documented in `PAPER_REPRODUCTION.md`.
