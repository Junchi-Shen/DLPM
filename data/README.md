# Data policy

The version-controlled repository does not include raw market data, processed
window tables, trained weights, or simulated path archives.  These artifacts
are large and some market-data sources impose redistribution constraints.

Place the processed chronological split under `data/processed/`, or point
`DLPM_DATA_ROOT` to an existing local directory.  The expected subdirectories
are `Trainning_Dataset/`, `Validation_Dataset/`, and `Testing_Dataset/`.

The final study uses eight indices: CSI 300, CSI 500, CSI 1000, SSE Composite,
S&P 500, NASDAQ, Dow Jones, and Russell 2000.  The build process uses only
information available before each forecast origin, with purged chronological
train/validation/test boundaries documented in the paper.
