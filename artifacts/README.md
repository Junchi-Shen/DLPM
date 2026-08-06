# Local artifacts

Keep checkpoints, fitted preprocessors, RN-Q path archives, and generated
paths here, or set `DLPM_ARTIFACT_ROOT` to an existing local artifact store.

For the frozen paper model, download the release asset
`DLPM_Final_EMA12000_Checkpoint_20260802.zip` and unpack it here. For an
interrupted run that must resume with its optimizer and scheduler state,
download `DLPM_Final_EMA12000_Resume_State_20260802.zip` as well. Checksums
and the exact configuration are in `../FINAL_RUN_MANIFEST.md`. The complete
per-file identity of the 40-path P archive, the 256-path RN-Q archive, and
both held-out/source-order tables is frozen in `ARCHIVE_MANIFEST.json`.

Artifacts are omitted from Git because they are large and are generated
outputs.  The evaluation code joins P and Q paths by the audited composite
key `(underlying, start date, start price, maturity, risk-free rate)` rather
than array position.  Do not substitute row-position joins when restoring an
archive: source testing rows and cleaned chronological tables need not share
the same order.
