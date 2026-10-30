# Original external Harness archive

The `exchange`, `electricity` and `pm25` directories contain byte-identical copies of the project-authored execution/observation code listed in the corresponding historical static review, plus the saved postrun verifier. `provenance/harness_sources.json` gives each source path, SHA-256 and role. The executed-code hashes were checked against the historical reviews before copying.

The files implement external execution control, read-only tracing, audit metadata, process supervision and evidence packaging. They call or inspect separately held author code; they do not redistribute the author's model or training implementation. The PM2.5 entry adapter makes only the approved data path and second `model_list` assignment changes in memory.

These are historical artifacts, not a new portable training pipeline. Absolute paths and no-retry guards remain as executed. Required local preflight/gate files, full snapshots and scientific evidence are excluded; helper imports can read or write the original runtime if executed. Do not execute or import these files for inspection, do not remove their locks, and do not replace the frozen runtime files with these copies. Use static reading or the read-only tools under `tools/`.

The two supplied standalone verification tools never import the archive. Their checks are publication checks; historical full-stream verification results are separately labeled in `provenance/historical_verification_extracts.json`.
