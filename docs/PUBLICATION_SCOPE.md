# Publication scope

Destination: `https://github.com/yaoyoli/DeLELSTM.git`, branch `track-a-dsc6013-release`. Independent root history; only this new branch is to be pushed. No main update, tag push, force push, research-worktree branch change or source/evidence rewrite is intended.

Included: reviewed project-authored external Harness and observers, postrun verifier sources, historical formal command records, platform-specific environment records, 75 metric rows, 15 sample-SD summaries, source/evidence/checkpoint hash indices, references and read-only publication tools.

Excluded: author source/preprocessing implementation and their Git history; raw/processed datasets; checkpoints and tensor files; full training/forward/gradient/epoch logs; evidence ZIPs; snapshots/full ASTs/diffs of author source; paper/supplementary PDFs; virtual environments; wheels/caches; resolver/package metadata dumps; earlier stopped/pilot execution artifacts; prior complete Word/figure handoff ZIPs. Historical exclusions are indexed where needed, rather than copied.

Results were extracted from original saved test CSVs and compared to formal summaries and byte-identical members of the three final evidence archives. Selected checkpoint files were hashed without loading them. The complete historical stream audit was not repeated. Files under `harness_archive/` retain original bytes and host-specific dependencies; only `tools/` provides standalone checks in this release.
