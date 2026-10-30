# Reproduction and verification guide

## What can be checked from this branch

```sh
python3 -B tools/verify_release.py
```

This checks file hashes and the included small result tables using only the Python standard library. It parses archived Python as syntax without executing it. It does not reproduce model training or independently establish scientific stability.

## Compare with separately held local source and evidence

If you already have authorized access to the original files, pass their paths explicitly:

```sh
python3 -B tools/audit_local_evidence.py --source /absolute/path/to/DeLELSTM-track-a --runtime /absolute/path/to/DeLELSTM-track-a-runtime
```

This reads source, original Harness, environment records, 75 saved test CSVs and selected checkpoint files for hashes. It never imports a model or unpickles a checkpoint. Add `--archives` to hash the three large evidence ZIPs and compare their indexed CSV members as well. The default check avoids rereading the ZIPs. Files are not created or modified by either tool.

The historical frozen source HEAD is `95ff20de5a2dea627c854e35a00f304c838fff12` with upstream parent `b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9`. Only the Exchange data path differs in that local commit. It is recorded here and deliberately not published as a commit. The upstream reference and per-file hashes are in `provenance/frozen_source_index.json`; the Exchange local path adaptation must be checked against that manifest. Acquire author source and datasets separately, subject to their rights and terms; this publication does not grant permission for redistribution.

## Historical execution

`commands/historical_formal_commands.json` records the exact formal child commands, working directories and environment overrides. Those commands are evidence, not instructions to rerun in the original runtime. They were launched through one-shot supervisors with persisted stdout/stderr, process identities and real wait exit codes. Prior Exchange Attempt01 stopped after 44 updates; the accepted Attempt02 was a distinct authorized replacement. The historical stopped attempt is not counted in the 75 formal workflows, and its evidence remains retained locally.

For any separately authorized fresh reproduction, use a new runtime outside the research worktrees, verify the exact source/data hashes and environment, regenerate fresh gate/preflight records, preserve model order/RNG and the approved AST substitutions, and use a new permanent one-run lock with supervision and recorded exit codes. The archive depends on historical guards, snapshots and paths and cannot be made portable merely by changing directory names. A fresh training run has not been validated by this release.

Do not change optimizer, solver, scaling, split, target, RNG, checkpoint selection or numerical behavior while describing a run as the same Track A protocol. PM2.5 must retain **Controlled Execution Deviation — Restored Five-model Comparison**. Track B and new training are outside this publication task.
