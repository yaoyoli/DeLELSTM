# DeLELSTM Track A — DSC6013 research handoff

这是 DSC6013 组员使用的独立发布分支：自有外部 Harness、历史环境、75 行正式结果、15 行统计及可追溯来源。先阅读 [中文交接指南](docs/GROUP_HANDOFF_ZH.md)。

This branch has its own root commit and contains project-authored research support material. Author model/training source, datasets, checkpoints and full logs are referenced rather than included. The existing `main` branch remains the upstream fork; switch to this release branch to see the handoff.

## Start here

```sh
python3 -B tools/verify_release.py
```

This command checks release hashes, Python syntax, all 75 result identities and the 15 sample-SD summaries. It performs no training, installation or network access. To compare against a separately available local evidence collection, see [the reproduction and verification guide](docs/REPRODUCTION.md).

| Material | Location |
| --- | --- |
| Results and report-ready table | [results](results/) |
| Original external execution/observation code | [harness_archive](harness_archive/) |
| Environment records | [environment](environment/) |
| Run counts, source and archive hashes | [provenance](provenance/) |
| Recorded historical formal commands | [commands](commands/) |
| Methodology and limitations | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Rights and attribution | [CODE_RIGHTS_AND_ATTRIBUTION.md](CODE_RIGHTS_AND_ATTRIBUTION.md) |

## Scope and findings

The historical program completed 75 model workflows, 23,750 epochs and 732,500 Adam updates across Exchange, Electricity and PM2.5. Exchange: Delelstm has the lowest mean RMSE, while Retain leads MAE/MAPE. Electricity: Delelstm leads all three. PM2.5: Retain leads all three. See the [complete table](results/REPORT_TABLE.md).

PM2.5 retains the label **Controlled Execution Deviation — Restored Five-model Comparison**. Numerical stability remains **UNRESOLVED**; paper-exact reproduction and leakage-controlled generalization remain **NOT ESTABLISHED**.

The archived Harness retains its original machine paths and one-run guards. It depends on the original local governance evidence and is not a self-contained training application. Do not rerun it in the frozen runtime. Publication performs no new training and grants no authorization for further experiments.

## Source

[Wang et al., IJCAI-23](https://doi.org/10.24963/ijcai.2023/478), [author repository](https://github.com/wangcq01/DeLELSTM/tree/b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9). See [references](docs/REFERENCES.md) and the [frozen-source index](provenance/frozen_source_index.json). The local frozen research commit `95ff20de5a2dea627c854e35a00f304c838fff12` is recorded for provenance and is not pushed by this release.
