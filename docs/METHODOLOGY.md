# Historical methodology and limits

The inputs are the already saved formal results, not newly trained models. Five experiments per dataset each execute five models. Seed 555 is set at the author entry; subsequent models/experiments consume a continuous RNG stream rather than resetting independent seeds. Preserve the recorded order in `provenance/run_summaries.json`: Exchange uses Delelstm, Retain, IMV_full, IMV_tensor, LSTM; Electricity/PM2.5 use Delelstm, IMV_full, IMV_tensor, Retain, LSTM.

| Dataset | Epochs per model | Batch size | Total epochs | Adam updates |
| --- | ---: | ---: | ---: | ---: |
| Exchange | 300 | 128 | 7,500 | 330,000 |
| Electricity | 350 | 64 | 8,750 | 87,500 |
| PM2.5 | 300 | 32 | 7,500 | 315,000 |

Original model/training functions, Adam, ReduceLROnPlateau, scaling, random split, targets and `drop_last` remain the historical protocol. The strict best-validation-RMSE comparison selects checkpoints; original testing occurs after the selected checkpoint is loaded. MAPE is the saved TorchMetrics ratio and is multiplied by 100 only for percent display. The source uses a 75%/15%/remaining-10% random split; full-data Min-Max fitting uses held-out information.

Exchange has overlapping sliding windows; Electricity and PM2.5 use daily windows with no shared timestamps across the recorded splits. Random day splits and temporal dependence still limit future extrapolation claims. PM2.5 preprocessing includes pre-split forward/backward filling. Near-zero/zero targets limit MAPE interpretation. Batches dropped by `drop_last` remain excluded under the original protocol.

Exchange uses the approved local path change in the frozen commit. Electricity uses an in-memory data-path substitution. PM2.5 has two approved in-memory substitutions: the data path and the effective second model-list assignment, which originally overrides the five-model list with Retain only. Its label remains **Controlled Execution Deviation — Restored Five-model Comparison**; the original PM training function remains unchanged.

## Evidence classifications

- **Evidence Verified** in the historical reports: documented complete workloads, saved checkpoint selection/loading, test results, finite-value records and scoped integrity checks. This release rechecks selected source/result/checkpoint/archive hashes, archived-code review hashes and result summaries; it does not repeat the historical multi-million-record full-stream audit.
- **Evidence-supported**: the model rankings in the complete five-repeat tables, within the historical source protocol.
- **UNVERIFIED**: unsampled solver paths and unresolved causes of historical noncritical metadata changes. Fixed profiler samples do not establish all natural forward paths.
- **Inconsistent / FAIL**: preserve any such historical classification if a check fails. Do not silently repair evidence or promote a failed or incomplete workflow into the 75 accepted records.

Overall numerical stability is **UNRESOLVED**. Finite tensors and exit code 0 do not establish well-conditioned Gram matrices, reliable least-squares explanations, paper-exact reproduction or leakage-controlled generalization. Published historical verification extracts retain these limits. No optimizer/solver modification, extra forward, new seed or Track B training is performed for publication.
