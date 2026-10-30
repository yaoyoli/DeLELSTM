# Result fields

`dataset`, `model`, `experiment_id` identify one completed formal model workflow. Experiment IDs are 0–4 and preserve the historical execution sequence.

`best_validation_epoch_zero_based` is the original selected checkpoint epoch. `test_mse`, `test_rmse` and `test_mae` are the author's saved test metrics in the historical target scale. `test_mape_raw_ratio` is the saved TorchMetrics ratio; `test_mape_percent = 100 * test_mape_raw_ratio`. No near-zero exclusion or replacement metric is introduced.

`source_run_id` names the formal run. The provenance index supplies the exact original CSV path, hash, evidence archive member and selected checkpoint hash. The published CSV is extracted from saved test CSVs, cross-checked against formal summaries and byte-identical archive members. No model output is recomputed.

The 15-row table uses arithmetic means and sample SD (`ddof=1`) over all five completed experiments, without filtering by performance. Summary columns ending `_sample_sd` use the same units as their corresponding means.
