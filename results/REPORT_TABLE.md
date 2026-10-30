# Track A results

Mean ± sample SD (n=5, ddof=1). MAPE is percent; best epochs in the detailed CSV are zero-based.

| Dataset | Model | RMSE | MAE | MAPE (%) |
| --- | --- | ---: | ---: | ---: |
| Exchange | Delelstm | 0.002343 ± 0.000062 | 0.001509 ± 0.000018 | 0.232240 ± 0.002793 |
| Exchange | IMV_full | 0.002620 ± 0.000007 | 0.001510 ± 0.000007 | 0.232464 ± 0.001061 |
| Exchange | IMV_tensor | 0.002634 ± 0.000003 | 0.001503 ± 0.000002 | 0.231406 ± 0.000400 |
| Exchange | Retain | 0.002483 ± 0.000014 | 0.001470 ± 0.000005 | 0.226086 ± 0.000781 |
| Exchange | LSTM | 0.002616 ± 0.000011 | 0.001509 ± 0.000004 | 0.232058 ± 0.000568 |
| Electricity | Delelstm | 1.727980 ± 0.053754 | 1.099732 ± 0.025901 | 1.693907 ± 0.044854 |
| Electricity | IMV_full | 2.057479 ± 0.153544 | 1.411528 ± 0.099397 | 2.251607 ± 0.174375 |
| Electricity | IMV_tensor | 2.676533 ± 0.937851 | 1.862280 ± 0.676486 | 2.943321 ± 1.071614 |
| Electricity | Retain | 2.663974 ± 0.262102 | 1.991283 ± 0.229469 | 3.211137 ± 0.378602 |
| Electricity | LSTM | 5.109491 ± 7.118701 | 3.955207 ± 5.957625 | 6.489831 ± 9.882802 |
| PM2.5 | Delelstm | 26.584807 ± 0.424520 | 13.575847 ± 0.189375 | 22.390449 ± 0.929176 |
| PM2.5 | IMV_full | 28.516684 ± 0.215945 | 13.922442 ± 0.074714 | 24.440463 ± 1.033138 |
| PM2.5 | IMV_tensor | 28.826824 ± 0.383711 | 14.501401 ± 0.150678 | 27.144082 ± 1.922716 |
| PM2.5 | Retain | 24.977720 ± 0.099626 | 13.343981 ± 0.070417 | 20.940206 ± 0.549458 |
| PM2.5 | LSTM | 26.537066 ± 0.204951 | 13.448053 ± 0.053167 | 22.089360 ± 0.450443 |

Exchange: Delelstm has the lowest mean RMSE; Retain has the lowest mean MAE and MAPE. Electricity: Delelstm leads all three metrics. PM2.5: Retain leads all three metrics.

Five experiments follow the author’s continuous RNG stream; they are not five independently seeded runs. Absolute RMSE values have different scales across datasets.

PM2.5: **Controlled Execution Deviation — Restored Five-model Comparison**.

Numerical stability: **UNRESOLVED**. Paper-exact reproduction and leakage-controlled generalization: **NOT ESTABLISHED**.
