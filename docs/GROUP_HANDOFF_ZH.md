# 给 DSC6013 组员的交接指南

本分支是 GitHub 可公开的科研支持材料，不是之前完整 Research Handoff ZIP 的镜像，也不包含完整原作者训练项目。原 ZIP 中的 Word、图表和报告草稿仍通过原交接渠道保留。

1. 负责结果和 PPT：查看 `results/REPORT_TABLE.md` 与 15 行汇总 CSV；需要逐次记录时使用 75 行明细 CSV。
2. 负责方法和讨论：查看 `docs/METHODOLOGY.md`；保留数据泄漏、连续 RNG、drop_last、MAPE 与数值稳定性的边界。
3. 负责代码和 Appendix：查看 `harness_archive/`、环境锁、历史命令和 `provenance/`。Appendix 应按课程要求写进报告，不能仅提供仓库链接。
4. 负责核验：运行 `python3 -B tools/verify_release.py`。有本地冻结源码和 runtime 的组员可按 `docs/REPRODUCTION.md` 进一步比对原始文件。

这是父提交为空的独立发布分支，不承接原作者源码和数据的 Git 历史。GitHub 默认仍显示 `main`，需要切换至 `track-a-dsc6013-release`。

三个数据集共 75 次完整模型流程，五次 Experiment 不是五个独立 Seed；统计为 Sample SD（ddof=1）。MAPE 明细同时保留原比率和百分比，避免混用。

结果应完整表述：Exchange 的 Delelstm 平均 RMSE 最低，但 Retain 的 MAE、MAPE 最低；Electricity 的 Delelstm 三项领先；PM2.5 的 Retain 三项领先。

PM2.5 必须保留 **Controlled Execution Deviation — Restored Five-model Comparison**。数值稳定性仍为 **UNRESOLVED**，Paper-exact 与泄漏控制泛化仍为 **NOT ESTABLISHED**。

历史 Harness 是原样归档，含机器路径和防重复运行保护，依赖未公开的原始审核记录。不要修改原路径后直接执行，也不要删除历史锁。新的训练需要新的外部目录、来源和环境核验及相应研究授权；这次发布没有运行任何模型。
