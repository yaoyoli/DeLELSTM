import json,gzip,hashlib,zipfile,statistics,traceback,math
from pathlib import Path
from audit_utils import A,B,R,D,now,capture,differences
from git_policy import git_identity
from scoped_metadata import compare_runtime,scoped_name,validate
OUT=R/'results/stage3a/attempt-02'
def save(n,x):
 with (D/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):x.update(b)
 return x.hexdigest()
def rows(n):
 p=D/n
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def main():
 with gzip.open(D/'preflight_snapshot.json.gz','rt') as f:pre=json.load(f)
 post=capture('stage3a_postflight')
 with gzip.open(D/'postflight_snapshot.json.gz','xt') as f:json.dump(post,f)
 protected=json.loads((D/'historical_scientific_registry.json').read_text())['protected_absolute_paths']
 dif=differences(pre,post);states={k:pre['states'][k]==post['states'][k] for k in ['A','B']};policy=compare_runtime(pre,post,protected)
 git_class=git_identity(pre['git_metadata'],post['git_metadata'])
 good=all(states.values()) and all(dif[k]['equal'] for k in ['files_A','files_B']) and git_class['identity_pass'] and policy['scientific_integrity_pass']
 save('track_a_b_non_interference.json',{'pass':good,'states_equal':states,'files':dif,'runtime_policy':policy,'Git_metadata_classification':git_class,'preflight':pre['started_at'],'postflight':post['completed_at'],'TrackB_file_count':post['files_B']['entries'],'TrackB_untracked_count':len(post['states']['B']['untracked']['stdout'].split('\0')[:-1]),'coverage':'Full pre/post and modelboundary/epochs0,100,200,299 content hashing; each epoch TrackB/stat+protectedhistory stat; each trainingbatch TrackAstat. No claim outside observation window.','historical_provenance_deviation':'Stage1I TrackB Codex Session operated on TrackA Worktree; preserved'})
 metas=[]
 for rel in sorted(set(p for p in pre['runtime_existing']['files'] if scoped_name(p))|set(p for p in post['runtime_existing']['files'] if scoped_name(p))|{str(p.relative_to(R)) for p in D.rglob('.DS_Store')}):
  metas.append({'relative_path':rel,'before':pre['runtime_existing']['files'].get(rel),'after':validate(rel,protected,pre['runtime_existing']['files'].get(rel)),'writer_cause':'UNVERIFIED'})
 save('metadata_exception_register_final.json',{'rules':['.DS_Store','diagnostics/**/.DS_Store'],'records':metas,'no_scientific_manifest_file_or_symlink_exempt':True,'writer_cause':'UNVERIFIED'})
 summary=json.loads((D/'formal_execution_summary.json').read_text()) if (D/'formal_execution_summary.json').exists() else {'status':'NOT RUN OR INTERRUPTED','counts':{},'runs_completed':[]}
 runs=rows('model_results.jsonl');epochs=rows('epoch_metrics.jsonl');groups={}
 for model in ['Delelstm','Retain','IMV_full','IMV_tensor','LSTM']:
  rr=[r for r in runs if r['experiment']['model_name']==model];group={'successful_repetitions':len(rr),'full_5_repetition_result':len(rr)==5,'experiments':[r['experiment']['exp_id'] for r in rr]}
  for name in ['rmse','mae','mape']:
   v=[r['test_metrics'][name] for r in rr]
   group['test_'+name]={'values':v,'mean':statistics.mean(v) if v else None,'sample_SD_ddof1':statistics.stdev(v) if len(v)>1 else None,'population_SD_ddof0':statistics.pstdev(v) if v else None}
  groups[model]=group
 save('experiment_aggregation.json',{'models':groups,'five_repeats_not_independent_seeds':True,'incomplete_runs_never_fill_success_statistics':True})
 selection={}
 for r in rows('checkpoint_selection.jsonl'):selection[(r['experiment']['exp_id'],r['experiment']['model_name'])]=r
 ck=[]
 for key,r in selection.items():
  p=Path(r['path']);ok=p.exists() and p.stat().st_size==r['bytes'] and h(p)==r['sha256']
  ck.append({'exp_id':key[0],'model_name':key[1],'current_checkpoint_matches_last_author_best_event':ok,**r})
  if not ok:good=False
 save('best_checkpoint_inventory.json',{'checkpoints':ck,'all_current_checkpoints_match':all(r['current_checkpoint_matches_last_author_best_event'] for r in ck)})
 artifacts=[]
 if OUT.exists():
  for p in sorted(OUT.rglob('*')):
   if p.is_symlink():raise RuntimeError('Results symlink '+str(p))
   if p.is_file():artifacts.append({'path':str(p),'relative_path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':h(p),'kind':'checkpoint' if p.suffix=='.pt' else 'author_result'})
 save('external_artifact_inventory.json',{'root':str(OUT),'files':artifacts,'raw_CSV_or_venv_copied':False})
 done=summary['status']=='COMPLETED' and len(runs)==25 and summary['counts'].get('optimizer_step_successes')==330000 and good
 status='PASS' if done else 'PARTIAL PASS' if runs or summary['counts'].get('optimizer_step_successes',0)>0 else 'FAIL / NOT RUN'
 stop=json.loads((D/'stop_condition_record.json').read_text()) if (D/'stop_condition_record.json').exists() else None
 save('final_decision.json',{'Stage3A':status,'complete_all25':done,'source_faithful_reproduction':'COMPLETED WITH LIMITATIONS' if done else 'PARTIAL / NOT ESTABLISHED','full_exchange_reproduction_readiness':'NOT ESTABLISHED' if not done else 'Source-faithful runs complete; paper-exact and general numerical stability not established','overall_numerical_stability':'UNRESOLVED','non_interference_pass':good,'execution':summary,'stop':stop,'no_additional_experiment':True})
 exps=sum(all(any(r['experiment']=={'exp_id':e,'model_name':m} for r in runs) for m in groups) for e in range(5))
 md=['# Stage 3A — Formal Source-faithful Exchange Reproduction Execution Report','',f'Recommendation: {status}. Independent review/Planner acceptance pending.','',
 '### A. Source / Data / Environment Fidelity','',
 'Frozen95ff20de5a2dea627c854e35a00f304c838fff12 parentb9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9 branchtrack-a-source-faithful. GateA verified20tracked files,CSVbytes,31deps and allhistoricalscientificevidence. Python3.9.6,Torch2.0.0,NumPy1.23.5,Pandas1.5.3,CPUarm64. IdentityJSONs are evidence.','',
 '### B. Actual Experiment Protocol','',
 'Unchanged original Exchange.py and exchange_training() through runpy. Exp0..4,Delelstm/Retain/IMV_full/IMV_tensor/LSTM;300epochs;Adam.01;MSE;originalReduceLROnPlateau;strictbestvalidationRMSE;scheduleronMSE;rawtargets;full-data scaling;randomwindow split;all drop_last;no per-model reseed.','',
 '### C. CPU Resource Feasibility and Usage','',
 'See resource_feasibility.json/rawobservations and epoch_metrics.jsonl. Pilot95.484218s includes profiler/audit. Scenario extrapolation is not fullworkload measurement. Otherfourmodels CPUtime and profiler overhead unknown. Per-process caffeinate only, no persistent settings change.','',
 '### D. Actual Experiments and Models Completed','',
 f'Complete Experiments {exps}/5; Model Runs {len(runs)}/25; complete Epochs {len(epochs)}. Actual cumulative counters: '+json.dumps(summary.get('counts',{})), '',
 '### E. Per-epoch Training / Validation Results','',
 'See epoch_metrics.jsonl. Original averagebatchMSE and its square-rootRMSE preserved; validationmetrics from originalsklearn calls. Distinctbatchlosses do not establish same-sample improvement.','',
 '### F. Best Validation Checkpoint Selection','',
 'checkpoint_selection.jsonl records strict min_val_loss > mse_val**0.5 events,epoch,metrics,hash,size and parameteridentity. best_checkpoint_inventory.json validates currentbest; checkpoint_load.jsonl records actual parameteridentity after original bestload. No testbasedselection or lastepochfallback.','',
 '### G. Test RMSE / MAE / MAPE','',
 '|Exp|Model|Epochs|BestEpoch|BestValRMSE|TestRMSE|TestMAE|TestMAPE|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in runs:
  c=r['best_checkpoint'];t=r['test_metrics'];md.append(f"|{r['experiment']['exp_id']}|{r['experiment']['model_name']}|{r['counts']['epochs']}|{c['epoch']}|{c['validation_rmse']:.12g}|{t['rmse']:.12g}|{t['mae']:.12g}|{t['mape']:.12g}|")
 if not runs:md.append('|—|NOT COMPLETED|—|—|—|—|—|—|')
 md+=['','### H. Repetition Mean / Standard Deviation','','experiment_aggregation.json includes completedresults only,sampleSDddof1 and populationSDddof0. Fiveexperiments shareauthorRNGstream,notfiveindependentseeds.','',
 '### I. Numerical Stability Monitoring','','Allnaturalforwards finitechecks;training_batch_monitor.jsonl allgradient,parameter,Adamfinite and global/B_j summaries. forward_monitor.jsonl coeffmagnitude. Solverfixedsamplestrain+valbatch1 epochs0,25,..275 andtest1 ofDelelstm. UnsampledUNVERIFIED. Internalinv_ex not counted as extra highlevelinv. OverallNumericalStabilityUNRESOLVED.','',
 '### J. Source-vs-Paper Discrepancies','','source_vs_paper.json and supplement_extracted_text.json: supplement75/15/15 sums105%,source75/15/remainder10; genericpaperhidden32 vs effectiveIMV_full16/Retain64. SupplementTable2 top50%features,not directfull8featurebenchmark. UnspecifiedpaperdetailsUNVERIFIED;mainpaper notprovided/fetched. CPUdifferentV100s.','',
 '### K. Leakage / Evaluation Limitations','','Full-data min-max usesallsplits;overlappingrandomwindows cansharetimeinformation;notstrictchronologicalextrapolation. drop_last drops37train per epoch,109val,117test. RawsameMAPE nearzerotargetrisk. NoTrackBpipelinechanges.','',
 '### L. Track A/B Non-interference','',f'Observedwindow pass={good};fullsnapshots and track_a_b_non_interference.json. TrackB30untracked retained ifPASS. HistoricalStage1I executorprovenancedeviation andauditfailures preserved. ScopedRuntime.DS_Store writer/causeUNVERIFIED,never scientificfile/symlink exemption.','',
 '### M. Artifact and Evidence Inventory','','external_artifact_inventory.json:all results/checkpoints externalRuntime/results/stage3a. evidence_manifest.json and stage3a_attempt02_evidence.zip. No rawCSVcopies/venv inZIP;no Stage2BPILOTstate loaded.','',
 '### N. Reproduction Status Recommendation','',f'Stage3A {status}; status={summary.get("status")}; Overall Numerical Stability UNRESOLVED. Source-faithful reproduction '+('completedwithreportedlimitations.' if done else 'PARTIAL / NOT ESTABLISHED.'),'Actualstop/failure: '+(json.dumps(stop,ensure_ascii=False) if stop else 'No recordedmodelstop;inspectactualstatus/exitreceipt.'),'No reinitialization,retry,solverreplacement,clipping,Gitwrite,TrackBexecution,additionalexperiments orautomaticnextstage. Await ResearchAnalyst andPlanner.']
 with (D/'stage3a_report.md').open('x') as f:f.write('\n'.join(md)+'\n')
 package()
def _package_original():
 excluded={'evidence_manifest.json','stage3a_attempt02_evidence.zip','package_verification.json','postflight.stdout','postflight.stderr'}
 records=[]
 for p in sorted(D.rglob('*')):
  if p.is_symlink():raise RuntimeError('Evidence symlink')
  if not p.is_file() or p.name in excluded or p.relative_to(D).parts[0]=='cache':continue
  if scoped_name(str(p.relative_to(R))):validate(str(p.relative_to(R)));continue
  records.append({'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':h(p)})
 save('evidence_manifest.json',{'created_at':now(),'files':records,'exclusions':'selfmanifest,ZIP,postpackagingreceipt exclude recursion; active postflight stdout/stderr + finalizer exitreceipt saved separately. cache externalnonscientific. Checkpoints/results indexed externally. Scopedmetadata separately registered.'})
 zp=D/'stage3a_attempt02_evidence.zip'
 with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for r in records:z.write(D/r['path'],'stage3a/'+r['path'])
  z.write(D/'evidence_manifest.json','stage3a/evidence_manifest.json')
 with zipfile.ZipFile(zp) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(records)+1
  for r in records:
   x=hashlib.sha256();size=0
   with z.open('stage3a/'+r['path']) as f:
    for b in iter(lambda:f.read(1048576),b''):x.update(b);size+=len(b)
   assert size==r['bytes'] and x.hexdigest()==r['sha256']
 save('package_verification.json',{'path':str(zp),'bytes':zp.stat().st_size,'sha256':h(zp),'manifest_sha256':h(D/'evidence_manifest.json'),'members':len(records)+1,'all_CRC_SHA256_size_pass':True})
 print(json.dumps({'ZIP':str(zp),'bytes':zp.stat().st_size,'sha256':h(zp)}),flush=True)

def write_attempt02_report():
 summary=json.loads((D/'formal_execution_summary.json').read_text()) if (D/'formal_execution_summary.json').exists() else {'status':'NOT RUN / INTERRUPTED','counts':{}}
 runs=rows('model_results.jsonl');epochs=rows('epoch_metrics.jsonl');decision=json.loads((D/'final_decision.json').read_text())
 first=json.loads((D/'epoch0_reproducibility_comparison.json').read_text()) if (D/'epoch0_reproducibility_comparison.json').exists() else None
 stop=json.loads((D/'stop_condition_record.json').read_text()) if (D/'stop_condition_record.json').exists() else None
 complete_exps=sum(all(any(r['experiment']=={'exp_id':e,'model_name':m} for r in runs) for m in ['Delelstm','Retain','IMV_full','IMV_tensor','LSTM']) for e in range(5))
 c=summary.get('counts',{})
 md=['# Stage 3A — Attempt 02 Formal Exchange Reproduction Execution Report','',
 '## A. Attempt 01 / 02 Provenance','',
 'Attempt01保持PARTIAL PASS/CONTROLLED STOP：44次历史Adam更新、8个Validation Batch、0个成功Epoch记录；原ZIP9505625d87fbb190cc5f4ac6613f99321b635fd100a70ac6a00736f42a049f5e及执行Observer/Candidate已逐文件核验。旧文件/Checkpoint没有覆盖或移动。Attempt02为独立新Seed555替代执行；历史44步不计入本次25模型预算，未加载旧Checkpoint或PilotState。','',
 '## B. Observer Serialization Repair and Regression Tests','',
 '仅对ReduceLROnPlateau(mode=min)顶层mode_worse=+inf采用明确类型标记，decode仅恢复记录含义，不回写Live Scheduler。非有限best、有效LR、未知嵌套字段均拒绝；全局allow_nan=False且删除default=str。16项隔离非模型回归测试PASS：包括真实DummyScheduler首次finite step、可逆性、LiveState不变、科学非有限值拒绝和日志写入失败计数不增加。DummyOptimizer未执行step；没有作者模型Import/Forward/Backward。见serialization_regression_results.json、Sources/Diff/Hash。','',
 '## C. Frozen Source / Environment / Dataset Fidelity','',
 'FrozenHEAD95ff20de5a2dea627c854e35a00f304c838fff12，Parentb9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9；20个tracked文件逐Blob二进制匹配；Exchange路径为唯一已批准适配。两份CSV匹配指定SHA。Python3.9.6/Torch2.0.0/NumPy1.23.5/Pandas1.5.3，arm64CPU，31项依赖Lock匹配；未安装或修改环境。','',
 '## D. Resource Preflight and Actual Resource Usage','',
 'resource_feasibility.json分别记录Measured/Derived/Assumed/Unknown。未额外测速。PID、启动/结束、stdout/stderr/exit和实际EpochResource记录保留；不以聊天中断推断子进程失败。每进程caffeinate只维持本次运行，未改电源配置；无自动重启。','',
 '## E. Formal Experiment and Model Execution Counts','',
 f'完整Model Runs：{len(runs)}/25；完整Experiments：{complete_exps}/5；成功记录Epoch：{len(epochs)}/7500；实际计数：'+json.dumps(c,ensure_ascii=False),
 '原始循环固定Exp0..4、Delelstm→Retain→IMV_full→IMV_tensor→LSTM；每模型300 Epoch，最大330000次训练更新。每模型不重新Seed。所有训练调用均来自未修改的作者入口/训练函数；无诊断性额外Forward。','',
 '## F. Epoch 0 Reproducibility Comparison','',
 'Source/CSV/Split/全部26初始参数/首批128个窗口及Input/TargetHash精确比对见对应comparison文件。数值比较来自本次自然Epoch0，没有重跑或新增batch。']
 if first:
  md+=['','|Metric|Reference|Observed|Abs Diff|Relative Diff|','|---|---:|---:|---:|---:|']
  for k,v in first['metrics'].items():md.append(f"|{k}|{v['reference']:.17g}|{v['observed']:.17g}|{v['absolute_difference']:.17g}|{v['relative_difference']:.17g}|")
  md+=['',f"Scheduler Step已返回、Epoch记录已成功写入：{first['scheduler_step_returned']} / {first['epoch_record_written']}；合法标记：{json.dumps(first['scheduler_mode_worse_marker'])}。未设置未经批准的科学稳定阈值。"]
 else:md+=['','NOT RUN / 未取得完整自然Epoch0记录；见实际Stop/Exit。']
 md+=['','## G. Per-model / Per-experiment Training and Validation','',
 '原作者Train MSE为平均Batch MSE，RMSE为其平方根。所有已完成Epoch记录在epoch_metrics.jsonl，按模型/Experiment可独立提取；部分流程不作为完整模型结果。','',
 '## H. Scheduler and Checkpoint Selection','',
 '原Adam(lr=.01)、ReduceLROnPlateau(min,factor.5,patience20,min_lr.0001)及原位置scheduler.step(mse_val)保持。原严格ValidationRMSE选模：min_val_loss > mse_val**.5。checkpoint_selection.jsonl与best_checkpoint_inventory.json记录Epoch、指标、Hash、Size；checkpoint_load.jsonl验证300Epoch后原始最佳加载身份。未根据Test选模。','',
 '## I. Original Test RMSE / MAE / MAPE','',
 '|Exp|Model|Epochs|Best Epoch|Best Val RMSE|Test RMSE|Test MAE|Test MAPE|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in runs:
  b=r['best_checkpoint'];t=r['test_metrics'];md.append(f"|{r['experiment']['exp_id']}|{r['experiment']['model_name']}|{r['counts']['epochs']}|{b['epoch']}|{b['validation_rmse']:.12g}|{t['rmse']:.12g}|{t['mae']:.12g}|{t['mape']:.12g}|")
 if not runs:md+=['|—|NOT RUN / 尚无完整模型|—|—|—|—|—|—|']
 md+=['','## J. Repeat Means / Standard Deviations','',
 'experiment_aggregation.json仅使用完整成功模型；报告成功重复数、Mean、sampleSD(ddof1)、populationSD(ddof0)。缺失或失败模型不填补统计；五次Experiment不是五个独立Seed。','',
 '## K. Numerical and Solver Observations','',
 '所有自然Forward进行Prediction/解释Tensor有限性检查；训练每批检查Loss、全部梯度、更新前后参数与AdamState。Delelstm保存Global/B_j梯度、平方范数贡献和系数量级。Solver仅固定采样：每25Epoch(含0)首个训练/验证Batch与正式Test首批；未采样UNVERIFIED，不额外Forward。高层inv与内部inv_ex不叠加。有限值不证明Gram良态或长期稳定，OverallNumericalStability仍UNRESOLVED。','',
 '## L. Source-vs-Paper Discrepancies','',
 'source_vs_paper.json保留补充材料对应差异：75/15/15合计105%与源码75/15/剩余10%；IMV_full有效hidden16、Retain64与通用hidden32描述；Table2为top50%feature结果，不能冒充当前全8特征目标。主论文未单独获取，补充未说明的协议细节UNVERIFIED。硬件CPU与作者V100s不同。','',
 '## M. Dataset Leakage and Evaluation Limitations','',
 '保留Full-dataMinMax、RandomWindowSplit、原始Target和所有drop_last。跨split重叠窗口可能共享原始时间信息；不是严格时间外推。每EpochTrain丢37窗口，Validation丢109，Test丢117；原MAPE对近零Target有解释风险。未继承TrackB的ChronologicalSplit或TrainOnlyScaler。','',
 '## N. Track A/B Non-interference','',
 '完整Pre/Post、关键阶段Hash、每EpochTrackB及历史科学文件stat、每训练BatchTrackAstat，范围据实记录。TrackB30Untracked和全部Ignored/文件纳入检查；无TrackB运行。旧Attempt01结果/历史Evidence受保护。GitMetadata差异逐项分类，关键refs/index/config无豁免；ScopedRuntime.DS_Store只限普通真实路径非科学文件，Writer/CauseUNVERIFIED。见track_a_b_non_interference.json。','',
 '## O. Evidence and Artifact Inventory','',
 'stage3a_attempt02_evidence.zip内含执行源码、完整Diff/Hash、回归、Preflight、原协议、自然比较、监控、异常、Manifest；Checkpoint和作者结果位于独立results/stage3a/attempt-02，external_artifact_inventory.json给出全部路径/Size/SHA。大Tensor不塞入ZIP。监督器及Finalizer结束后Receipt单独保存Hash，避免自引用。','',
 '## P. Final Reproduction Recommendation','',
 f"Stage3A Attempt02建议：{decision['Stage3A']}；实际状态：{summary.get('status')}。完整Runs {len(runs)}/25，完整Experiments {complete_exps}/5，记录Epoch {len(epochs)}，Optimizer Step {c.get('optimizer_step_successes',0)}。",
 'Stop Condition：'+(json.dumps(stop,ensure_ascii=False) if stop else '没有记录到模型Stop；请结合实际Exit/Completed状态判断。'),
 'Source-faithful Outcome：'+str(decision['source_faithful_reproduction'])+'。OverallNumericalStability：UNRESOLVED。FullExchangeReadiness：'+str(decision['full_exchange_reproduction_readiness'])+'。',
 '本次替代运行不自动重启，不进行Attempt03，不加载部分Checkpoint接续；等待ResearchAnalyst独立审核及Planner正式验收。']
 with (D/'stage3a_attempt02_execution_report_zh.md').open('x') as f:f.write('\n'.join(md)+'\n')
def package():
 write_attempt02_report()
 _package_original()

if __name__=='__main__':
 try:main()
 except BaseException:
  save('postflight_packaging_error.json',{'traceback':traceback.format_exc(),'no_model_retry':True});raise
