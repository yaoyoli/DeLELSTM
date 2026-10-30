"""Electricity postflight/aggregation/report/package. Standard library only; no model import."""
import os,sys,json,gzip,hashlib,zipfile,statistics,math,traceback
from pathlib import Path
from audit_utils import A,B,R,D,now,capture,differences
from scoped_metadata import scoped_name,validate,compare_runtime
from git_policy import git_identity
OUT=R/'results/stage3b/electricity/run-01'
NAMES=['Delelstm','IMV_full','IMV_tensor','Retain','LSTM']
def save(n,x):
 with (D/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):x.update(b)
 return x.hexdigest()
def rows(n):
 p=D/n
 if not p.exists():return []
 out=[]
 with p.open() as f:
  for i,l in enumerate(f,1):
   if not l.strip():continue
   try:out.append(json.loads(l))
   except json.JSONDecodeError:out.append({'unresolved_partial_log_line':i,'raw':l[:500]})
 return out
def main():
 with gzip.open(D/'preflight_snapshot.json.gz','rt') as f:pre=json.load(f)
 post=capture('electricity_run01_postflight')
 with gzip.open(D/'postflight_snapshot.json.gz','xt') as f:json.dump(post,f)
 protected=json.loads((D/'historical_scientific_registry.json').read_text())['protected_absolute_paths']
 dif=differences(pre,post);states={k:pre['states'][k]==post['states'][k] for k in ['A','B']}
 runtime=compare_runtime(pre,post,protected);git=git_identity(pre['git_metadata'],post['git_metadata'])
 good=all(states.values()) and all(dif[k]['equal'] for k in ['files_A','files_B']) and runtime['scientific_integrity_pass'] and git['identity_pass']
 save('track_a_b_non_interference.json',{'pass':good,'preflight':pre['started_at'],'postflight':post['completed_at'],'states_equal':states,'files':dif,'runtime_policy':runtime,'Git_metadata_classification':git,'TrackB_files':post['files_B']['entries'],'TrackB_untracked':len(post['states']['B']['untracked']['stdout'].split('\0')[:-1]),'historical_Exchange_Pilot_and_prior_scientific_files_preserved':runtime['scientific_integrity_pass'],'coverage':'Allregularfilebytes/SHA,size/mode/mtime/inode andsymlinktarget atfullboundaries; perEpochTrackB/historicalstats, perBatchTrackAstats. ACL/xattrs/ctime not audited. No claim outsidewindow.'})
 meta=[]
 candidates=set(p for p in pre['runtime_existing']['files'] if scoped_name(p))|set(p for p in post['runtime_existing']['files'] if scoped_name(p))|{str(p.relative_to(R)) for p in D.rglob('.DS_Store')}
 for rel in sorted(candidates):meta.append({'relative_path':rel,'before':pre['runtime_existing']['files'].get(rel),'after':validate(rel,protected,pre['runtime_existing']['files'].get(rel)),'writer_cause':'UNVERIFIED'})
 save('metadata_exception_register_final.json',{'rules':['.DS_Store','diagnostics/**/.DS_Store'],'records':meta,'writer_cause':'UNVERIFIED','no_scientific_or_symlink_exemption':True})
 summary=json.loads((D/'formal_execution_summary.json').read_text()) if (D/'formal_execution_summary.json').exists() else {'status':'NOT RUN / INTERRUPTED','counts':{}}
 runs=[r for r in rows('model_results.jsonl') if r.get('status')=='COMPLETED'];epochs=[r for r in rows('epoch_metrics.jsonl') if 'epoch' in r]
 order=[(e,m) for e in range(5) for m in NAMES]
 run_order=[(r['experiment']['exp_id'],r['experiment']['model_name']) for r in runs]
 orderpass=run_order==order[:len(runs)]
 groups={}
 for m in NAMES:
  rr=[r for r in runs if r['experiment']['model_name']==m]
  g={'successful_repetitions':len(rr),'experiments':[r['experiment']['exp_id'] for r in rr],'complete5':len(rr)==5}
  for k in ['rmse','mae','mape']:
   v=[r['test_metrics'][k] for r in rr];g['test_'+k]={'values':v,'mean':statistics.mean(v) if v else None,'sample_SD_ddof1':statistics.stdev(v) if len(v)>1 else None,'population_SD_ddof0':statistics.pstdev(v) if v else None}
  groups[m]=g
 save('experiment_aggregation.json',{'models':groups,'MAPE_author_units':'rawratio; percent=rawratio*100','five_repeats_not_independent_seeds':True,'incomplete_models_not_imputed':True})
 selections={}
 for r in rows('checkpoint_selection.jsonl'):
  if 'experiment' in r:selections[(r['experiment']['exp_id'],r['experiment']['model_name'])]=r
 cp=[]
 for key,r in selections.items():
  p=Path(r['path']);ok=p.is_file() and p.stat().st_size==r['bytes'] and h(p)==r['sha256']
  if not ok:good=False
  completed=key in run_order
  cp.append({'experiment':r['experiment'],'current_hash_matches_latest_selection':ok,'eligible_final_model_completed350epochs_and_test':completed,**r})
 save('best_checkpoint_inventory.json',{'checkpoints':cp,'all_current_files_match':all(x['current_hash_matches_latest_selection'] for x in cp)})
 artifacts=[]
 if OUT.exists():
  for p in sorted(OUT.rglob('*')):
   if p.is_symlink():raise RuntimeError('Results symlink '+str(p))
   if p.is_file():artifacts.append({'path':str(p),'relative_path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':h(p),'kind':'checkpoint' if p.suffix=='.pt' else 'author_result'})
 save('external_artifact_inventory.json',{'root':str(OUT),'files':artifacts,'original_DATA_CSV_or_venv_included':False})
 # Independently stream stored successful-update and natural-forward records.
 counts=summary.get('counts',{});tr=0;fr=0;gradients=0;bad=[];maxcoeff=0.;maxgrad=0.;maxbj=0.
 p=D/'training_batch_monitor.jsonl'
 if p.exists():
  with p.open() as f:
   for i,l in enumerate(f,1):
    try:r=json.loads(l)
    except json.JSONDecodeError:bad.append('partialtrainingline'+str(i));continue
    tr+=1
    if not r.get('step_success') or not r.get('post_parameter_finite') or not r.get('post_Adam_state_finite'):bad.append('badupdate'+str(i))
    for g in r['gradients']:
     if g.get('classification')=='legitimate_frozen_parameter_no_gradient':continue
     gradients+=1
     if not g.get('present') or not g.get('finite'):bad.append('badgradient'+str(i)+g['name'])
    maxgrad=max(maxgrad,r['global_gradient_l2'])
    if r.get('B_j'):maxbj=max(maxbj,r['B_j']['l2'])
 p=D/'forward_monitor.jsonl';phases={'train':0,'val':0,'test':0}
 if p.exists():
  with p.open() as f:
   for i,l in enumerate(f,1):
    try:r=json.loads(l)
    except json.JSONDecodeError:bad.append('partialforwardline'+str(i));continue
    fr+=1;phases[r['phase']]+=1
    if not r['prediction']['finite']:bad.append('badprediction'+str(i))
    for k in ['coefficients_or_alpha','third_or_beta']:
     x=r[k]
     if x is not None and not x['finite']:bad.append('badcoefficient'+str(i))
    if r['experiment']['model_name']=='Delelstm' and r['coefficients_or_alpha'] is not None:maxcoeff=max(maxcoeff,r['coefficients_or_alpha']['max_abs'] or 0.)
 success=summary.get('status')=='COMPLETED'
 expected={'model_initializations':25,'optimizer_creations':25,'training_batches':87500,'validation_batches':17500,'test_batches':25,'forwards':105025,'losses':87500,'backwards':87500,'optimizer_step_attempts':87500,'optimizer_step_successes':87500,'epochs_completed':8750,'models_completed':25}
 full=success and counts==expected and len(runs)==25 and len(epochs)==8750 and tr==87500 and fr==105025 and phases=={'train':87500,'val':17500,'test':25} and orderpass and not bad and good
 solver=[r for r in rows('solver_samples.jsonl') if 'solver' in r]
 st={k:sum(r['solver'][k] for r in solver) for k in ['high_level_inv','internal_inv_ex','pinv']}
 check={'stored_successful_updates':tr,'recorded_natural_forwards':fr,'phases':phases,'gradient_records_verified':gradients,'all_recorded_numeric_gates_finite':not bad,'discrepancies':bad[:100],'counts_match_complete_budget':counts==expected,'model_order_matches_prefix':orderpass,'max_Delelstm_coefficient_max_abs':maxcoeff,'max_global_gradient_l2':maxgrad,'max_B_j_gradient_l2':maxbj,'solver_samples':len(solver),'solver_event_totals':st,'solver_coverage':'Only actual fixed natural samples; unprofiledUNVERIFIED;inv_ex not added to highlevelinv'}
 save('execution_evidence_crosscheck.json',check)
 status='PASS' if full else 'PARTIAL PASS / CONTROLLED STOP' if counts.get('optimizer_step_successes',0)>0 else 'FAIL / NOT RUN'
 stop=json.loads((D/'stop_condition_record.json').read_text()) if (D/'stop_condition_record.json').exists() else None
 save('final_decision.json',{'ElectricityRun01':status,'complete25':full,'actual_counts':counts,'completed_models':len(runs),'completed_experiments':sum(all((e,m) in run_order for m in NAMES) for e in range(5)),'formal_status':summary.get('status'),'non_interference_pass':good,'overall_numerical_stability':'UNRESOLVED','source_faithful_Electricity_outcome':'COMPLETED WITH LIMITATIONS' if full else 'PARTIAL / NOT ESTABLISHED','stop':stop,'no_relaunch_or_PM_Exchange_TrackB_execution':True})
 report(summary,runs,epochs,groups,cp,check,status,good,stop)
 package(artifacts)
def report(summary,runs,epochs,groups,cp,check,status,good,stop):
 data=json.loads((D/'dataset_identity.json').read_text());counts=summary.get('counts',{})
 md=['# Stage 3B Run 1 — Electricity Final Execution Report','',f'执行建议：{status}。等待Research Analyst Electricity Review，不代表Planner已最终验收。','',
 '## 1. Source / Data / Environment Fidelity','',
 'Branchtrack-a-source-faithful；FrozenHEAD95ff20de5a2dea627c854e35a00f304c838fff12；20tracked文件与GitBlob二进制一致；工作区/IndexClean。Python3.9.6、Torch2.0.0、NumPy1.23.5、Pandas1.5.3、CPUarm64、31项Lock匹配，无依赖或源码修改。']
 for r in data['files']:md.append(r['path']+'；'+str(r['bytes'])+'bytes；SHA-256 '+r['sha256'])
 md+=['','915日窗口，16特征顺序与原X.columns[2:]一致；Date/idx按原逻辑排除，Y仅取target。原始Consumption第2..23位置与22个Target逐字节数值对齐；Y保留未用的Unnamed索引列，未改CSV。','',
 '## 2. Windows Path Compatibility Mapping','',
 '外部AST仅把Electricity.py第36行data_path的Windows字面量定向到TrackA/DATA/Electricity。反向恢复后全部AST节点与冻结源码一致；原文件/行号/训练函数不变，没有全局pd.read_csv替换。完整Diff、映射和Hash见对应文件。结果CLI目录仅重定向至独立Run01。','',
 '## 3. Observer Regression Tests','',
 '最终20项无作者模型回归PASS，含真实虚拟Scheduler350次finite指标记录、合法mode=min/mode_worse=+inf可逆类型标记、LiveState未改、未知NaN/Inf/best/LR拒绝、日志失败计数不虚增和10批/350Epoch预算。虚拟Optimizer没有step；不导入或运行作者模型。全局严格JSON，无allow_nan=True或default=str。','',
 '## 4. CPU Resource and Process Lifecycle','',
 'resource_feasibility.json区分Measured/Derived/Assumed/Unknown，不把Exchange耗时当Electricity速度。独立无模型进程已验证启动器退出后继续记录，以及实际0/7退出码；原子O_EXCL锁拒绝重复创建。正式Run保持永久防重Lock，记录实际PID/PPID/PGID/SID、命令、cwd、环境、时间、stdout/stderr/exit。没有自动重启或平台限制绕过；聊天中断不是模型失败依据。']
 if epochs:md.append('已记录最大进程RSS：'+str(max(r['process_peak_rss_bytes'] for r in epochs))+'bytes；Epoch wall累计：'+str(sum(r['wall_seconds_including_observer'] for r in epochs))+'秒。')
 md+=['','## 5. Actual Experiment / Model / Epoch / Step Counts','',
 f'完整模型{len(runs)}/25；已记录Epoch{len(epochs)}/8750；实际计数：'+json.dumps(counts,ensure_ascii=False),
 '固定模型顺序Delelstm→IMV_full→IMV_tensor→Retain→LSTM，外层Exp0..4。每模型350Epoch，不独立reseed，不加载任何旧数据集/Pilot/历史训练状态。','',
 '## 6. Per-model Training and Validation Metrics','',
 'epoch_metrics.jsonl保存每Epoch原平均BatchMSE及平方根RMSE、ValidationMSE/RMSE/MAE、LR、SchedulerState、资源、最佳选择。原数据Train/Val/Test686/137/92；每轮10/2/1批，实际640/128/64，drop_last排除46/9/28。完整历史见作者CSV和原始JSONL。','',
 '## 7. Best Checkpoint Selection','',
 '严格min_val_loss > mse_val**0.5，初值1e6；按原位置scheduler.step(mse_val)，Adam(lr=.01,weight_decay=.001)及其他默认值保持。仅350Epoch完整结束后加载本模型最佳Checkpoint，再执行原Test。Checkpoint事件、Hash/Size、参数身份和加载记录见Ledger/Inventory。未以Test选模；部分Checkpoint不冒充完整350Epoch最佳。','',
 '## 8. Original Test RMSE / MAE / MAPE','',
 '|Exp|Model|Epochs|BestEpoch|BestValRMSE|TestRMSE|TestMAE|TestMAPE rawratio|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in runs:
  b=r['best_checkpoint'];t=r['test_metrics'];md.append(f"|{r['experiment']['exp_id']}|{r['experiment']['model_name']}|{r['counts']['epochs']}|{b['epoch']}|{b['validation_rmse']:.12g}|{t['rmse']:.12g}|{t['mae']:.12g}|{t['mape']:.12g}|")
 if not runs:md.append('|—|NOT COMPLETED / Test unavailable|—|—|—|—|—|—|')
 md+=['','## 9. Five-repeat Mean ± Sample SD','',
 '以下n仅计完整成功模型；sampleSD为ddof1，populationSDddof0另在JSON保存；五次为连续RNG重复，不是独立Seed。MAPE百分比=rawratio×100。','',
 '|Model|n|TestRMSE Mean±SD|TestMAE Mean±SD|TestMAPE % Mean±SD|','|---|---:|---:|---:|---:|']
 for m in NAMES:
  g=groups[m];n=g['successful_repetitions']
  def fmt(k,scale=1):
   x=g['test_'+k];return 'NOT RUN' if x['mean'] is None else ('%.8g'% (x['mean']*scale))+' ± '+('%.8g'%(x['sample_SD_ddof1']*scale) if x['sample_SD_ddof1'] is not None else 'UNDEFINED n<2')
  md.append('|%s|%d|%s|%s|%s|'%(m,n,fmt('rmse'),fmt('mae'),fmt('mape',100)))
 md+=['','## 10. Numerical and Solver Observations','',json.dumps(check,ensure_ascii=False),
 '动态逐参数检查，不将Delelstm26参数规则套给其他模型。仅source requires_gradFalse允许合法None；未识别到合法未参与的trainable路径，异常None记录后STOP。Solver采样只附着自然Forward：Delelstm每25Epoch首个Train/ValBatch与Test首批。无额外Forward、Solver替换、Ridge、float64求解或Clipping。合法Scheduler元数据不等于科研Tensor非有限。OverallNumericalStability仍UNRESOLVED。','',
 '## 11. Source-vs-Paper Discrepancies','',
 '源码发布915窗口vs补充914；Electricity350Epochvs补充统一300；sourceSplit75/15/remainder10vs补充75/15/15；IMV_full16、RetainEmbedding256/Attention128必须保留；Target长度T-2=22，补充公式文字T-1需报告。缩放/drop_last/WD在补充未全部明确，不能宣称Paper-exact。CPU与V100s不同。详见source_vs_paper.json。','',
 '## 12. Leakage and Evaluation Limitations','',
 '全数据MinMax含heldout信息；随机日窗口划分不是时间外推。实际Electricity按每日24小时分组，21960时间戳唯一，未观察跨窗口共享时间戳，不能照搬Exchange滑窗重叠结论；时间相邻/自相关与随机日期划分局限仍保留。drop_last丢弃样本已逐项记录。RawTarget零值数为'+str(data['raw_target_zero_count'])+'，原MAPE近零/零Target的epsilon行为不修改。','',
 '## 13. Track A/B Non-interference','',f'观察窗口核验PASS={good}；完整Pre/Post/Hash、Index、工作区、TrackB30untracked、ignored和全部文件记录均保存。历史Stage3A/Pilot等科学Evidence保持；Git非关键Metadata逐项分类，writer/causeUNVERIFIED，关键refs/index/config无豁免。ScopedDS_Store仅普通真实路径非科学文件，ACL/xattrs/ctime不在本次文件快照范围。','',
 '## 14. Evidence and Checkpoint Inventory','',
 'stage3b_electricity_evidence.zip包含本Run证据、源码/Diff、无模型回归、进程Probe、命令/PID/exit、全部日志与作者已生成结果；external_artifact_inventory.json记录外部真实路径/Size/SHA256。原始DATA CSV、虚拟环境和历史Checkpoint不复制。Manifest/包后Receipt避免自引用；原子RunLock保留。','',
 '## 15. Final Electricity Recommendation','',f'ElectricityRun01：{status}。Source-faithful Electricity：'+('COMPLETED WITH LIMITATIONS' if status=='PASS' else 'PARTIAL / NOT ESTABLISHED')+'；Overall Numerical Stability：UNRESOLVED。','Stop/异常：'+(json.dumps(stop,ensure_ascii=False) if stop else '没有记录的模型Stop；结合实际完整性/Exit/计数确认。'),
 '本任务到此停止，不运行PM2.5、Exchange或TrackB，不自动追加新尝试。等待Research Analyst Electricity Review。']
 with (D/'stage3b_electricity_final_report.md').open('x') as f:f.write('\n'.join(md)+'\n')
def package(artifacts):
 excluded={'evidence_manifest.json','stage3b_electricity_evidence.zip','package_verification.json','postflight.stdout','postflight.stderr','supervisor.stdout','supervisor.stderr','outer.stdout','outer.stderr'}
 files=[]
 for p in sorted(D.rglob('*')):
  if p.is_symlink():raise RuntimeError('Evidence symlink')
  if not p.is_file() or p.name in excluded or p.relative_to(D).parts[0]=='cache':continue
  if scoped_name(str(p.relative_to(R))):validate(str(p.relative_to(R)));continue
  files.append({'archive_path':'stage3b_electricity/'+str(p.relative_to(D)),'path':str(p),'bytes':p.stat().st_size,'sha256':h(p),'role':'scientific_evidence'})
 for row in artifacts:files.append({'archive_path':'stage3b_electricity/results/'+row['relative_path'],'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'role':row['kind']})
 save('evidence_manifest.json',{'files':files,'file_count':len(files),'author_artifacts':len(artifacts),'exclusions':'Self/ZIP/postpackagingreceipts andactiveWrapper/finalizerstreams avoid recursive seals; authorizednon-scientificcache excluded;ScopedDS_Store registered separately. No inputDATA copied.'})
 zp=D/'stage3b_electricity_evidence.zip'
 with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for r in files:z.write(r['path'],r['archive_path'])
  z.write(D/'evidence_manifest.json','stage3b_electricity/evidence_manifest.json')
 with zipfile.ZipFile(zp) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)+1
  for r in files:
   hh=hashlib.sha256();sz=0
   with z.open(r['archive_path']) as f:
    for b in iter(lambda:f.read(1048576),b''):hh.update(b);sz+=len(b)
   assert sz==r['bytes'] and hh.hexdigest()==r['sha256']
 save('package_verification.json',{'path':str(zp),'bytes':zp.stat().st_size,'sha256':h(zp),'manifest_sha256':h(D/'evidence_manifest.json'),'members':len(files)+1,'all_CRC_size_SHA256_pass':True,'no_new_model_computation':True})
 print(json.dumps(json.loads((D/'package_verification.json').read_text())),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:save('postflight_packaging_error.json',{'traceback':traceback.format_exc(),'no_model_restart':True});raise
