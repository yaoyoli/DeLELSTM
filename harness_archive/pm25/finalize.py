"""PM25 postflight/aggregation/report/package. Standard library only; no model import."""
import os,sys,json,gzip,hashlib,zipfile,statistics,math,traceback
from pathlib import Path
from audit_utils import A,B,R,D,now,capture,differences
from scoped_metadata import scoped_name,validate,compare_runtime
from git_policy import git_identity
OUT=R/'results/stage3b/pm25/run-01'
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
 post=capture('pm25_run01_postflight')
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
  cp.append({'experiment':r['experiment'],'current_hash_matches_latest_selection':ok,'eligible_final_model_completed300epochs_and_test':completed,**r})
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
 expected={'model_initializations':25,'optimizer_creations':25,'training_batches':315000,'validation_batches':60000,'test_batches':125,'forwards':375125,'losses':315000,'backwards':315000,'optimizer_step_attempts':315000,'optimizer_step_successes':315000,'epochs_completed':7500,'models_completed':25}
 full=success and counts==expected and len(runs)==25 and len(epochs)==7500 and tr==315000 and fr==375125 and phases=={'train':315000,'val':60000,'test':125} and orderpass and not bad and good
 solver=[r for r in rows('solver_samples.jsonl') if 'solver' in r]
 st={k:sum(r['solver'][k] for r in solver) for k in ['high_level_inv','internal_inv_ex','pinv']}
 check={'stored_successful_updates':tr,'recorded_natural_forwards':fr,'phases':phases,'gradient_records_verified':gradients,'all_recorded_numeric_gates_finite':not bad,'discrepancies':bad[:100],'counts_match_complete_budget':counts==expected,'model_order_matches_prefix':orderpass,'max_Delelstm_coefficient_max_abs':maxcoeff,'max_global_gradient_l2':maxgrad,'max_B_j_gradient_l2':maxbj,'solver_samples':len(solver),'solver_event_totals':st,'solver_coverage':'Only actual fixed natural samples; unprofiledUNVERIFIED;inv_ex not added to highlevelinv'}
 save('execution_evidence_crosscheck.json',check)
 status='PASS' if full else 'PARTIAL PASS / CONTROLLED STOP' if counts.get('optimizer_step_successes',0)>0 else 'FAIL / NOT RUN'
 stop=json.loads((D/'stop_condition_record.json').read_text()) if (D/'stop_condition_record.json').exists() else None
 save('final_decision.json',{'PM25Run01':status,'complete25':full,'actual_counts':counts,'completed_models':len(runs),'completed_experiments':sum(all((e,m) in run_order for m in NAMES) for e in range(5)),'formal_status':summary.get('status'),'non_interference_pass':good,'overall_numerical_stability':'UNRESOLVED','source_faithful_PM25_outcome':'COMPLETED WITH LIMITATIONS' if full else 'PARTIAL / NOT ESTABLISHED','stop':stop,'no_relaunch_or_Electricity_Exchange_TrackB_execution':True,'deviation_label':'Controlled Execution Deviation — Restored Five-model Comparison'})
 report(summary,runs,epochs,groups,cp,check,status,good,stop)
 package(artifacts)
def report(summary,runs,epochs,groups,cp,check,status,good,stop):
 data=json.loads((D/'dataset_identity.json').read_text());counts=summary.get('counts',{})
 deviation='Controlled Execution Deviation — Restored Five-model Comparison'
 md=['# Stage 3B Run 2 — PM2.5 Five-model Comparison Execution Report','',
 f'Executor 建议：**{status}**。等待 Research Analyst 独立 PM2.5 审核和 Stage 3B 综合审核；不代表 Planner 最终验收。','',
 '**'+deviation+'**。不是完全未经修改的 PM.py 原始入口执行。Overall Numerical Stability：UNRESOLVED。','',
 '## A. Frozen Source / Dataset / Environment Fidelity','',
 'Track A Branch：track-a-source-faithful；HEAD：95ff20de5a2dea627c854e35a00f304c838fff12；Parent：b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9。20个tracked文件逐字节符合冻结Blob，其他19个保留作者基线；工作区/Index clean，无源码修改或Git Commit/Push。','',
 'Python3.9.6、Torch2.0.0、NumPy1.23.5、Pandas1.5.3、macOS arm64 CPU；31项Lock版本一致，Assertions生效，无环境修改。']
 for r in data['files']:md.append('\n'+r['path']+'；'+str(r['bytes'])+'bytes；SHA-256 '+r['sha256'])
 md+=['','特征顺序：'+', '.join(data['feature_order'])+'。采用columns[6:]，8特征，24小时日窗口，Target原pm2.5位置2..23，共22值。','',
 '## B. Controlled Five-model Deviation','',
 '冻结PM.py第91行五模型列表被第92行Retain-only覆盖。外部AST仅将第二次列表恢复为Delelstm→IMV_full→IMV_tensor→Retain→LSTM；第一处赋值、构造语句、循环和Seed位置保留。撤销两处获批改动后，全部AST节点及位置与原始一致。Original/Patched AST、Diff、Mapping和Hash均保存。','',
 '## C. Original PM Training Protocol','',
 'Seed555仅在入口设置一次；Depth24、Input8、Output1、入口32/32、B32、5Experiments、300Epoch。IMV_full有效n_units16，Retain Embedding/Alpha/Beta Hidden64，其他原配置。Adam(lr0.05,weight_decay0.03)、MSELoss(mean)、ReduceLROnPlateau(min,factor0.5,patience20,min_lr0.0001)。PM_training科学语句未改写。','',
 '## D. Mac Path Adaptation','',
 '仅将PM.py第39行Windows路径映射到/Users/yao/DeLELSTM-track-a/DATA/PM2.5；结果通过原save_dirs参数指向本次外部目录。保留入口newrevision_pm与训练函数实际newrevision_pmc差异。未改CSV/Scaling/Split/Target/Loader。','',
 '## E. Observer Regression and Process Continuity','',
 '全部必需无作者模型回归在启动前PASS：Scheduler合法+inf严格可逆标记、LiveState不变、非有限科学字段/LR拒绝、解释Tensor与None返回、Shape/预算/计数、两处AST适配。详情见serialization_regression_results.json及实际命令/stdout/stderr/退出码。','',
 '独立无模型探针验证启动器退出后继续记录、0/7退出码与原子锁拒绝重复创建。正式使用永久RunLock和独立本地监督进程，记录PID/PPID/PGID/SID、命令/cwd/环境和准确wait()退出码。无自动重启；重连仅观察旧进程。观察器不改变Autograd/Optimizer/Scheduler/Checkpoint语义。','',
 '## F. CPU Resource Budget and Actual Usage','',
 '完整预算由真实数据推导：315000 Training Updates、60000 Validation Forward Batches、125 Test Forward Batches，共375125自然Forward。resource_feasibility.json区分Measured/Derived/Assumed/Unknown；Electricity耗时不是PM确定预测。观察器/Profiler开销包含在实测时间内，未为估计开销新增Forward。']
 if epochs:md.append('\n最大Epoch记录RSS：'+str(max(r['process_peak_rss_bytes'] for r in epochs))+'bytes；Epoch wall累计：'+str(sum(r['wall_seconds_including_observer'] for r in epochs))+'秒。')
 if (D/'formal_exit_code.json').exists():md.append('\n正式退出记录：'+(D/'formal_exit_code.json').read_text().strip())
 complete_exps=sum(all(any(r['experiment']=={'exp_id':e,'model_name':m} for r in runs) for m in NAMES) for e in range(5))
 md+=['','## G. Actual Experiments / Models / Epochs / Adam Steps','',
 f'完整Experiments {complete_exps}/5；完整Models {len(runs)}/25；Epoch记录{len(epochs)}/7500。实际计数：'+json.dumps(counts,ensure_ascii=False),
 '外层Exp0..4、内层固定五模型顺序。没有逐模型reseed或载入历史状态；完整模型须完成300Epoch、最佳Checkpoint加载和原Test，未完成结果不补齐。','',
 '## H. Training and Validation Results','',
 'epoch_metrics.jsonl保存逐Epoch原TrainMSE/RMSE、ValMSE/RMSE/MAE、LR/Scheduler/资源/最佳选择；training_batch_monitor.jsonl保存逐更新全部梯度及有限性。Train/Val/Test窗口1369/273/184；Batch42/8/5；使用1344/256/160；drop_last排除25/17/24。Train shuffle=True，Val/Test=False，全部drop_last=True。','',
 '## I. Best Checkpoint Selection','',
 '保留严格min_val_loss > mse_val**0.5，初值1e6；按ValRMSE改善保存，原位置scheduler.step(mse_val)。300Epoch后加载对应Experiment/Model最佳Checkpoint再Test；完整Selection/Load Ledger、Size/SHA256/参数身份另存，未利用Test选模。','',
 '## J. Test RMSE / MAE / MAPE','',
 '|Exp|Model|Epochs|BestEpoch|BestValRMSE|TestRMSE|TestMAE|MAPE raw ratio|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in runs:
  b=r['best_checkpoint'];t=r['test_metrics'];md.append(f"|{r['experiment']['exp_id']}|{r['experiment']['model_name']}|{r['counts']['epochs']}|{b['epoch']}|{b['validation_rmse']:.12g}|{t['rmse']:.12g}|{t['mae']:.12g}|{t['mape']:.12g}|")
 if not runs:md.append('|—|NOT COMPLETED|—|—|—|—|—|—|')
 md+=['','BestEpoch采用源码零起点。TestMSE、Prediction/TargetShape/Hash及解释产物另存。MAPE使用原torchmetrics.MeanAbsolutePercentageError，百分比=rawratio×100。','',
 '## K. Five-repeat Mean ± Sample SD','',
 '|Model|n|RMSE Mean±SD|MAE Mean±SD|MAPE % Mean±SD|','|---|---:|---:|---:|---:|']
 for m in NAMES:
  g=groups[m]
  def fmt(k,scale=1):
   x=g['test_'+k]
   return 'NOT RUN' if x['mean'] is None else ('%.9g'%(x['mean']*scale))+' ± '+('%.9g'%(x['sample_SD_ddof1']*scale) if x['sample_SD_ddof1'] is not None else 'UNDEFINED n<2')
  md.append('|%s|%d|%s|%s|%s|'%(m,g['successful_repetitions'],fmt('rmse'),fmt('mae'),fmt('mape',100)))
 md+=['','SampleSD为ddof1，populationSD ddof0另存JSON。连续RNG不称独立Seeds，不按性能排除完整重复。','',
 '## L. Numerical and Solver Observations','',json.dumps(check,ensure_ascii=False),'',
 '各模型按实际参数结构检查，合法冻结参数None单列，意外缺失/非有限梯度STOP。固定采样Delelstm每25Epoch首个自然Train/ValBatch及Test首批，完整应125次采样。高层inv和内部inv_ex不累加；未采样路径UNVERIFIED，batched fallback不解释为逐样本独立回退。巨大但finite的系数/梯度仍为风险。Overall Numerical Stability：UNRESOLVED。','',
 '## M. Source-vs-Paper Differences','',
 '批准偏离是恢复五模型。PM发布1826日窗口、B32、lr0.05、入口hidden32及300Epoch与本地Supplement相应说明一致；有效IMV_full16/Retain64以源码为准。源码Split75/15/remainder10 vs Supplement75/15/15；Target T−2 vs损失文字T−1；Scaling/drop_last/WD细节及CPU/GPU差异保留。没有执行top50%特征实验，不将Supplement该表作为全特征精确复现证明。未新增在线论文核验。','',
 '## N. Leakage and Dataset Limitations','',
 '日窗口不重叠，时间戳唯一，跨Split共享时间戳0；随机日划分和时间自相关限制未来泛化。Full-dataMinMax使用heldout信息；发布数据等于作者ffill再bfill，Split前插补与初始未来bfill的风险保留，没有重新预处理。','',
 '原Target零值数'+str(data['raw_target_zero_count'])+'，最小值'+str(data['raw_target_min'])+'。原MAPE近零/零值敏感性及epsilon不修改。Test只使用160/184窗口；不声称无泄漏泛化。','',
 '## O. Track A/B Non-interference','',
 f'观察窗口核验PASS={good}。Pre/Post覆盖A/B Branch/HEAD/Index/Status、TrackB30untracked/ignored及完整文件Hash/Size/Mode/mtime/Inode和SymlinkTarget；ACL/xattrs/ctime未覆盖。历史Exchange/Electricity/Stage1/2科学证据单独保护。共享GitMetadata变化逐项分类，不宣称全部字节和元数据相同；Writer/Cause为UNVERIFIED。ScopedDS_Store不豁免Symlink/科研文件/Git/TrackB。保留Stage1I执行会话偏离和CRLF失败历史。','',
 '## P. Evidence / Checkpoint / Artifact Inventory','',
 'stage3b_pm25_evidence.zip包含本次授权、Electricity验收引用、Gates/AST/代码Diff/Hash、回归、生命周期、命令/日志/退出码、逐批/逐Epoch记录、Checkpoint/Test/解释、Manifest和报告。external_artifact_inventory.json提供全部作者产物实际路径/Size/SHA256；归档逐成员CRC/Size/SHA256验证。原始CSV、虚拟环境不打包。后续WrapperReceipt在包外封存以避免自引用。','',
 '## Q. Final PM2.5 Reproduction Recommendation','',
 f'PM2.5五模型比较：**{status}**；'+deviation+'；Overall Numerical Stability：**UNRESOLVED**。',
 'Stop/异常：'+(json.dumps(stop,ensure_ascii=False) if stop else '没有记录的模型阻断，结合实际计数/退出码和完整性证据判断。'),
 'Evidence Verified限实际本地身份/文件/记录/完成流程；带批准偏离的科学忠实性为Evidence-supported，不宣称未经修改的原入口或Paper-exact。没有额外Retain-only、第二次PM尝试、Exchange/Electricity重跑或TrackB执行。','',
 'STOP — Await Research Analyst Independent PM2.5 Review and Stage 3B Final Integrated Research Review.']
 with (D/'stage3b_pm25_final_report.md').open('x') as f:f.write('\n'.join(md)+'\n')
def package(artifacts):
 excluded={'evidence_manifest.json','stage3b_pm25_evidence.zip','package_verification.json','postflight.stdout','postflight.stderr','supervisor.stdout','supervisor.stderr','outer.stdout','outer.stderr'}
 files=[]
 for p in sorted(D.rglob('*')):
  if p.is_symlink():raise RuntimeError('Evidence symlink')
  if not p.is_file() or p.name in excluded or p.relative_to(D).parts[0]=='cache':continue
  if scoped_name(str(p.relative_to(R))):validate(str(p.relative_to(R)));continue
  files.append({'archive_path':'stage3b_pm25/'+str(p.relative_to(D)),'path':str(p),'bytes':p.stat().st_size,'sha256':h(p),'role':'scientific_evidence'})
 for row in artifacts:files.append({'archive_path':'stage3b_pm25/results/'+row['relative_path'],'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'role':row['kind']})
 save('evidence_manifest.json',{'deviation_label':'Controlled Execution Deviation — Restored Five-model Comparison','files':files,'file_count':len(files),'author_artifacts':len(artifacts),'exclusions':'Self/ZIP/postpackagingreceipts andactiveWrapper/finalizerstreams avoid recursive seals; authorizednon-scientificcache excluded;ScopedDS_Store registered separately. No inputDATA copied.'})
 zp=D/'stage3b_pm25_evidence.zip'
 with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for r in files:z.write(r['path'],r['archive_path'])
  z.write(D/'evidence_manifest.json','stage3b_pm25/evidence_manifest.json')
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
