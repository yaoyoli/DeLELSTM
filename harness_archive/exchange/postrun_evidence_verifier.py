"""Post-run independent evidence verification. Standard library only; no model import/load."""
import json,math,hashlib,zipfile,csv,struct,statistics,re,collections,gzip,sys,time
from pathlib import Path
D=Path(__file__).parent;R=Path('/Users/yao/DeLELSTM-track-a-runtime');OUT=R/'results/stage3a/attempt-02'
NAMES=['Delelstm','Retain','IMV_full','IMV_tensor','LSTM'];ORDER=[(e,m) for e in range(5) for m in NAMES]
def get(n):return json.loads((D/n).read_text())
def lines(n):
 with (D/n).open() as f:
  for i,l in enumerate(f,1):
   if l.strip():yield i,json.loads(l)
def hf(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):x.update(b)
 return x.hexdigest()
def pair(r):return (r['experiment']['exp_id'],r['experiment']['model_name'])
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def check(c,m):
 if not c:raise AssertionError(m)
def main():
 summary=get('formal_execution_summary.json')
 if summary['status']!='COMPLETED':
  return {'status':'FULL-RUN VERIFICATION NOT RUN','reason':'Formal replacement did not complete; raw stop/final reports govern. No retry.','actual':summary}
 expected={'model_initializations':25,'optimizer_creations':25,'training_batches':330000,'validation_batches':60000,'test_batches':125,'forwards':390125,'losses':330000,'backwards':330000,'optimizer_step_attempts':330000,'optimizer_step_successes':330000,'epochs_completed':7500,'models_completed':25}
 check(summary['counts']==expected,'Actual cumulative counters')
 runs=[r for _,r in lines('model_results.jsonl')];check([pair(r) for r in runs]==ORDER,'25 run order')
 ep=collections.defaultdict(list)
 for _,r in lines('epoch_metrics.jsonl'):ep[pair(r)].append(r)
 marker={'__float_sentinel__':'+Infinity','field':'mode_worse','source':'ReduceLROnPlateau'}
 selections=collections.defaultdict(list)
 for _,r in lines('checkpoint_selection.jsonl'):selections[pair(r)].append(r)
 loads={pair(r):r for _,r in lines('checkpoint_load.jsonl')};check(len(loads)==25,'25 best checkpoint loads')
 initial={};valid_epoch_count=0;expected_cp_events=0
 for key in ORDER:
  e,m=key;rows=ep[key]
  check([r['epoch'] for r in rows]==list(range(300)),'300 epoch sequence '+str(key))
  pars=get('exp%d_%s_initialization.json'%key)['parameters']
  initial[key]={p['name']:p for p in pars}
  wanted=[];best=1000000.;previous_lr=.01
  for r in rows:
   check(r['train_batches']==44 and r['val_batches']==8 and r['train_used_windows']==5632 and r['val_used_windows']==1024,'Per-epoch loader counts')
   check(all(finite(r[k]) for k in ['train_mse','train_rmse','val_mse','val_rmse','val_mae','learning_rate_before_scheduler','learning_rate_after_scheduler']),'Finite epoch metrics/LR')
   s=r['scheduler_state']
   check(s['mode']=='min' and s['mode_worse']==marker and finite(s['best']),'Only legal scheduler sentinel')
   check(s['last_epoch']==r['epoch']+1,'Exactly one scheduler step per epoch')
   check(r['learning_rate_before_scheduler']==previous_lr and 1e-4<=r['learning_rate_after_scheduler']<=previous_lr,'Scheduler LR continuity')
   previous_lr=r['learning_rate_after_scheduler']
   if r['val_rmse']<best:best=r['val_rmse'];wanted.append(r['epoch'])
  check([r['epoch'] for r in selections[key]]==wanted,'Exact strict-RMSE checkpoint events '+str(key))
  rr=next(r for r in runs if pair(r)==key)
  check(rr['counts']=={'train_batches':13200,'val_batches':2400,'test_batches':5,'epochs':300,'backwards':13200,'step_attempts':13200,'step_successes':13200},'Model run counts')
  check(rr['best_checkpoint']['validation_rmse']==best and rr['best_checkpoint']['epoch']==wanted[-1],'Final best validation choice')
  check(loads[key]['pass'] and loads[key]['checkpoint']['sha256']==rr['best_checkpoint']['sha256'],'Loaded best checkpoint identity')
  cp=Path(rr['best_checkpoint']['path']).resolve(strict=True);cp.relative_to(OUT.resolve(strict=True))
  check(hf(cp)==rr['best_checkpoint']['sha256'] and cp.stat().st_size==rr['best_checkpoint']['bytes'],'Current checkpoint bytes')
  valid_epoch_count+=len(rows);expected_cp_events+=len(wanted)
 # Stream all large per-update logs; never hold tensor/log arrays for all steps in memory.
 grads=collections.Counter();train_count=0;max_coeff=0.;max_grad=0.;bjmax=0.
 for n,r in lines('training_batch_monitor.jsonl'):
  i=n-1;key=ORDER[i//13200];local=i%13200+1
  check(pair(r)==key and r['global_train_batch']==n and r['epoch']==(local-1)//44 and r['train_batch']==(local-1)%44+1,'Training update sequence')
  check(r['step_success'] and r['post_parameter_finite'] and r['post_Adam_state_finite'] and finite(r['loss']),'Update numeric gate')
  check(r['Adam_step_values']==[float(local)]*len(initial[key]),'Actual Adam state step values')
  check({g['name'] for g in r['gradients']}==set(initial[key]) and len(r['gradients'])==len(initial[key]),'Complete parameter gradient names')
  for g in r['gradients']:
   p=initial[key][g['name']]
   check(g['present'] and g['finite'] and g['nan']==g['positive_inf']==g['negative_inf']==0,'Gradient finite/present')
   check(g['shape']==p['shape'] and g['dtype']==p['dtype'] and g['device']==p['device'],'Gradient shape/dtype/device')
   check(re.fullmatch('[0-9a-f]{64}',g['sha256']) and finite(g['l2']) and finite(g['max_abs']),'Gradient digest/statistics')
  if key[1]=='Delelstm':
   check(r['B_j'] is not None and finite(r['B_j']['l2']) and 0<=r['B_j']['squared_norm_contribution']<=1,'B_j diagnostics')
   bjmax=max(bjmax,r['B_j']['l2'])
  grads[key[1]]+=len(r['gradients']);max_grad=max(max_grad,r['global_gradient_l2']);train_count+=1
 check(train_count==330000,'330000 stored successful updates')
 phases=collections.Counter();forward_count=0;delelstm_forwards=0
 for n,r in lines('forward_monitor.jsonl'):
  idx=n-1;key=ORDER[idx//15605];within=idx%15605
  phase='test' if within>=15600 else 'train' if within%52<44 else 'val'
  batch=(within-15600+1) if phase=='test' else within%52+1 if phase=='train' else within%52-43
  epoch=299 if phase=='test' else within//52
  check(pair(r)==key and r['phase']==phase and r['batch']==batch and r['epoch']==epoch,'Natural forward order/count')
  check(r['prediction']['finite'] and r['prediction']['shape']==[128,28,1] and r['requires_grad']==(phase=='train'),'Prediction finite/shape/autograd')
  for name in ['coefficients_or_alpha','third_or_beta']:
   x=r[name]
   if x is not None:
    check(x['finite'] and x['nan']==x['positive_inf']==x['negative_inf']==0,'Coefficient/explanation finite')
    if name=='coefficients_or_alpha' and key[1]=='Delelstm':max_coeff=max(max_coeff,x['max_abs'])
  if key[1]=='Delelstm':delelstm_forwards+=1
  phases[phase]+=1;forward_count+=1
 check(forward_count==390125 and phases=={'train':330000,'val':60000,'test':125},'Forward coverage')
 solver=list(r for _,r in lines('solver_samples.jsonl'));check(len(solver)==125,'Fixed solver sample count')
 for r in solver:
  check(pair(r)[1]=='Delelstm' and r['batch']==1 and (r['phase']=='test' or r['epoch']%25==0),'Fixed nonselective solver coverage')
 totals={k:sum(r['solver'][k] for r in solver) for k in ['high_level_inv','internal_inv_ex','pinv']}
 # Verify saved test outcomes independently as raw IEEE float32 bytes, not new evaluations.
 outcome=[];targets=set();csv_metric_diffs=[]
 for r in runs:
  e,m=pair(r);folder=Path(r['outdir']);true_h=hashlib.sha256();pred_h=hashlib.sha256();used=0
  with (folder/(str(e)+'_Explain_outcome.csv')).open(newline='') as f:
   reader=csv.reader(f);header=next(reader);check(len(header)==57,'Original true+prediction CSV width')
   for values in reader:
    check(len(values)==57,'Outcome row width')
    vv=[float(x) for x in values[1:]];check(all(math.isfinite(x) for x in vv),'Outcome finite')
    true_h.update(struct.pack('<28f',*vv[:28]));pred_h.update(struct.pack('<28f',*vv[28:]));used+=1
  check(used==640 and true_h.hexdigest()==r['target_sha256'] and pred_h.hexdigest()==r['prediction_sha256'],'Saved CSV matches actual test tensor hashes')
  targets.add(true_h.hexdigest());outcome.append({'experiment':r['experiment'],'rows':used,'true_and_prediction_float32_hash_match':True})
  with (folder/(str(e)+'_Explain_test_results.csv')).open(newline='') as f:cr=next(csv.DictReader(f))
  for col,k in [('mse_test','mse'),('mae_test','mae'),('mape_test','mape'),('rmse_test','rmse')]:
   v=float(cr[col]);ref=r['test_metrics'][k];same=(v==ref)
   semantic_same=same or (k!='rmse' and struct.pack('<f',v)==struct.pack('<f',ref))
   check(semantic_same,'Saved test metric value mismatch '+str((e,m,k,v,ref)))
   if not same:csv_metric_diffs.append({'experiment':r['experiment'],'metric':k,'CSV':v,'actual_tensor_metric':ref,'IEEE_float32_identical':True,'role':'Formatting roundtrip, not a stability tolerance'})
 check(len(targets)==1,'Same fixed raw test targets across models')
 agg=get('experiment_aggregation.json')['models']
 for m in NAMES:
  rr=[r for r in runs if pair(r)[1]==m];check(len(rr)==agg[m]['successful_repetitions']==5,'Five completed repetitions')
  for k in ['rmse','mae','mape']:
   v=[r['test_metrics'][k] for r in rr];a=agg[m]['test_'+k]
   check(a['values']==v and a['mean']==statistics.mean(v) and a['sample_SD_ddof1']==statistics.stdev(v) and a['population_SD_ddof0']==statistics.pstdev(v),'Aggregation exact')
 check(get('track_a_b_non_interference.json')['pass'],'Postflight non-interference')
 check(get('epoch0_reproducibility_comparison.json')['epoch_record_written'],'Original bug regression in natural epoch0')
 for n in ['first_batch_reproducibility_comparison.json','initial_model_reproducibility_comparison.json','split_reproducibility_comparison.json']:get(n)
 # All scientific files and archive seal are checked after the supervised package is closed.
 manifest=get('evidence_manifest.json')
 for row in manifest['files']:
  p=D/row['path'];check(p.stat().st_size==row['bytes'] and hf(p)==row['sha256'],'Evidence manifest file '+row['path'])
 receipt=get('package_verification.json');zp=Path(receipt['path'])
 check(hf(zp)==receipt['sha256'] and zp.stat().st_size==receipt['bytes'],'Final ZIP seal')
 with zipfile.ZipFile(zp) as z:
  check(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==receipt['members'],'ZIP CRC/unique count')
 return {'pass':True,'counts':expected,'epochs_verified':valid_epoch_count,'checkpoint_selection_events_verified':expected_cp_events,'all_gradient_tensors_present_finite':True,'gradient_records_by_model':dict(grads),'all_training_updates_stream_verified':train_count,'all_natural_forwards_verified':forward_count,'phases':dict(phases),'max_Delelstm_coefficient_max_abs':max_coeff,'max_global_gradient_l2_all_models':max_grad,'max_Delelstm_B_j_gradient_l2':bjmax,'solver_profiled_samples':len(solver),'Delelstm_natural_forwards':delelstm_forwards,'solver_sample_fraction':len(solver)/delelstm_forwards,'solver_event_totals':totals,'unsampled_solver_paths':'UNVERIFIED; no inv-only claim','25_saved_test_outcome_float32_hash_checks':outcome,'CSV_metric_float32_formatting_differences':csv_metric_diffs,'all5_repetition_aggregations_exact':True,'postflight_non_interference_pass':True,'archive_size':receipt['bytes'],'archive_sha256':receipt['sha256'],'author_model_import_or_forward_in_this_verifier':False,'overall_numerical_stability':'UNRESOLVED'}
if __name__=='__main__':
 import traceback
 started=time.time()
 try:r=main();r['wall_seconds']=time.time()-started
 except BaseException:r={'pass':False,'traceback':traceback.format_exc(),'no_model_executed':True,'no_data_or_source_repaired':True}
 with (D/'postpackage_independent_verification.json').open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(r,ensure_ascii=False),flush=True)
 if r.get('pass') is False:sys.exit(1)
