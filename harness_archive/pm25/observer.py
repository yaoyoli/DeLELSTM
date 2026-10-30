"""PM2.5 Run01 read-only line observer: original PM_training executes unchanged.
No solver hooks, replacement functions, RNG draws, extra model calls or loader iterators.
Only torch.profiler on fixed natural Forward samples. Outputs exclusively D.
"""
import os,sys,json,hashlib,time,math,gzip,traceback,resource,contextlib
from pathlib import Path
import numpy as np
import torch
from audit_utils import A,B,R,D,sha,now,capture,differences,tree
from git_policy import git_identity
from scoped_metadata import scoped_name,validate,compare_runtime
OUT=R/'results/stage3b/pm25/run-01'
RUN_ID='stage3b-pm25-run-01'
NAMES=['Delelstm','IMV_full','IMV_tensor','Retain','LSTM']
FROZEN='95ff20de5a2dea627c854e35a00f304c838fff12'
from scheduler_metadata import strict_json,scheduler_state_record
def encode(x):return strict_json(x)
def atomic(n,x):
 p=D/n;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:f.write(encode(x)+'\n')
 os.replace(t,p)
def once(n,x):
 with (D/n).open('x') as f:f.write(encode(x)+'\n')
def append(n,x):
 with (D/n).open('a') as f:f.write(encode(x)+'\n')
def fin(t):
 d=t.detach()
 return bool(torch.isfinite(d).all().item())
def ti(t):
 d=t.detach().cpu().contiguous()
 return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'sha256':sha(d.numpy().tobytes()),'requires_grad':t.requires_grad}
def numeric(t):
 d=t.detach()
 return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'finite':fin(d),'nan':int(torch.isnan(d).sum().item()),'positive_inf':int(torch.isposinf(d).sum().item()),'negative_inf':int(torch.isneginf(d).sum().item()),'max_abs':float(d.abs().max().item()) if fin(d) else None}
def rng():
 s=np.random.get_state()
 return {'torch_cpu_sha256':sha(torch.get_rng_state().numpy().tobytes()),'numpy_sha256':sha(s[1].tobytes()+repr(s[2:]).encode())}
class ScientificStop(RuntimeError):pass
class Observer:
 def __init__(self):
  assert json.loads((D/'gateA.json').read_text())['pass']
  assert json.loads((D/'gateBC.json').read_text())['pass']
  assert json.loads((D/'harness_static_review.json').read_text())['pass']
  with gzip.open(D/'preflight_snapshot.json.gz','rt') as f:self.baseline=json.load(f)
  self.protected=json.loads((D/'historical_scientific_registry.json').read_text())['protected_absolute_paths']
  self.order=[];self.runs=[];self.epochs=[];self.checkpoints={};self.init_pending=None;self.current=None
  self.counts={'model_initializations':0,'optimizer_creations':0,'training_batches':0,'validation_batches':0,'test_batches':0,'forwards':0,'losses':0,'backwards':0,'optimizer_step_attempts':0,'optimizer_step_successes':0,'epochs_completed':0,'models_completed':0}
  self.pending_step=False;self.pending_forward=None;self.pending_scheduler=False;self.pending_save=None;self.pending_load=False;self.prof=None;self.mon=None;self.final_status='RUNNING';self.frame_exception=False
  self.source_stats={p:(A/p).stat() for p in self.baseline['files_A']['files'] if self.baseline['files_A']['files'][p]['kind']=='file'}
  self.ref=None # No Exchange/Pilot model state or sample identity is reused.
  self.started=now();self.started_monotonic=time.monotonic()
  once('execution_started.json',{'time':self.started,'pid':os.getpid(),'source':FROZEN,'rng_before_author_entry':rng(),'author_source_execution':'PM AST with approved path and second model_list changes only; Controlled Execution Deviation — Restored Five-model Comparison; original PM_training unchanged; read-only line observer','run_id':RUN_ID,'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'fixed_profiler_sampling':'Delelstm: batch1 of train+validation in epochs 0,25,...275; test batch1. No other forwards profiled. No extra forwards.'})
  self.status()
 def need(self,c,m):
  if not c:raise ScientificStop(m)
 def status(self):
  atomic('live_status.json',{'time':now(),'status':self.final_status,'current':self.current,'epoch':getattr(self,'epoch',None),'run_id':RUN_ID,'counts':self.counts,'elapsed_seconds':time.monotonic()-self.started_monotonic,'pid':os.getpid(),'readiness':'NOT ESTABLISHED','overall_numerical_stability':'UNRESOLVED'})
 def source_quick(self):
  for p,s in self.source_stats.items():
   v=(A/p).stat()
   self.need((s.st_ino,s.st_size,s.st_mtime_ns)==(v.st_ino,v.st_size,v.st_mtime_ns),'Unexpected TrackA file metadata '+p)
 def full_integrity(self,label,full=True):
  current=capture(label) if full else None
  if full:
   d=differences(self.baseline,current);states={k:self.baseline['states'][k]==current['states'][k] for k in ['A','B']}
   policy=compare_runtime(self.baseline,current,self.protected)
   git_class=git_identity(self.baseline['git_metadata'],current['git_metadata'])
   good=all(states.values()) and all(d[k]['equal'] for k in ['files_A','files_B']) and git_class['identity_pass'] and policy['scientific_integrity_pass']
   append('integrity_windows.jsonl',{'label':label,'time':now(),'pass':good,'states_equal':states,'differences':d,'runtime_policy':policy,'Git_metadata_classification':git_class,'full_snapshot_sha256':sha(encode(current).encode())})
   self.need(good,'Scientific/Git/TrackB interference '+label)
  else:
   # Epoch boundary: complete Track B inventory/stat; changed records are never exempt.
   a=self.baseline['files_B']['files'];seen={}
   def walk(p,rel=''):
    for c in p.iterdir():
     r=(rel+'/'+c.name).lstrip('/')
     if r=='.git':continue
     s=c.lstat()
     if c.is_symlink():seen[r]=(s.st_ino,s.st_size,s.st_mtime_ns,os.readlink(c))
     elif c.is_dir():walk(c,r)
     else:seen[r]=(s.st_ino,s.st_size,s.st_mtime_ns,None)
   walk(B)
   self.need(set(seen)==set(a),'TrackB path inventory changed')
   bad=[p for p,v in seen.items() if v!=(a[p]['inode'],a[p]['size'],a[p]['mtime_ns'],a[p].get('link_target'))]
   self.need(not bad,'TrackB file metadata changed '+str(bad[:20]))
   for p in self.protected:
    q=Path(p);rel=str(q.relative_to(R))
    rec=self.baseline['runtime_existing']['files'].get(rel)
    self.need(rec is not None and q.is_file() and not q.is_symlink(),'Historical file missing/symlink '+p)
    s=q.stat()
    self.need((s.st_ino,s.st_size,s.st_mtime_ns)==(rec['inode'],rec['size'],rec['mtime_ns']),'Historical scientific metadata changed '+p)
   append('epoch_integrity.jsonl',{'time':now(),'model':self.current,'epoch':getattr(self,'epoch',None),'all_TrackB_file_stat_records':len(seen),'historical_scientific_file_stats':len(self.protected),'pass':True,'content_hash_scope':'Full hashes each model start/end and epoch0/100/200/299; full snapshots before/after whole stage.'})
  self.source_quick()
 def data_ready(self,L):
   self.need([len(L[k+'_idx']) for k in ['train','val','test']]==[1369,273,184],'PM2.5 split counts')
   split={};self.ids={};self.lookup={}
   expected=json.loads((D/'expected_split_manifest.json').read_text())['split']
   for phase in ['train','val','test']:
    raw=L[phase+'_X']['idx'].to_numpy().reshape(-1,24)[:,0].astype(int).tolist()
    self.ids[phase]=raw;self.need(len(raw)==len(set(raw)),'Split window identity')
    loader=L[phase+'_loader'];xs,ys=loader.dataset.tensors
    self.need(loader.batch_size==32 and loader.drop_last and loader.num_workers==0 and len(loader)=={'train':42,'val':8,'test':5}[phase],'PM2.5 loader protocol')
    self.need(raw==expected[phase]['dataset_order_ids'] and L[phase+'_idx'].astype(int).tolist()==expected[phase]['choice_order_ids'],'Natural split differs exact source-faithful preflight')
    if phase=='train':
     for i in range(len(xs)):
      key=sha(xs[i].contiguous().numpy().tobytes());self.lookup.setdefault(key,[]).append(i)
    split[phase]={'choice_order_ids':L[phase+'_idx'].astype(int).tolist(),'dataset_order_ids':raw,'input':ti(xs),'target':ti(ys),'windows':len(raw),'batch_count':len(loader),'used_windows':len(loader)*32,'dropped_windows':len(raw)-len(loader)*32,'sampler':type(loader.sampler).__name__}
   once('training_split_manifest.json',{'seed':555,'split':split,'RNG_after_author_data':rng(),'source_preflight_identity_match':True,'no_additional_loader_iterator':True,'protocol':'Original PM normalization/split/tensor/loaders; approved model-list restoration and directed path only'})
   self.loaded_data=True
 def before_init(self,L):
  self.need(len(self.order)<25,'Model budget exceeded')
  expected=[(e,m) for e in range(5) for m in NAMES][len(self.order)]
  self.need((L['exp_id'],L['model_name'])==expected,'Model order mismatch')
  self.init_pending={'exp_id':L['exp_id'],'model_name':L['model_name'],'rng_before_initialization':rng(),'time':now()}
 def after_init(self,L):
   self.need(self.init_pending is not None,'Unregistered model initialization')
   model=L['model'];self.counts['model_initializations']+=1
   pars=[{'name':n,**ti(p),'elements':p.numel(),'finite':fin(p),'grad_is_none':p.grad is None} for n,p in model.named_parameters()]
   self.need(all(p['finite'] and p['grad_is_none'] for p in pars),'Initial parameter state')
   self.need(all(p.device.type=='cpu' for p in model.parameters()),'CPU required')
   self.current={'exp_id':L['exp_id'],'model_name':L['model_name']};self.runkey='exp%d_%s'%(L['exp_id'],L['model_name'])
   self.order.append(self.current.copy())
   once(self.runkey+'_initialization.json',{**self.init_pending,'run_id':RUN_ID,'rng_after_initialization':rng(),'class':str(type(model)),'parameters':pars,'trainable_elements':sum(p.numel() for p in model.parameters() if p.requires_grad),'buffers':[{'name':n,**ti(b)} for n,b in model.named_buffers()],'args':vars(L['args']),'config':L['config'].config(L['model_name'],L['args']),'source':FROZEN,'gradient_policy':'All declared trainable parameters have source prediction paths; no trainable exclusion identified. Classify frozen/no-grad separately; never impose Delelstm tensor count on other models.'})
   self.init_pending=None
   self.full_integrity(self.runkey+'_start',True)
   self.run_counts={'train_batches':0,'val_batches':0,'test_batches':0,'epochs':0,'backwards':0,'step_attempts':0,'step_successes':0}
   self.run_started=now();self.best=None;self.status()
 def run_start(self,L):
  self.need(L['args'].log is True and Path(L['args'].save_dirs)==OUT,'log/output identity')
 def epoch_start(self,L):
  self.epoch=L['i'];self.need(0<=self.epoch<300,'Epoch budget')
  self.epoch_time=time.monotonic();self.train_batch=0;self.val_batch=0;self.test_batch=0
  self.profiler_rows=[];self.epoch_grad_max=0.;self.epoch_coeff_max=0.
  self.epoch_start_lr=L['opt'].param_groups[0]['lr']
  self.need(self.counts['optimizer_creations']<=25,'Optimizer budget')
  self.full_integrity(self.runkey+'_epoch_'+str(self.epoch),self.epoch in [0,100,200,299])
  self.status()
 def opt_ready(self,L):
  self.counts['optimizer_creations']+=1
  opt=L['opt'];self.need(type(opt) is torch.optim.Adam and opt.param_groups[0]['lr']==.05,'Adam identity')
  self.need(opt.param_groups[0]['weight_decay']==.03,'Adam weight decay')
  once(self.runkey+'_optimizer_identity.json',{'class':str(type(opt)),'defaults':opt.defaults,'initial_param_groups':[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]})
 def adam_finite(self,opt):
  bad=[{'state_index':i,'key':k,**numeric(v)} for i,st in enumerate(opt.state.values()) for k,v in st.items() if isinstance(v,torch.Tensor) and not fin(v)]
  if bad:append('nonfinite_Adam_diagnostics.jsonl',{'time':now(),'current':self.current,'epoch':getattr(self,'epoch',None),'states':bad})
  return not bad
 def parameters_finite(self,model):
  bad=[{'name':n,'kind':kind,**numeric(p)} for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())] for n,p in items if not fin(p)]
  if bad:append('nonfinite_parameter_diagnostics.jsonl',{'time':now(),'current':self.current,'epoch':getattr(self,'epoch',None),'tensors':bad})
  return not bad
 def before_batch(self,L):
   self.source_quick();self.need(__import__('shutil').disk_usage(R).free>2*1024**3,'RESOURCE STOP: free disk below2GiB')
   self.train_batch+=1
   self.need(self.train_batch<=42 and self.run_counts['train_batches']<12600 and self.counts['training_batches']<315000,'PM2.5 training batch budget')
   self.need(L['j']==self.train_batch and torch.is_grad_enabled(),'Training order/autograd')
   self.need(all(math.isfinite(float(g['lr'])) for g in L['opt'].param_groups),'Non-finite Learning Rate')
   self.need(self.parameters_finite(L['model']) and self.adam_finite(L['opt']),'Non-finite parameters/Adam')
   self.need(list(L['batch_x'].shape)==[32,24,8] and list(L['batch_y'].shape)==[32,22],'PM2.5 input/target shape')
   self.counts['training_batches']+=1;self.run_counts['train_batches']+=1
   if self.current=={'exp_id':0,'model_name':'Delelstm'} and self.epoch==0:
    samples=[]
    for i,x in enumerate(L['batch_x']):
     key=sha(x.contiguous().numpy().tobytes());pos=self.lookup.get(key,[])
     self.need(len(pos)==1,'Training sample mapping ambiguous')
     p=pos[0];samples.append({'batch_position':i,'dataset_position':p,'idx':self.ids['train'][p]})
    append('first_epoch_training_batches.jsonl',{'epoch':self.epoch,'batch':self.train_batch,'samples':samples,'input_float64':ti(L['batch_x']),'input_float32':ti(L['batch_x'].float()),'target':ti(L['batch_y']),'extra_iterator':False})
   self.mon={'time':now(),'run_id':RUN_ID,'experiment':self.current,'epoch':self.epoch,'train_batch':self.train_batch,'global_train_batch':self.counts['training_batches'],'input_shape':list(L['batch_x'].shape),'target_shape':list(L['batch_y'].shape)}
 def begin_forward(self,L,phase):
  if phase=='val':self.val_batch+=1;self.counts['validation_batches']+=1;self.run_counts['val_batches']+=1
  elif phase=='test':self.test_batch+=1;self.counts['test_batches']+=1;self.run_counts['test_batches']+=1
  b={'train':self.train_batch,'val':self.val_batch,'test':self.test_batch}[phase]
  self.need(b<={'train':42,'val':8,'test':5}[phase],'Forward phase budget')
  self.need(self.parameters_finite(L['model']),'Non-finite parameters before '+phase)
  self.need(L['model'].training==(phase=='train'),'Model mode')
  self.need(torch.is_grad_enabled()==(phase=='train'),'Autograd mode')
  self.counts['forwards']+=1;self.pending_forward=phase
  self.prof_phase=phase;self.prof_batch=b
  # Fixed, nonselective coverage; no diagnostic Forward beyond author call.
  sample=self.current['model_name']=='Delelstm' and b==1 and (phase=='test' or self.epoch%25==0)
  if sample:
   self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False)
   self.prof.__enter__()
  self.forward_time=time.monotonic()
 def end_forward(self,L):
  phase=self.pending_forward;self.pending_forward=None
  elapsed=time.monotonic()-self.forward_time
  solver={'observed':False,'status':'UNVERIFIED: unsampled natural forward','policy':'batch1 train/validation epochs0,25,...275 and test1'}
  if self.prof:
   self.prof.__exit__(None,None,None)
   ev=self.prof.key_averages()
   counts={e.key:e.count for e in ev if 'linalg_inv' in e.key or 'pinv' in e.key or 'pinverse' in e.key}
   solver={'observed':True,'events':counts,'high_level_inv':counts.get('aten::linalg_inv',0),'internal_inv_ex':counts.get('aten::linalg_inv_ex',0),'pinv':counts.get('aten::linalg_pinv',0)+counts.get('aten::pinverse',0),'profiler_overhead':'Not independently measured; included in wall duration. No control forward.','batched_fallback_not_per_sample':True}
   self.prof=None
   append('solver_samples.jsonl',{'experiment':self.current,'epoch':self.epoch,'phase':phase,'batch':self.prof_batch,'solver':solver,'profiled_wall_seconds':elapsed})
  pred=L['y_pred'] if phase=='train' else L['output']
  coeff=L['unorm_weight'];third=L['_']
  pr=numeric(pred);co=numeric(coeff) if isinstance(coeff,torch.Tensor) else None
  th=numeric(third) if isinstance(third,torch.Tensor) else None
  info={'experiment':self.current,'epoch':self.epoch,'phase':phase,'batch':self.prof_batch,'prediction':pr,'coefficients_or_alpha':co,'third_or_beta':th,'requires_grad':pred.requires_grad,'grad_fn':str(pred.grad_fn),'forward_seconds':elapsed,'solver':solver}
  append('forward_monitor.jsonl',info)
  self.need(pr['shape']==[32,22,1] and pr['finite'],'Prediction shape/non-finite')
  self.need((co is None or co['finite']) and (th is None or th['finite']),'Non-finite coefficient/explanation')
  if phase=='train':
   self.mon.update(forward=info)
   self.need(pred.requires_grad,'Training prediction graph missing')
   self.epoch_coeff_max=max(self.epoch_coeff_max,co['max_abs'] if co else 0.)
 def before_loss(self,L):
  self.need(list(L['y_pred'].shape)==list(L['batch_y'].shape)==[32,22],'Unexpected broadcasting')
 def before_backward(self,L):
  loss=L['l'];self.counts['losses']+=1
  self.need(loss.ndim==0 and fin(loss) and loss.requires_grad and loss.grad_fn is not None,'Invalid loss')
  self.mon.update(loss=float(loss.item()),loss_dtype=str(loss.dtype),loss_reduction=L['loss'].reduction)
  self.counts['backwards']+=1;self.run_counts['backwards']+=1
 def after_backward(self,L):
  records=[];sum_sq=0.;maxabs=0.;bj=None
  for name,p in L['model'].named_parameters():
   g=p.grad
   rec={'name':name,'requires_grad':p.requires_grad,'present':g is not None}
   if g is None and not p.requires_grad:
    rec.update(classification='legitimate_frozen_parameter_no_gradient',source_reason='requires_gradFalse from original model');records.append(rec);continue
   if g is None:rec.update(classification='unexpected_missing_trainable_gradient',source_reason='No legitimate unused trainable parameter identified in this frozen prediction graph; not a fixed26parameter rule')
   if g is None:append('gradient_stop_diagnostics.jsonl',{'time':now(),'current':self.current,'epoch':self.epoch,'batch':self.train_batch,'bad_gradient':rec,'already_observed':records})
   self.need(g is not None,'Missing gradient '+name)
   rec.update(shape=list(g.shape),dtype=str(g.dtype),device=str(g.device),finite=fin(g),nan=int(torch.isnan(g).sum()),positive_inf=int(torch.isposinf(g).sum()),negative_inf=int(torch.isneginf(g).sum()))
   if not rec['finite'] or g.shape!=p.shape:append('gradient_stop_diagnostics.jsonl',{'time':now(),'current':self.current,'epoch':self.epoch,'batch':self.train_batch,'bad_gradient':rec,'already_observed':records})
   self.need(rec['finite'] and g.shape==p.shape,'Non-finite/wrong-shape gradient '+name)
   d=g.detach();sq=float(d.double().square().sum().item());ma=float(d.abs().max().item())
   rec.update(l2=math.sqrt(sq),max_abs=ma,zero_count=int((d==0).sum().item()),sha256=sha(d.cpu().contiguous().numpy().tobytes()))
   records.append(rec);sum_sq+=sq;maxabs=max(maxabs,ma)
   if name=='B_j':bj={'l2':math.sqrt(sq),'max_abs':ma,'squared_norm':sq}
  if bj:bj['squared_norm_contribution']=bj['squared_norm']/sum_sq if sum_sq else None
  self.mon.update(gradients=records,global_gradient_l2=math.sqrt(sum_sq),global_gradient_max_abs=maxabs,B_j=bj)
  self.epoch_grad_max=max(self.epoch_grad_max,math.sqrt(sum_sq))
 def before_step(self,L):
  self.need(all(math.isfinite(float(g['lr'])) for g in L['opt'].param_groups),'Non-finite effective Learning Rate')
  self.need(self.mon and 'gradients' in self.mon,'Step before gradient diagnostics')
  self.need(self.parameters_finite(L['model']) and self.adam_finite(L['opt']),'Invalid pre-step parameters/Adam')
  self.need(self.counts['optimizer_step_attempts']<315000,'Global update budget')
  self.pending_step=True
  self.counts['optimizer_step_attempts']+=1;self.run_counts['step_attempts']+=1
 def after_step(self,L):
  self.pending_step=False
  # The original step call has returned: count it even if post-state checks now fail.
  self.counts['optimizer_step_successes']+=1;self.run_counts['step_successes']+=1
  self.need(self.parameters_finite(L['model']) and self.adam_finite(L['opt']),'Non-finite post-step parameters/Adam')
  values=[float(st['step'].item()) for st in L['opt'].state.values()]
  self.need(values and all(s==self.run_counts['step_successes'] for s in values),'Adam step state mismatch')
  self.mon.update(step_success=True,post_parameter_finite=True,post_Adam_state_finite=True,Adam_step_values=values)
  append('training_batch_monitor.jsonl',self.mon)
  if self.train_batch==1 or self.train_batch==42:self.status()
 def validation_metrics_ready(self,L):
  self.need(math.isfinite(float(L['mse_val'])) and math.isfinite(float(L['mae_val'])),'Non-finite validation metric')
 def before_save(self,L):
  self.need(L['min_val_loss']==L['mse_val']**.5,'Checkpoint criterion')
  path=Path(L['model_save_path'])/(str(L['exp_id'])+'_pm_predict.pt')
  if path.exists():
   old=self.checkpoints.get(self.runkey)
   self.need(old and sha(path.read_bytes())==old['sha256'],'Unauthorized checkpoint modification before overwrite')
  self.pending_save={'path':str(path),'epoch':self.epoch,'validation_mse':float(L['mse_val']),'validation_rmse':float(L['mse_val']**.5),'validation_mae':float(L['mae_val']),'parameter_hashes':{n:ti(p)['sha256'] for n,p in L['model'].named_parameters()},'time':now(),'selection':'min_val_loss > mse_val ** 0.5, original strict inequality'}
 def after_save(self):
  c=self.pending_save;self.pending_save=None;p=Path(c['path']);raw=p.read_bytes();c.update(bytes=len(raw),sha256=sha(raw),save_success=True)
  self.checkpoints[self.runkey]=c;self.best=c
  append('checkpoint_selection.jsonl',{'experiment':self.current,**c})
 def epoch_end(self,L):
   self.pending_scheduler=False
   self.need(self.train_batch==42 and self.val_batch==8,'PM2.5 epoch loader counts')
   self.need(math.isfinite(float(L['mse_val'])) and math.isfinite(float(L['mae_val'])),'Non-finite validation metrics')
   sch=L['epoch_scheduler']
   self.need(type(sch) is torch.optim.lr_scheduler.ReduceLROnPlateau and sch.mode=='min','Scheduler identity')
   self.need(all(math.isfinite(float(g['lr'])) for g in L['opt'].param_groups),'Non-finite Learning Rate')
   row={'run_id':RUN_ID,'experiment':self.current,'epoch':self.epoch,'train_batches':42,'train_used_windows':1344,'train_dropped_windows':25,'val_batches':8,'val_used_windows':256,'val_dropped_windows':17,'train_mse':L['mse_train']/L['j'],'train_rmse':(L['mse_train']/L['j'])**.5,'val_mse':float(L['mse_val']),'val_rmse':float(L['mse_val']**.5),'val_mae':float(L['mae_val']),'learning_rate_before_scheduler':self.epoch_start_lr,'learning_rate_after_scheduler':L['opt'].param_groups[0]['lr'],'scheduler_state':scheduler_state_record(sch.state_dict()),'best_validation_rmse':L['min_val_loss'],'best_epoch':self.best['epoch'] if self.best else None,'best_checkpoint_sha256':self.best['sha256'] if self.best else None,'wall_seconds_including_observer':time.monotonic()-self.epoch_time,'author_train_wall_seconds':L['time_elapsed'],'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'free_disk_bytes':__import__('shutil').disk_usage(R).free,'coefficient_max_abs':self.epoch_coeff_max,'gradient_l2_max':self.epoch_grad_max,'time':now()}
   append('epoch_metrics.jsonl',row)
   self.epochs.append(row);self.counts['epochs_completed']+=1;self.run_counts['epochs']+=1
   self.status()
   print('PM25_EPOCH '+encode({k:row[k] for k in ['experiment','epoch','train_mse','val_rmse','wall_seconds_including_observer']}),flush=True)
 def before_load(self,L):
  self.need(self.run_counts['epochs']==300 and self.best is not None,'Test requires 300epochs and best checkpoint')
  raw=Path(self.best['path']).read_bytes()
  self.need(sha(raw)==self.best['sha256'],'Unauthorized checkpoint modification before test')
  self.pending_load=True
 def after_load(self,L):
  self.pending_load=False
  self.need({n:ti(p)['sha256'] for n,p in L['model'].named_parameters()}==self.best['parameter_hashes'],'Loaded checkpoint parameter identity')
  append('checkpoint_load.jsonl',{'experiment':self.current,'time':now(),'pass':True,'checkpoint':self.best})
 def run_end(self,L):
  self.need(self.run_counts['epochs']==300 and self.run_counts['step_successes']==12600 and self.test_batch==5,'Run budget incomplete')
  metrics={'mse':float(L['mse_test']),'rmse':float(L['mse_test']**.5),'mae':float(L['mae_test']),'mape':float(L['mape_test'].item())}
  self.need(all(math.isfinite(x) for x in metrics.values()),'Non-finite final test metric')
  self.full_integrity(self.runkey+'_end',True)
  rec={'experiment':self.current.copy(),'status':'COMPLETED','started_at':self.run_started,'ended_at':now(),'counts':self.run_counts.copy(),'best_checkpoint':self.best,'test_metrics':metrics,'test_windows':160,'test_dropped':24,'test_prediction_shape':list(L['preds'].shape),'test_target_shape':list(L['true'].shape),'prediction_sha256':sha(L['preds'].tobytes()),'target_sha256':sha(L['true'].tobytes()),'outdir':L['model_save_path'],'rng_after_test_and_outputs':rng()}
  self.runs.append(rec);append('model_results.jsonl',rec);self.counts['models_completed']+=1;self.status()
 def training_trace(self,frame,event,arg):
  L=frame.f_locals;n=frame.f_lineno
  if event=='exception':
   if not issubclass(arg[0],StopIteration):self.frame_exception=True
   if self.prof:
    self.prof.__exit__(*arg);self.prof=None
   self.pending_forward=None
   return self.training_trace
  if event=='return':
   if not self.frame_exception and arg is None and self.run_counts.get('epochs')==300 and 'mape_test' in L:self.run_end(L)
   return self.training_trace
  if event!='line':return self.training_trace
  if self.pending_step:self.after_step(L)
  if self.pending_forward:self.end_forward(L)
  if self.pending_save:self.after_save()
  if self.pending_load:self.after_load(L)
  if self.pending_scheduler:self.epoch_end(L)
  if n==427:self.opt_ready(L)
  elif n==447:self.epoch_start(L)
  elif n==454:self.before_batch(L)
  elif n==456:self.begin_forward(L,'train')
  elif n==459:self.before_loss(L)
  elif n==460:self.before_backward(L)
  elif n==461:self.after_backward(L)
  elif n==462:self.before_step(L)
  elif n==483:self.begin_forward(L,'val')
  elif n==519:self.validation_metrics_ready(L)
  elif n==525:self.before_save(L)
  elif n==548:self.pending_scheduler=True
  elif n==556:self.before_load(L)
  elif n==565:self.begin_forward(L,'test')
  return self.training_trace
 def entry_trace(self,frame,event,arg):
  if event!='line':return self.entry_trace
  L=frame.f_locals;n=frame.f_lineno
  if n==86:self.data_ready(L)
  elif n==97:self.before_init(L)
  elif n==113:self.after_init(L)
  return self.entry_trace
 def trace(self,frame,event,arg):
  if event!='call':return None
  path=frame.f_code.co_filename
  if path==str(A/'model_training.py') and frame.f_code.co_name=='PM_training':
   self.frame_exception=False;self.run_start(frame.f_locals);return self.training_trace
  if path==str(A/'PM.py') and frame.f_code.co_name=='<module>':return self.entry_trace
  return None
 def complete(self):
  self.need(self.counts['models_completed']==25 and self.counts['epochs_completed']==7500 and self.counts['optimizer_step_successes']==315000,'Final workload count mismatch')
  self.final_status='COMPLETED';self.status()
 def fail(self):
  self.final_status='STOPPED';self.status()
  once('stop_condition_record.json',{'time':now(),'traceback':traceback.format_exc(),'current':self.current,'epoch':getattr(self,'epoch',None),'train_batch':getattr(self,'train_batch',None),'counts':self.counts,'no_restart':True,'no_source_or_algorithm_repair':True})
