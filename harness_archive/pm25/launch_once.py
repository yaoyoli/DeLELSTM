import os,json,time,hashlib,subprocess,sys
from pathlib import Path
D=Path('/Users/yao/DeLELSTM-track-a-runtime/diagnostics/stage3b/pm25/run-01')
RID='stage3b-pm25-run-01'
for n in ['formal_run.lock','formal_run_started_once.json','outer_launch_once.json','supervisor_launch_once.json','formal_process.json']:assert not (D/n).exists(),'Already claimed/started: '+n
assert Path('/Users/yao/DeLELSTM-track-a-runtime/results/stage3b/pm25/run-01').is_dir() and not any(Path('/Users/yao/DeLELSTM-track-a-runtime/results/stage3b/pm25/run-01').iterdir()),'Results not empty'
r=json.loads((D/'harness_static_review.json').read_text());assert r['pass']
for n,h in r['sources_sha256'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
assert json.loads((D/'serialization_regression_results_final.json').read_text())['pass']
assert json.loads((D/'immediate_prelaunch_recheck.json').read_text())['pass']
claim={'run_id':RID,'launcher_pid':os.getpid(),'launcher_ppid':os.getppid(),'launcher_pgid':os.getpgid(0),'launcher_sid':os.getsid(0),'created_unix':time.time(),'frozen_commit':'95ff20de5a2dea627c854e35a00f304c838fff12','permanent_once_token':True,'remove_or_reclaim_automatically':False}
fd=os.open(str(D/'formal_run.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f:json.dump(claim,f,indent=2);f.flush();os.fsync(f.fileno())
cmd=['/usr/bin/python3','-B',str(D/'outer_launcher.py')]
with (D/'outer_launch_once.json').open('x') as f:json.dump({'run_id':RID,'argv':cmd,'cwd':str(D),'time_unix':time.time(),'max_models':25,'max_epochs':7500,'max_updates':315000,'no_automatic_restart':True},f,indent=2)
try:
 with (D/'outer.stdout').open('xb') as out,(D/'outer.stderr').open('xb') as err:
  p=subprocess.Popen(cmd,cwd=D,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',DELELSTM_RUN_ID=RID),stdout=out,stderr=err,start_new_session=True)
 with (D/'outer_process.json').open('x') as f:json.dump({'run_id':RID,'pid':p.pid,'ppid_launcher':os.getpid(),'pgid':os.getpgid(p.pid),'sid':os.getsid(p.pid),'started_unix':time.time(),'cwd':str(D),'detached_local_session':True,'no_restart':True},f,indent=2)
 print(json.dumps({'run_id':RID,'launched_once':True,'outer_pid':p.pid,'atomic_run_lock':True,'max_models':25,'max_epochs':7500,'max_updates':315000}))
except BaseException:
 import traceback
 with (D/'formal_launch_error.json').open('x') as f:json.dump({'traceback':traceback.format_exc(),'lock_preserved':True,'no_retry':True},f)
 raise
