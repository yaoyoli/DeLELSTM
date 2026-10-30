"""One-launch Stage3B PM2.5 supervisor: logs, exit and stdlib postflight; no restart."""
import os,sys,json,subprocess,hashlib,time,traceback
from pathlib import Path
D=Path('/Users/yao/DeLELSTM-track-a-runtime/diagnostics/stage3b/pm25/run-01')
PY='/Users/yao/DeLELSTM-track-a-runtime/.venv/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',TMPDIR=str(D/'cache/tmp'),XDG_CACHE_HOME=str(D/'cache/xdg'),MPLCONFIGDIR=str(D/'cache/matplotlib'),TORCH_HOME=str(D/'cache/torch'),GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0',GIT_PAGER='cat')
def once(n,x):
 with (D/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
assert not (D/'supervisor_launch_once.json').exists(),'No automatic restart'
once('supervisor_launch_once.json',{'pid':os.getpid(),'time_unix':time.time()})
lock=json.loads((D/'formal_run.lock').read_text());assert lock['run_id']=='stage3b-pm25-run-01'
review=json.loads((D/'harness_static_review.json').read_text());assert review['pass']
for n,h in review['sources_sha256'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h,'Reviewed source changed '+n
cmd=[PY,'-B',str(D/'formal_harness.py')]
once('formal_execution_command.json',{'argv':cmd,'cwd':str(D),'environment_overrides':{k:env[k] for k in ['PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','GIT_OPTIONAL_LOCKS']},'no_git_write':True,'one_continuous_run':True})
start=time.time();p=None;c=None;code=99
try:
 with (D/'formal.stdout').open('xb') as out,(D/'formal.stderr').open('xb') as err:
  p=subprocess.Popen(cmd,cwd=D,env=env,stdout=out,stderr=err)
  once('formal_process.json',{'pid':p.pid,'supervisor_pid':os.getpid(),'supervisor_ppid':os.getppid(),'supervisor_pgid':os.getpgid(0),'supervisor_sid':os.getsid(0),'started_unix':start,'run_id':'stage3b-pm25-run-01'})
  c=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(p.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  once('power_assertion.json',{'argv':['/usr/bin/caffeinate','-i','-w',str(p.pid)],'pid':c.pid,'scope':'Only authorchildlifetime;no persistentpowerconfigwrite'})
  code=p.wait()
except BaseException:
 once('supervisor_error.json',{'traceback':traceback.format_exc(),'no_restart':True,'child_pid':p.pid if p else None})
 if p is not None and p.poll() is None:code=p.wait()
finally:
 if c is not None:
  try:c.wait(timeout=15)
  except subprocess.TimeoutExpired:once('power_assertion_exit_unresolved.json',{'pid':c.pid,'child_finished':True})
 once('formal_exit_code.json',{'exit_code':code,'wall_seconds':time.time()-start,'one_launch':True})
 if not (D/'formal_execution_summary.json').exists():once('startup_or_interrupt_record.json',{'status':'NOTRUN OR INTERRUPTED;actualcountersunproven inspectlogs','exit_code':code,'constructor_or_OS_failure_possible':True,'no_model_retry':True})
 with (D/'postflight.stdout').open('xb') as out,(D/'postflight.stderr').open('xb') as err:q=subprocess.run(['/usr/bin/python3','-B',str(D/'finalize.py')],cwd=D,env=env,stdout=out,stderr=err)
 once('finalizer_exit_code.json',{'exit_code':q.returncode,'model_execution_finished':True})
sys.exit(0 if code==0 and q.returncode==0 else 1)
