"""Outer supervisor capture and post-package receipt. No model retry."""
import os,sys,json,subprocess,hashlib,time,traceback
from pathlib import Path
D=Path('/Users/yao/DeLELSTM-track-a-runtime/diagnostics/stage3b/electricity/run-01')
def write(n,o):
 with (D/n).open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2)
def hf(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',TMPDIR=str(D/'cache/tmp'),XDG_CACHE_HOME=str(D/'cache/xdg'),MPLCONFIGDIR=str(D/'cache/matplotlib'),TORCH_HOME=str(D/'cache/torch'),GIT_OPTIONAL_LOCKS='0')
code=99;child=None
try:
 with (D/'supervisor.stdout').open('xb') as out,(D/'supervisor.stderr').open('xb') as err:
  child=subprocess.Popen(['/usr/bin/python3','-B',str(D/'supervisor.py')],cwd=D,env=env,stdout=out,stderr=err)
  write('supervisor_process.json',{'pid':child.pid,'outer_pid':os.getpid(),'time_unix':time.time(),'one_launch_no_retry':True})
  code=child.wait()
except BaseException:
 write('outer_launcher_error.json',{'traceback':traceback.format_exc(),'no_retry':True})
 if child is not None and child.poll() is None:code=child.wait()
finally:
 write('supervisor_exit_code.json',{'exit_code':code,'time_unix':time.time()})
 records=[]
 for n in ['supervisor.stdout','supervisor.stderr','supervisor_exit_code.json','postflight.stdout','postflight.stderr','finalizer_exit_code.json','package_verification.json']:
  p=D/n
  if p.exists():records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hf(p)})
 write('external_execution_receipt.json',{'files':records,'reason':'These post-package receipts/active finalizer streams cannot be recursively sealed inside the same ZIP. Their hashes supplied outside archive.','model_retry':False})
sys.exit(code)
