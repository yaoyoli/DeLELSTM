"""Unique Electricity formal run; only one approved path constant changes in memory."""
import os,sys,json,hashlib,traceback
from pathlib import Path
D=Path('/Users/yao/DeLELSTM-track-a-runtime/diagnostics/stage3b/electricity/run-01')
A=Path('/Users/yao/DeLELSTM-track-a')
OUT=Path('/Users/yao/DeLELSTM-track-a-runtime/results/stage3b/electricity/run-01')
assert __debug__ and sys.flags.optimize==0 and sys.dont_write_bytecode
lock=json.loads((D/'formal_run.lock').read_text());assert lock['run_id']=='stage3b-electricity-run-01'
assert not (D/'formal_run_started_once.json').exists(),'Run01 already started; no retry'
with (D/'formal_run_started_once.json').open('x') as f:json.dump({'run_id':lock['run_id'],'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'one_formal_entry':True},f)
sys.path.insert(0,str(A))
from observer import Observer,once
from entry_adapter import make_code,ORIGINAL
observer=None;code=1
try:
 entry,proof=make_code()
 expected=json.loads((D/'path_compatibility_mapping.json').read_text())
 assert proof==expected,'Entry source/path mapping drift'
 observer=Observer()
 sys.argv=[str(ORIGINAL),'--save_dirs',str(OUT)]
 namespace={'__name__':'__main__','__file__':str(ORIGINAL),'__package__':None,'__spec__':None,'__cached__':None,'__builtins__':__builtins__}
 sys.settrace(observer.trace)
 exec(entry,namespace,namespace)
 sys.settrace(None)
 observer.complete();code=0
except BaseException:
 sys.settrace(None)
 if observer is not None:observer.fail()
 else:once('startup_error.json',{'traceback':traceback.format_exc(),'formal_author_models_known_started':0,'no_retry':True})
 traceback.print_exc()
finally:
 sys.settrace(None)
 once('formal_execution_summary.json',{'run_id':'stage3b-electricity-run-01','status':observer.final_status if observer else 'NOT RUN / STARTUP FAILED','counts':observer.counts if observer else {},'runs_completed':observer.runs if observer else [],'order':observer.order if observer else [],'exit_code':code,'overall_numerical_stability':'UNRESOLVED','no_restart':True,'no_PM_Exchange_TrackB_execution':True})
sys.exit(code)
