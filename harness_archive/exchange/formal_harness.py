"""Runs unchanged frozen Exchange.py with read-only Python line observation."""
import os,sys,json,runpy,traceback
from pathlib import Path
D=Path('/Users/yao/DeLELSTM-track-a-runtime/diagnostics/stage3a/attempt-02')
A=Path('/Users/yao/DeLELSTM-track-a')
R=A.parent/'DeLELSTM-track-a-runtime'
assert __debug__ and sys.flags.optimize==0 and sys.dont_write_bytecode
assert not (D/'formal_run_started_once.json').exists(),'One continuous run only; no automatic restart'
with (D/'formal_run_started_once.json').open('x') as f:json.dump({'pid':os.getpid(),'single_launch':True},f)
sys.path.insert(0,str(A))
from observer import Observer,once
observer=Observer()
sys.argv=[str(A/'Exchange.py'),'--save_dirs',str(R/'results/stage3a/attempt-02')]
exit_code=0
try:
 sys.settrace(observer.trace)
 runpy.run_path(str(A/'Exchange.py'),run_name='__main__')
 sys.settrace(None)
 observer.complete()
except BaseException:
 sys.settrace(None)
 observer.fail()
 traceback.print_exc()
 exit_code=1
finally:
 sys.settrace(None)
 once('formal_execution_summary.json',{'status':observer.final_status,'counts':observer.counts,'runs_completed':observer.runs,'order':observer.order,'exit_code':exit_code,'overall_numerical_stability':'UNRESOLVED','no_retry':True})
sys.exit(exit_code)
