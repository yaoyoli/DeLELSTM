"""Strict, record-only metadata encoding for PyTorch2.0 ReduceLROnPlateau(mode=min).
Only top-level mode_worse=+inf is a legal sentinel. Never mutates/loads live state.
"""
import json,math
SENTINEL={'__float_sentinel__':'+Infinity','field':'mode_worse','source':'ReduceLROnPlateau'}
def strict_json(value):
 return json.dumps(value,ensure_ascii=False,allow_nan=False)
def scheduler_state_record(state):
 if not isinstance(state,dict) or state.get('mode')!='min':
  raise ValueError('Only verified ReduceLROnPlateau mode=min metadata is supported')
 best=state.get('best')
 if not isinstance(best,(int,float)) or isinstance(best,bool) or not math.isfinite(best):
  raise ValueError('Non-finite/invalid scheduler best is not a legal sentinel')
 sentinel=state.get('mode_worse')
 if not isinstance(sentinel,float) or not math.isinf(sentinel) or sentinel<=0:
  raise ValueError('Unexpected mode_worse for mode=min')
 def walk(v,path=()):
  if isinstance(v,float) and not math.isfinite(v):
   if path==('mode_worse',) and math.isinf(v) and v>0:return dict(SENTINEL)
   raise ValueError('Unknown non-finite scheduler field '+repr(path))
  if isinstance(v,dict):return {k:walk(x,path+(k,)) for k,x in v.items()}
  if isinstance(v,list):return [walk(x,path+(i,)) for i,x in enumerate(v)]
  if isinstance(v,tuple):return tuple(walk(x,path+(i,)) for i,x in enumerate(v))
  return v
 record=walk(state)
 strict_json(record) # Also rejects opaque scientific objects; no default=str.
 return record
def decode_scheduler_record(record):
 if not isinstance(record,dict) or record.get('mode')!='min':
  raise ValueError('Unrecognized scheduler record')
 if record.get('mode_worse')!=SENTINEL:
  raise ValueError('Unrecognized sentinel marker')
 def walk(v,path=()):
  if path==('mode_worse',):return float('inf')
  if isinstance(v,dict):
   if '__float_sentinel__' in v:raise ValueError('Sentinel marker outside permitted field')
   return {k:walk(x,path+(k,)) for k,x in v.items()}
  if isinstance(v,list):return [walk(x,path+(i,)) for i,x in enumerate(v)]
  if isinstance(v,float) and not math.isfinite(v):raise ValueError('Non-JSON float in record')
  return v
 restored=walk(record)
 if not math.isfinite(restored['best']):raise ValueError('Non-finite best')
 return restored
