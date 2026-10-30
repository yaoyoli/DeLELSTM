"""Classify only observed non-scientific Git metadata deltas; never hide content changes."""
import re
def git_identity(old,new):
 a=old['files'];b=new['files'];known={v.get('sha256') for p,v in a.items() if p.startswith('refs/codex/turn-diffs/')}
 ds=[];bad=[]
 for p in sorted(set(a)|set(b)):
  x=a.get(p);y=b.get(p)
  if x==y:continue
  kind=None
  if re.fullmatch(r'objects/[0-9a-f]{2}/[0-9a-f]{38}',p) and x is not None and y is not None and x.get('kind')=='file' and y.get('kind')=='file' and {k:v for k,v in x.items() if k!='mtime_ns'}=={k:v for k,v in y.items() if k!='mtime_ns'}:
   kind='Existing loose Object: mtime only; exact compressed bytes/SHA256/size/inode/mode preserved'
  elif p.startswith('refs/codex/turn-diffs/') and (x is None or y is None) and (x or y).get('kind')=='file' and (x or y).get('sha256') in known:
   kind='Codex namespaced reference path changed; exact payload hash already present in baseline'
  if kind is None:bad.append(p)
  ds.append({'path':p,'before':x,'after':y,'classification':kind,'writer_cause':'UNVERIFIED'})
 return {'identity_pass':not bad and not new['errors'],'byte_and_metadata_exact':old==new,'differences':ds,'unexplained_or_key_content_changes':bad,'scope':'Only existingObject mtime and content-identical known Codexref path deltas classified; no newObject,changedpayload,branchref,index,config,source,data orTrackB exception','Runtime_DS_Store_rule_not_used_for_Git':True}
