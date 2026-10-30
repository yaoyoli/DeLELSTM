"""Exact, component-based Runtime metadata policy; no blanket directory exclusions."""
import os,stat,json
from pathlib import Path,PurePosixPath
from audit_utils import R,D,sha
def scoped_name(rel):
 p=PurePosixPath(rel)
 if p.is_absolute() or '..' in p.parts or not p.parts:return False
 return p.name=='.DS_Store' and (p.parts==('.DS_Store',) or (len(p.parts)>=2 and p.parts[0]=='diagnostics'))
def validate(rel,protected=(),historical=None):
 if not scoped_name(rel):return {'eligible':False,'reason':'Name/components outside approved rule'}
 root=R.resolve(strict=True)
 if R.is_symlink() or root!=R:raise RuntimeError('Runtime root path/symlink mismatch')
 candidate=R/rel
 protected={str(Path(p).absolute()) for p in protected}
 if str(candidate.absolute()) in protected:raise RuntimeError('Registered scientific file cannot be exempt: '+rel)
 # Inspect all path components, never traverse a symlink.
 cursor=R
 for part in PurePosixPath(rel).parts:
  cursor=cursor/part
  try:s=cursor.lstat()
  except FileNotFoundError:
   if cursor!=candidate:raise RuntimeError('Missing metadata ancestor '+str(cursor))
   if not historical or historical.get('kind')!='file':return {'eligible':False,'reason':'Missing candidate without prior regular-file evidence'}
   resolved=cursor.parent.resolve(strict=True)/cursor.name
   resolved.relative_to(root)
   return {'eligible':True,'exists':False,'path':str(candidate),'resolved':str(resolved),'relative_path':rel,'prior_regular_file_verified':True,'writer_cause':'UNVERIFIED'}
  if stat.S_ISLNK(s.st_mode):raise RuntimeError('Metadata path contains symlink: '+str(cursor))
  if cursor!=candidate and not stat.S_ISDIR(s.st_mode):raise RuntimeError('Non-directory metadata ancestor')
 resolved=candidate.resolve(strict=True);resolved.relative_to(root)
 s=candidate.lstat()
 if not stat.S_ISREG(s.st_mode):raise RuntimeError('Metadata candidate is not regular file '+rel)
 raw=candidate.read_bytes()
 end=candidate.stat()
 if (s.st_size,s.st_mtime_ns,s.st_ino)!=(end.st_size,end.st_mtime_ns,end.st_ino):raise RuntimeError('Metadata changed while hashing '+rel)
 return {'eligible':True,'exists':True,'relative_path':rel,'path':str(candidate),'resolved':str(resolved),'ordinary_file':True,'no_symlink_components':True,'inside_real_runtime':True,'not_registered_scientific_evidence':True,'bytes':len(raw),'sha256':sha(raw),'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'writer_cause':'UNVERIFIED'}
def compare_runtime(old,new,protected=(),registered_new=()):
 approved=[];bad=[]
 a=old['runtime_existing']['files'];b=new['runtime_existing']['files'];registered_new=set(registered_new)
 for rel in sorted(set(a)|set(b)):
  x=a.get(rel);y=b.get(rel)
  if x==y:continue
  if rel in registered_new and x is None:continue
  if scoped_name(rel):
   validation=validate(rel,protected,historical=x)
   if validation['eligible']:
    approved.append({'relative_path':rel,'before':x,'after':y,'validation':validation});continue
  bad.append({'relative_path':rel,'before':x,'after':y})
 return {'scientific_integrity_pass':not bad,'non_exempt_changes':bad,'approved_metadata_changes':approved,'full_runtime_byte_for_byte_equal':old['runtime_existing']==new['runtime_existing'],'rules':['.DS_Store','diagnostics/**/.DS_Store'],'no_scoped_rule_applies_to_A_or_B':True}
