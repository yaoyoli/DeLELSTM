import os,json,gzip,hashlib,stat,subprocess,datetime
from pathlib import Path
A=Path('/Users/yao/DeLELSTM-track-a');B=Path('/Users/yao/DeLELSTM');R=Path('/Users/yao/DeLELSTM-track-a-runtime');D=R/'diagnostics/stage3b/pm25/run-01'
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0',GIT_PAGER='cat')
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(n,o):
 p=D/n
 assert p.parent==D and not p.exists()
 with p.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def gz(n,o):
 assert not (D/n).exists()
 with gzip.open(D/n,'xt',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False)
def git(root,*args):
 assert args and args[0] in ['symbolic-ref','rev-parse','status','diff','ls-files','worktree']
 if args[0]=='worktree':assert args[1:] == ('list','--porcelain')
 p=subprocess.run(['git','-C',str(root),*args],env=ENV,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
 return {'command':p.args,'exit_code':p.returncode,'stdout':p.stdout.decode('utf-8'),'stderr':p.stderr.decode(errors='replace')}
def tree(root,exclude=()):
 files={};dirs=[];errors=[]
 def visit(p,rel):
  if rel and (rel=='.git' or any(rel==x or rel.startswith(x+'/') for x in exclude)):return
  try:
   st=p.lstat();rec={'mode':stat.S_IMODE(st.st_mode),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino}
   if stat.S_ISLNK(st.st_mode):rec.update(kind='symlink',link_target=os.readlink(p));files[rel]=rec
   elif stat.S_ISREG(st.st_mode):
    h=hashlib.sha256()
    with p.open('rb') as f:
     for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    end=p.stat()
    if (st.st_size,st.st_mtime_ns,st.st_ino)!=(end.st_size,end.st_mtime_ns,end.st_ino):raise RuntimeError('changed during read')
    rec.update(kind='file',sha256=h.hexdigest());files[rel]=rec
   elif stat.S_ISDIR(st.st_mode):
    dirs.append(rel)
    for c in sorted(p.iterdir()):visit(c,rel+'/'+c.name if rel else c.name)
   else:raise RuntimeError('unsupported entry')
  except Exception as e:errors.append({'path':str(p),'error':repr(e)})
 visit(root,'')
 return {'root':str(root),'files':files,'directories':dirs,'errors':errors,'entries':len(files),'bytes':sum(v['size'] for v in files.values() if v['kind']=='file')}
def state(root):
 calls={'branch':('symbolic-ref','--short','HEAD'),'head':('rev-parse','HEAD'),'status':('status','--porcelain=v1','--untracked-files=all'),'tracked_diff':('diff','--no-ext-diff','--no-textconv','--name-status'),'staged_diff':('diff','--cached','--no-ext-diff','--no-textconv','--name-status'),'unmerged':('ls-files','--unmerged'),'untracked':('ls-files','--others','--exclude-standard','-z'),'ignored':('ls-files','--others','--ignored','--exclude-standard','-z'),'worktrees':('worktree','list','--porcelain'),'git_dir':('rev-parse','--absolute-git-dir'),'common_dir':('rev-parse','--path-format=absolute','--git-common-dir')}
 s={k:git(root,*v) for k,v in calls.items()}
 gd=Path(s['git_dir']['stdout'].strip());cd=Path(s['common_dir']['stdout'].strip())
 s['index_sha256']=sha((gd/'index').read_bytes())
 s['operation_markers']={str(p/n):(p/n).exists() for p in {gd,cd} for n in ['MERGE_HEAD','MERGE_AUTOSTASH','REBASE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD','BISECT_LOG','rebase-apply','rebase-merge','sequencer','index.lock','HEAD.lock','config.lock','packed-refs.lock']}
 return s
def capture(label):
 return {'label':label,'started_at':now(),'states':{'A':state(A),'B':state(B)},'files_A':tree(A),'files_B':tree(B),'git_metadata':tree(B/'.git'),'runtime_existing':tree(R,('diagnostics/stage3b/pm25/run-01','results/stage3b/pm25/run-01')),'completed_at':now()}
def differences(old,new):
 return {k:{'equal':old[k]==new[k],'changed_files':sorted(p for p in set(old[k]['files'])|set(new[k]['files']) if old[k]['files'].get(p)!=new[k]['files'].get(p)),'errors':new[k]['errors']} for k in ['files_A','files_B','git_metadata','runtime_existing']}
