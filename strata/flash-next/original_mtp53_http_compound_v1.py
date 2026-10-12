"""Explicit original-compound HTTP bridge. Default OFF; no legacy flat-session view."""
import contextlib,hashlib,json,os,queue,re,stat,struct,threading,time
from pathlib import Path
MAGIC='ORIGINAL-SHIFTED-JOINT58-DIR2';CAP=1<<30
class CompoundBusy(ValueError):pass
class CompoundRefused(ValueError):
 def __init__(self,message,published=False):super().__init__(message);self.published=published

def mode(engine):
 spawn=getattr(engine,'spawn',None)
 return bool(spawn and len(spawn)>4 and (spawn[4]or {}).get('STRATA_ORIGINAL_MTP53_SERVING')=='1')
def gate(engine):return engine.__dict__.setdefault('_original_compound_gate',{'cv':threading.Condition(),'active':0,'exclusive':False})
@contextlib.contextmanager
def stream(engine,cancel):
 if not mode(engine):yield;return
 g=gate(engine)
 with g['cv']:
  while g['exclusive']:
   if cancel.is_set():raise CompoundBusy('original file operation owns admission')
   g['cv'].wait(.25)
  g['active']+=1
 try:yield
 finally:
  with g['cv']:g['active']-=1;g['cv'].notify_all()
@contextlib.contextmanager
def exclusive(engine):
 g=gate(engine)
 with g['cv']:
  if g['active']or g['exclusive']:raise CompoundBusy('original compound operation requires all streams retired')
  g['exclusive']=True
 try:
  # No generate owns ctl once stream gate is idle. Retain ctl through answer+
  # manifest verification, preventing admission on the same producer pipe.
  with engine.ctl:
   with engine.slot_cv:
    if any(engine.slot_busy):raise CompoundBusy('original native slots still busy')
   yield
 finally:
  with g['cv']:g['exclusive']=False;g['cv'].notify_all()
def identity(p):
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('compound regular single-owner file required')
 return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def file_digest(p):
 before=identity(p)
 if before[2]>CAP:raise ValueError('compound file byte budget')
 h=hashlib.sha256()
 with p.open('rb')as f:
  if os.fstat(f.fileno()).st_ino!=before[1]:raise ValueError('compound opened inode changed')
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 if identity(p)!=before:raise ValueError('compound file changed during read')
 return {'sha256':h.hexdigest(),'stat5':before,'bytes':before[2]}
def bounded_bytes(path,cap):
 before=identity(path)
 if before[2]>cap:raise ValueError('compound bounded metadata size')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  current=os.fstat(fd)
  if [current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns]!=before:raise ValueError('opened compound metadata changed')
  with os.fdopen(fd,'rb',closefd=False)as f:raw=f.read(cap+1)
  if len(raw)!=before[2]or identity(path)!=before:raise ValueError('compound metadata changed')
 finally:os.close(fd)
 return raw
def prefix_metadata(path):
 # Read only the bounded shifted header. Native LOAD independently validates
 # complete CP2 banks/hidden/head/digests and BOTH target images before restore.
 before=identity(path)
 with path.open('rb')as f:
  def number():
   raw=f.read(8)
   if len(raw)!=8:raise ValueError('original metadata truncated')
   return struct.unpack('<Q',raw)[0]
  n=number()
  if n!=len(b'ORIGINAL-SHIFTED-CP1')or f.read(n)!=b'ORIGINAL-SHIFTED-CP1':raise ValueError('original shifted schema changed')
  n=number()
  if not 1<=n<=4096:raise ValueError('original target prefix bound')
  raw=f.read(n*4)
  if len(raw)!=n*4:raise ValueError('original prefix truncated')
  ids=list(struct.unpack('<%dI'%n,raw));look=number()
  if any(t>=248320 for t in ids)or look>=248320:raise ValueError('original prefix vocabulary bound')
  current_bytes=number();magic_size=number()
  if not 64<=current_bytes<=20<<20 or magic_size!=len(b'ORIGINAL34-CP2-LE4096')or f.read(magic_size)!=b'ORIGINAL34-CP2-LE4096':raise ValueError('original CP2 header changed')
  rid,generation,cells,head=number(),number(),number(),number()
  if not rid or not generation or cells!=n or head!=1:raise ValueError('original full current head/owner/count required')
 if identity(path)!=before:raise ValueError('original prefix metadata changed')
 return {'target_input_tokens':n,'target_prefix_sha256':hashlib.sha256(raw).hexdigest(),'original_lookahead':look,'source_rid':rid,'source_slot_generation':generation}
def inspect_tree(path,pinned=None):
 path=Path(path)
 if not path.is_absolute()or path.resolve()!=path or not path.is_dir()or path.is_symlink():raise ValueError('canonical original compound directory required')
 roster={};bytes_total=0;nodes=0
 def visit(folder,depth=0,want=None):
  nonlocal bytes_total,nodes
  if depth>64:raise ValueError('compound branch depth')
  lines=bounded_bytes(folder/'manifest.txt',8192).decode('ascii').splitlines()
  if len(lines)<4 or lines[0]!=MAGIC or not all(re.fullmatch('[0-9a-f]{64}',x)for x in lines[1:3])or not re.fullmatch('[0-9]{1,2}',lines[3]):raise ValueError('compound manifest schema')
  count=int(lines[3])
  if count>64 or len(lines)!=4+count or any(not re.fullmatch('[0-9a-f]{64}',x)for x in lines[4:])or want is not None and lines[1]!=want:raise ValueError('compound manifest pin/branches')
  expected={'target0.bin','target1.bin','original.bin','ple-previous.bin','manifest.txt'}|{'branch'+str(i)for i in range(count)}
  if {p.name for p in folder.iterdir()}!=expected:raise ValueError('compound exact directory roster')
  for name in sorted(expected):
   p=folder/name
   if p.is_symlink():raise ValueError('compound alias refused')
   if name.startswith('branch'):
    if not p.is_dir():raise ValueError('compound branch directory required')
    visit(p,depth+1,lines[4+int(name[6:])])
   else:
    nodes+=1
    if nodes>325:raise ValueError('compound bounded complete branch roster')
    row=file_digest(p);bytes_total+=row['bytes']
    if bytes_total>CAP:raise ValueError('compound directory byte budget')
    roster[str(p.relative_to(path))]=row
  return lines[1]
 digest=visit(path,want=pinned);meta=prefix_metadata(path/'original.bin')
 return {'kind':'original-compound','compound_sha256':digest,'bytes':bytes_total,'raw_roster':roster,**meta,'native_full_payload_validation_pending':True}
def sidecar(path):return Path(str(path)+'.original-meta.json')
def write_sidecar(path,data):
 p=sidecar(path);raw=(json.dumps(data,sort_keys=True,ensure_ascii=True)+'\n').encode('ascii');fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  at=0
  while at<len(raw):
   n=os.write(fd,raw[at:])
   if n<=0:raise OSError('compound metadata write')
   at+=n
  os.fsync(fd)
 finally:os.close(fd)
def session_file(engine,action,path,pinned,EngineDied):
 path=Path(path);env=engine.spawn[4]or {};start=time.monotonic()
 if action not in('save','restore')or not mode(engine)or not path.is_absolute()or path.parent.resolve()!=path.parent or any(x in str(path)for x in '\r\n\0'):raise ValueError('explicit original compound action/canonical path required')
 if action=='restore'and not re.fullmatch('[0-9a-f]{64}',pinned or ''):raise ValueError('restore requires independently pinned compound_sha256')
 with exclusive(engine):
  if not engine.alive():raise EngineDied('original engine is not loaded')
  original_proc=engine.proc;born=engine.gen
  if action=='save':
   if path.exists()or sidecar(path).exists():raise ValueError('original publication requires new destination')
   before=None;packet='ORIGINAL_SAVE '+str(path);want='ORIGINAL_SAVED compound=1'
  else:
   metadata=sidecar(path)
   if metadata.is_symlink()or metadata.resolve()!=metadata:raise ValueError('original metadata alias')
   published=json.loads(bounded_bytes(metadata,2<<20));before=inspect_tree(path,pinned)
   if published['compound_sha256']!=pinned or published['raw_roster']!=before['raw_roster']or published['draft_contract_sha256']!=env['STRATA_ORIGINAL_MTP53_DRAFT_CONTRACT_SHA256']or published['target_contract_sha256']!=env['STRATA_ORIGINAL_MTP53_TARGET_CONTRACT_SHA256']:raise ValueError('original independent compound/source/byte roster changed')
   packet='ORIGINAL_LOAD '+pinned+' '+str(path);want='ORIGINAL_LOADED compound=1 activation=0'
  engine._send(packet);deadline=time.monotonic()+900;last_progress=-1
  while True:
   if time.monotonic()>deadline:raise engine._silent('original compound operation exceeded bounded900 seconds')
   if engine.proc is not original_proc or engine.gen!=born:raise EngineDied('original session owner changed')
   try:line=engine.lines.get(timeout=max(.001,deadline-time.monotonic()))
   except queue.Empty:raise engine._silent('original compound operation exceeded bounded900 seconds')from None
   if line is None:raise EngineDied('original producer EOF during compound operation')
   text=line.strip()
   if text==want:break
   if text.startswith('ERR original compound file:'):raise CompoundRefused(text,path.exists()if action=='save'else False)
   if text.startswith('FATAL')or text.startswith('MTP53_SERVING_FATAL'):raise EngineDied(text)
   if text.startswith(('INFO ','WARN ')):continue
   parts=text.split()
   if len(parts)==3 and parts[0]=='SESSION'and parts[1].isdigit()and parts[2].isdigit():
    done,total=map(int,parts[1:])
    if not last_progress<=done<=total<=CAP:raise EngineDied('original compound progress invalid')
    last_progress=done;continue
   if len(parts)==3 and parts[0]=='SWAIT'and parts[1].isascii()and parts[1].isalpha()and parts[2].isdigit()and 0<int(parts[2])<=900:continue
   engine.ended=True;raise EngineDied('original compound protocol lost step: '+text[:200])
  after=inspect_tree(path,pinned)
  if before is not None and before!=after:raise EngineDied('original consumed compound changed')
  if action=='save':
   metadata={**after,'draft_contract_sha256':env['STRATA_ORIGINAL_MTP53_DRAFT_CONTRACT_SHA256'],'target_contract_sha256':env['STRATA_ORIGINAL_MTP53_TARGET_CONTRACT_SHA256'],'original_engine_generation':born,'native_original_reply':want,'native_full_payload_validation_pending':False}
   try:write_sidecar(path,metadata)
   except BaseException as e:raise CompoundRefused('compound published but HTTP metadata publication failed: '+str(e),True)from e
  return {**{k:v for k,v in after.items()if k!='raw_roster'},'native_full_payload_validation_pending':False,'activation':'deferred_next_request'if action=='restore'else'none','snapshot_scope':'latest_complete_publication','file_operation_host_ms':(time.monotonic()-start)*1000}
def slot_action(service,slot,action,filename,pinned,EngineDied,filename_problem):
 def error(code,message,**extra):return code,{'error':{'code':code,'message':str(message),'type':'invalid_request_error'if code<500 else'server_error',**extra}}
 if slot!='0':return error(400,'original compound control endpoint is0; saved source_rid identifies actual private publication')
 if not service.slot_save_path:return error(501,'original compound directory store is disabled')
 why=filename_problem(filename)
 if why:return error(400,'filename '+why)
 path=str(Path(service.slot_save_path)/filename)
 with service.fifo:
  try:value=session_file(service.engine,action,path,pinned,EngineDied)
  except CompoundBusy as e:return error(409,e)
  except CompoundRefused as e:return error(500,e,published=e.published,durable=False)
  except(ValueError,OSError)as e:return error(400,e)
 return 200,{'id_slot':0,'filename':filename,**value,'legacy_flat_session_schema':False}
