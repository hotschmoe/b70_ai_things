"""Capture exact owned terminal container logs before removal; bounded failures are explicit."""
import hashlib,json,os,subprocess,threading,time
from pathlib import Path
MAX_STREAM_BYTES=32<<30

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 path=Path(path);before=path.stat();digest=hashlib.sha256()
 with path.open('rb')as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
 after=path.stat();require(before==after,'Original log/source changed during bounded hash');return digest.hexdigest()
def capture(root,inspection,write,*,popen=subprocess.Popen,max_bytes=MAX_STREAM_BYTES,timeout=180):
 root=Path(root);identity=inspection['Id'];require(type(identity)is str and len(identity)==64 and all(c in '0123456789abcdef'for c in identity),'Exact owned inspected Docker ID required');require(inspection['State']['Running']is False,'Logs capture requires original owned terminal')
 require(type(max_bytes)is int and 1<=max_bytes<=MAX_STREAM_BYTES and type(timeout)is int and 1<=timeout<=180,'Typed bounded log capture required')
 receipt=root/'owned-container-logs-receipt.json'
 if receipt.exists():
  value=json.loads(receipt.read_bytes());binding(root,inspection,value,require_complete=False);return value
 command=['docker','logs','--timestamps',identity];paths={k:root/('owned-container-'+k+'.raw')for k in ('stdout','stderr')};sinks={k:p.open('xb')for k,p in paths.items()};row={'command':command,'container_id':identity,'terminal_inspection_sha256':sha(root/'owned-terminal-inspection.json'),'started_epoch':time.time(),'finished_epoch':None,'return_code':None,'error':None,'complete':False,'streams':{},'max_bytes_per_stream':max_bytes};process=None;threads=[];failures=[];lock=threading.Lock()
 def pump(kind,pipe):
  count=0;written=0;digest=hashlib.sha256();overflow=False
  try:
   while True:
    data=pipe.read(1<<16)
    if not data:break
    count+=len(data)
    if written+len(data)>max_bytes:
     overflow=True
     with lock:failures.append(kind+' capture byte limit exceeded')
     if process.poll()is None:process.kill()
    allowed=data[:max(0,max_bytes-written)];sinks[kind].write(allowed);digest.update(allowed);written+=len(allowed)
   sinks[kind].flush();os.fsync(sinks[kind].fileno())
  except BaseException as error:
   with lock:failures.append(kind+': '+type(error).__name__+': '+str(error))
  finally:
   pipe.close();row['streams'][kind]={'path':str(paths[kind].resolve()),'bytes':written,'observed_bytes':count,'sha256':digest.hexdigest(),'truncated':overflow}
 for kind,path in paths.items():row['streams'][kind]={'path':str(path.resolve()),'bytes':0,'observed_bytes':0,'sha256':hashlib.sha256(b'').hexdigest(),'truncated':False}
 try:
  process=popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  for kind in ('stdout','stderr'):
   thread=threading.Thread(target=pump,args=(kind,getattr(process,kind)));threads.append(thread);thread.start()
  try:row['return_code']=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   failures.append('Original Docker logs capture timeout');process.kill();row['return_code']=process.wait()
 except BaseException as error:
  failures.append(type(error).__name__+': '+str(error))
 finally:
  if process is not None and process.poll()is None:process.kill();process.wait()
  for thread in threads:thread.join()
  for sink in sinks.values():sink.close()
  row['finished_epoch']=time.time();row['error']='; '.join(failures)or None;row['complete']=row['return_code']==0 and row['error']is None and set(row['streams'])=={'stdout','stderr'}and all(not v['truncated']for v in row['streams'].values());write(receipt,row)
 return row

def binding(root,inspection,row,*,require_complete=True):
 root=Path(root);require(row['command']==['docker','logs','--timestamps',inspection['Id']]and row['container_id']==inspection['Id']and inspection['State']['Running']is False,'Original exact owned terminal log command differs');require(row['terminal_inspection_sha256']==sha(root/'owned-terminal-inspection.json'),'Original inspected log owner changed');require((type(row['return_code'])is int or row['return_code']is None)and type(row['complete'])is bool and type(row['max_bytes_per_stream'])is int and 1<=row['max_bytes_per_stream']<=MAX_STREAM_BYTES,'Typed original log capture required');require(type(row['started_epoch'])in(int,float)and type(row['finished_epoch'])in(int,float)and 0<row['started_epoch']<=row['finished_epoch'],'Original log chronology differs');require(set(row['streams'])=={'stdout','stderr'},'Both original Docker log streams required')
 for kind,value in row['streams'].items():
  path=root/('owned-container-'+kind+'.raw');require(path.is_file()and not path.is_symlink()and value['path']==str(path.resolve())and type(value['bytes'])is int and value['bytes']==path.stat().st_size<=row['max_bytes_per_stream']and type(value['observed_bytes'])is int and value['observed_bytes']>=value['bytes']and type(value['truncated'])is bool and value['sha256']==sha(path),'Original captured Docker log bytes changed')
 if require_complete:require(row['complete']is True and row['return_code']==0 and row['error']is None and all(v['truncated']is False and v['bytes']==v['observed_bytes']for v in row['streams'].values()),'Incomplete/failed Docker logs cannot qualify normal closure')
 return {'actual_original_owned_logs_before_removal':row['complete'],'bounded_failure_without_complete_logs_as_normal':False}
