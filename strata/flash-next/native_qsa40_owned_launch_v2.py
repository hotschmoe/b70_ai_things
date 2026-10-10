"""Original owned attach/session fence; absence alone never retires a launch."""
import contextlib,os,signal,time
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical

def require(ok,msg):
 if not ok:raise ValueError(msg)
def identity(pid):
 text=Path('/proc',str(pid),'stat').read_text();tail=text[text.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(tail[19]),'session':int(tail[3]),'state':tail[0]}
def session_members(session):
 found=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:row=identity(int(path.name))
  except(FileNotFoundError,ProcessLookupError):continue
  if row['session']==session and row['state']not in('Z','X'):found.append(row)
 return sorted(found,key=lambda r:r['pid'])
def write(path,value):Path(path).write_text(canonical(value)+'\n',encoding='ascii')
@contextlib.contextmanager
def deferred_signals():
 pending=[];old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 for sig in old:signal.signal(sig,lambda number,frame:pending.append(number))
 try:yield
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
  if pending:raise InterruptedError('Deferred owned launch signal '+str(pending[0]))
def intent(directory,command):
 write(directory/'launch-intent.json',{'owner':identity(os.getpid()),'command':command,'started_epoch':time.time(),'launch_planned':True})
def launched(directory,proc,command):
 row={'owner':identity(os.getpid()),'client':identity(proc.pid),'command':command,'started_epoch':time.time()};require(row['client']['session']==proc.pid,'Original owned attach must create independent session');write(directory/'launch.json',row);return row

def retired(directory,proc):
 saved=read_unique(directory/'launch.json');require(proc.poll()is not None,'Original attach not terminal');members=session_members(saved['client']['session']);require(not members,'Original attach session/descendants still live');row={'client':saved['client'],'command':saved['command'],'return_code':proc.returncode,'session_members':members,'retired_epoch':time.time()};write(directory/'launch-retired.json',row);return row

def fence(directory):
 directory=Path(directory)
 if not(directory/'launch-intent.json').exists():return True
 if not(directory/'launch.json').exists():
  if(directory/'launch-failure.json').exists():
   failure=read_unique(directory/'launch-failure.json');intent=read_unique(directory/'launch-intent.json');return failure['actual_client_started']is False and failure['command']==intent['command']
  return False
 if not(directory/'launch-retired.json').exists():return False
 row=read_unique(directory/'launch-retired.json');saved=read_unique(directory/'launch.json');return row['client']==saved['client']and row['command']==saved['command']and row['session_members']==[]and not session_members(saved['client']['session'])
