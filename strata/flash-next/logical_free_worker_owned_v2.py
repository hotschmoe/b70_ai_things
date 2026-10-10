"""Owned CPU worker session retirement; no GPU/Docker/model operations."""
import os,signal,subprocess,threading,time
from pathlib import Path

def require(ok,message):
 if not ok:raise ValueError(message)
def session_rows(sid):
 rows=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:text=(path/'stat').read_text()
  except FileNotFoundError:continue
  end=text.rfind(')');fields=text[end+2:].split()
  require(end>=0 and len(fields)>=20,'Malformed process stat in owned-session census')
  if int(fields[3])==sid:rows.append({'pid':int(path.name),'state':fields[0],'session':sid,'start_ticks':int(fields[19])})
 return rows

def process_start(pid):
 text=Path('/proc',str(pid),'stat').read_text();end=text.rfind(')');fields=text[end+2:].split();require(end>=0 and len(fields)>=20,'Owned worker stat malformed');return int(fields[19])

def retire(child,start_ticks):
 if child is None:return {'actual_worker_launched':False,'owned_session_empty':True}
 # A new session can be inherited only by this worker's descendants. Census
 # metadata is PID/session/start/state only, never environment or command line.
 while child.poll()is None:
  child.kill();child.wait()
 child.wait();sid=child.pid;observed=[]
 while True:
  rows=session_rows(sid)
  if not rows:break
  leader=[r for r in rows if r['pid']==sid]
  if leader and leader[0]['start_ticks']!=start_ticks:return {'actual_worker_launched':True,'child_pid':child.pid,'return_code':child.returncode,'process_terminal':True,'owned_session_empty':True,'foreign_reused_session_observed':True,'unexpected_descendant_rows':rows}
  observed.extend(rows)
  for row in rows:
   if row['state']in ('Z','X'):continue
   current=[r for r in session_rows(sid)if r['pid']==row['pid']and r['start_ticks']==row['start_ticks']]
   if current:
    try:os.kill(row['pid'],signal.SIGKILL)
    except ProcessLookupError:pass
  # Failure/interruption cannot authorize return/reuse while the owned session
  # still exists. Frozen parser/worker source has no child-process spawn route.
  time.sleep(.01)
 return {'actual_worker_launched':True,'child_pid':child.pid,'return_code':child.returncode,'process_terminal':True,'owned_session_empty':True,'foreign_reused_session_observed':False,'unexpected_descendant_rows':observed}

def execute(command,raw,stdout_path,stderr_path,empty,*,timeout=120):
 require(threading.current_thread()is threading.main_thread(),'Owned parser invocation requires main-thread signal ownership');child=None;start_ticks=None;rows=None;old={};interrupted=[];pending=None;out=None;err=None
 def stop(sig,frame):
  interrupted.append(sig);prior=old.get(sig)
  try:
   if callable(prior):prior(sig,frame)
  finally:raise KeyboardInterrupt('Owned parser interrupted by signal '+str(sig))
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):old[sig]=signal.getsignal(sig);signal.signal(sig,stop)
 try:
  try:
   out=Path(stdout_path).open('xb');err=Path(stderr_path).open('xb')
   child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=empty,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},start_new_session=True)
   start_ticks=process_start(child.pid);child.communicate(raw,timeout=timeout)
  except BaseException as exc:pending=exc
  finally:
   # Every catchable interruption/error waits for actual worker/session
   # retirement and closed stdin/output descriptors. Cleanup interruptions
   # cannot authorize a partial return or reuse.
   while True:
    try:
     if child is not None and start_ticks is None:start_ticks=process_start(child.pid)
     rows=retire(child,start_ticks)
     if child is not None and child.stdin is not None and not child.stdin.closed:
      try:child.stdin.close()
      except BrokenPipeError:pass
     for handle in (out,err):
      if handle is not None and not handle.closed:handle.close()
     break
    except BaseException as exc:
     if pending is None:pending=exc
     if child is not None and child.poll()is None:child.kill()
     time.sleep(.01)
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
 if pending is not None:
  pending.owned_worker_retirement={'child_pid':child.pid if child is not None else None,'worker_start_ticks':start_ticks,'actual_worker_launched':child is not None,'process_terminal':child is None or child.poll()is not None,'owned_session_empty':rows is not None and rows['owned_session_empty']is True,'retirement':rows,'stdout_stderr_regular_sinks_closed':all(handle is None or handle.closed for handle in (out,err)),'stop_signals':interrupted}
  raise pending
 require(rows is not None and rows['owned_session_empty']is True and not rows.get('foreign_reused_session_observed',False) and not rows.get('unexpected_descendant_rows',[]),'Actual empty owned worker session without foreign reuse required');rows.update(worker_start_ticks=start_ticks,stdout_stderr_regular_sinks_closed=True,stdout_stderr_pipe_drain_claimed=False,stop_signals=interrupted);return rows
