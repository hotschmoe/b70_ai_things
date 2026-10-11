"""Original merged native protocol; source41 only adds bounded pipe recording."""
import contextlib,json,os,queue,signal,selectors,subprocess,threading,time
from pathlib import Path
from serial_prefix_qualification_v8 import Protocol as Commands
from merged_numerical_protocol_v2 import MergedProtocol
from serial37_stdout_capture_v2 import completed
@contextlib.contextmanager
def deferred_source_signals():
 pending=[];old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1)}
 for sig in old:signal.signal(sig,lambda number,frame:pending.append(number))
 try:yield
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
  if pending:raise InterruptedError('Deferred source41 owned launch signal '+str(pending[0]))

class Protocol(Commands):
 is_protocol=staticmethod(MergedProtocol.is_protocol)
 def send(self,line):
  self.p.stdin.write(line+"\n");self.p.stdin.flush();self.sent.append({"seq":len(self.sent),"packet":line+"\n","monotonic_ns":time.monotonic_ns()});(self.directory/"native.stdin.jsonl").write_text("".join(json.dumps(row)+"\n"for row in self.sent),encoding="ascii")
 def __init__(self,command,directory,cleanup,ready_timeout=1800,drain_timeout=10,active=None,pipe_bytes=64<<20,pipe_lines=100000):
  if not 0<drain_timeout<=10:raise ValueError('Bounded owned stdout drain timeout')
  if type(pipe_bytes)is not int or type(pipe_lines)is not int or (pipe_bytes,pipe_lines)not in((64<<20,100000),(128<<20,300000)):raise ValueError('Exact normal/fidelity pipe quota required')
  self.pipe_bytes=pipe_bytes;self.pipe_lines=pipe_lines
  self.drain_timeout=drain_timeout
  from native_qsa40_owned_launch_v9 import intent,launched
  self.sent=[];self.threads=[];self.launch=None;self.command=command;self.setup_error=None
  self.directory=directory;self.stdout=[];self.stderr=[];self.q=queue.Queue();self.cleanup=cleanup;self.status={'eof':False,'error':None,'recorded_bytes':0,'recorded_lines':0,'max_line_bytes':0,'observed_bytes':0,'observed_lines':0,'pipe_byte_limit':pipe_bytes,'pipe_line_limit':pipe_lines,'line_byte_limit':1<<20};self.started_epoch=time.time();self.err=(directory/'engine.combined.log').open('w',encoding='ascii');intent(directory,command)
  defer=deferred_source_signals();defer.__enter__()
  try:self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
  except BaseException as exc:
   from native_qsa40_owned_launch_v9 import write
   write(directory/'launch-failure.json',{'command':command,'actual_client_started':False,'error':type(exc).__name__+': '+str(exc)})
   self.err.close();defer.__exit__(None,None,None);raise
  if active is not None:active.append(self)
  try:
   self.launch=launched(directory,self.p,command)
   def pump():
    self.status['started_epoch']=time.time()
    try:
     recorded=0;lines=0;failed=False
     while True:
      source=self.p.stdout.readline((1<<20)+1)
      if not source:break
      line_bytes=len(source.encode('ascii'));lines+=1;recorded+=line_bytes;self.status['observed_bytes']=recorded;self.status['observed_lines']=lines;self.status['max_line_bytes']=max(self.status['max_line_bytes'],line_bytes)
      if len(source)>(1<<20)or recorded>self.pipe_bytes or lines>self.pipe_lines or not source.endswith('\n'):
       if not failed:self.status['error']='Bounded/truncated producer pipe';self.q.put((time.monotonic(),None));failed=True
       continue
      if failed:continue
      self.err.write(source);self.err.flush();self.status['recorded_bytes']+=line_bytes;self.status['recorded_lines']+=1;line=source[:-1]
      if self.is_protocol(line):self.stdout.append(line);self.q.put((time.monotonic(),line))
      else:self.stderr.append(line)
     self.status['eof']=True
    except BaseException as e:self.status['error']=type(e).__name__+': '+str(e)
    finally:self.status['completed_epoch']=time.time();self.q.put((time.monotonic(),None))
   reader=threading.Thread(target=pump,daemon=True);self.threads=[reader]
   reader.start()
   deadline=time.monotonic()+ready_timeout
   try:
    defer.__exit__(None,None,None)
    if not self.launch['client_identity_observed']:raise ValueError('Original started client identity unavailable; failed launch must retire')
    while True:
     _,line=self.q.get(timeout=max(.001,deadline-time.monotonic()))
     if line is None:raise ValueError('Engine ended or pipe quota failed before READY')
     if line.startswith('ERR'):raise ValueError(line)
     if line.startswith('READY'):self.ready=line;break
   except BaseException:raise
  except BaseException as exc:
   self.setup_error=type(exc).__name__+': '+str(exc)
   try:self.close(failed=True)
   finally:
    try:defer.__exit__(None,None,None)
    except BaseException:pass
   raise
 def close(self,failed=False):
  if hasattr(self,'capture'):return self.capture['return_code']
  reader=self.threads[0]if self.threads else None
  started=reader is not None and reader.ident is not None
  forced=failed or not started or self.launch is None
  if not started:
   # The original Popen is retained even if launch observer or Thread setup failed.
   # Drain synchronously while the exact owned cleanup callback resolves late creation.
   self.status.update(started_epoch=time.time(),error=self.setup_error or 'Native reader did not start')
   selector=selectors.DefaultSelector();os.set_blocking(self.p.stdout.fileno(),False);selector.register(self.p.stdout,selectors.EVENT_READ)
   size=0;next_cleanup=0
   try:
    with (self.directory/'setup-stdout.raw').open('xb')as recovery:
     while selector.get_map()or self.p.poll()is None:
      if time.monotonic()>=next_cleanup:
       try:self.cleanup()
       except BaseException as exc:self.status['cleanup_error']=type(exc).__name__+': '+str(exc)
       next_cleanup=time.monotonic()+1
      for key,_ in selector.select(.1):
       try:raw=os.read(key.fileobj.fileno(),65536)
       except BlockingIOError:continue
       if not raw:selector.unregister(key.fileobj);self.status['eof']=True
       elif size+len(raw)<=self.pipe_bytes:recovery.write(raw);size+=len(raw)
       else:self.status['quota_failed']=True
   finally:selector.close();self.status['completed_epoch']=time.time()
  else:
   try:
    if self.p.poll()is None:
     if forced:self.cleanup()
     else:self.send('QUIT')
    self.p.wait(timeout=90)
   except BaseException:
    forced=True
    while self.p.poll()is None:
     try:self.cleanup()
     except BaseException as exc:self.status['cleanup_error']=type(exc).__name__+': '+str(exc)
     try:self.p.wait(timeout=1)
     except subprocess.TimeoutExpired:pass
   reader.join(self.drain_timeout)
   while reader.is_alive():
    forced=True
    try:self.cleanup()
    except BaseException as exc:self.status['cleanup_error']=type(exc).__name__+': '+str(exc)
    reader.join(1)
  rc=self.p.wait()
  from native_qsa40_owned_launch_v9 import retired,session_members,write
  if (self.directory/'launch.json').is_file():
   while session_members(self.p.pid):
    try:self.cleanup()
    except BaseException as exc:self.status['cleanup_error']=type(exc).__name__+': '+str(exc)
    time.sleep(.2)
   self.launch_retirement=retired(self.directory,self.p)
  else:
   members=session_members(self.p.pid)
   while members:
    try:self.cleanup()
    except BaseException as exc:self.status['cleanup_error']=type(exc).__name__+': '+str(exc)
    time.sleep(.2);members=session_members(self.p.pid)
   self.launch_retirement={'client_pid':self.p.pid,'session':self.p.pid,'return_code':rc,'session_members':members,'retired_epoch':time.time(),'original_launch_observer_completed':False}
   write(self.directory/'setup-launch-retired.json',self.launch_retirement)
  self.p.stdout.close();self.p.stdin.close();self.err.close();(self.directory/'engine.protocol.log').write_text('\n'.join(self.stdout)+'\n',encoding='ascii')
  self.capture=completed(reader if started else None,self.status,self.directory/'engine.combined.log');self.capture.update(return_code=rc,forced_cleanup=forced,process_started_epoch=self.started_epoch,setup_error=self.setup_error)
  if forced or self.launch is None or not self.launch['client_identity_observed']:self.capture['passed']=False
  return rc
