"""Original merged native protocol; source41 only adds bounded pipe recording."""
import contextlib,json,os,queue,signal,subprocess,threading,time
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
 def __init__(self,command,directory,cleanup,ready_timeout=1800,drain_timeout=10,active=None):
  if not 0<drain_timeout<=10:raise ValueError('Bounded owned stdout drain timeout')
  self.drain_timeout=drain_timeout
  from native_qsa40_owned_launch_v9 import intent,launched
  self.sent=[];self.threads=[]
  self.directory=directory;self.stdout=[];self.stderr=[];self.q=queue.Queue();self.cleanup=cleanup;self.status={'eof':False,'error':None};self.started_epoch=time.time();self.err=(directory/'engine.combined.log').open('w',encoding='ascii');intent(directory,command)
  defer=deferred_source_signals();defer.__enter__()
  try:self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
  except BaseException as exc:
   from native_qsa40_owned_launch_v9 import write
   write(directory/'launch-failure.json',{'command':command,'actual_client_started':False,'error':type(exc).__name__+': '+str(exc)})
   self.err.close();defer.__exit__(None,None,None);raise
  if active is not None:active.append(self)
  self.launch=launched(directory,self.p,command)
  def pump():
   self.status['started_epoch']=time.time()
   try:
    recorded=0;lines=0;failed=False
    while True:
     source=self.p.stdout.readline((1<<20)+1)
     if not source:break
     lines+=1;recorded+=len(source.encode('ascii'))
     if len(source)>(1<<20)or recorded>(64<<20)or lines>100000 or not source.endswith('\n'):
      if not failed:self.status['error']='Bounded/truncated producer pipe';self.q.put((time.monotonic(),None));failed=True
      continue
     if failed:continue
     self.err.write(source);self.err.flush();line=source[:-1]
     if self.is_protocol(line):self.stdout.append(line);self.q.put((time.monotonic(),line))
     else:self.stderr.append(line)
    self.status['eof']=True
   except BaseException as e:self.status['error']=type(e).__name__+': '+str(e)
   finally:self.status['completed_epoch']=time.time();self.q.put((time.monotonic(),None))
  reader=threading.Thread(target=pump,daemon=True);self.threads=[reader]
  try:reader.start()
  except BaseException as exc:
   from run_decode40_host_profile_v3 import setup_failure_drain
   self.cleanup();name=command[command.index('--name')+1];label=command[command.index('--label')+1].split('=',1)[1]
   recovery=setup_failure_drain(directory,name,command,command[-3],label,self.p,[('stdout',self.p.stdout)],type(exc).__name__+': '+str(exc))
   from native_qsa40_owned_launch_v9 import retired
   self.launch_retirement=retired(directory,self.p);self.p.stdin.close();self.err.close();self.capture={'passed':False,'return_code':self.p.returncode,'setup_failure':recovery};defer.__exit__(None,None,None);raise
  deadline=time.monotonic()+ready_timeout
  try:
   defer.__exit__(None,None,None)
   if not self.launch['client_identity_observed']:raise ValueError('Original started client identity unavailable; failed launch must retire')
   while True:
    _,line=self.q.get(timeout=max(.001,deadline-time.monotonic()))
    if line is None:raise ValueError('Engine ended or pipe quota failed before READY')
    if line.startswith('ERR'):raise ValueError(line)
    if line.startswith('READY'):self.ready=line;break
  except BaseException:self.close(failed=True);raise
 def close(self,failed=False):
  if hasattr(self,'capture'):return self.capture['return_code']
  forced=failed
  try:
   if self.p.poll() is None:
    if failed:self.cleanup()
    else:self.send('QUIT')
   rc=self.p.wait(timeout=90)
  except BaseException:
   forced=True;self.cleanup()
   while self.p.poll() is None:
    print('QSA40 failure retains leases until original attach settles',flush=True)
    self.cleanup()
    try:self.p.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
   rc=self.p.wait()
  self.threads[0].join(self.drain_timeout)
  if self.threads[0].is_alive():forced=True;self.cleanup()
  while self.threads[0].is_alive():self.threads[0].join(1)
  from native_qsa40_owned_launch_v9 import retired
  self.launch_retirement=retired(self.directory,self.p)
  self.p.stdout.close();self.p.stdin.close();self.err.close();(self.directory/'engine.protocol.log').write_text('\n'.join(self.stdout)+'\n',encoding='ascii')
  self.capture=completed(self.threads[0],self.status,self.directory/'engine.combined.log');self.capture.update(return_code=rc,forced_cleanup=forced,process_started_epoch=self.started_epoch)
  if forced or not self.launch['client_identity_observed']:self.capture['passed']=False
  return rc
