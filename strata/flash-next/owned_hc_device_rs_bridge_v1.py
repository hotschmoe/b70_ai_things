"""Root-owned interactive arithmetic phase; cleanup precedes pipe retirement."""
import queue,re,subprocess,threading,time
from pathlib import Path
import serial37_stdout_capture_v2 as capture
from owned_hc_device_rs_protocol_v1 import OwnDeviceRsClient,require
def run_phase(argv,path,work,phase_cleanup,start_admission,active,timeout=7200):
 path=Path(path);status={'eof':False,'error':None};messages=queue.Queue(maxsize=1024);started=time.time();error=None;cleanup_error=None;result=None;client=None;ready=None;done=None
 with path.open('w',encoding='ascii') as log:
  proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,pass_fds=tuple(fd for fd in (8,9) if __import__('os').path.exists('/proc/self/fd/'+str(fd))))
  def emit(line):
   require(len(line)<=4096 and line.endswith('\n'),'Complete bounded helper stdout line required')
   if line.startswith('OWNRS37_'):messages.put_nowait(line)
  reader=threading.Thread(target=capture.forward,args=(proc.stdout,log,emit,status),daemon=True);reader.start();active.append((proc,reader,str(path)))
  def receive():
   require(time.time()-started<timeout,'Owned arithmetic phase deadline exceeded');line=messages.get(timeout=min(30,max(.01,timeout-(time.time()-started))));require(not line.startswith('OWNRS37_ERROR'),'Actual arithmetic helper error '+line);return line
  def exchange(command):proc.stdin.write(command);proc.stdin.flush();return receive()
  try:
   ready=receive();require(re.fullmatch(r'OWNRS37_READY pid=[1-9][0-9]* backend=level_zero affinity=0 width=4 maximum=512\n',ready) is not None,'Exact actual owned device READY required');qualification=start_admission(ready);client=OwnDeviceRsClient(exchange,qualification);result=work(client);model_terminal=time.time();require(model_terminal-started<timeout,'Original model work exceeded owned phase deadline');proc.stdin.write('QUIT\n');proc.stdin.flush();proc.stdin.close();done=receive();require(done==f'OWNRS37_DONE requests={len(client.records)} graph_retired=1 owned_allocations_freed=1\n','Actual arithmetic request/free/graph terminal differs');rc=proc.wait(timeout=30)
  except BaseException as exc:
   error=type(exc).__name__+': '+str(exc);model_terminal=time.time();proc.terminate()
   try:phase_cleanup()
   except BaseException as exc:cleanup_error=type(exc).__name__+': '+str(exc)
   if proc.poll() is None:proc.kill()
   rc=proc.wait()
  reader.join(10)
  if reader.is_alive():
   error=(error+'; ' if error else '')+'stdout drain deadline'
   try:phase_cleanup()
   except BaseException as exc:cleanup_error=type(exc).__name__+': '+str(exc)
  while reader.is_alive():reader.join(1)
  if not proc.stdin.closed:
   try:proc.stdin.close()
   except BrokenPipeError:pass
  proc.stdout.close()
 row=capture.completed(reader,status,path);row.update(command=argv,started_command_epoch=started,return_code=rc,command_error=error,phase_cleanup_error=cleanup_error);row['passed']=row['passed'] and rc==0 and error is None and cleanup_error is None
 require(messages.empty(),'Foreign/duplicate arithmetic marker after terminal')
 return row,{'ready':ready,'done':done,'records':client.records if client else [],'qualification':client.qualification if client else None,'model_computation_terminal_epoch':model_terminal,'work':result}
