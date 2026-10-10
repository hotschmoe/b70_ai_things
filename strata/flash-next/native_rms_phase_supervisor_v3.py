"""Owned phase timeout: cleanup Docker phase BEFORE stdout/parent drain."""
import subprocess,threading,time
import serial37_stdout_capture_v2 as capture

def supervise(argv,path,timeout,phase_cleanup=None,active=None,drain_timeout=10):
 status={'eof':False,'error':None};error=None;cleanup_error=None;begin=time.time()
 with path.open('w') as log:
  proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,pass_fds=tuple(fd for fd in (8,9) if __import__('os').path.exists('/proc/self/fd/'+str(fd))))
  reader=threading.Thread(target=capture.forward,args=(proc.stdout,log,lambda line:None,status),daemon=True);reader.start()
  if active is not None:active.append((proc,reader,str(path)))
  try:rc=proc.wait(timeout=timeout)
  except BaseException as exc:
   error=type(exc).__name__+': '+str(exc);proc.terminate()
   # Docker attach can keep the client and pipe alive after client TERM.
   # Inspect/stop/remove exact owned phase before waiting for inherited EOF.
   if phase_cleanup is not None:
    try:phase_cleanup()
    except BaseException as failure:cleanup_error=type(failure).__name__+': '+str(failure)
   if phase_cleanup is not None and cleanup_error is not None:
    # Only kill our own client, never the rejected foreign container. Outer
    # owner cleanup retains leases until that remaining phase is terminal.
    proc.kill()
   while proc.poll() is None:
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   rc=proc.wait()
  reader.join(drain_timeout)
  if reader.is_alive():
   error=(error+'; ' if error else '')+'stdout drain deadline'
   if phase_cleanup is not None:
    try:phase_cleanup()
    except BaseException as failure:cleanup_error=type(failure).__name__+': '+str(failure)
  while reader.is_alive():reader.join(1)
  proc.stdout.close()
 row=capture.completed(reader,status,path);row.update(return_code=rc,started_command_epoch=begin,command=argv,command_error=error,phase_cleanup_error=cleanup_error)
 row['passed']=row['passed'] and rc==0 and error is None and cleanup_error is None
 return row
