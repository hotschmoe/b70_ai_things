#!/usr/bin/env python3
"""Numerical protocol with one producer-merged OS pipe; never reorder two readers."""
import subprocess,threading,time,queue
from pathlib import Path
from serial_prefix_qualification_v6 import Protocol as Previous

class MergedProtocol(Previous):
 def __init__(self,command,directory,ready_timeout=900):
  self.directory=directory;self.stdout=[];self.stderr=[];self.q=queue.Queue();self.err=(directory/'engine.combined.log').open('w',encoding='ascii');self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True);self.threads=[]
  def pump():
   for source in self.p.stdout:
    line=source.rstrip('\n');self.err.write(line+'\n');self.err.flush()
    if self.is_protocol(line):self.stdout.append(line);self.q.put((time.monotonic(),line))
    else:self.stderr.append(line)
   self.q.put((time.monotonic(),None))
  thread=threading.Thread(target=pump,daemon=True);thread.start();self.threads.append(thread);deadline=time.monotonic()+ready_timeout
  while True:
   _,line=self.q.get(timeout=max(.001,deadline-time.monotonic()))
   if line is None:raise ValueError('Engine ended before readiness')
   if line.startswith('ERR'):raise ValueError(line)
   if line.startswith('READY'):self.ready=line;break
 @staticmethod
 def is_protocol(line):return line.startswith(('READY','ERR','RESUME ','REUSED ','PP ','T ','LP ','DONE '))
 def close(self):
  if self.p.poll() is None:self.send('QUIT')
  try:rc=self.p.wait(timeout=90)
  except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=20);rc=self.p.returncode
  for thread in self.threads:thread.join(timeout=5)
  self.err.close();(self.directory/'engine.protocol.log').write_text('\n'.join(self.stdout)+'\n',encoding='ascii');return rc
