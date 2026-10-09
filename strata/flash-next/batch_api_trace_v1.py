#!/usr/bin/env python3
"""Diagnostic actual API/native boundary observer; no cache/scheduler/math edits."""
import hashlib,json,os,struct,subprocess,sys,threading,time
from pathlib import Path

def token_sha(ids):return hashlib.sha256(b''.join(struct.pack('<I',int(t)) for t in ids)).hexdigest()

class PipeObserver:
 def __init__(self,raw,callback,direction):self.raw=raw;self.callback=callback;self.direction=direction
 def __getattr__(self,name):return getattr(self.raw,name)
 def write(self,text):
  for line in text.splitlines():self.callback(self.direction,line)
  return self.raw.write(text)
 def __iter__(self):
  for line in self.raw:self.callback(self.direction,line.rstrip('\n'));yield line
 def readline(self,*a):
  line=self.raw.readline(*a)
  if line:self.callback(self.direction,line.rstrip('\n'))
  return line

def main():
 sys.path.insert(0,'/src');from serve import server
 from c1_trace_contract import pinned_eos
 cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());eos=pinned_eos(cfg);trace=Path(os.environ['B70_BATCH_TRACE']);combined=Path(os.environ['B70_BATCH_NATIVE_LOG']);lock=threading.Lock();local=threading.local();counter=[0];seq=[0]
 def emit(row):
  with lock:
   seq[0]+=1;row=dict(row,sequence=seq[0],epoch=time.time())
   with trace.open('a',encoding='ascii') as f:f.write(json.dumps(row,ensure_ascii=True,sort_keys=True)+'\n')
 encode_original=server.Service.encode_prompt
 def encode(self,messages,tools,kwargs):
  ids=encode_original(self,messages,tools,kwargs);local.prompt={'ids':list(ids),'sha256':token_sha(ids),'messages':messages,'tools':tools,'template_kwargs':kwargs};return ids
 popen_original=server.popen
 def popen(name,command,*a,**kw):
  if '--serve' not in command:return popen_original(name,command,*a,**kw)
  # One producer OS stream before the real API readers; preserving their routing.
  kw['stderr']=subprocess.STDOUT;proc=popen_original(name,command,*a,**kw)
  def native(direction,line):
   emit({'kind':'native_'+direction,'engine_pid':proc.pid,'call':getattr(local,'call',None) if direction=='send' else None,'line':line})
   if direction=='receive':
    with lock:
     with combined.open('a',encoding='ascii') as f:f.write(line.encode('ascii','backslashreplace').decode('ascii')+'\n')
  proc.stdin=PipeObserver(proc.stdin,native,'send');proc.stdout=PipeObserver(proc.stdout,native,'receive');return proc
 generate_original=server.StrataEngine.generate
 def generate(self,ids,max_new,sampling,cancel,embeddings=None):
  with lock:counter[0]+=1;call=counter[0]
  local.call=call;sent=list(ids);prompt=getattr(local,'prompt',None);generated=[];closed=False;error=None;born=self.gen;pid=self.proc.pid
  emit({'kind':'engine_begin','call':call,'engine_pid':pid,'engine_generation':born,'submitted_ids':sent,'submitted_ids_sha256':token_sha(sent),'rendered_prompt':prompt,'rendered_matches_submitted':bool(prompt and prompt['ids']==sent),'max_new':max_new,'sampling':sampling,'embeddings':embeddings})
  iterator=generate_original(self,ids,max_new,sampling,cancel,embeddings)
  try:
   for token in iterator:
    if type(token) is int:generated.append(token)
    yield token
  except GeneratorExit:
   closed=True;iterator.close();raise
  except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
   last=dict(self.last or {});cancelled=cancel.is_set();rid=last.get('request_id');normal_eos=bool(generated) and generated[-1] in eos and last.get('finish')=='stop' and not cancelled
   if closed and not cancelled and not normal_eos and error is None:error={'type':'GeneratorExit','message':'Noncancelled consumer close without pinned EOS native stop'}
   if self.gen!=born and error is None:error={'type':'EngineGenerationChanged','message':'Native incarnation changed during API request'}
   emit({'kind':'engine_end','call':call,'engine_pid':pid,'engine_generation':born,'rid':rid,'generated_ids':generated,'generated_ids_sha256':token_sha(generated),'engine_last':last,'consumer_closed':closed,'cancelled':cancelled,'terminal_pinned_eos':normal_eos,'error':error,'pinned_eos_ids':eos,'scope':'API yielded IDs/segments only; actual consumed native history separately reconstructed'})
   local.call=None
 server.Service.encode_prompt=encode;server.popen=popen;server.StrataEngine.generate=generate
 # Observe original close without changing its QUIT/drain/state semantics.
 close_original=server.StrataEngine.close
 def close(self):
  proc=self.proc;error=None
  try:return close_original(self)
  except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
  finally:emit({'kind':'engine_close','engine_pid':getattr(proc,'pid',None),'exit_code':proc.poll() if proc else None,'error':error})
 server.StrataEngine.close=close;return server.main()
if __name__=='__main__':raise SystemExit(main())
