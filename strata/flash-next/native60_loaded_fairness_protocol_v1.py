"""Owned native2 token-receipt adapter; default scheduler and delayed second arrival."""
import json,re,time
from pathlib import Path
from native_source41_cli_protocol_v4 import Protocol as Protected
from batch_numerical_protocol_v2 import Roster

def require(ok,msg):
 if not ok:raise ValueError(msg)
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def command_for(row):
 prefix='GEN %d'%row['max_new'] if row['slot']<0 else 'BGEN %d %d'%(row['slot'],row['max_new'])
 return prefix+' fresh=1 pin=1536 ckpt=1 temperature=0 rid=%d '%row['rid']+' '.join(map(str,row['ids']))
class LongRoster(Roster):
 def submit(self,rid,slot,ids,max_new):
  require(max_new in (64,512),'Only declared incoming64 or long512 bound')
  super().submit(rid,slot,ids,64);self.requests[rid]['max_new']=max_new
class Protocol(Protected):
 @staticmethod
 def is_protocol(line):return line.startswith(('READY','INFO ','ERR','FATAL','RESUME ','REUSED ','PP ','T ','LP ','DONE ','BT ','BADM ','BDONE ','SBF batch_event '))
 def __init__(self,command,directory,*args,**kwargs):
  self.schedule=json.loads((Path(directory)/'recipe.json').read_bytes())['requests'];self.batch=LongRoster(2);self.completed={}
  require(not (Path(directory)/'NO_CAPTURE_ARM').exists() and not (Path(directory)/'ARM').exists(),'Raw observer must never be armed')
  super().__init__(command,directory,*args,**kwargs)
  for line in self.stdout:
   if line.startswith('INFO '):self.batch.consume(line)
  self.batch.capacity()
 def request(self,label,ids,**kwargs):
  index=int(label.removeprefix('request-'))-1;want=self.schedule[index];require(ids==want['ids'],'Actual authentic original owner input differs')
  if index in self.completed:return self.completed[index]
  serial=want['slot']<0;group=[(index,want)] if serial else [(i,row) for i,row in enumerate(self.schedule) if row['cohort']==want['cohort']];require(len(group)==2,'Each loaded cohort must have two original owners');pending=list(group);lines=[];times=[];sends={};deadline=time.monotonic()+1800;arrival=None
  def submit(i,row):
   if not serial:self.batch.submit(row['rid'],row['slot'],row['ids'],row['max_new'])
   command=command_for(row);before=time.monotonic_ns();self.send(command);after=time.monotonic_ns();sent=self.sent[-1]
   require(before<=sent['monotonic_ns']<=after,'Original same-host send chronology differs')
   sends[i]=dict(command=command,send_before_monotonic_ns=before,send_monotonic_ns=sent['monotonic_ns'],send_after_monotonic_ns=after)
  first_i,first_row=pending.pop(0);submit(first_i,first_row)
  while True:
   stamp,line=self.q.get(timeout=max(.001,deadline-time.monotonic()));require(line is not None,'Native EOF/quota failure inside latency request');require(time.monotonic()<deadline,'Bounded native2 latency deadline');require(not line.startswith(('ERR','FATAL')),'Native latency refusal '+line);lines.append(line);times.append(stamp)
   if serial:
    if line.startswith('DONE ') and fields(line).get('rid')==str(want['rid']):break
   else:
    self.batch.consume(line);long=self.batch.requests[first_row['rid']]
    if pending and len(long['generated'])>=256:
     require(len(long['generated'])==256 and long['admitted'] and not long['done'],'Incoming prompt must arrive after actual256th token while long owner remains live')
     i,row=pending.pop(0);arrival=dict(long_emitted_ids=list(long['generated']),long_slotgen=long['slotgen'],trigger_receive_monotonic_seconds=stamp,trigger_line=line);submit(i,row);arrival.update(sends[i])
    if long['done'] and pending:raise ValueError('Long natural output ended before actual256-token incoming arrival')
    if not pending and all(self.batch.requests[row['rid']]['done'] for _,row in group):break
  require(not (self.directory/'NO_CAPTURE_ARM').exists() and not (self.directory/'ARM').exists(),'Unexpected heavy observer ARM')
  receipt=dict(cohort=want['cohort'],commands=[sends[i]['command'] for i,_ in group],sends=[sends[i] for i,_ in group],events=[dict(line=line,host_monotonic_seconds=stamp) for line,stamp in zip(lines,times)],arrival=arrival,clock_domain='same producer host Python time.monotonic/monotonic_ns')
  (self.directory/('cohort-%d.json'%want['cohort'])).write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
  for i,row in group:
   own=[(line,stamp) for line,stamp in zip(lines,times) if fields(line).get('rid')==str(row['rid']) and line.startswith(('T ','BT ','DONE ','BADM ','BDONE '))];output=[int(line.split()[1 if line.startswith('T ') else 2]) for line,_ in own if line.startswith(('T ','BT '))];terminal=next(line for line,_ in reversed(own) if line.startswith('DONE ' if serial else 'BDONE '));send=sends[i]['send_monotonic_ns']
   raw=dict(command=sends[i]['command'],ids=row['ids'],fresh=1,pin=1536,rid=row['rid'],slot=row['slot'],cohort=row['cohort'],output_ids=output,stdout=[line for line,_ in own],stdout_monotonic_seconds=[stamp for _,stamp in own],stdout_seconds=[stamp-send/1e9 for _,stamp in own],done=terminal,LP=[],stop_sent=False,cancel_requested=None,send_monotonic_ns=send,phase_directory=str(self.directory.resolve()))
   self.completed[i]=raw;(self.directory/('request-%d.json'%(i+1))).write_text(json.dumps(raw,indent=2)+'\n',encoding='ascii')
  return self.completed[index]
