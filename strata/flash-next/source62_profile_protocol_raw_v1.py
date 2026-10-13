"""Original bounded native pipe plus CPU RAW receive points and explicit RIDs."""
import json,os,time
from pathlib import Path
from source61_ple_clean_abba_contract_v1 import require

def raw_ns():return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
class ObservedQueue:
 def __init__(self,owner,queue):self.owner=owner;self.queue=queue
 def put(self,value,*args,**kwargs):
  point=raw_ns();owner=self.owner
  require(isinstance(value,tuple) and len(value)==2,'Exact original native pump tuple required')
  mono,line=value;row=dict(seq=len(owner.raw_receives),rid=owner.active_rid,host_RAW_ns=point,original_MONOTONIC_seconds=mono,line=line,clock='CLOCK_MONOTONIC_RAW',observer='CPU point after original stdout validation/before original queue put')
  encoded=json.dumps(row,separators=(',',':'),ensure_ascii=True)+'\n';owner.raw_bytes+=len(encoded)
  require(owner.raw_bytes<=128<<10 and len(encoded)<=4096,'Bounded host RAW observer quota exceeded')
  owner.raw_receives.append(row)
  with (owner.directory/'host-protocol-RAW.jsonl').open('a',encoding='ascii') as f:f.write(encoded)
  return self.queue.put(value,*args,**kwargs)
 def get(self,*args,**kwargs):return self.queue.get(*args,**kwargs)

def protocol_class(base,c,life,closure):
 class Protocol(base):
  def __init__(self,command,directory,*args,**kwargs):
   self.roster=c.read(Path(directory)/'recipe.json')['requests'];self.active_rid=0;self.raw_receives=[];self.raw_bytes=0
   super().__init__(command,directory,*args,**kwargs)
   m=c.read(self.directory.parent/'prepared.snapshot.json');name=command[command.index('--name')+1];obj=life.inspect_owned(name,command,m['runtime']['image'],closure());pid=obj['State']['Pid']
   own=os.readlink('/proc/self/ns/time');native=os.readlink('/proc/%d/ns/time'%pid);require(own==native,'Actual native/client time namespace must match')
   c.write(self.directory/'host-clock-owner.json',dict(passed=True,client_pid=os.getpid(),native_host_pid=pid,container_id=obj['Id'],native_namespace_pid=1,client_time_namespace=own,native_time_namespace=native,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),RAW_clock='CLOCK_MONOTONIC_RAW',recorded_epoch=time.time(),this_CPU_observer_GPU_clock_queries_added=False,this_CPU_observer_GPU_waits_added=False,internal_sampling_or_preparation_coverage_qualified=False))
   self.q=ObservedQueue(self,self.q)
  def send(self,line):
   if line.startswith('GEN '):
    require(self.active_rid in (1,2),'Explicit current serial RID required');words=line.split();line=' '.join(words[:2]+['rid='+str(self.active_rid)]+words[2:])
    self.request_packet=line;self.request_send_RAW_before_ns=raw_ns();super().send(line);self.request_send_RAW_after_ns=raw_ns()
   else:super().send(line)
  def request(self,label,ids,**kwargs):
   index=int(label.removeprefix('request-'))-1;want=self.roster[index];require(ids==want['ids'],'Exact authentic cached profile input required');self.active_rid=want['rid'];start=len(self.raw_receives);began=raw_ns()
   kwargs.update(fresh=want['fresh'],pin=1536,max_new=8)
   try:raw=super().request(label,ids,**kwargs)
   finally:self.active_rid=0
   raw.update(command=self.request_packet,profile_request_rid=want['rid'],profile_send_RAW_before_ns=self.request_send_RAW_before_ns,profile_send_RAW_after_ns=self.request_send_RAW_after_ns,profile_call_RAW_begin_ns=began,profile_call_RAW_end_ns=raw_ns(),profile_RAW_receive_rows=list(self.raw_receives[start:]),clock_basis='Actual client/native shared RAW namespace; original MONOTONIC intervals preserved')
   c.write(self.directory/(label+'.json'),raw);return raw
 return Protocol
