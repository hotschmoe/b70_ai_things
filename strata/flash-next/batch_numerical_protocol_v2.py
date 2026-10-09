#!/usr/bin/env python3
"""Leased actual native producer stream and strict per-RID greedy BGEN parser."""
import json,queue,re,subprocess,threading,time
from pathlib import Path
from c1_serve_controller_combined_v8 import leased

def tags(line):
 fields=dict(re.findall(r'(\w+)=([^ ]+)',line));rid=int(fields.get('rid','0'));generation=int(fields.get('slotgen','-1'))
 if rid<=0 or generation<0:raise ValueError('Native request/slot identity absent')
 return rid,generation

class Roster:
 def __init__(self,slots):
  if slots not in (2,4,6):raise ValueError('Exact batch2/4/6 required')
  self.slots=slots;self.bt_slots=set();self.info={};self.requests={};self.owners={};self.events=[]
 def submit(self,rid,slot,ids,max_new):
  if rid in self.requests or type(rid) is not int or rid<=0 or not 0<=slot<self.slots or slot in self.owners or not 2<=max_new<=64 or not 1<=len(ids)<=2048 or any(type(t) is not int or not 0<=t<248320 for t in ids):raise ValueError('Invalid bounded request/private slot')
  self.requests[rid]={'rid':rid,'slot':slot,'ids':list(ids),'max_new':max_new,'generated':[],'slotgen':None,'done':None,'admission_done':None,'admitted':False,'cancel_requested':False};self.owners[slot]=rid
 def cancel(self,rid):
  r=self.requests[rid]
  if not r['admitted'] or r['done'] or r['slotgen'] is None:raise ValueError('Cancel needs a current live admitted identity')
  r['cancel_requested']=True;return f'BSTOP {r["slot"]} rid={rid} slotgen={r["slotgen"]}'
 def consume(self,line):
  if line.startswith('INFO '):
   for item in line.split()[1:]:
    if '=' in item:k,v=item.split('=',1);self.info[k]=int(v) if v.isdigit() else v
   return
  if line.startswith(('ERR ','FATAL ')):raise ValueError('Native refusal/failure: '+line)
  if line.startswith('SBF batch_event '):
   f=dict(re.findall(r'(\w+)=([^ ]+)',line));rows=int(f['rows']);mask=int(f['active_mask'])
   if f['completed']!='1' or not 1<=rows<=self.slots or mask.bit_count()!=rows or mask>=1<<self.slots:raise ValueError('Actual completed batch geometry invalid')
   self.events.append(f);return
  if not line.startswith(('T ','BT ','DONE ','BADM ','BDONE ')):return
  rid,generation=tags(line);r=self.requests[rid];part=line.split();kind=part[0]
  if r['done']:raise ValueError('Native token/completion after request terminal')
  if r['slotgen'] is None:
   if kind!='T' or generation<=0:raise ValueError('First admission token/generation absent')
   r['slotgen']=generation
  if generation!=r['slotgen']:raise ValueError('Stale native slot generation')
  if kind in ('BT','BADM','BDONE') and (int(part[1])!=r['slot'] or self.owners.get(r['slot'])!=rid):raise ValueError('Foreign private slot attribution')
  if kind in ('T','BT'):
   if kind=='T' and r['generated'] or kind=='BT' and not r['admitted']:raise ValueError('Unexpected admission/batch token ordering')
   if kind=='BT':self.bt_slots.add(int(part[1]))
   token=int(part[1 if kind=='T' else 2]);assert 0<=token<248320;r['generated'].append(token)
   if len(r['generated'])>r['max_new']:raise ValueError('Native generated bound exceeded')
  elif kind=='DONE':
   if r['admission_done'] or int(part[1])!=1 or int(part[2])!=len(r['ids']):raise ValueError('Incomplete/duplicate native admission DONE')
   if len(part)<15 or int(part[8])+int(part[14])!=len(r['ids']):raise ValueError('Actual admission consumed/reused counts missing')
   r['admission_done']=line
  elif kind=='BADM':
   if not r['admission_done'] or r['admitted']:raise ValueError('Missing/duplicate actual admission')
   continuation=int(part[2]);assert continuation in (0,1);r['admitted']=True
   if not continuation:r['done']=r['admission_done'];del self.owners[r['slot']]
  elif kind=='BDONE':
   if not r['admitted'] or int(part[2])!=len(r['generated']) or part[3] not in ('stop','length','cancel'):raise ValueError('Native completion count/reason mismatch')
   if (part[3]=='cancel')!=r['cancel_requested']:raise ValueError('Unexpected/missing real cancellation')
   r['done']=line;del self.owners[r['slot']]
 def capacity(self):
  if self.info.get('batch_slots')!=self.slots or self.info.get('batch_requested')!=self.slots or self.info.get('batch_protocol')!=2 or self.info.get('batch_groups',1)!=1:raise ValueError('INFO fallback/reduced/legacy capacity refused')
  # Protocol2 source startup rejects groups!=1; current INFO omits group when1.
 def prefix(self,rid,position,token):
  r=self.requests[rid];take=position+1-len(r['ids'])
  if take<0 or take>len(r['generated']):raise ValueError('Observed input position not in actual consumed history')
  ids=r['ids']+r['generated'][:take]
  if len(ids)!=position+1 or ids[-1]!=token:raise ValueError('Observed token differs from actual full consumed prefix')
  return ids

class NativeStream:
 def __init__(self,command,directory,slots,ready_timeout=900):
  leased([0,1]);self.directory=Path(directory);self.roster=Roster(slots);self.queue=queue.Queue();self.lock=threading.Lock();self.lines=[];self.arm_event_index=0;self.log=(self.directory/'engine.combined.log').open('w',encoding='ascii');self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
  def pump():
   for text in self.proc.stdout:
    line=text.rstrip('\n').encode('ascii','backslashreplace').decode('ascii')
    with self.lock:self.log.write(line+'\n');self.log.flush();self.lines.append(line)
    self.queue.put(line)
   self.queue.put(None)
  self.thread=threading.Thread(target=pump,daemon=True);self.thread.start();deadline=time.monotonic()+ready_timeout
  while True:
   line=self.next(deadline)
   if line.startswith('READY'):self.roster.capacity();break
 def next(self,deadline):
  left=deadline-time.monotonic()
  if left<=0:raise TimeoutError('Native bounded deadline')
  line=self.queue.get(timeout=left)
  if line is None:raise ValueError('Unexpected native producer EOF')
  self.roster.consume(line);return line
 def send(self,line):self.proc.stdin.write(line+'\n');self.proc.stdin.flush()
 def submit(self,rid,slot,ids,max_new=32):
  self.roster.submit(rid,slot,ids,max_new);self.send(f'BGEN {slot} {max_new} temperature=0 ckpt=0 rid={rid} '+','.join(map(str,ids)))
 def drain(self,rids,timeout=1800,cancel_rid=None,diagnostic=True):
  deadline=time.monotonic()+timeout;stopped=False
  while any(not self.roster.requests[r]['done'] for r in rids):
   self.next(deadline)
   if cancel_rid is not None and not stopped and self.roster.requests[cancel_rid]['admitted'] and (any(int(e['rows'])>=2 and int(e['active_mask'])&(1<<self.roster.requests[cancel_rid]['slot']) for e in self.roster.events[self.arm_event_index:]) if diagnostic else len(self.roster.bt_slots)>=2):self.send(self.roster.cancel(cancel_rid));stopped=True
  if cancel_rid is not None and not stopped:raise ValueError('No actual asynchronous multi-row cancellation occurred')
 def arm(self,path):
  if self.roster.owners or not any(int(e['rows'])>=2 for e in self.roster.events):raise ValueError('Warmup needs real completed multi-row work and all requests drained')
  if any(l.startswith(('SBF request ','SBF replay ','SBF vector ')) for l in self.lines):raise ValueError('Unarmed warmup unexpectedly observed/allocated producer data')
  with self.lock:
   marker='HARNESS ARM after actual unarmed multi-row terminal';self.log.write(marker+'\n');self.log.flush();self.lines.append(marker)
  self.arm_event_index=len(self.roster.events)
  Path(path).write_text('ARM warm complete\n',encoding='ascii')
 def close(self):
  if self.proc.poll() is None:self.send('QUIT')
  rc=self.proc.wait(timeout=90);self.thread.join(timeout=10);self.log.close()
  if self.thread.is_alive() or rc!=0:raise ValueError('Native clean terminal/producer reader required')
  return rc
