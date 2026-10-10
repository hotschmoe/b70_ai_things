"""Separate actual producer terminal proof; preserves frozen observer raw snapshots/errors."""
import re
from batch_api_prefixes_v2 import command

def require(ok,msg):
 if not ok:raise ValueError(msg)
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))

class Terminals:
 def __init__(self):self.begins={};self.active={};self.history={};self.ends={};self.last_sequence=0
 def consume(self,row):
  require(type(row['sequence']) is int and row['sequence']>self.last_sequence,'Actual sparse native sequence order required');self.last_sequence=row['sequence'];kind=row['kind']
  if kind=='engine_begin':
   require(row['call'] not in self.begins and row['rendered_matches_submitted'] is True,'Actual request begin/render ownership invalid');self.begins[row['call']]=row
  if kind=='native_send':
   line=row['line'];c=command(line)
   if c:
    call=row['call'];require(call in self.begins and row['engine_pid']==self.begins[call]['engine_pid'],'Native leg belongs to exact API call/PID');key=(row['engine_pid'],c['rid']);past=self.history.setdefault(key,[]);require(not past or past[-1]['terminal'],'Actual next leg before previous terminal')
    expected=self.begins[call]['submitted_ids'] if not past else past[0]['ids']+[t for seg in past for t in seg['generated']];require(c['ids']==expected and all(seg['call']==call for seg in past),'Actual native extendedprefix/source call differs');c.update(call=call,send_sequence=row['sequence'],terminal_record=None,stop_records=[]);past.append(c);self.active[key]=c
   elif line=='STOP' or line.startswith('BSTOP '):
    call=row.get('call');matches=[s for s in self.active.values() if s['call']==call and not s['terminal'] and row['engine_pid']==self.begins[call]['engine_pid']]
    # An EOS terminal may already be produced before frontend's final drain STOP.
    if not matches:matches=[s for s in self.active.values() if s['call']==call and row['engine_pid']==self.begins[call]['engine_pid']]
    require(len(matches)==1,'Actual STOP mustbind one owned request/native leg');s=matches[0]
    if line.startswith('BSTOP '):
     f=fields(line);require(s['mode']=='BGEN' and int(line.split()[1])==s['slot'] and int(f['rid'])==s['rid'] and int(f['slotgen'])==s['slotgen'],'Actual BSTOP stale owner/generation')
    else:require(s['mode']=='GEN','Bare STOP cannotbind a batched slot')
    s['stop_records'].append({'sequence':row['sequence'],'line':line})
  if kind=='native_receive':
   line=row['line'];words=line.split();head=words[0] if words else ''
   if head not in ('T','BT','DONE','BDONE','BADM'):return
   f=fields(line);rid=int(f.get('rid','0'));generation=int(f.get('slotgen','-1'));key=(row['engine_pid'],rid);require(key in self.active,'Actual source terminal/token has no owned leg');seg=self.active[key]
   require(not seg['terminal'],'Late source output on terminal leg');require(generation>=0 and (seg['slotgen'] is None or seg['slotgen']==generation),'Actual source slot generation changed');seg['slotgen']=generation
   if head in ('T','BT'):
    if head=='BT':require(seg['mode']=='BGEN' and int(words[1])==seg['slot'],'Foreign actual slot token')
    token=int(words[1 if head=='T' else 2]);require(0<=token<248320,'Actual token outside vocabulary');seg['generated'].append(token);seg['last_token_sequence']=row['sequence']
   if head=='DONE':
    require(int(words[1])==len(seg['generated']) and int(words[2])==len(seg['ids']) and int(words[8])+int(words[14])==len(seg['ids']),'Actual DONE counts/input/reuse differ')
    seg['admission_DONE_record']={'sequence':row['sequence'],'line':line,'finish':words[5],'slotgen':generation}
    if seg['mode']=='GEN':seg['terminal']=True;seg['terminal_record']={'sequence':row['sequence'],'line':line,'finish':words[5],'slotgen':generation}
   if head=='BADM':
    require(seg['mode']=='BGEN' and int(words[1])==seg['slot'],'Actual admission slot differs')
    if words[2]=='0':require('admission_DONE_record' in seg,'Actual BADM0 needs completed admissionDONE');seg['terminal']=True;seg['terminal_record']={'sequence':row['sequence'],'line':line,'finish':seg['admission_DONE_record']['finish'],'slotgen':generation,'admission_DONE_record':seg['admission_DONE_record']}
   if head=='BDONE':
    require(seg['mode']=='BGEN' and int(words[1])==seg['slot'] and int(words[2])==len(seg['generated']),'Actual BDONE slot/token count differs');seg['terminal']=True;seg['terminal_record']={'sequence':row['sequence'],'line':line,'finish':words[3],'slotgen':generation}
  if kind=='engine_end':
   call=row['call'];require(call in self.begins and call not in self.ends,'Actual end begin/uniqueness invalid');self.ends[call]=row
 def association(self,call):
  end=self.ends[call];begin=self.begins[call];require(end['engine_pid']==begin['engine_pid'] and end['engine_generation']==begin['engine_generation'] and end['rid'] is not None,'Actual API PID/incarnation/request differs');key=(end['engine_pid'],end['rid']);past=self.history.get(key,[]);require(past and all(s['call']==call and s['terminal'] and s['terminal_record'] for s in past),'All actual request legs need owned producer terminals');tokens=[t for s in past for t in s['generated']];require(tokens==end['generated_ids'],'Actual yielded/native fulltoken trajectory differs');last=past[-1];terminal=last['terminal_record'];require(terminal['sequence']<end['sequence'],'Actual native terminal mustprecede completed observerend')
  pinned=bool(tokens) and tokens[-1] in end['pinned_eos_ids'];cancelled=end['cancelled'];legacy=end['error'];legacy_generator_exit=isinstance(legacy,dict) and legacy=={'type':'GeneratorExit','message':'Noncancelled consumer close without pinned EOS native stop'}
  for segment in past:
   require(segment['terminal_record']['finish'] in ('stop','length','cancel'),'Actual unknown producer finish')
   if segment['terminal_record']['finish']=='cancel':require(any(s['sequence']<segment['terminal_record']['sequence'] for s in segment['stop_records']),'Actual canceled leg needs source-bound precedingSTOP')
  require(legacy is None or legacy_generator_exit,'Unknown actual observer/transport error cannotbe admitted')
  eos_close_drain=False
  if cancelled:require(terminal['finish']=='cancel' and last['stop_records'],'Actual clientcancel mustbind nativecancel/stop')
  elif end['consumer_closed']:
   require(pinned,'Noncancelled consumer close without actual yielded pinned EOS')
   if terminal['finish']=='cancel':
    eos_close_drain=any(last['last_token_sequence']<s['sequence']<terminal['sequence'] for s in last['stop_records']);require(eos_close_drain,'EOSclose cancellation needs actual owned postEOS STOP/drain')
   else:require(terminal['finish']=='stop','Noncancelled EOS close requires actual native stop')
  else:require(terminal['finish'] in ('stop','length') and not legacy_generator_exit,'Actual normal terminal cannotborrow unknown legacyerror')
  return {'call':call,'engine_pid':end['engine_pid'],'engine_generation':end['engine_generation'],'rid':end['rid'],'actual_native_terminal':terminal,'actual_generated_ids':tokens,'actual_yielded_pinned_eos':pinned,'actual_client_cancelled':cancelled,'source_bound_EOS_close_stop_drain':eos_close_drain,'legacy_observer_error_preserved':legacy,'legacy_engine_last_preserved':end['engine_last'],'frozen_raw_engine_end_rewritten':False,'terminal_association_qualified':True,'cache_or_math_qualified':False}

def semantic_events(path):
 from pathlib import Path
 import json
 wanted=('"kind": "engine_begin"','"kind": "engine_end"','"kind": "native_send"','"line": "T ','"line": "BT ','"line": "DONE ','"line": "BDONE ','"line": "BADM ')
 with Path(path).open() as f:
  for line in f:
   require(line.endswith('\n'),'Actual raw JSON trace truncated line')
   if any(word in line for word in wanted):yield json.loads(line)

def finalized_associations(path):
 tracker=Terminals()
 for row in semantic_events(path):tracker.consume(row)
 require(set(tracker.begins)==set(tracker.ends),'Actual request begin/end roster incomplete')
 return [tracker.association(call) for call in sorted(tracker.ends)]

def associate_events(events,calls=None):
 tracker=Terminals()
 for row in events:tracker.consume(row)
 wanted=set(tracker.ends) if calls is None else set(calls)
 require(wanted<=set(tracker.begins)&set(tracker.ends),'Actual named call terminals incomplete')
 return {call:tracker.association(call) for call in sorted(wanted)}
