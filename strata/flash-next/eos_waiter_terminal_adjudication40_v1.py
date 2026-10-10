#!/usr/bin/env python3
"""Named source40 EOS waiter chronology adjudication; no raw errors rewritten."""
import math
from eos_waiter_observer40_v1 import SERVER_SHA256,digest
from full_cache_shared_terminal_v2 import derived_source

LEGACY={'type':'GeneratorExit','message':'Noncancelled consumer close without pinned EOS native stop'}
OLD="require(terminal['sequence']<end['sequence'],'Actual native terminal mustprecede completed observerend')"
NEW="self.waiter_terminal_binding(call,terminal,end)"

def require(ok,message):
 if not ok:raise ValueError(message)
def fields(line):
 pairs=[x.split('=',1) for x in line.split() if '=' in x];require(len({k for k,v in pairs})==len(pairs),'Duplicate native tags');return dict(pairs)
def positive(value):return type(value)is int and value>0

def grammar():
 source=derived_source();require(source.count(OLD)==1,'Original named chronology boundary changed');ns={};exec(compile(source.replace(OLD,NEW),'[source40-eos-waiter-chronology]','exec'),ns);return ns['Terminals']
Base=grammar()

class Terminals(Base):
 def __init__(self,events,proof):
  super().__init__();self.events=events;self.proof=proof;self.waiter_bindings=[]
 def waiter_terminal_binding(self,call,terminal,end):
  candidates=[x for x in self.events if x['kind']=='eos_waiter_created' and x.get('call')==call]
  if not candidates:
   require(terminal['sequence']<end['sequence'],'Late native terminal requires actual source40 waiter proof');return
  require(len(candidates)==1,'One exact final background waiter required per adjudicated API call');created=candidates[0];wid=created['waiter'];rows=[x for x in self.events if x['kind'].startswith('eos_waiter_') and x.get('waiter')==wid]
  enter=[x for x in rows if x['kind']=='eos_waiter_enter'];exits=[x for x in rows if x['kind']=='eos_waiter_exit'];gets=[x for x in rows if x['kind']=='eos_waiter_queue_return']
  require(len(enter)==len(exits)==1 and gets,'Exact waiter enter/exit/queue observations required');enter=enter[0];exit=exits[0]
  owner_keys=('call','waiter','engine_pid','engine_generation','rid','slot_generation','slot','server_pid','queue_object','busy_object','held_object','server_sha256')
  require(all(positive(created[k]) for k in owner_keys if k not in ('slot','server_sha256')) and type(created['slot'])is int and created['slot']>=0,'Typed actual waiter owner required')
  require(created['server_sha256']==SERVER_SHA256 and all(all(type(x[k])is type(created[k]) and x[k]==created[k] for k in owner_keys) for x in rows),'Waiter actual owner/source changed')
  require(created['engine_pid']==end['engine_pid'] and created['engine_generation']==end['engine_generation'] and created['rid']==end['rid'],'Waiter/API native incarnation differs')
  require(positive(enter['thread_ident']) and positive(enter['thread_native_id']) and all(x['thread_ident']==enter['thread_ident'] and type(x['thread_ident'])is int and x['thread_native_id']==enter['thread_native_id'] and type(x['thread_native_id'])is int for x in gets+[exit]),'Waiter actual thread identity differs')
  require(created['sequence']<enter['sequence']<gets[0]['sequence']<=gets[-1]['sequence']<exit['sequence'],'Waiter event chronology differs')
  require(all(type(x.get('epoch'))in (int,float) and math.isfinite(x['epoch']) for x in rows),'Finite actual waiter epochs required')
  require([x['epoch'] for x in rows]==sorted(x['epoch'] for x in rows),'Actual waiter timestamps move backward')
  line=gets[-1]['line'];require(type(line)is str and line.startswith('BDONE ') and line==terminal['line'],'Waiter must actually consume exact native terminal last')
  words=line.split();tags=fields(line);require(int(words[1])==created['slot'] and tags.get('rid')==str(created['rid']) and tags.get('slotgen')==str(created['slot_generation']),'Queued BDONE slot/request generation differs')
  receives=[x for x in self.events if x['kind']=='native_receive' and x['engine_pid']==created['engine_pid'] and x['line']==line]
  require(len(receives)==1 and receives[0]['sequence']==terminal['sequence'] and terminal['sequence']<gets[-1]['sequence'],'Actual native/queue BDONE association differs')
  require(exit['error']is None and exit['same_original_queue']is True and exit['same_original_busy_array']is True and type(exit['busy'])is bool,'Waiter target did not normally return on original arrays')
  require(type(exit['held_ids'])is list and all(type(x)is int and 0<=x<2**32 for x in exit['held_ids']) and exit['held_sha256']==digest(exit['held_ids']),'Held snapshot bytes differ')
  # Snapshot follows source's busy=False/notify block and can see a newer owner.
  # Do not turn such a race into an old-slot false claim without actual evidence.
  require(exit['same_current_busy_array']is True and exit['same_current_queue']is True and type(exit['current_engine_generation'])is int and exit['current_engine_generation']==created['engine_generation'] and type(exit['current_engine_pid'])is int and exit['current_engine_pid']==created['engine_pid'],'Native/slot array incarnation changed before snapshot')
  current=exit['current_slot_identity'];require(type(current)is list and len(current)==3 and all(positive(x) for x in current),'Exact current slot identity snapshot required')
  if exit['busy'] is True:
   require(current[0]==created['engine_generation'] and current[1:]!=[created['rid'],created['slot_generation']],'Busy snapshot lacks distinct new-owner identity')
   new_admissions=[x for x in self.events if x['kind']=='native_receive' and x['engine_pid']==created['engine_pid'] and x['line'].startswith('BADM '+str(created['slot'])+' 1 ') and fields(x['line']).get('rid')==str(current[1]) and fields(x['line']).get('slotgen')==str(current[2]) and gets[-1]['sequence']<x['sequence']<exit['sequence']]
   require(len(new_admissions)==1,'Busy true requires actual uniquely observed new-owner admission before snapshot')
  else:require(current==[created['engine_generation'],created['rid'],created['slot_generation']],'Idle snapshot owner unexpectedly changed')
  threads=[x for x in self.proof['threads'] if x['waiter']==wid];require(len(threads)==1,'Exact retained Thread proof required');thread=threads[0]
  require(thread['thread_ident']==enter['thread_ident'] and type(thread['thread_ident'])is int and thread['thread_native_id']==enter['thread_native_id'] and type(thread['thread_native_id'])is int and thread['started']is True and thread['alive']is False and thread['daemon']is True,'Actual waiter Thread not retired')
  require(end['consumer_closed']is True and type(end['cancelled'])is bool,'Only actual consumer-close waiter supported')
  if end['cancelled']:require(end['error']is None and words[3]=='cancel','Actual client cancellation cannot borrow EOS exception')
  else:require(end['generated_ids'] and end['generated_ids'][-1] in end['pinned_eos_ids'] and end['error'] in (None,LEGACY),'Only actual noncancelled pinned-EOS close supports named error adjudication')
  self.waiter_bindings.append({'call':call,'waiter':wid,'actual_native_BDONE':line,'actual_waiter_thread_retired':True,'original_busy_false_transition_source_bound':True,'busy_snapshot':exit['busy'],'actual_client_cancelled':end['cancelled'],'named_EOS_error_adjudicated':end['error']==LEGACY,'legacy_observer_error_preserved':end['error'],'legacy_engine_last_preserved':end['engine_last'],'original_raw_records_rewritten':False})

def adjudicate(events,observer_proof):
 require(type(observer_proof)is dict and observer_proof.get('schema')==1 and type(observer_proof['schema'])is int and observer_proof.get('server_sha256')==SERVER_SHA256 and observer_proof.get('errors')==[] and observer_proof.get('observer_passed')is True and observer_proof.get('all_waiter_threads_retired')is True,'Actual final observer/source/Thread proof required')
 created=[x for x in events if x['kind']=='eos_waiter_created'];require(type(observer_proof['waiters'])is int and observer_proof['waiters']==len(created) and len({x['waiter'] for x in created})==len(created),'Full waiter roster differs')
 require(type(observer_proof['queue_returns'])is int and observer_proof['queue_returns']==sum(x['kind']=='eos_waiter_queue_return' for x in events),'Full actual queue return roster differs')
 tracker=Terminals(events,observer_proof)
 for row in events:tracker.consume(row)
 require(set(tracker.begins)==set(tracker.ends),'All original API calls need exact terminals')
 if observer_proof.get('calls')is not None:require(type(observer_proof['calls'])is list and all(positive(x) for x in observer_proof['calls']) and observer_proof['calls']==sorted(tracker.begins),'Actual phase observer call roster differs')
 associations=[tracker.association(call) for call in sorted(tracker.ends)]
 require({x['waiter'] for x in tracker.waiter_bindings}=={x['waiter'] for x in created},'Unassociated actual background waiter')
 return {'schema':1,'actual_API_terminal_associations':associations,'actual_EOS_waiter_retirement':tracker.waiter_bindings,'named_source40_waiter_adjudication_passed':True,'original_raw_errors_rewritten':False,'original_V6_passed_promoted':False,'cache_or_math_qualified':False}
