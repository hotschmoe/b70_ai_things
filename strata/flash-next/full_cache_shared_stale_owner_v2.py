"""Declared old-owner BSTOP negative, using actual same-actor owner records."""
from full_cache_shared_history_v2 import canonical_sha,require

def owner_rows(events):
 """Source-bound actual submitted legs, not caller-created owner identities."""
 from full_cache_shared_terminal_v2 import Terminals
 tracker=Terminals()
 for event in events:tracker.consume(event)
 result=[]
 for (pid,rid),legs in tracker.history.items():
  for leg in legs:
   if leg['mode']!='BGEN' or leg['slotgen'] is None:continue
   row={'pid':pid,'rid':rid,'slotgen':leg['slotgen'],'slot':leg['slot'],'engine_generation':tracker.begins[leg['call']]['engine_generation'],'send_sequence':leg['send_sequence'],'mode':'BGEN','terminal':leg['terminal'],'admitted':leg['admitted']}
   if leg['terminal']:row['terminal_sequence']=leg['terminal_record']['sequence']
   result.append(row)
 return result

def admit_proposal(proposal,events):
 rows=owner_rows(events);require(proposal==command(proposal['old_owner'],proposal['current_owner']) and proposal['old_owner'] in rows and proposal['current_owner'] in rows,'Stale control owners must equal actual original submitted native legs')
 return proposal

def command(old,current):
 for row in (old,current):
  require(type(row)is dict and all(type(row[k])is int and row[k]>0 for k in ('pid','rid','slotgen','engine_generation','send_sequence')) and type(row['slot'])is int and 0<=row['slot']<2,'Exact actual two-slot owner integers required')
 require(old['pid']==current['pid'] and old['engine_generation']==current['engine_generation'] and old['slot']==current['slot'] and old['rid']!=current['rid'] and old['slotgen']<current['slotgen'],'Old identity must differ from actual new same-actor slot owner')
 require(old['mode']==current['mode']=='BGEN' and old['terminal'] is True and current['terminal'] is False and current['admitted'] is True and type(old['terminal_sequence'])is int and old['send_sequence']<old['terminal_sequence']<current['send_sequence'],'Completed prior owner and currently continued new owner required')
 return {'command':'BSTOP '+str(old['slot'])+' rid='+str(old['rid'])+' slotgen='+str(old['slotgen']),'old_owner_sha256':canonical_sha(old),'current_owner_sha256':canonical_sha(current),'engine_pid':current['pid'],'old_owner':dict(old),'current_owner':dict(current),'expected_source_rejection':'strata batch: stale/unidentified BSTOP'+str(old['slot'])+' rejected','current_owner_must_continue':True,'full_cache_runtime_qualified':False}

def recollect(proposal,send,events,current_terminal):
 require(proposal==command(proposal['old_owner'],proposal['current_owner']),'Preregistered old/new owner proposal changed')
 require(send['kind']=='fullcache_control_send' and send['line']==proposal['command'] and send['engine_pid']==proposal['engine_pid'] and type(send['sequence'])is int and send['sequence']>proposal['current_owner']['send_sequence'],'Actual stale control must follow admitted new owner')
 rows=[e for e in events if e['kind']=='native_receive' and e['engine_pid']==proposal['engine_pid'] and e['sequence']>send['sequence']]
 rejected=[e for e in rows if e['line']==proposal['expected_source_rejection']];require(len(rejected)==1,'Unique actual exact source stale-owner refusal required')
 tokens=[e for e in rows if e['line'].startswith('BT '+str(proposal['current_owner']['slot'])+' ') and ' rid='+str(proposal['current_owner']['rid'])+' ' in e['line'] and e['line'].endswith('slotgen='+str(proposal['current_owner']['slotgen'])) and e['sequence']>rejected[0]['sequence']]
 require(tokens and current_terminal['call']>0 and current_terminal['engine_pid']==proposal['engine_pid'] and current_terminal['rid']==proposal['current_owner']['rid'] and current_terminal['actual_client_cancelled'] is False and current_terminal['terminal_association_qualified'] is True and current_terminal['actual_native_terminal']['finish'] in ('stop','length') and current_terminal['actual_native_terminal']['sequence']>tokens[-1]['sequence'],'Actual new owner must emit after rejected stale stop and retire normally')
 return {'actual_old_owner_refused':True,'actual_new_owner_continued':True,'actual_source_refusal_record':rejected[0],'full49_independent_control_still_required':True,'full_cache_runtime_qualified':False}
