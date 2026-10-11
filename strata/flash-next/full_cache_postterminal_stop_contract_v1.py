"""Named inactive-slot stop housekeeping; never live cancel or state transfer."""
from full_cache_memory39_contract_v1 import batch_terminal_binding as original
from full_cache_shared_terminal_v2 import fields

def require(ok,msg):
 if not ok:raise ValueError(msg)
def binding(rows,rid,slotgen,slot,events,pid):
 terminals=[i for i,r in enumerate(rows)if r['phase']=='BDONE_emitted'];require(len(terminals)==1,'Exactly one original FC39 batch terminal required');cut=terminals[0]+1;core=rows[:cut];tail=rows[cut:];value=original(core,rid,slotgen,slot)
 if not tail:return {**value,'postterminal_stop_housekeeping':[]}
 require(len(tail)==1 and tail[0]['phase']=='BSTOP_applied','Only one exact inactive same-owner stop housekeeping observation allowed');last=core[-1];row=tail[0]
 require(last['finish']in('stop','length') and row['event']>last['event'],'Late housekeeping requires original normal EOS/limit terminal')
 for key in ('pid','rid','slotgen','slot','position','consumed_tokens','generated'):
  require(type(row[key])is int and type(last[key])is int and row[key]==last[key],'Inactive same-owner terminal/count/position changed '+key)
 require(row['pid']==pid and row['rid']==rid and row['slotgen']==slotgen and row['slot']==slot and type(row['batch_window'])is int and row['batch_window']==0 and row['token']==-1 and type(row['token'])is int and row['finish']=='cancel'and row['continuation']==-1 and type(row['continuation'])is int and row['actual_HTTP_client_terminal_qualified']is False,'Exact consumed source inactive-stop metadata shape changed')
 records=[]
 for event in events:
  if event['kind']!='native_receive' or event['engine_pid']!=pid or not event['line'].startswith('FC39 '):continue
  import json
  native=json.loads(event['line'][5:])
  if native==last:records.append(('terminal',event))
  if native==row:records.append(('late',event))
 require([kind for kind,_ in records]==['terminal','late'],'Original unique source terminal/late-stop record order required');terminal_seq=records[0][1]['sequence'];late_seq=records[1][1]['sequence']
 sends=[]
 for event in events:
  if event['kind']!='native_send' or event['engine_pid']!=pid or not event['line'].startswith('BSTOP '):continue
  words=event['line'].split();tags=fields(event['line'])
  if len(words)>=2 and words[1]==str(slot)and tags.get('rid')==str(rid)and tags.get('slotgen')==str(slotgen):sends.append(event)
 require(len(sends)==1 and sends[0]['sequence']<late_seq,'Actual unique strict same-owner BSTOP command must precede observed apply')
 if last['finish']=='stop':
  ends=[e for e in events if e['kind']=='engine_end'and e['engine_pid']==pid and e.get('rid')==rid]
  tokens=[e for e in events if e['kind']=='native_receive'and e['engine_pid']==pid and e['line'].startswith('BT '+str(slot)+' '+str(last['token'])+' ')and fields(e['line']).get('rid')==str(rid)and fields(e['line']).get('slotgen')==str(slotgen)and e['sequence']<terminal_seq]
  require(len(tokens)==1,'Actual source owned final BT EOS before native terminal required')
  require(any(type(e.get('pinned_eos_ids'))is list and last['token']in e['pinned_eos_ids'] for e in ends),'Postterminal stop housekeeping requires actual admitted owner EOS evidence')
 require(not any(event['kind']=='native_send'and event['engine_pid']==pid and event['sequence']>terminal_seq and event['sequence']<late_seq and event['line'].startswith(('GEN ','BGEN '))for event in events),'Cannot borrow housekeeping across a new native request')
 return {**value,'postterminal_stop_housekeeping':[{'source_event':row,'actual_BSTOP_send_sequence':sends[0]['sequence'],'original_terminal_sequence':terminal_seq,'observed_apply_sequence':late_seq,'inactive_stop_flag_set':True,'live_cancel_or_migration_qualified':False,'original_terminal_finish_or_count_rewritten':False}]}
