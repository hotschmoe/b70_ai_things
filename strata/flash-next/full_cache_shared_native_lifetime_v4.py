"""FC39 actual continued-window proof joined to immutable protocol observations."""
import json
from eos_waiter_terminal_adjudication40_v1 import Terminals
from full_cache_shared_terminal_v2 import fields
from full_cache_postterminal_stop_contract_v1 import binding as batch_terminal_binding
from full_cache_shared_metadata_v2 import unique

def require(ok, message):
 if not ok: raise ValueError(message)

def recollect(events,observer_proof):
 tracker = Terminals(events,observer_proof)
 for event in events: tracker.consume(event)
 require(set(tracker.begins) == set(tracker.ends), 'Every actual phase request must retire')
 associations = [tracker.association(call) for call in sorted(tracker.ends)]
 metadata = []
 for event in events:
  if event['kind'] != 'native_receive' or not event['line'].startswith('FC39 '): continue
  row = json.loads(event['line'][5:], object_pairs_hook=unique)
  if row.get('kind') != 'native_lifetime': continue
  require(type(row['pid']) is int and row['pid'] == event['engine_pid'] and row['actual_HTTP_client_terminal_qualified'] is False, 'FC39 actual producer owner/scope differs')
  metadata.append((event['sequence'], row))
 accepted = set(); results = []
 for (pid,rid), legs in tracker.history.items():
  for number,leg in enumerate(legs):
   end = leg['terminal_record']['sequence']; seq = leg['send_sequence']; owner = (rid,leg['slotgen'],-1 if leg['mode']=='GEN' else leg['slot'])
   # Source39 emits FC39 AFTER printing/flushing native DONE/BADM/BT/BDONE.
   # API end is therefore not an upper bound on the metadata observation.
   next_send = legs[number+1]['send_sequence'] if number+1<len(legs) else float('inf')
   selected = [(i,s,r) for i,(s,r) in enumerate(metadata) if r['pid'] == pid and (r['rid'],r['slotgen'],r['slot']) == owner and seq < s < next_send]
   require(selected, 'Actual batch leg has no FC39 lifetime observations')
   rows = [r for _,_,r in selected]
   if leg['mode'] == 'GEN':
    require(len(rows)==1 and rows[0]['phase']=='DONE_emitted' and rows[0]['generated']==len(leg['generated']) and rows[0]['finish']==leg['terminal_record']['finish'] and rows[0]['actual_HTTP_client_terminal_qualified'] is False, 'Direct/solo GEN requires exact own FC39 DONE')
    accepted.update(i for i,_,_ in selected); results.append({'rid':rid,'slotgen':leg['slotgen'],'slot':leg['slot'],'generated':len(leg['generated']),'native_terminal_observed':True,'source_route':'GEN','actual_HTTP_client_terminal_qualified':False});continue
   protocol = [e for e in events if e['kind'] == 'native_receive' and e['engine_pid'] == pid and seq < e['sequence'] <= end and e['line'].split(' ',1)[0] in ('DONE','BADM','BT','BDONE') and fields(e['line']).get('rid') == str(rid) and fields(e['line']).get('slotgen') == str(leg['slotgen'])]
   mapped = [r for r in rows if r['phase'] in ('DONE_emitted','BADM_emitted','BT_emitted','BDONE_emitted')]
   require(len(mapped) == len(protocol), 'FC39/protocol terminal-token roster differs')
   count = 1
   for row,event in zip(mapped,protocol):
    words = event['line'].split(); head = words[0]
    require(row['phase'] == head+'_emitted' and row['generated'] == (count+1 if head=='BT' else int(words[1]) if head=='DONE' else count), 'FC39/protocol phase/generated count differs')
    if head=='BT':
     require(row['token'] == int(words[2]) and row['slot'] == int(words[1]), 'FC39 emitted token/slot differs'); count += 1
    if head=='DONE':
     require(row['finish'] == words[5] and row['generated'] == len(leg['generated'][:1]), 'FC39 admission finish/count differs');count=row['generated']
    if head=='BADM': require(row['continuation'] == int(words[2]) and row['slot'] == int(words[1]), 'FC39 continuation/slot differs')
    if head=='BDONE': require(row['generated'] == int(words[2]) and row['finish'] == words[3] and row['slot'] == int(words[1]), 'FC39 final count/finish differs')
   if leg['admitted']:
    binding = batch_terminal_binding(rows,*owner,events,pid)
    require(binding['generated'] == len(leg['generated']), 'Actual complete FC39 leg count differs')
   else:
    require([r['phase'] for r in rows] == ['DONE_emitted','BADM_emitted'] and rows[-1]['continuation'] == 0, 'Uncontinued actual admission must contain exactly DONE/BADM0')
    binding = {'rid':rid,'slotgen':leg['slotgen'],'slot':leg['slot'],'generated':len(leg['generated']),'native_terminal_observed':True,'actual_HTTP_client_terminal_qualified':False}
   accepted.update(i for i,_,_ in selected); results.append(binding)
 require(accepted == set(range(len(metadata))), 'Unassociated FC39 batch lifetime record')
 return {'actual_batch_legs':results,'actual_API_terminal_associations':associations,'native_batch_metadata_joined':True,'full_cache_runtime_qualified':False,'physical_memory_qualified':False}
