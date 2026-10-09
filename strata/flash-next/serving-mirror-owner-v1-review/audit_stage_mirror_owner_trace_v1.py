#!/usr/bin/env python3
"""Actual serving mirror logical ownership; separate-process upload proof excluded."""
import argparse,json,re,hashlib
from pathlib import Path
from parse_usm_logical_free_trace import HEX,pointer
CALL=re.compile(r'<--- (urUSMDeviceAlloc|urUSMHostAlloc|urUSMSharedAlloc|urUSMFree|urContextGetNativeHandle|urDeviceGetNativeHandle)\((.*)\) -> (\w+);')

def audit(text,expected_owners):
 active={};contexts={};devices={};owners={};owned_history=set();generations=set();errors=[];allocations=0
 def fail(condition,message):
  if not condition:raise ValueError(message)
 for number,line in enumerate(text.splitlines(),1):
  try:
   call=CALL.search(line)
   if call:
    kind,args,result=call.groups();fail(result=='UR_RESULT_SUCCESS','UR operation failed')
    if kind in ('urContextGetNativeHandle','urDeviceGetNativeHandle'):
     scope='Context' if kind.startswith('urContext') else 'Device';table=contexts if scope=='Context' else devices
     h=re.search(r'\.h'+scope+r' = ('+HEX+r')',args);native=re.search(r'\.phNative'+scope+r' = '+HEX+r' \(('+HEX+r'|\d+)\)',args)
     fail(h and native,'Actual native context/device bridge missing');key=pointer(h.group(1));value=pointer(native.group(1))
     fail(key not in table or table[key]==value,'Native bridge changed');table[key]=value
    else:
     h=re.search(r'\.hContext = ('+HEX+r')',args);fail(h,'UR owning context missing');context=pointer(h.group(1))
     if kind=='urUSMFree':
      ptr=re.search(r'\.pMem = ('+HEX+r')(?=,|$)',args);fail(ptr,'UR free pointer missing');key=(context,pointer(ptr.group(1)));allocation=active.pop(key,None)
      if allocation and 'owner' in allocation:
       owner=owners[allocation['owner']];fail(owner.get('release_begin') and allocation.get('free_begin') and not allocation.get('free_line'),'Owned free missing/duplicate begin');allocation['free_line']=number
      elif not allocation and key in owned_history:raise ValueError('Missing/double/stale-context owned UR free')
     else:
      ptr=re.search(r'\.ppMem = '+HEX+r' \(('+HEX+r')\)',args);size=re.search(r'\.size = (\d+)(?=,|$)',args);fail(ptr and size,'UR allocation extent missing');key=(context,pointer(ptr.group(1)))
      fail(key not in active,'Duplicate live UR allocation');allocations+=1;dev=re.search(r'\.hDevice = ('+HEX+r')',args)
      active[key]={'context':context,'pointer':key[1],'bytes':int(size.group(1)),'kind':kind,'line':number,'UR_generation':allocations,'device':pointer(dev.group(1)) if dev else None}
   if not line.startswith('MIRROR_USM '):continue
   mark=json.loads(line[len('MIRROR_USM '):]);event=mark['event'];key=(mark['pid'],mark['stage'],mark['device'],mark['owner_generation'])
   fail(all(type(x) is int and x>=0 for x in key) and key[0]>0 and key[3]>0,'Invalid PID/stage/device/owner generation')
   identity={name:mark[name] for name in ('pid','stage','device','owner_generation','layer_begin','layer_end','n_layers','n_expert','payload_bytes','ze_context','ze_device')}
   fail(all(type(mark[n]) is int for n in ('layer_begin','layer_end','n_layers','n_expert','payload_bytes')) and 0<=mark['layer_begin']<mark['layer_end']<=mark['n_layers'] and mark['n_expert']>0 and mark['payload_bytes']>=0,'Invalid mirror geometry')
   if event=='owner_begin':
    fail(key not in owners and any(v==pointer(mark['ze_context']) for v in contexts.values()) and any(v==pointer(mark['ze_device']) for v in devices.values()),'Mirror owner duplicate/missing actual native bridges')
    owners[key]={'identity':identity,'allocations':{},'consumers':{},'begin':number,'serving':False};continue
   fail(key in owners,'Marker without actual mirror owner');owner=owners[key];fail(owner['identity']==identity,'Mirror stage/device/context/geometry changed')
   fail(not owner.get('release_returned'),'Late marker after returned release')
   fail(type(mark['registered_owners']) is int and mark['registered_owners']>=0,'Invalid marker allocation count')
   if event=='owner_register':
    fail(not owner.get('release_begin'),'Late owned registration');address=pointer(mark['pointer']);matches=[(ak,a) for ak,a in active.items() if ak[1]==address and contexts.get(ak[0])==pointer(mark['ze_context'])]
    fail(len(matches)==1,'Owned pointer does not bind unique actual allocation/context');ak,allocation=matches[0];role=mark['role'];segment=mark['segment'];generation=mark['allocation_generation']
    fail(type(generation) is int and generation>0 and (key[0],generation) not in generations,'Allocation generation duplicate/invalid')
    fail(role in ('device_table','segment','contiguous_host') and type(segment) is int and (segment>=0 if role=='segment' else segment==-1),'Borrowed/unknown role or segment')
    fail(allocation['bytes']==mark['bytes'] and allocation['bytes']>0 and 'owner' not in allocation and allocation['kind']==('urUSMDeviceAlloc' if role=='device_table' else 'urUSMHostAlloc'),'Allocation role/extent/ownership differs')
    if role=='device_table':fail(devices.get(allocation['device'])==pointer(mark['ze_device']),'Actual device table native device differs')
    generations.add((key[0],generation));allocation.update(owner=key,allocation_generation=generation,role=role,segment=segment,register_line=number);owner['allocations'][ak]=allocation;owned_history.add(ak)
    fail(mark['registered_owners']==len(owner['allocations']),'Registered owner count differs')
   elif event in ('graphs_bound','graphs_retired'):
    fail(not owner.get('release_begin') and mark['consumer_kind'] in ('verifier','source_aliases'),'Late/foreign graph consumer');consumer=(mark['consumer_kind'],pointer(mark['consumer']));fail(consumer[1]>0,'Null graph/source consumer')
    if event=='graphs_bound':fail(not owner['consumers'].get(consumer,False),'Duplicate active graph binding');owner['consumers'][consumer]=True
    else:fail(owner['consumers'].get(consumer,False),'Foreign/stale graph retirement');owner['consumers'][consumer]=False
   elif event=='serving_bound':
    fail(not owner['serving'] and not owner.get('release_begin') and mark['serving_bound'] is True,'Duplicate/late serving bind');owner['serving']=True
   elif event=='serving_graphs_retired':
    fail(owner['serving'] and not owner.get('serving_retired') and not owner.get('release_begin') and mark['serving_graphs_retired'] is True and all(not live or kind=='source_aliases' for (kind,_),live in owner['consumers'].items()),'Serving verifier/global graph retirement missing/late')
    owner['serving_retired']=number
   elif event=='release_begin':
    fail(not owner.get('release_begin') and mark['registered_owners']==len(owner['allocations']) and all(not live for live in owner['consumers'].values()),'Release with live graph/source alias or wrong roster')
    fail(not owner['serving'] or owner.get('serving_retired'),'Serving global graphs not retired');owner['release_begin']=number
   elif event in ('free_begin','free_returned'):
    matches=[a for a in owner['allocations'].values() if a['pointer']==pointer(mark['pointer']) and a['allocation_generation']==mark['allocation_generation']]
    fail(owner.get('release_begin') and len(matches)==1,'Free marker has no owned pointer/generation');allocation=matches[0]
    fail((mark['bytes'],mark['role'],mark['segment'])==(allocation['bytes'],allocation['role'],allocation['segment']),'Free pointer role/extent/segment changed')
    if event=='free_begin':fail(not allocation.get('free_begin'),'Duplicate free begin');allocation['free_begin']=number
    else:fail(allocation.get('free_line') and allocation['free_begin']<allocation['free_line']<number and not allocation.get('free_returned'),'Free returned without unique context-matched successful UR free');allocation['free_returned']=number
   elif event=='release_returned':
    fail(owner.get('release_begin') and mark['registered_owners']==len(owner['allocations']) and all(a.get('free_returned') for a in owner['allocations'].values()),'Release returned missing owning free');owner['release_returned']=number
   else:raise ValueError('Unknown mirror owner event')
  except (ValueError,KeyError,TypeError) as error:errors.append({'line':number,'error':str(error)})
 roster=[]
 for owner in owners.values():
  ident=owner['identity'];records=list(owner['allocations'].values());tables=[a for a in records if a['role']=='device_table'];host=[a for a in records if a['role']!='device_table'];segments=[a['segment'] for a in host if a['role']=='segment']
  if not owner.get('release_returned') or any(not a.get('free_returned') for a in records):errors.append({'error':'Incomplete actual owning release','owner':ident})
  if owner['serving']:
   if not any(kind=='verifier' for kind,_ in owner['consumers']) or len([1 for kind,_ in owner['consumers'] if kind=='source_aliases'])!=1:errors.append({'error':'Serving owner missing actual verifier/source bindings','owner':ident})
   roster.append(tuple(ident[n] for n in ('stage','device','layer_begin','layer_end')))
   if len(tables)!=1 or tables[0]['bytes']!=ident['n_layers']*ident['n_expert']*8 or sum(a['bytes'] for a in host)!=ident['payload_bytes'] or sorted(segments)!=list(range(len(segments))) or (segments and any(a['role']=='contiguous_host' for a in host)):errors.append({'error':'Incomplete mirror table/segment extent accounting','owner':ident})
 if not expected_owners or len(roster)!=len(expected_owners) or set(roster)!=set(map(tuple,expected_owners)):errors.append({'error':'Actual serving stage/device/range owner roster differs'})
 pids={key[0] for key in owners}
 if len(pids)!=1:errors.append({'error':'Expected one actual engine PID'})
 return {'schema':1,'passed':not errors,'owners':[{**{k:v for k,v in o.items() if k not in ('allocations','consumers')},'allocations':list(o['allocations'].values()),'consumers':[{'kind':k,'pointer':p,'live':v} for (k,p),v in o['consumers'].items()]} for o in owners.values()],'expected_owners':list(expected_owners or []),'errors':errors,'postfree_pointer_type_gate_used':False,'physical_allocator_reclamation_qualified':False,'engine_all_allocations_retired':False,'separate_upload_oracle_proof_used':False,'scope':'Actual serving stage mirror owned device table and host contiguous/segment allocations only; actual source/verifier and global shutdown retirement markers + chronological owning UR frees. Borrowed aliases and other engine allocations excluded.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--expected-owners',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve existing evidence')
 result=audit(a.log.read_text(),json.loads(a.expected_owners.read_text()));result.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps({'passed':result['passed'],'owners':len(result['owners'])}));
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
