#!/usr/bin/env python3
"""Exact chronological slot arena/owned-host UR registrations and frees."""
import json,re
from parse_usm_logical_free_trace import CALL,HEX,pointer

def audit(text,expected_stage_slots=None):
 active={};bridges={};owners={};errors=[];counter=0
 for number,line in enumerate(text.splitlines(),1):
  try:
   match=CALL.search(line)
   if match:
    kind,args,result=match.groups()
    if result!='UR_RESULT_SUCCESS':raise ValueError('UR owner operation failed')
    ctxmatch=re.search(r'\.hContext = ('+HEX+r')',args)
    if not ctxmatch:raise ValueError('UR context missing')
    ctx=pointer(ctxmatch.group(1))
    if kind=='urContextGetNativeHandle':
     native=re.search(r'\.phNativeContext = '+HEX+r' \((0x[0-9a-fA-F]+|\d+)\)',args)
     if not native:raise ValueError('Native context bridge missing')
     value=pointer(native.group(1))
     if ctx in bridges and bridges[ctx]!=value:raise ValueError('Native owning context bridge changed')
     bridges[ctx]=value
    elif kind=='urUSMFree':
     match=re.search(r'\.pMem = ('+HEX+r')(?=,|$)',args)
     if not match:raise ValueError('UR free pointer missing')
     key=(ctx,pointer(match.group(1)));allocation=active.pop(key,None)
     if allocation and 'owner' in allocation:
      owner=owners[allocation['owner']]
      if 'release_begin' not in owner or 'release_returned' in owner:raise ValueError('Slot owner free outside release')
      if allocation.get('free_line'):raise ValueError('Double slot free')
      allocation['free_line']=number
     elif any(key in owner['allocations'] for owner in owners.values()):raise ValueError('Double/context-stale owned free')
    else:
     value=re.search(r'\.ppMem = '+HEX+r' \(('+HEX+r')\)',args);size=re.search(r'\.size = (\d+)(?=,|$)',args)
     if not value or not size:raise ValueError('UR allocation shape missing')
     key=(ctx,pointer(value.group(1)));counter+=1
     if key in active:raise ValueError('Duplicate live allocation')
     active[key]={'context':ctx,'pointer':key[1],'bytes':int(size.group(1)),'kind':kind,'alloc_line':number,'allocation_generation':counter}
   if not line.startswith('SLOT_USM '):continue
   mark=json.loads(line[len('SLOT_USM '):]);kind=mark['event'];key=(mark['pid'],mark['stage'],mark['slot'],mark['generation']);assert all(type(n) is int and n>=0 for n in key) and mark['generation']>0
   if kind=='allocation_null':
    if key in owners:raise ValueError('Null attempt already owned')
    continue
   owner=owners.setdefault(key,{'pid':mark['pid'],'stage':mark['stage'],'slot':mark['slot'],'generation':mark['generation'],'device':mark['device'],'allocations':{}})
   if owner['device']!=mark['device']:raise ValueError('Owner device changed')
   if kind=='owner_register':
    native=pointer(mark['ze_context']);address=pointer(mark['pointer']);matches=[(k,a) for k,a in active.items() if k[1]==address and bridges.get(k[0])==native]
    if len(matches)!=1:raise ValueError('Registered owner does not bind unique live allocation/context')
    ak,allocation=matches[0];expected='urUSMDeviceAlloc' if mark['role']=='slot_arena' else 'urUSMHostAlloc'
    if allocation['kind']!=expected or allocation['bytes']!=mark['bytes'] or 'owner' in allocation or 'release_begin' in owner:raise ValueError('Wrong/double/late owned registration')
    allocation.update(owner=key,register_line=number,role=mark['role']);owner['allocations'][ak]=allocation
   elif kind=='graphs_bound':
    if 'bound' in owner or 'release_begin' in owner:raise ValueError('Duplicate/late graph bind')
    owner['bound']=number
   elif kind=='graphs_retired':
    if 'retired' in owner or 'release_begin' in owner:raise ValueError('Duplicate/late graph retirement')
    owner['retired']=number
   elif kind=='release_begin':
    if 'release_begin' in owner or not owner['allocations'] or mark['registered_owners']!=len(owner['allocations']):raise ValueError('Release owner roster missing/duplicate')
    if bool(owner.get('bound'))!=mark['graphs_bound']:raise ValueError('Release graph-bound claim differs from source history')
    if mark['graphs_bound'] and (not mark['graphs_retired'] or not owner.get('retired') or owner['retired']>=number):raise ValueError('Graph references not retired before slot release')
    owner['release_begin']=number
   elif kind=='release_returned':
    if 'release_returned' in owner or 'release_begin' not in owner or any(not a.get('free_line') or not owner['release_begin']<a['free_line']<number for a in owner['allocations'].values()):raise ValueError('Owner returned without each unique context-matched UR free')
    owner['release_returned']=number
   else:raise ValueError('Unknown slot ownership marker')
  except (ValueError,KeyError,TypeError,AssertionError) as error:errors.append({'line':number,'error':str(error)})
 for owner in owners.values():
  if not owner.get('release_returned') or len([a for a in owner['allocations'].values() if a['role']=='slot_arena'])!=1:errors.append({'error':'Incomplete arena owning lifecycle','owner':{k:v for k,v in owner.items() if k!='allocations'}})
 if expected_stage_slots is not None and {(o['stage'],o['slot']) for o in owners.values() if o.get('bound')}!=set(map(tuple,expected_stage_slots)):errors.append({'error':'Requested actual stage/private-slot owner roster differs'})
 return {'passed':not errors and bool(owners),'owners':[{**{k:v for k,v in o.items() if k!='allocations'},'allocations':list(o['allocations'].values())} for o in owners.values()],'errors':errors,'postfree_pointer_type_gate_used':False,'scope':'owned slot arena and explicitly owned host side allocations only; borrowed aliases not registered; logical UR frees and graph retirement markers; physical allocator reclamation unproved'}
