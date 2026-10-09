#!/usr/bin/env python3
"""Source29 batch snapshot owners, matched chronological UR allocations/frees."""
import re
from parse_usm_logical_free_trace import CALL,HEX,pointer

def audit(text,stages):
 active={};owners={};errors=[];serial=0
 expected={s:(lo,hi,6*((hi-lo)*10240+(248320 if hi==48 else 0))*4+4736) for s,lo,hi in stages}
 for at,line in enumerate(text.splitlines(),1):
  try:
   match=CALL.search(line)
   if match:
    kind,args,result=match.groups()
    if result!='UR_RESULT_SUCCESS':raise ValueError('UR operation failed')
    ctx=re.search(r'\.hContext = ('+HEX+r')',args)
    if not ctx:raise ValueError('UR owning context missing')
    ctx=pointer(ctx.group(1))
    if kind=='urContextGetNativeHandle':continue
    if kind=='urUSMFree':
     m=re.search(r'\.pMem = ('+HEX+r')(?=,|$)',args)
     if not m:raise ValueError('Free pointer missing')
     key=(ctx,pointer(m.group(1)));item=active.pop(key,None)
     if item and 'stage' in item:
      owner=owners[item['stage']]
      if not owner.get('release_begin') or owner.get('free'):raise ValueError('Observer free outside owning release')
      owner['free']=at
     elif any(v['key']==key for v in owners.values()):raise ValueError('Double/stale observer free')
    else:
     addr=re.search(r'\.ppMem = '+HEX+r' \(('+HEX+r')\)',args);size=re.search(r'\.size = (\d+)(?=,|$)',args)
     if not addr or not size:raise ValueError('UR allocation arguments missing')
     key=(ctx,pointer(addr.group(1)));serial+=1
     if key in active:raise ValueError('Duplicate live allocation')
     active[key]={'bytes':int(size.group(1)),'kind':kind,'allocation_line':at,'generation':serial}
   if line.startswith('SBF allocation '):
    f=dict(re.findall(r'(\w+)=([^ ]+)',line));stage=int(f['stage'])
    if stage not in expected or stage in owners:raise ValueError('Missing/duplicate/foreign stage snapshot')
    lo,hi,size=expected[stage]
    if (int(f['lb']),int(f['le']),int(f['bytes']),int(f['rows']),int(f['head']))!=(lo,hi,size,6,int(hi==48)) or f['owner_queue']!='verifier_cs':raise ValueError('Exact raw plus4736B control extent differs')
    candidates=[(key,v) for key,v in active.items() if key[1]==pointer(f['pointer']) and v['bytes']==size and v['kind']=='urUSMDeviceAlloc']
    if len(candidates)!=1:raise ValueError('Owning allocation/context ambiguous')
    key,item=candidates[0];item['stage']=stage;owners[stage]=dict(item,key=key,owner_line=at)
   if line.startswith('SBF release_begin '):
    f=dict(re.findall(r'(\w+)=([^ ]+)',line));v=owners[int(f['stage'])]
    if v.get('release_begin') or pointer(f['pointer'])!=v['key'][1] or int(f['bytes'])!=v['bytes']:raise ValueError('Release pointer/extent/generation differs')
    v['release_begin']=at
   if line.startswith('SBF release_returned '):
    f=dict(re.findall(r'(\w+)=([^ ]+)',line));v=owners[int(f['stage'])]
    if not v.get('free') or v.get('release_returned'):raise ValueError('Release returned without exact successful free')
    v['release_returned']=at
  except (ValueError,KeyError,TypeError) as exc:errors.append({'line':at,'error':str(exc)})
 if set(owners)!=set(expected):errors.append({'error':'Complete actual stage owner roster absent'})
 for owner in owners.values():
  if not owner['allocation_line']<owner['owner_line']<owner.get('release_begin',0)<owner.get('free',0)<owner.get('release_returned',0):errors.append({'error':'Chronological owner lifecycle incomplete'})
 return {'passed':not errors,'owners':list(owners.values()),'errors':errors,'postfree_pointer_type_gate_used':False,'physical_driver_retirement_qualified':False,'scope':'actual batch observer allocations/free only; slot arenas and model mirror owners separate'}
