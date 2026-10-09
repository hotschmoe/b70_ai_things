#!/usr/bin/env python3
"""P30/SFD allocation-byte budget and actual UR owning frees; no graph handles."""
import re
from parse_usm_logical_free_trace import CALL,HEX,pointer
CAP=128<<20;D=10240

def audit_text(text,stages,p30_enabled=True):
 active={};owners=[];errors=[];owned_history=set();generation=0;maximum=0;live_bytes=0
 for number,line in enumerate(text.splitlines(),1):
  try:
   m=CALL.search(line)
   if m:
    kind,args,result=m.groups()
    if kind=='urContextGetNativeHandle':continue
    if result!='UR_RESULT_SUCCESS':raise ValueError('UR operation failed')
    context=re.search(r'\.hContext = ('+HEX+r')',args)
    if not context:raise ValueError('UR context missing')
    ctx=pointer(context.group(1))
    if kind=='urUSMFree':
     ptr=re.search(r'\.pMem = ('+HEX+r')(?=,|$)',args)
     if not ptr:raise ValueError('UR free pointer missing')
     key=(ctx,pointer(ptr.group(1)));allocation=active.pop(key,None)
     if allocation is None and key in owned_history:raise ValueError('Double/stale owned UR free')
     if allocation and 'owner' in allocation:
      owner=allocation['owner']
      if owner.get('free_line') or (owner['kind']=='P30' and not owner.get('begin_line')):raise ValueError('Duplicate/unordered observer free')
      owner['free_line']=number;live_bytes-=owner['bytes']
    else:
     ptr=re.search(r'\.ppMem = '+HEX+r' \(('+HEX+r')\)',args);size=re.search(r'\.size = (\d+)(?=,|$)',args);device=re.search(r'\.hDevice = ('+HEX+r')',args)
     if not ptr or not size:raise ValueError('UR allocation args missing')
     key=(ctx,pointer(ptr.group(1)));generation+=1
     if key in active:raise ValueError('Duplicate live allocation')
     active[key]={'pointer':key[1],'context':ctx,'device':pointer(device.group(1)) if device else None,'bytes':int(size.group(1)),'generation':generation,'alloc_line':number,'function':kind}
   if line.startswith(('P30 allocation ','SFD allocation ')):
    kind='P30' if line.startswith('P30 ') else 'SFD';mark=dict(re.findall(r'(\w+)=([^ ]+)',line));stage=int(mark['stage']);bounds=(int(mark['lb']),int(mark['le']));extent=int(mark['bytes'])
    if stages.get(stage)!=bounds:raise ValueError('Observer stage/range differs')
    if kind=='P30':
     if not p30_enabled or extent!=(bounds[1]-bounds[0])*3*8*D*4+16:raise ValueError('P30 source owner extent/flag differs')
     matches=[a for a in active.values() if a['pointer']==pointer(mark['pointer']) and a['bytes']==extent and 'owner' not in a]
    else:
     flags=re.search(r'residuals(\d+) logits(\d+)',line)
     if not flags:raise ValueError('SFD source flags missing')
     mark.update(residuals=flags[1],logits=flags[2])
     if int(mark['residuals'])!=1 or extent!=(bounds[1]-bounds[0])*D*4+int(mark['logits'])*248320*4:raise ValueError('SFD complete owner extent differs')
     matches=[a for a in active.values() if a['bytes']==extent and 'owner' not in a]
     if matches:matches=[max(matches,key=lambda a:a['alloc_line'])]
    if len(matches)!=1 or matches[0]['function']!='urUSMDeviceAlloc' or matches[0]['device'] is None:raise ValueError('Observer does not bind actual owning device allocation')
    allocation=matches[0];owner={**{k:v for k,v in allocation.items() if k!='owner'},'SFD_logits':int(mark['logits']) if kind=='SFD' else None,'kind':kind,'stage':stage,'bounds':bounds,'owner_line':number,'pid':int(mark['pid']) if kind=='P30' else None,'SFD_pointer_association':'source-bound immediate allocation chronology' if kind=='SFD' else 'explicit pointer marker'};owners.append(owner);allocation['owner']=owner;owned_history.add((owner['context'],owner['pointer']));live_bytes+=extent;maximum=max(maximum,live_bytes)
   if line.startswith('P30 release_begin '):
    mark=dict(re.findall(r'(\w+)=([^ ]+)',line));matches=[o for o in owners if o['kind']=='P30' and o['stage']==int(mark['stage']) and o['pid']==int(mark['pid']) and o['pointer']==pointer(mark['pointer']) and o['bytes']==int(mark['bytes']) and not o.get('free_line')]
    if len(matches)!=1 or matches[0].get('begin_line'):raise ValueError('P30 release begin owner differs')
    matches[0]['begin_line']=number
   if line.startswith('P30 release_returned '):
    mark=dict(re.findall(r'(\w+)=([^ ]+)',line));matches=[o for o in owners if o['kind']=='P30' and o['stage']==int(mark['stage']) and o['pid']==int(mark['pid']) and o.get('begin_line') and o.get('free_line') and not o.get('end_line')]
    if len(matches)!=1:raise ValueError('P30 returned without unique successful owning UR free')
    matches[0]['end_line']=number
  except (ValueError,KeyError,TypeError) as error:errors.append({'line':number,'error':str(error)})
 devices={}
 for stage,bounds in stages.items():
  selected=[o for o in owners if o['stage']==stage];p30=[o for o in selected if o['kind']=='P30'];sfd=[o for o in selected if o['kind']=='SFD']
  if len(p30)!=(2 if p30_enabled else 0) or len(sfd)!=2:errors.append({'error':'Exact two Prefill/Verifier source owner sets perstage required','stage':stage})
  if sum(o['SFD_logits'] for o in sfd)!=int(bounds[1]==48) or any(o['SFD_logits'] not in (0,1) for o in sfd):errors.append({'error':'Actual SFD head-owner roster differs','stage':stage})
  contexts={(o['context'],o['device']) for o in selected}
  if len(contexts)!=1:errors.append({'error':'P30/SFD stage owning context/device differs','stage':stage})
  elif contexts:devices[stage]=next(iter(contexts))[1]
 for owner in owners:
  if not owner.get('free_line') or (owner['kind']=='P30' and not owner.get('end_line')):errors.append({'error':'Actual observer owning free missing','owner':owner})
 if len(set(devices.values()))!=len(stages):errors.append({'error':'Distinct actual stage devices required'})
 if maximum>CAP or live_bytes!=0:errors.append({'error':'128MiB aggregate observer budget/terminal live owners exceeded'})
 return {'schema':1,'passed':not errors,'owners':owners,'errors':errors,'aggregate_peak_bytes':maximum,'aggregate_cap_bytes':CAP,'terminal_live_observer_bytes':live_bytes,'stage_context_devices':{str(k):v for k,v in devices.items()},'graph_retirement_runtime_handle_association_observed':False,'P30_route_to_owner_pointer_observed':False,'queue_wait_completion_marker_observed':False,'queue_wait_before_P30_release_marker_source_bound':True,'postfree_type_gate_used':False,'physical_reclamation_qualified':False,'whole_engine_allocations_retired':False,'scope':'All explicit P30 owner pointers plus source-bound SFD allocation chronology; original owning-context UR frees and aggregate observer bytes. Prefill/Verifier roles follow exact compiled source census; runtime route-to-pointer/graph-handle/wait-event associations UNOBSERVED.'}
