#!/usr/bin/env python3
"""Targeted chronological UR owner/free audit; does not require all engine USM freed."""
import argparse,json,re
from pathlib import Path
from parse_usm_logical_free_trace import CALL,HEX,pointer


def audit_text(text):
 active={};targets={};released={};errors=[];generation=0
 expected_extent=7023104
 for number,line in enumerate(text.splitlines(),1):
  try:
   match=CALL.search(line)
   if match:
    name,args,result=match.groups()
    if name=='urContextGetNativeHandle':continue
    if result!='UR_RESULT_SUCCESS':raise ValueError('UR allocation/free returned error')
    context_match=re.search(r'\.hContext = ('+HEX+r')',args)
    if not context_match:raise ValueError('UR owner context unparsed')
    context=pointer(context_match.group(1))
    if name=='urUSMFree':
     value=re.search(r'\.pMem = ('+HEX+r')(?=,|$)',args)
     if not value:raise ValueError('UR free pointer unparsed')
     key=(context,pointer(value.group(1)));entry=active.pop(key,None)
     if key in targets:
      target=targets[key]
      if entry is None or entry['generation']!=target['generation'] or 'release_begin_line' not in target or target.get('free_line'):raise ValueError('Target absent/double/context-wrong/outside owner free')
      target['free_line']=number
    else:
     value=re.search(r'\.ppMem = '+HEX+r' \(('+HEX+r')\)',args);size=re.search(r'\.size = (\d+)(?=,|$)',args)
     if not value or not size:raise ValueError('UR target allocation args unparsed')
     key=(context,pointer(value.group(1)));generation+=1
     if key in active:raise ValueError('Duplicate live UR context/pointer')
     active[key]={'context':context,'pointer':key[1],'bytes':int(size.group(1)),'generation':generation,'alloc_line':number,'function':name}
   if line.startswith('L0Q8 allocation '):
    values=dict(re.findall(r'(\w+)=([^ ]+)',line));address=pointer(values['pointer']);
    if int(values['bytes'])!=expected_extent or values.get('owner_queue')!='verifier_cs':raise ValueError('Layer0 complete33field buffer plus56B nonce/control/key extent or owning queue differs')
    matches=[(k,v) for k,v in active.items() if k[1]==address and v['bytes']==int(values['bytes'])]
    if len(matches)!=1:raise ValueError('Layer0 owner does not uniquely bind live allocation/context/extent')
    key,entry=matches[0]
    if key in targets:raise ValueError('Duplicate target owner allocation marker')
    targets[key]={**entry,'stage':int(values['stage']),'owner_line':number}
   if line.startswith('L0Q8 release_begin '):
    values=dict(re.findall(r'(\w+)=([^ ]+)',line));matches=[v for v in targets.values() if v['pointer']==pointer(values['pointer']) and v['stage']==int(values['stage'])]
    if len(matches)!=1 or 'release_begin_line' in matches[0] or matches[0]['bytes']!=int(values['bytes']):raise ValueError('Layer0 release owner/extent differs')
    matches[0]['release_begin_line']=number
   if line.startswith('L0Q8 release_returned '):
    values=dict(re.findall(r'(\w+)=([^ ]+)',line));matches=[v for v in targets.values() if v['stage']==int(values['stage'])]
    if len(matches)!=1 or not matches[0].get('free_line') or matches[0].get('release_end_line'):raise ValueError('Layer0 owning release did not contain successful unique UR free')
    matches[0]['release_end_line']=number
  except (ValueError,KeyError,TypeError) as error:errors.append({'line':number,'error':str(error)})
 if len(targets)!=1:errors.append({'error':'exactly one global-layer0 observer owner required'})
 for target in targets.values():
  if target['stage']!=0 or not(target['alloc_line']<target['owner_line']<target.get('release_begin_line',0)<target.get('free_line',0)<target.get('release_end_line',0)):errors.append({'error':'chronological owning allocation/release/context proof missing','owner':target})
 return {'passed':not errors,'scope':'target observer UR logical allocation/context/free only; physical release/whole engine allocator/state/math unproved','owners':list(targets.values()),'errors':errors,'unrelated_engine_allocations_not_required_retired':True,'postfree_type_gate_used':False,'explicit_new_allocation_scope':'single7023104B source allocation including complete33field storage and all56B nonce/control/keys; no separate explicit constants allocations in0021','full_model_math_qualified':False,'graph_retirement_runtime_handle_association_observed':False}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit_text(a.log.read_text(errors='replace'));result['plan_sha256']=a.plan_sha256;result['log']=str(a.log);a.output.write_text(json.dumps(result,indent=2)+'\n');print('PASS' if result['passed'] else 'FAIL','target layer0 logical lifecycle')
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
