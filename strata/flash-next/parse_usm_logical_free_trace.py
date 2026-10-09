#!/usr/bin/env python3
"""Verify chronological UR logical allocation/free evidence; metadata type is not liveness."""
import argparse
import hashlib
import json
from pathlib import Path
import re

CALL=re.compile(r'<--- (urUSMDeviceAlloc|urUSMHostAlloc|urUSMSharedAlloc|urUSMFree|urContextGetNativeHandle)\((.*)\) -> (\w+);')
HEX=r'0x[0-9a-fA-F]+'


def pointer(value):
    return int(str(value),0)


def parse_trace(text, require_owners=True):
    active={};history={};bridges={};owners={};begins={};ends={};events=[];errors=[]
    counts={'allocations':0,'frees':0,'owners':0,'context_bridges':0}
    for number,line in enumerate(text.splitlines(),1):
        try:
            if line.startswith('UPLOAD_USM '):
                mark=json.loads(line[len('UPLOAD_USM '):]);kind=mark['event'];stage=mark['stage']
                if kind=='owner_register':
                    native=pointer(mark['ze_context']);address=pointer(mark['pointer'])
                    contexts={ctx for ctx,value in bridges.items() if value==native}
                    matches=[(key,item) for key,item in active.items() if key[0] in contexts and key[1]==address]
                    if len(matches)!=1:raise ValueError('owner pointer has no unique live allocation/context bridge')
                    key,item=matches[0]
                    if item['generation'] in owners:raise ValueError('duplicate owner registration')
                    if item['bytes']!=mark['bytes']:raise ValueError('owned image/scratch extent differs from UR allocation')
                    owners[item['generation']]={**mark,'context':key[0],'register_line':number,'generation':item['generation']}
                    counts['owners']+=1
                elif kind=='destroy_begin':
                    if stage in begins:raise ValueError('duplicate destructor begin')
                    begins[stage]=number
                elif kind=='destroy_end':
                    if stage not in begins or stage in ends or not mark.get('owning_destructor_returned'):raise ValueError('missing/duplicate/failed destructor end')
                    ends[stage]=number
                elif kind not in ['postfree_query','probe_begin','probe_end']:
                    raise ValueError('unknown ownership marker')
                continue
            match=CALL.search(line)
            if not match:continue
            name,args,result=match.groups()
            if result!='UR_RESULT_SUCCESS':raise ValueError(name+' did not return success')
            context_match=re.search(r'\.hContext = ('+HEX+r')',args)
            if not context_match:raise ValueError('unparsed UR context')
            context=pointer(context_match.group(1))
            if name=='urContextGetNativeHandle':
                native=re.search(r'\.phNativeContext = '+HEX+r' \((0x[0-9a-fA-F]+|\d+)\)',args)
                if not native:raise ValueError('unparsed native context bridge')
                value=pointer(native.group(1))
                if context in bridges and bridges[context]!=value:raise ValueError('UR context bridge changed')
                bridges[context]=value;counts['context_bridges']+=1;continue
            if name=='urUSMFree':
                value=re.search(r'\.pMem = ('+HEX+r')(?=,|$)',args)
                if not value:raise ValueError('unparsed free pointer')
                key=(context,pointer(value.group(1)))
                if key not in active:raise ValueError('free without matching live allocation or double free')
                item=active.pop(key);item['free_line']=number;history[item['generation']]=item;counts['frees']+=1
                events.append({'line':number,'function':name,'context':context,'pointer':key[1],'generation':item['generation']})
            else:
                value=re.search(r'\.ppMem = '+HEX+r' \(('+HEX+r')\)',args)
                size=re.search(r'\.size = (\d+)(?=,|$)',args)
                if not value or not size:raise ValueError('unparsed allocation pointer/size')
                key=(context,pointer(value.group(1)))
                if key in active:raise ValueError('duplicate live allocation/context/pointer')
                generation=counts['allocations']+1
                active[key]={'generation':generation,'alloc_line':number,'bytes':int(size.group(1)),'context':context,'pointer':key[1],'function':name}
                counts['allocations']+=1;events.append({'line':number,'function':name,'context':context,'pointer':key[1],'generation':generation})
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
            errors.append({'line':number,'error':str(error),'record':line})
    if not counts['allocations'] or counts['allocations']!=counts['frees'] or active:
        errors.append({'error':'allocation/free coverage missing, unequal, or live ledger nonempty','live_count':len(active)})
    if require_owners:
        if not owners:errors.append({'error':'owned source allocation markers missing'})
        if set(begins)!=set(ends):errors.append({'error':'destructor marker coverage incomplete'})
        for generation,owner in owners.items():
            record=history.get(generation);stage=owner['stage']
            if (record is None or stage not in begins or stage not in ends
                    or not (record['alloc_line'] < owner['register_line'] < begins[stage] < record['free_line'] < ends[stage])):
                errors.append({'error':'owned allocation not successfully retired inside its matching destructor','owner':owner,'allocation':record})
    return {'schema':1,'scope':'UR logical ownership/free evidence only; no numerical, source-byte or physical-release conclusion',
            'passed':not errors,'counts':counts,'owners':list(owners.values()),'events':events,'live':list(active.values()),'errors':errors,
            'postfree_type_gate_used':False,'physical_backing_release_proven':False}


def negative_controls(text, require_owners):
    if not parse_trace(text,require_owners)['passed']:raise ValueError('positive trace required before negative controls')
    lines=text.splitlines();indices=[i for i,line in enumerate(lines) if '<--- urUSMFree(' in line]
    if not indices:raise ValueError('free events missing')
    remove=lines[:];remove.pop(indices[-1]);duplicate=lines[:];duplicate.insert(indices[-1]+1,lines[indices[-1]])
    failed=lines[:];failed[indices[-1]]=failed[indices[-1]].replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN')
    return {name:not parse_trace('\n'.join(rows),require_owners)['passed'] for name,rows in [('missing_free',remove),('double_free',duplicate),('failed_free',failed)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--control-smoke',action='store_true',help='Actual allocator control without model owner markers; cannot qualify model lifecycle')
    p.add_argument('--oracle-json',type=Path,help='v2 raw source/probe receipt; mandatory for model lifecycle interpretation')
    a=p.parse_args()
    if a.output.exists():p.error('New output receipt path required')
    if not a.control_smoke and a.oracle_json is None:p.error('Model lifecycle needs --oracle-json')
    raw=a.log.read_bytes();text=raw.decode('utf-8',errors='replace');r=parse_trace(text,not a.control_smoke)
    r['log_sha256']=hashlib.sha256(raw).hexdigest();r['control_smoke']=a.control_smoke
    if r['passed']:
        r['negative_controls']=negative_controls(text,not a.control_smoke)
        r['passed']=all(r['negative_controls'].values())
    if a.oracle_json is not None:
        oracle=json.loads(a.oracle_json.read_text());r['oracle_sha256']=hashlib.sha256(a.oracle_json.read_bytes()).hexdigest()
        if not oracle.get('source_and_probe_passed') or not oracle.get('all_owners_destructor_returned'):
            r['passed']=False;r['errors'].append({'error':'v2 source/probe/destructor receipt incomplete'})
        rows=sum(s['unique_allocations']+1 for s in oracle.get('stages',[]))
        if r['counts']['owners']!=rows:
            r['passed']=False;r['errors'].append({'error':'owned image/scratch marker coverage differs from raw oracle','expected':rows})
    r['lifecycle_qualified_by_this_receipt']=False
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='ascii')
    print(json.dumps({'passed':r['passed'],'counts':r['counts'],'receipt':str(a.output)}))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
