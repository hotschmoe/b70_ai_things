#!/usr/bin/env python3
"""Validate bounded SFD raw files and compare first observed layer divergence."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def parse(line):
    return dict(word.split('=',1) for word in line.split()[2:] if '=' in word)


def load(log):
    requests={};vectors=[];skips=[]
    for line in log.read_text(errors='replace').splitlines():
        if not line.startswith('SFD '):continue
        kind=line.split()[1];record=parse(line)
        if kind=='request':requests[(record['pid'],record['request'])]=record
        elif kind=='skip':skips.append(record)
        elif kind=='vector':
            key=(record['pid'],record['request'])
            if key not in requests:raise ValueError('vector lacks GEN token correlation')
            raw=Path(record['file']).read_bytes()
            expected=248320*4 if record['phase']=='first_logits_before_sampler' else 10240*4
            if len(raw)!=expected or len(raw)!=int(record['bytes']):raise ValueError('incomplete canonical F32 vector')
            ids=[int(x) for x in requests[key]['ids'].split(',')]
            pos=int(record['pos'])
            if pos<0 or pos>=len(ids) or ids[pos]!=int(record['token']):raise ValueError('token position/ID mismatch')
            record['sha256']=hashlib.sha256(raw).hexdigest();record['ids']=ids
            record['nonfinite']=sum(((bits>>23)&255)==255 for (bits,) in struct.iter_unpack('<I',raw))
            record['positive_zero']=sum(bits==0 for (bits,) in struct.iter_unpack('<I',raw))
            record['negative_zero']=sum(bits==0x80000000 for (bits,) in struct.iter_unpack('<I',raw))
            vectors.append(record)
    if sum(int(r['bytes']) for r in vectors)>64<<20:raise ValueError('capture byte bound exceeded')
    return dict(scope='observed raw residual/logit samples only; GDN/PLE/QSA state unobserved',
                log=str(log),requests=list(requests.values()),vectors=vectors,skips=skips)


def compare(a,b,ra,rb):
    def chosen(trace,request):
        result={}
        for v in trace['vectors']:
            if int(v['request'])!=request:continue
            key=(v['phase'],int(v['layer']),int(v['pos']))
            if key in result:raise ValueError('duplicate observation key')
            result[key]=v
        return result
    x=chosen(a,ra);y=chosen(b,rb);rows=[];missing=[]
    for key in sorted(set(x)|set(y)):
        if key not in x or key not in y:missing.append(list(key));continue
        left,right=x[key],y[key]
        if left['ids']!=right['ids'] or left['token']!=right['token'] or left['bytes']!=right['bytes']:
            raise ValueError('unmatched actual GEN IDs/token geometry')
        rows.append(dict(key=list(key),byte_equal=left['sha256']==right['sha256'],
                         a_sha256=left['sha256'],b_sha256=right['sha256']))
    first={}
    for phase in ['prefill_last_residual','first_window_residual']:
        different=[r for r in rows if r['key'][0]==phase and not r['byte_equal']]
        first[phase]=min(different,key=lambda r:(r['key'][2],r['key'][1])) if different else None
    return dict(matched=rows,unobserved_keys=missing,first_observed_layer_difference=first,
                complete_state_fidelity_proven=False,caveat='Same-layer residuals do not prove all token/state components equal.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--other',type=Path)
    p.add_argument('--request-a',type=int);p.add_argument('--request-b',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=load(a.log)
    if a.request_b is not None:
        if a.request_a is None:p.error('--request-a required for comparison')
        result['comparison']=compare(result,load(a.other) if a.other else result,a.request_a,a.request_b)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
    print('Validated',len(result['vectors']),'observed canonical vectors;',len(result['skips']),'explicit skips')


if __name__=='__main__':main()
