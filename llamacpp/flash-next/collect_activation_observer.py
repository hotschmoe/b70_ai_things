#!/usr/bin/env python3
"""Validate bounded FNACT raw vectors and compare only matched observation keys."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def fields(line):
    result={}
    for word in line.split()[2:]:
        if '=' in word:
            key,value=word.split('=',1);result[key]=value
    return result


def vector(path):
    raw=path.read_bytes()
    if not raw or len(raw)%4:raise ValueError('invalid canonical F32 byte count: '+str(path))
    summary=dict(elements=len(raw)//4,nan=0,pos_inf=0,neg_inf=0,pos_zero=0,neg_zero=0,
                 finite_nonzero=0,max_abs_finite=0.0,sha256=hashlib.sha256(raw).hexdigest())
    for (bits,) in struct.iter_unpack('<I',raw):
        exp=(bits>>23)&255;mant=bits&0x7fffff;sign=bits>>31
        if exp==255:
            if mant:summary['nan']+=1
            else:summary['neg_inf' if sign else 'pos_inf']+=1
        elif bits&0x7fffffff==0:summary['neg_zero' if sign else 'pos_zero']+=1
        else:
            summary['finite_nonzero']+=1
            f=struct.unpack('<f',struct.pack('<I',bits))[0]
            summary['max_abs_finite']=max(summary['max_abs_finite'],abs(f))
    return raw,summary


def load(log):
    records=[];ubatches={};skips=[];first=[];ends=[]
    for line in log.read_text(errors='replace').splitlines():
        if not line.startswith('FNACT '):continue
        data=fields(line);kind=line.split()[1]
        if kind=='ubatch':ubatches[(data['request'],data['ubatch'],data['ordinal'])]=data
        elif kind=='skip':skips.append(data)
        elif kind=='first_logits':first.append(data)
        elif kind=='graph_end':ends.append(data)
        elif kind=='vector':
            path=Path(data['file']);raw,summary=vector(path)
            if len(raw)!=int(data['bytes']):raise ValueError('metadata/raw byte mismatch')
            if len(raw)>16<<20:raise ValueError('per-vector capture bound exceeded')
            data['summary']=summary;data['path']=str(path);records.append(data)
    if sum(int(r['bytes']) for r in records)>128<<20:raise ValueError('process capture bound exceeded')
    for record in records:
        key=(record['request'],record['ubatch'],record['ordinal'])
        if key not in ubatches:raise ValueError('vector missing correlated ubatch metadata')
        record['ubatch_metadata']=ubatches[key]
    closed={(e['request'],e['ubatch'],e['ordinal']):e for e in ends}
    for record in records:
        close=closed.get((record['request'],record['ubatch'],record['ordinal']))
        if close is None or int(close['status'])!=0:raise ValueError('missing or failed graph completion for vector')
        record['semantic_point']=record['point']
        record['observation_index']=records.index(record)
    files={r['path']:r for r in records}
    for record in first:
        if record['file'] not in files:raise ValueError('first logits marker lacks vector record')
        if files[record['file']]['summary']['elements']!=248320:raise ValueError('incomplete first-vocabulary vector')
        files[record['file']]['semantic_point']='first_generated_logits'
    return dict(log=str(log),records=records,skips=skips,first_logits=first,graph_ends=ends,
                caveat='Observed samples only; skipped/absent points never imply zero. No equivalence or fidelity claim without matched controls.')


def key(record):
    return (record['request'],record['ubatch'],record['semantic_point'],record['batch'],record['token'],record['pos'])


def compare(a,b,request_a=None,request_b=None):
    def selected(trace,request):
        result={}
        for r in trace['records']:
            if request is not None and int(r['request'])!=request:continue
            if r['semantic_point']=='logits_candidate':continue
            k=list(key(r));k[0]='selected_request' if request is not None else k[0]
            result[tuple(k)]=r
        return result
    left=selected(a,request_a);right=selected(b,request_b)
    matched=[];unmatched=[]
    for k in sorted(set(left)|set(right),key=lambda k:left.get(k,right.get(k))['observation_index']):
        if k not in left or k not in right:unmatched.append(list(k));continue
        x,y=left[k],right[k]
        for field in ['ids','tokens','pos_min','pos_max','outputs']:
            if x['ubatch_metadata'].get(field)!=y['ubatch_metadata'].get(field):
                raise ValueError('unmatched token/ubatch geometry for '+str(k))
        if x['bytes']!=y['bytes'] or (x['semantic_point']!='first_generated_logits' and (x.get('ne')!=y.get('ne') or x.get('type')!=y.get('type'))):
            raise ValueError('unmatched canonical tensor descriptor for '+str(k))
        rx,sx=vector(Path(x['path']));ry,sy=vector(Path(y['path']))
        item={'key':list(k),'byte_equal':rx==ry,'a_sha256':sx['sha256'],'b_sha256':sy['sha256']}
        if rx!=ry:
            errors=[];reference=[];nonfinite_mismatch=0
            for (bx,),(by,) in zip(struct.iter_unpack('<I',rx),struct.iter_unpack('<I',ry)):
                if (bx>>23)&255!=255 and (by>>23)&255!=255:
                    vx=struct.unpack('<f',struct.pack('<I',bx))[0];vy=struct.unpack('<f',struct.pack('<I',by))[0]
                    errors.append((vx-vy)**2);reference.append(vx*vx)
                elif bx!=by:nonfinite_mismatch+=1
            item.update(nmse_finite=sum(errors)/max(sum(reference),1e-300),nonfinite_mismatch=nonfinite_mismatch)
        matched.append(item)
    return dict(matched=matched,unmatched=unmatched,
                first_observed_byte_difference=next((m for m in matched if not m['byte_equal']),None),
                caveat='Ordering follows first-arm observed callback order, not proof of earliest global graph divergence; inspect original graph order/coverage.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--other',type=Path);p.add_argument('--request-a',type=int);p.add_argument('--request-b',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=load(a.log)
    if a.other or a.request_b is not None:result['comparison']=compare(result,load(a.other) if a.other else result,a.request_a,a.request_b)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
    print('Validated',len(result['records']),'observed vectors;',len(result['skips']),'explicit skips;',len(result['first_logits']),'complete first-logit markers')


if __name__=='__main__':main()
