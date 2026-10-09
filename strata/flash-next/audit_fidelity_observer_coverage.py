#!/usr/bin/env python3
"""Actual-source observer audit and strict observed coverage; no GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
from collect_fidelity_observer import load

DEFAULT_SOURCE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T080647Z-6_ja3xd5/source')

def coverage(trace,expected_requests,activations,expected_stages,cancelled=()):
    errors=[];requests=trace['requests'];groups={}
    if expected_requests<=0:errors.append('armed diagnostic requires a positive expected request count')
    if not expected_stages:errors.append('explicit expected stage ranges required')
    expected_stages=dict(expected_stages)
    ranges=[layer for lb,le in expected_stages.values() for layer in range(lb,le)]
    if sorted(ranges)!=list(range(48)):errors.append('expected stage ranges must partition all48 layers')
    heads=[stage for stage,(lb,le) in expected_stages.items() if le==48]
    if len(heads)!=1:errors.append('exactly one expected final head stage required')
    for request in requests:
        key=(request['pid'],request['request'])
        if key in groups:errors.append('duplicate request identity')
        groups[key]=[]
    if len(groups)!=expected_requests:errors.append('observed request count differs from required count')
    seen_files=set()
    for vector in trace['vectors']:
        if vector.get('file'):
            if vector['file'] in seen_files:errors.append('reused raw capture filename')
            seen_files.add(vector['file'])
        if int(vector.get('nonfinite',0)):errors.append('nonfinite observed vector')
        key=(vector['pid'],vector['request'])
        if key not in groups:errors.append('vector without request');continue
        groups[key].append(vector)
    for key,vectors in groups.items():
        req=next(r for r in requests if (r['pid'],r['request'])==key)
        tokens=int(req['tokens']);ids=[int(i) for i in req['ids'].split(',')]
        if int(req['request']) in cancelled:continue
        for vector in vectors:
            stage=int(vector['stage']);bounds=(int(vector['lb']),int(vector['le']))
            if stage not in expected_stages or bounds!=expected_stages[stage]:
                errors.append(str(key)+': unexpected stage identity/range')
            if vector['phase']!='first_logits_before_sampler' and not bounds[0]<=int(vector['layer'])<bounds[1]:
                errors.append(str(key)+': residual layer outside stage range')
            if vector['phase'] not in ('first_logits_before_sampler','first_window_residual','prefill_last_residual'):
                errors.append(str(key)+': unknown observed phase')
        if tokens!=len(ids):errors.append(str(key)+': declared token count mismatch')
        logits=[v for v in vectors if v['phase']=='first_logits_before_sampler']
        if len(logits)!=1 or any(int(v['pos'])!=tokens-1 or int(v['layer'])!=-1 or int(v['le'])!=48 or int(v['stage']) not in heads for v in logits):
            errors.append(str(key)+': requires exactly one last-prompt-position full head logits vector')
        first=[v for v in vectors if v['phase']=='first_window_residual']
        if activations:
            layers=[int(v['layer']) for v in first]
            if sorted(layers)!=list(range(48)):errors.append(str(key)+': first-window layer coverage incomplete or duplicated')
            if any(int(v['pos'])!=tokens-1 or not int(v['lb'])<=int(v['layer'])<int(v['le']) for v in first):
                errors.append(str(key)+': first-window position/stage ownership mismatch')
        elif first:errors.append(str(key)+': activations present in logits-only arm')
        prefill=[v for v in vectors if v['phase']=='prefill_last_residual']
        points={}
        for v in prefill:points.setdefault(int(v['pos']),[]).append(int(v['layer']))
        for pos,layers in points.items():
            if sorted(layers)!=list(range(48)):errors.append(str(key)+': prefill position '+str(pos)+' layer coverage incomplete or duplicated')
        # Zero prefill points is valid for short-window or fully reused prompts.
        # This must never be represented as complete prefill/state coverage.
    return {'passed':not errors,'errors':errors,'required_requests':expected_requests,
            'required_first_window_layers':48 if activations else 0,'required_stage_ranges':expected_stages,'cancelled_requests_unobserved':list(cancelled),
            'prefill_absence_is_unobserved':True,'complete_state_fidelity_proven':False}


def source_audit(source):
    names=['sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp','sycl/src/program/generate.cpp',
           'sycl/include/strata/core/fidelity_observer.hpp','sycl/include/strata/core/fidelity_observer_contract.hpp']
    text={name:(source/name).read_text() for name in names};v,p,g,h,c=(text[n] for n in names)
    peer=p[p.index('struct PeerPrefill {'):p.index('struct Prefill::Impl {')]
    impl=p[p.index('struct Prefill::Impl {'):p.index('// chunk buffers',p.index('struct Prefill::Impl {'))]
    copy=v.index('fidelity_snapshot_->logits(head_logits_,*cs)')
    checks={
        'prefill_impl_owner': 'fidelity_snapshot' not in peer and 'fidelity_snapshot' in impl,
        'logit_copy_after_head_matvec_before_argmax':v.rindex('native_mmvq(head_->type()',0,copy)<copy<v.index('argmax_rows(head_logits_',copy),
        'prefill_last_contiguous_row':'m.R+size_t(T-1)*D' in p and 'const int64_t at=p0+T-1' in p,
        'prefill_capture_after_ffn_half':'if (half==1 && fidelity_chunk && m.fidelity_snapshot)' in p,
        'prefill_native_disables_fused_pending_write':'return strata::core::native_hc_requested() || v;' in p,
        'first_decode_uses_last_prompt_token':'int64_t p = n - 1;' in g and 'int32_t x = (int32_t) ids[(size_t) (n - 1)]' in g,
        'head_row_guard':'position==prompt_tokens-1' in c and 'rows==1 && row==0' in c,
        'bounded_first_window_selection':'produced_n==0 && strata::core::fidelity_diag::current().active && T==1' in g,
        'request_bounds':'REQUESTS=6,CHUNKS=2' in c and 'r.ordinal>REQUESTS' in h,
        'prefill_chunk_quota_per_owner':'if(m.fidelity_request!=request){m.fidelity_request=request;m.fidelity_chunks=0;}' in p,
        'verifier_default_off_allocation_gate':'if (fidelity_diag::settings().enabled)' in v,
        'prefill_default_off_allocation_gate':'settings().enabled && core::fidelity_diag::settings().activations' in p,
    }
    return {'source':str(source),'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names},
            'checks':checks,'passed':all(checks.values()),'scope':'static consumed-source checks; no execution or graph ordering proof',
            'max_output_bytes_six_requests':6*(3*48*10240*4+248320*4),
            'unobserved':['GDN persistent state','PLE history','QSA/indexer/KV','pending-state internals','other token rows']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--log',type=Path);p.add_argument('--expected-requests',type=int,default=6)
    p.add_argument('--stage',action='append',default=[],help='Required observed stage:lower-layer:exclusive-upper-layer');p.add_argument('--cancelled-request',type=int,action='append',default=[]);p.add_argument('--activations',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'source_audit':source_audit(a.source)}
    if a.log:
        stages={}
        for raw in a.stage:
            stage,lb,le=map(int,raw.split(':'));stages[stage]=(lb,le)
        result['coverage']=coverage(load(a.log),a.expected_requests,a.activations,stages,a.cancelled_request)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    passed=result['source_audit']['passed'] and result.get('coverage',{'passed':True})['passed']
    print('PASS' if passed else 'FAIL','observed-source/coverage audit; full state and GPU equivalence unqualified')
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
