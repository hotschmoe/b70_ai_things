"""Offline closure, strict gates, and actual pre-capture native-call proof."""
import argparse
import json
from pathlib import Path
import re

REQUIRED_GATE=['identity','greedy_17x19','greedy_23x29','exact_repeat','long_8192',
               'long_32768','identity_after','identity_unchanged']

def branch_proof(directory,arm):
    rows=[]
    for p in directory.glob('observer-*.jsonl'):
        rows.extend(json.loads(line)for line in p.read_text().splitlines()if line)
    required={'b70_o4.w8_gemv'}if arm=='control'else{
        'b70_ep.moe_ep_route','b70_o4.w8_down_act','b70_o4.w8_up_collapse'}
    forbidden=set()if arm!='control'else{
        'b70_ep.moe_ep_route','b70_o4.w8_down_act','b70_o4.w8_up_collapse'}
    seen={r['target']for r in rows if r['event']=='op_return'}
    witnesses=[]
    for cap in rows:
        if cap['event']!='capture_return'or cap.get('profile_only')is not False:continue
        before={r['target']for r in rows if r['event']=='op_return'and r['pid']==cap['pid']
                and r['monotonic_ns']<cap['monotonic_ns']}
        if required<=before:witnesses.append(cap)
    errors=[r for r in rows if r['event']in ('hook_error','op_raise')]
    return dict(pass_proof=bool(witnesses)and not forbidden&seen and not errors,
                required=sorted(required),observed_return=sorted(seen),forbidden_observed=sorted(forbidden&seen),
                full_capture_return_witnesses=witnesses,errors=errors,
                scope='Python native op returned before full capture_model returned in same workerPID; no device-sync claim')

def classify(aout,outer,arm):
    issues=[]
    log=(aout/'window.log').read_text()
    for key in ['SERVE_RC','ENGINE_ERROR_RC','HEALTH_AFTER_RC']:
        values=re.findall(r'\b'+key+r':(-?\d+)\b',log)
        if not values or values[-1]!='0':issues.append('lifecycle '+key+' not0')
    if 'WINDOW_DONE'not in log:issues.append('lifecycle missing WINDOW_DONE')
    if (aout/'engine-errors.txt').read_text().strip():issues.append('fatal engine evidence')
    if 'xpu-health: HEALTHY'not in(aout/'health-after.txt').read_text():issues.append('nativepost nothealthy')
    for name in ['so-after-check.txt']:
        lines=(aout/name).read_text().splitlines()
        if len(lines)!=4 or not all(l.endswith(': OK')for l in lines):issues.append(name+' incomplete/failed')
    source=json.loads((outer/'source-after.json').read_text())
    if source.get('bad')!=[]:issues.append('source guard failed')
    outerlog=(outer/'status.txt').read_text()
    for name in ['SOURCE_RC:0','PRIVATE_RC:0']:
        if name not in outerlog:issues.append('private/source guard missing '+name)
    clients=json.loads((aout/'clients-status.json').read_text())
    if not clients.get('completed_all_phases')or clients.get('overall_bound_expired'):issues.append('postREADY client phase incomplete/bounded')
    phases=clients.get('phases',{})
    for name in ['bench','probe','canonical']:
        if phases.get(name,{}).get('rc')!=0:issues.append('client phase '+name+' failed')
    gate=json.loads((aout/'gate.json').read_text());checks=gate['checks']
    strict={k:checks.get(k,{}).get('pass')for k in REQUIRED_GATE}
    gate_complete=all(type(strict[k])is bool and not checks[k].get('error')for k in REQUIRED_GATE)
    if not gate_complete:issues.append('promo phase missing/errored checks')
    for k in ['identity','greedy_17x19','greedy_23x29','identity_after','identity_unchanged']:
        if strict[k]is not True:issues.append('identity/math '+k+' failed')
    expected_gate_rc=0 if all(v is True for v in strict.values())else 3
    if phases.get('gate',{}).get('rc')!=expected_gate_rc:issues.append('promo RC/checks mismatch')
    branch=branch_proof(aout/'branches',arm)
    if not branch['pass_proof']:issues.append('native branch proof failed')
    config=json.loads((aout/'lmx/server-config.json').read_text())
    proof={'max_model_len':config.get('model_config',{}).get('max_model_len'),
           'speculative_config':config.get('speculative_config','MISSING'),
           'TP':config.get('parallel_config',{}).get('tensor_parallel_size'),
           'max_num_batched_tokens':config.get('scheduler_config',{}).get('max_num_batched_tokens'),
           'max_num_seqs':config.get('scheduler_config',{}).get('max_num_seqs'),
           'prefix_caching':config.get('cache_config',{}).get('enable_prefix_caching'),
           'compilation_mode':config.get('compilation_config',{}).get('mode'),
           'cudagraph_mode':config.get('compilation_config',{}).get('cudagraph_mode'),
           'capture_sizes':config.get('compilation_config',{}).get('cudagraph_capture_sizes')}
    expected={'max_model_len':131072,'speculative_config':None,'TP':1,'max_num_batched_tokens':12288,
              'max_num_seqs':4,'prefix_caching':False,'compilation_mode':0,
              'cudagraph_mode':[2,0],'capture_sizes':[1,2,3,4,8]}
    if proof!=expected:issues.append('actual engine config mismatch/incomplete')
    canonical={}
    for suite in ['reasoning-v1','code-v1']:
        p=aout/'lmx'/('no-spec-'+suite)
        v=json.loads((p/'validation.json').read_text());payload=json.loads((p/'payload.json').read_text())
        complete=all((p/n/'sample.json').is_file()for n in ['warmup','iteration-1','iteration-2','iteration-3'])
        canonical[suite]={'valid':v.get('valid'),'all_samples_saved':complete,'tok_s':payload.get('tokSOut')}
        if v.get('valid')is not True or not complete:issues.append('canonical '+suite+' incomplete/invalid')
    return dict(operational_complete=not issues,issues=issues,strict_gates=strict,
                strict_all_pass=all(v is True for v in strict.values()),branch_proof=branch,
                client_status=clients,actual_config=proof,canonical=canonical,qualification_pass=False,
                scope='finite dirty diagnostic; operational closure does not qualify fidelity/stability/promotion',
                bound_interpretation='overall expiration is not proof of deadlock; preserve progress/last phase')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--aout',type=Path,required=True);p.add_argument('--outer',type=Path,required=True)
    p.add_argument('--arm',choices=['control','routeq8f'],required=True);a=p.parse_args()
    try:r=classify(a.aout,a.outer,a.arm)
    except Exception as e:r=dict(operational_complete=False,issues=['incomplete classifier input: '+type(e).__name__],qualification_pass=False)
    (a.outer/'outcome.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');print(json.dumps(r))
    raise SystemExit(0 if r['operational_complete']else 1)
