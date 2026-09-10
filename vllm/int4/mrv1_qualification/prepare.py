#!/usr/bin/env python3
"""Freeze an unlaunched MRV1 TP2 qualification plan; never touch a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
REPO=Path(__file__).resolve().parents[3]
RAW=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910')
ALIAS='qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276phase-mrv1fix-tp2-mtp3-fp8kv-freshcal-qualify-ctx100k'
IMAGE='sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1'

def prepare(dest):
    dest.mkdir(parents=True,exist_ok=False)
    old=RAW/'fresh-calibration-plan'
    shutil.copytree(old/'config',dest/'config')
    config=json.loads((dest/'config/Config.json').read_text())
    config['Env']=[v if not v.startswith('CCL_TOPO_P2P_ACCESS=') else 'CCL_TOPO_P2P_ACCESS=0' for v in config['Env']]
    (dest/'config/Config.json').write_text(json.dumps(config,indent=2)+'\n')
    shutil.copy2(old/'fresh-scales.json',dest/'fresh-scales.json')
    files=['vllm/int4/mrv1_qualification/agent_probe.py','vllm/int4/mrv1_qualification/gate.py',
           'vllm/int4/replay_pi_history.py','vllm/int4/audit_replay_quality.py',
           'vllm/fp8/kv_campaign_agent_probe.py','vllm/fp8/kv_campaign_probe.py']
    for rel in files:
        target=dest/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,target)
    tiny=dest/'source/tiny.py';shutil.copy2(RAW/'source-review/steve-probe_endpoint.py',tiny)
    plan=json.loads((old/'qualify-fp8_e4m3.plan.json').read_text());out=dest/'run'
    oldout=plan['out'];plan['out']=str(out)
    plan['server']=[str(out) if x==oldout else x for x in plan['server']]
    for flag,value in [('--preservation',dest/'config'),('--image',IMAGE),('--name','b70-mrv1-tp2-fp8-qualification'),('--served-alias',ALIAS),('--scales',dest/'fresh-scales.json')]:
        plan['server'][plan['server'].index(flag)+1]=str(value)
    plan['server']+=['--tp-host-trace','phase-mrv1']
    jobs=[]
    def job(name,cmd,timeout):jobs.append(dict(name=name,command=list(map(str,cmd)),timeout=timeout))
    replay=dest/'source/vllm/int4/replay_pi_history.py';gate=dest/'source/vllm/int4/mrv1_qualification/gate.py'
    for index,manifest,name,repeats,count in [(1,RAW/'reconstructed-payloads/manifest.json','pi-replay',2,9),(3,RAW/'early-clean-history-payloads/manifest.json','early-history',3,3)]:
        if index==3:job('02-tiny24',['python3',tiny,'--base-url','http://127.0.0.1:18125','--model','hotschmoe-dd','--out',out/'tiny-prefill.json','--timeout','90'],900)
        manifest_copy=dest/(name+'-manifest.json')
        shutil.copy2(manifest,manifest_copy)
        job(f'{index:02d}-{name}',['python3',replay,'run','--manifest',manifest_copy,'--model','hotschmoe-dd','--out',out/name,'--timeout','240','--repeats',repeats],2250 if index==1 else 750)
        job(f'{index:02d}-{name}-quality',['python3',gate,'replay',out/name,'--count',count],60)
    for temperature,seeds,label in [(0,[42,42,42],'deterministic'),(.7,[42,43,44],'sampled')]:
        for round_,seed in enumerate(seeds):
            name=f'{label}-round{round_+1}'
            probe=dest/'source/vllm/int4/mrv1_qualification/agent_probe.py'
            job('04-'+name,['python3',probe,'--model','hotschmoe-dd','--out',out/name,'--records','360','--stream','--shared-cache','--cache-namespace',dest.name+'-'+name,'--turns','4','--timeout','180','--temperature',temperature,'--seed',seed],1500)
            job('04-'+name+'-quality',['python3',gate,'agent',out/name],60)
    plan['jobs']=jobs
    plan['notes']='Prepared, not GPU-qualified. Six fresh salted history rounds: three t0 seed42, three t0.7 seeds42/43/44. Same salt shared within each round; distinct between rounds. Positive per-session hits required. Pi is reconstructed, heuristic review not semantic proof. Startup host trace is not device completion or graph replay count.'
    (dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    dependencies=[REPO/'vllm/cache800k/run_arm.py',REPO/'vllm/fp8/kv_campaign_server.py',REPO/'vllm/int4/diagnostics/xpu_health_strict.sh']
    dependencies+=sorted((REPO/'vllm/fp8/kv_hooks').glob('*.py'))
    for manifest in [dest/'pi-replay-manifest.json',dest/'early-history-manifest.json']:
        dependencies += [Path(c['payload']) for c in json.loads(manifest.read_text())['cases']]
    (dest/'manifest.json').write_text(json.dumps(dict(files=hashes,external={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}),indent=2)+'\n')
    return dest/'plan.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();print(prepare(a.destination.resolve()))
