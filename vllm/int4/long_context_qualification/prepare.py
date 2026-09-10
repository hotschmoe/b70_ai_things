#!/usr/bin/env python3
"""Freeze the separate 200K arm after CPU tokenization; do not launch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
REPO=Path(__file__).resolve().parents[3]
RAW=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910')
ALIAS='qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276phase-mrv1fix-tp2-mtp3-fp8kv-freshcal-long200k'

def prepare(dest):
    prior=RAW/'mrv1-tp2-fp8-qualification'
    assert not (dest/'plan.json').exists()
    identity=json.loads((dest/'tokenized/tokenizer-identity.json').read_text())
    assert hashlib.sha256((dest/'tokenized/corpus.json').read_bytes()).hexdigest()==identity['corpus_sha256']
    assert len(identity['counts'])==8 and all(180000<=r['prompt_tokens']<=190000 for r in identity['counts'])
    shutil.copytree(prior/'config',dest/'config')
    config=json.loads((dest/'config/Config.json').read_text());cmd=config['Cmd'];cmd[cmd.index('--max-model-len')+1]='200000'
    assert cmd[cmd.index('--max-num-seqs')+1]=='4' and cmd[cmd.index('--max-num-batched-tokens')+1]=='32768'
    assert 'CCL_TOPO_P2P_ACCESS=0' in config['Env']
    (dest/'config/Config.json').write_text(json.dumps(config,indent=2)+'\n')
    shutil.copy2(prior/'fresh-scales.json',dest/'fresh-scales.json')
    assert hashlib.sha256((dest/'fresh-scales.json').read_bytes()).hexdigest()=='be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062'
    files=['vllm/int4/long_context_qualification/probe.py','vllm/fp8/kv_campaign_probe.py','vllm/int4/diagnostics/review_tp_host_trace.py']
    for rel in files:
        target=dest/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,target)
    plan=json.loads((prior/'plan.json').read_text());server=plan['server'];out=dest/'run'
    for flag,value in [('--out',out),('--preservation',dest/'config'),('--name','b70-mrv1-tp2-fp8-long200k'),('--served-alias',ALIAS),('--scales',dest/'fresh-scales.json')]:server[server.index(flag)+1]=str(value)
    assert server[server.index('--image')+1]=='sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1'
    assert server[server.index('--tp-host-trace')+1]=='phase-mrv1'
    jobs=[]
    for i,mode in enumerate(['retrieval','reuse','guide','cancel','recovery']):
        jobs.append(dict(name=f'{i+1:02d}-{mode}',command=list(map(str,['python3',dest/'source/vllm/int4/long_context_qualification/probe.py','--alias',ALIAS,'--corpus',dest/'tokenized/corpus.json','--out',out/mode,'--mode',mode,'--namespace',dest.name])),timeout=1350))
    jobs.append(dict(name='06-startup-host-boundaries',command=list(map(str,['python3',dest/'source/vllm/int4/diagnostics/review_tp_host_trace.py',out/'tp-host-trace','--out',out/'tp-host-review.json'])),timeout=60))
    plan.update(out=str(out),jobs=jobs,notes='PREPARED_NOT_RUN. Requires successful100K predecessor including lifecycle. Calibration covered <=96K,185K is separate validation. Exact tokenizer corpus, four independent long retrievals then reuse, four long guides, bounded single cancellation and concurrent recovery. Guide gate is structural, not full semantic proof. No shelf promotion.')
    (dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (dest/'prerequisite.json').write_text(json.dumps(dict(plan=str(prior/'plan.json'),run=str(prior/'run'),lifecycle=str(prior/'plan.lifecycle-rc')),indent=2)+'\n')
    files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    external=[REPO/'vllm/cache800k/run_arm.py',REPO/'vllm/fp8/kv_campaign_server.py',REPO/'vllm/int4/diagnostics/xpu_health_strict.sh']+sorted((REPO/'vllm/fp8/kv_hooks').glob('*.py'))
    (dest/'manifest.json').write_text(json.dumps(dict(files=files,external={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in external}),indent=2)+'\n')
    return dest/'plan.json'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();print(prepare(a.destination.resolve()))
