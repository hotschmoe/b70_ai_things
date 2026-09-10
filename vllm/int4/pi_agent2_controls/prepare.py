#!/usr/bin/env python3
"""Prepare matched TP1 phase/fixed fresh-scale Pi controls without GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
REPO=Path(__file__).resolve().parents[3]
RAW=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910')
IMAGES=['sha256:0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077','sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1']

def prepare(root):
 root.mkdir(parents=True,exist_ok=False)
 previous=RAW/'mrv1-tp2-fp8-clean100k'
 manifest=json.loads((RAW/'reconstructed-payloads/manifest.json').read_text());manifest['cases']=[c for c in manifest['cases'] if c['agent']=='agent_2'];assert len(manifest['cases'])==2
 for card,label in enumerate(['phase','phase-mrv1fix']):
  dest=root/label;dest.mkdir();shutil.copytree(previous/'config',dest/'config');shutil.copy2(previous/'fresh-scales.json',dest/'fresh-scales.json')
  (dest/'pi-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  alias=f'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276{label}-tp1-mtp3-fp8kv-freshcal-agent2-ctx100k'
  plan=json.loads((previous/'plan.json').read_text());server=plan['server'];out=dest/'run'
  i=server.index('--tp-host-trace');del server[i:i+2]
  for flag,value in [('--out',out),('--preservation',dest/'config'),('--name','b70-agent2-'+label),('--image',IMAGES[card]),('--served-alias',alias),('--scales',dest/'fresh-scales.json'),('--tensor-parallel-size','1')]:server[server.index(flag)+1]=str(value)
  server+=['--card',str(card),'--port',str(18151+card)]
  files=['vllm/int4/pi_agent2_controls/gate.py','vllm/int4/replay_pi_history.py','vllm/int4/audit_replay_quality.py','vllm/fp8/kv_campaign_agent_probe.py','vllm/fp8/kv_campaign_probe.py']
  for rel in files:
   target=dest/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,target)
  gate=dest/'source/vllm/int4/pi_agent2_controls/gate.py'
  jobs=[dict(name='01-geometry',command=list(map(str,['python3',gate,out,'--startup'])),timeout=30),dict(name='02-agent2-warm-target2',command=list(map(str,['python3',dest/'source/vllm/int4/replay_pi_history.py','run','--manifest',dest/'pi-manifest.json','--model','hotschmoe-dd','--base',f'http://127.0.0.1:{18151+card}','--out',out/'replay','--timeout','240','--repeats','2'])),timeout=750),dict(name='03-strict-quality',command=list(map(str,['python3',gate,out])),timeout=60)]
  plan.update(out=str(out),jobs=jobs,notes='PREPARED_NOT_RUN. Matched TP1 phase vs MRV1, freshbe02 FP8 MTP3 FULL/prefixON100K. Same exact agent2 warm+target2 inputs; frozen prior outputs are never appended to history. Block1600 and positive target hits required. Strict quality retains malformed/gibberish failures. Cards differ; cross over if outcome differs. No historical Pi failure relabel or fullfix claim.')
  (dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  external=[REPO/'vllm/fp8/kv_campaign_server.py',REPO/'vllm/cache800k/run_arm.py',REPO/'vllm/int4/diagnostics/xpu_health_strict.sh']+list((REPO/'vllm/fp8/kv_hooks').glob('*.py'))+[Path(c['payload']) for c in manifest['cases']]
  (dest/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()},external={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in external}),indent=2)+'\n')
 return root
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();print(prepare(a.destination.resolve()))
