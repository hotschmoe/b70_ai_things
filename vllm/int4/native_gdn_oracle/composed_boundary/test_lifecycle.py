"""CPU-only lifecycle outcomes; every GPU helper is mocked."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import lifecycle as runner

p=Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/composed-boundary-plan-card0/plan.json');original=json.loads(p.read_text())
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);lock=root/'gpu.lock.0';lock.touch();fd=os.open(lock,os.O_WRONLY);os.dup2(fd,8)
 if fd!=8:os.close(fd)
 probe=root/'probe';probe.write_text('CPU fixture');prior=root/'PAIR_PASS';prior.touch();pub=root/'prior.json';pub.write_text('{"passed":true}')
 for mode in ['pass','numeric_fail','crash','post_fail']:
  plan=copy.deepcopy(original);plan.update(output=str(root/mode),pair_preflight_pass=str(prior),prerequisite_outcomes=[str(pub)],frozen_files={},health_probe=str(probe),health_sha256=runner.sha(probe));stages=[]
  rows=[]
  for case in plan['cases']:
   for sync in ['unsynced','synced']:
    checks={n:True for n in plan['check_names']}
    if mode=='numeric_fail':checks['nomove_output_exact']=False
    rows.append(dict(case=case,mode=sync,checks=checks,metadata=plan['metadata_receipt'],errors={'first':{'max_abs':0.,'relative_l2':0.}},synced_unsynced_exact=True,fingerprint='same',passed=all(checks.values())))
  result=dict(image=plan['image'],native_sha256=plan['native_sha256'],source_sha256=plan['oracle_sha256'],worker_sha256=plan['worker_sha256'],model_sha256=plan['model_sha256'],cases=rows,passed=mode!='numeric_fail')
  def run(cmd,path,timeout,owned):
   assert owned;stages.append(path.name);path.write_text('CPU fake')
   if path.name=='oracle.log':
    if mode!='crash':(Path(plan['output'])/'results/result.json').write_text(json.dumps(result))
    return 2 if mode=='crash' else int(mode=='numeric_fail')
   assert cmd[-2:]==['--card','0'];return int(mode=='post_fail' and path.name=='post-health.log')
  with patch.dict(os.environ,B70_GPU_LOCK=str(root/'gpu.lock')),patch.object(runner.health,'run',side_effect=run),patch.object(runner.health,'capture',return_value=NS(returncode=0,stdout=json.dumps([{'Id':plan['image']}]))):rc=runner.execute(plan)
  assert rc==(0 if mode=='pass' else 1);assert stages==['pre-health.log','oracle.log','post-health.log']
  if mode!='crash':assert len(json.loads((Path(plan['output'])/'oracle-result.json').read_text())['cases'])==10
 os.close(8)
print('PASS complete numeric failure retention, crash/posthealth handling, selected-card-only and no success masking')
