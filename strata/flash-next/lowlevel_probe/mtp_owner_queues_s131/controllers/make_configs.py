#!/usr/bin/env python3
"""Bind a completed S130 mode0 read-only control; create only S131 owner-ON."""
import argparse,copy,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--control-dir',type=Path,required=True)
p.add_argument('--exe',type=Path,required=True)
p.add_argument('--sha256',required=True)
a=p.parse_args()
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
control=a.control_dir.resolve();exe=a.exe.resolve()
assert control==D.parent/'q5-runtime-ones-s130-serving-prep/mode0','Explicit completed S130 mode0 only'
assert not(D/'serving-inputs.json').exists()and not(D/'mode1').exists()and not(D/'mode0').exists()
assert len(a.sha256)==64 and sha(exe)==a.sha256
base=read(control/'config.json');prior=read(control.parent/'serving-inputs.json')
assert base['exe']==str(exe)==prior['arms']['0']['exe']
assert prior['arms']['0']['sha256']==a.sha256
assert base['env']['STRATA_BATCH_MTP_OWNER_QUEUES']=='0'
assert base['env']['STRATA_Q5_HEAD_RUNTIME_ONES']=='0'and base['env']['STRATA_MTP_PARALLEL_ARGMAX']=='0'
stop=read(control/'normal-stop.json');health=read(control/'post-health.json');gone=read(control/'container-removal.json')
assert stop['status']=='PASS_NORMAL_TEARDOWN'and stop['lease_wrapper_exit']==0 and stop['container_removed']
assert set(health)=={'card','collective'}and all(v['exit']==0 for v in health.values())
assert gone['docker_ps_exit']==0 and gone['full_cid_absent']and gone['name_absent']
assert read(control/'aggregate-proof.json')['present_once']and read(control/'functional/verdict.json')['status']=='PASS'
critical=['config.json','container-ready.json','health-ready.json','native-process-identity.json','normal-stop.json','post-health.json','container-removal.json','aggregate-proof.json','functional/verdict.json','engine-final.log']
for label in ['warm','repeat1','repeat2','repeat3']:
 pair=read(control/f'coding/{label}/pair.json')
 assert pair['status']=='PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED'and pair['paired']>0
 assert pair['q5_runtime_ones_host_selection_marker']is None
 assert pair['admission_gate']['status']=='PASS_NATIVE_PROMPT_START_ORDER_AND_CLIENT_OVERLAP'
 for name in ['pair.json','window-start.json','coding-0.json','coding-1.json']:
  critical.append(f'coding/{label}/{name}')
pins={n:{'bytes':(control/n).stat().st_size,'sha256':sha(control/n)}for n in critical}
out=D/'mode1';out.mkdir(exist_ok=False)
c=copy.deepcopy(base);c['log']=str(out/'engine.log');c['env']['STRATA_BATCH_MTP_OWNER_QUEUES']='1'
(out/'config.json').write_text(json.dumps(c,indent=2)+'\n',encoding='ascii')
inputs=copy.deepcopy(prior)
inputs.update(control_dir=str(control),control_read_only=True,control_artifact_pins=pins,
 arms={'0':{'exe':str(exe),'sha256':a.sha256},'1':{'exe':str(exe),'sha256':a.sha256}},
 containers={'0':prior['containers']['0'],'1':'strata-s131-owner-mode1'},
 arm_difference='STRATA_BATCH_MTP_OWNER_QUEUES 0/1 and arm-local log/container/service names only',
 candidate_scope='Owner-queue head-free submission with host joins; Q5 runtime ones and MTP parallel argmax OFF; no overlap/speed claim',
 capture_scope='Verifier capture layouts/work equal; MTP owner head-free queue/graph identities are not required equal',
 owner_queues_marker_scope='Once first eligible HOST selection by warm, startup capture permitted; no observed hardware overlap',
 control_provenance='Completed S130 mode0 same binary/configuration except owner queues; no control regeneration or launch')
inputs.pop('q5_runtime_ones_marker_scope',None)
(D/'serving-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='ascii')
print(json.dumps(inputs))
