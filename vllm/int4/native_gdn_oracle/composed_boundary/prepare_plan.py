#!/usr/bin/env python3
"""Freeze CPU-reviewed composition sources; no GPU execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--card',type=int,choices=[0,1],default=0);a=p.parse_args();here=Path(__file__).resolve().parent;repo=here.parents[3];a.root.mkdir(parents=True,exist_ok=False);inputs=a.root/'inputs';inputs.mkdir()
    for name in ['oracle.py','test_cpu.py']:shutil.copyfile(here/name,inputs/name)
    base=Path('/mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/native-gdn-tp1-plan-card0/inputs/oracle.py');shutil.copyfile(base,inputs/'base_oracle.py')
    spec=importlib.util.spec_from_file_location('oracle',inputs/'oracle.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o);assert sha(inputs/'base_oracle.py')==o.BASE_ORACLE_SHA
    out=a.root.parent/('composed-boundary-adapter-card'+str(a.card));health=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/health-repair/xpu-health');helper=repo/'sglang/refresh/20260910_main/preflight.py'
    pub=Path('/mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/native-gdn-publication-card0/OUTCOME.json');assert json.loads(pub.read_text())['diagnostic_completed'] is True
    frozen=[here/'lifecycle.py',helper,health,pub,inputs/'oracle.py',inputs/'base_oracle.py']
    cmd=['docker','run','--name','b70-composed-gdn-adapter-card'+str(a.card),'--network','none','--device','/dev/dri','--cpus','4','--memory','8g','--memory-swap','8g','--ulimit','core=0','-v','/dev/dri/by-path:/dev/dri/by-path:ro','-v',str(inputs)+':/candidate:ro','-v',str(out/'results')+':/out','-v',str(out/('card'+str(a.card))/'cache')+':/cache']
    for env in ['ZE_AFFINITY_MASK='+str(a.card),'ONEAPI_DEVICE_SELECTOR=level_zero:gpu','CCL_TOPO_P2P_ACCESS=0','PYTORCH_ALLOC_CONF=expandable_segments:True','B70_XPU_GDN_PREFIX_CONV_COPY=1','TRITON_CACHE_DIR=/cache/triton','TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor','NEO_CACHE_DIR=/cache/neo']:cmd+=['-e',env]
    cmd+=['--entrypoint','python3',o.IMAGE,'/candidate/oracle.py','--run-xpu','--base-oracle','/candidate/base_oracle.py','--output','/out/result.json']
    plan=dict(card=a.card,image=o.IMAGE,output=str(out),pair_preflight_pass='/mnt/vm_8tb/b70/results/bang_isolation_20260910/conv-contract-adapter-preflight/PASS',prerequisite_outcomes=[str(pub)],frozen_files={str(f):sha(f) for f in frozen},health_probe=str(health),health_sha256=sha(health),native_sha256=o.NATIVE_SHA,worker_sha256=o.WORKER_SHA,model_sha256=o.MODEL_SHA,oracle_sha256=sha(inputs/'oracle.py'),cases=o.cases(),check_names=['finite','math','z_exact','conv_exact','padding_exact','inactive_ssm_exact','accepted_exact','nomove_output_exact','nomove_z_exact','nomove_conv_exact','nomove_ssm_exact'],metadata_receipt=dict(copy_functions=['get_xpu_gdn_conv_copy_spec','get_temporal_copy_spec'],conv_widths=[0,0],inner_sizes=[30720,786432],block_strides=[3276800,3276800]),oracle_command=cmd,status='PREPARED_NOT_RUN; independent source review pending; selectedlease parent owns',launch=['bin/gpu-run','--card',str(a.card),'python3',str(here/'lifecycle.py'),str(a.root/'plan.json'),'--run'])
    (a.root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(a.root/'plan.json')
if __name__=='__main__':main()
