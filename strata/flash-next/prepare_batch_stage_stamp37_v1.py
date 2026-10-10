#!/usr/bin/env python3
"""CPU source reconstruction only; no compilation, GPU or model access."""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source')
BASE=HERE/'current-ple-prompt35-engine-build-plan-v1.json'
BASE_SHA='82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7'
V='sycl/src/core/verify.cpp'
PATCH=HERE/'patches/0037-default-off-batch-observer-earlier-stage-stamp.patch'
ANCHOR='        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, inj2_, (int64_t) T * HC, cs);\n        return true;'
HOOK='        if(batch_observe_ && batch_snapshot_)batch_snapshot_->stamp(*cs);\n'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def reconstruct():
 if sha(BASE)!=BASE_SHA:raise ValueError('Frozen source35 plan changed')
 base=json.loads(BASE.read_text())
 for path,digest in base['expected_patched_source_sha256'].items():
  if sha(SDK/path)!=digest:raise ValueError('Exact source35 closure changed: '+path)
 old=(SDK/V).read_text()
 if old.count(ANCHOR)!=1:raise ValueError('Exact earlier-stage handoff anchor changed')
 new=old.replace(ANCHOR,ANCHOR.replace('        return true;',HOOK+'        return true;'))
 return old,new
def patch_bytes():
 old,new=reconstruct()
 return ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+V,tofile='b/'+V)).encode()
def build_plan():
 old,new=reconstruct();p=json.loads(BASE.read_text())
 if PATCH.read_bytes()!=patch_bytes():raise ValueError('Patch differs from exact reconstruction')
 p['status']='SOURCE/CPU only; new native ABI build and paired observer qualification unexecuted'
 p['derived_from_plan']={'path':str(BASE.relative_to(ROOT)),'sha256':BASE_SHA}
 p['generation']='source35 plus observer earlier-stage stamp37; excludes host tracer36'
 p['patches'].append({'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)})
 p['expected_patched_source_sha256'][V]=hashlib.sha256(new.encode()).hexdigest()
 p['new_source_contracts']['0037']={'scope':'Default-OFF existing batch observer only; stamp exact non-head stage roster after existing handoff copies and before early return. No math/cache/graph selection/quantization change.', 'patch_sha256':sha(PATCH),'earlier_stage_batch_observer_stamp37':True,'fresh_all_eight_ABI_and_new_C113_derived_admission_required':True,'old_source35_runtime_proof_transfer':False}
 p['required_before_model_run'].append('Require new observer stamp37 code binding and fresh SDK source/eightABI/upload390/full4/pages/health/owned teardown; prior source35 activated batch proofs cannot qualify this generation. STRATA_VERIFY_EAGER absent. Matched OFF/ON fresh paired native2 functional and full observer roster/head evidence required; native4/6 remain pending until actually observed.')
 p['actual_paired_batch_observer37_qualified']=False
 return p
def combined_plan():
 """Independent optional composition; immutable trace36 proposal not modified."""
 import prepare_host_critical_path36_v1 as h
 trace=HERE/'host-critical-path36-engine-build-plan-v1.json'
 if sha(trace)!='c2b80b7e7b0bccdfe98e5771e88a604888e86443d5e2f6c3468a8ed0d7b46709':raise ValueError('Frozen trace36 plan changed')
 plan=json.loads(trace.read_text());old,new=h.reconstruct()
 if (HERE/'patches/0036-default-off-bounded-host-critical-path-trace.patch').read_bytes()!=h.patch_bytes():raise ValueError('Frozen trace36 patch changed')
 for name,data in new.items():
  if hashlib.sha256(data.encode()).hexdigest()!=plan['expected_patched_source_sha256'][name]:raise ValueError('Trace36 reconstruction changed '+name)
 if new[V].count(ANCHOR)!=1:raise ValueError('Combined source anchor changed')
 result=new[V].replace(ANCHOR,ANCHOR.replace('        return true;',HOOK+'        return true;'))
 plan['expected_patched_source_sha256'][V]=hashlib.sha256(result.encode()).hexdigest()
 plan['patches'].append({'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)})
 plan['derived_from_plan']={'path':str(trace.relative_to(ROOT)),'sha256':sha(trace)}
 plan['generation']='source35 plus independently composed trace36 and earlier-stage observer stamp37'
 plan['new_source_contracts']['0037']=build_plan()['new_source_contracts']['0037']
 plan['actual_paired_batch_observer37_qualified']=False
 plan['status']='SOURCE/CPU composition only; no native ABI build or activated trace/observer qualification'
 plan['required_before_model_run'].append('All stamp37 new-generation ABI/admission gates required independently of trace36. Trace36 stays OFF during paired observer runs because trace36 ON supports onecard serial only; paired observer37 remains diagnostic until fresh actual complete proof.')
 return plan
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--combined36',action='store_true');args=a.parse_args()
 if args.output.exists():raise ValueError('New output required')
 args.output.write_text(json.dumps(combined_plan() if args.combined36 else build_plan(),indent=2)+'\n')
 print(json.dumps({'actual_build':False,'actual_GPU_run':False,'plan_sha256':sha(args.output)}))
