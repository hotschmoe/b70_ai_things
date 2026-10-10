#!/usr/bin/env python3
"""CPU metadata/source preparation only. Print fresh leaf commands; never execute them."""
import argparse,hashlib,json,re
import hc_projection_arithmetic35_fixture_v1 as fixture
import c1_serve_controller_combined_v13 as c1
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ENGINE_PLAN=ROOT/'strata/flash-next/current-ple-prompt35-engine-build-plan-v1.json'
ENGINE_PLAN_SHA='82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7'
SOURCE_RELS=['sycl/include/strata/kernels/hc_native_projection.hpp','sycl/src/kernels/hc_native_projection.cpp','sycl/src/kernels/hc_native_composition.cpp','sycl/include/strata/core/native_hc_dispatch.hpp','sycl/include/strata/sycl_queue.hpp','sycl/CMakeLists.txt']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(1<<20),b''):h.update(data)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,msg):
 if not ok:raise ValueError(msg)
def prepare(engine):
 engine=Path(engine).resolve();c1.combined_generation_gate(engine);require(sha(ENGINE_PLAN)==ENGINE_PLAN_SHA,'Frozen source35 plan changed');fixture.source_binding();lock=ROOT/'strata/flash-next/model-lock.json';sdk=read(ENGINE_PLAN);r=read(engine/'receipt.json')
 require(r.get('build_rc')==0 and r.get('external_source_unchanged') is True and r.get('plan_snapshot_unchanged') is True,'Current SDK did not complete')
 require(r['plan_sha256']==ENGINE_PLAN_SHA and sha(engine/'plan.snapshot.json')==ENGINE_PLAN_SHA and r['patches']==sdk['patches'],'Current exact source35 SDK recipe differs')
 require(r['patched_source_sha256']==sdk['expected_patched_source_sha256'],'Full current SDK source ledger differs')
 require(Path(r['source_copy']).resolve()==engine/'source' and Path(r['build']).resolve()==engine/'build','SDK source/build association differs')
 for rel,digest in sdk['expected_patched_source_sha256'].items():require(sha(engine/'source'/rel)==digest,'Current source changed: '+rel)
 for rel,digest in r['binary_sha256'].items():require(sha(rel)==digest,'Actual compiled SDK target changed: '+rel)
 require(len(r['binary_sha256'])==8,'Actual eight rebuilt SDK targets required')
 files=[]
 for rel in SOURCE_RELS+['../build/libstrata_kernels.a','../build/libstrata_core.a','../build/build.ninja','../build/CMakeCache.txt']:
  path=(engine/'source'/rel).resolve();require(path.is_file(),'Current leaf dependency missing: '+str(path));files.append({'path':str(path),'sha256':sha(path)})
 ninja=(engine/'build/build.ninja').read_text();require('hc_native_projection.cpp' in ninja and '-fsycl-default-sub-group-size=32' in ninja and '-cl-fp32-correctly-rounded-divide-sqrt' in ninja,'Current consumed SYCL compile/link recipe differs')
 source=ROOT/'strata/flash-next/hc_projection_arithmetic35_gpu_v1.cpp'
 compile_cmd=['icpx','-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-I/sdk/source/sycl/include','-I/sdk/source/include','/leaf/hc_projection_arithmetic35_gpu_v1.cpp','/sdk/build/libstrata_kernels.a','/sdk/build/libstrata_core.a','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/out/hc_projection_arithmetic35_gpu_v1']
 return {'schema':1,'status':'CPU_SOURCE_PREPARED; no compile or GPU run','engine_root':str(engine),'engine_receipt_sha256':sha(engine/'receipt.json'),'engine_plan_sha256':ENGINE_PLAN_SHA,'image':r['image'],'synthetic_geometry':{'K':[32,64,96,320,10240],'T':[1,2],'cases':13,'output_words':38},'model_lock_sha256':sha(lock),'leaf_source':str(source),'leaf_source_sha256':sha(source),'producer_sha256':sha(Path(__file__)),'collector_source_sha256':sha(ROOT/'strata/flash-next/hc_projection_arithmetic35_fixture_v1.py'),'fixture_source_plan_sha256':sha(ROOT/'strata/flash-next/hc-projection-arithmetic35-fixture-source-plan-v1.json'),'current_SDK_dependencies':files,'actual_eight_SDK_ELFs':r['binary_sha256'],'compile_argv_inside_pinned_image':compile_cmd,'run_argv_inside_pinned_image':['/out/hc_projection_arithmetic35_gpu_v1','--inputs','/inputs','--output','/results/raw-new'],'environment':{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_UR_TRACE':'2'},'required_absent_environment':['STRATA_VERIFY_EAGER'],'cases':[case['id'] for case in fixture.fixtures()],'expected_raw_cases':13,'expected_raw_words':38,'negative_controls':2,'scope':'synthetic public HC projection exact host-schedule comparison only; no composed HC/deviceintrinsic/model qualification','model_math_qualified':False,'real_model_inputs_observed':False,'device_intrinsics_qualified':False,'full_HC_qualified':False,'backend_patch_required':False,'root_owned_execution_requirements':['bin/gpu-run --card N plus matching workload device pin','strict per-card and compiled P2P0 pre/post health under parent pair lease','fresh leaf compile in pinned current SDK baseimage','owned compile/run container clean terminal and removed','fresh all4 publisher hashes after terminal and posthealth with both known source pages bracketing','actual USM logicalfree audit and no kernel faults','recheck all current SDK/leaf/source/archive/ELF hashes before compile and after run'],'archive_association_limit':'Current static archive bytes are freshly pinned here, but original SDK receipt binds eight target ELFs rather than archives; no retroactive original-build archive hash claim'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Output must be new');j=prepare(a.engine_root);a.output.write_text(json.dumps(j,indent=2)+'\n',encoding='ascii');print(json.dumps({'plan':str(a.output),'GPU_executed':False,'compile_executed':False}))
if __name__=='__main__':main()
