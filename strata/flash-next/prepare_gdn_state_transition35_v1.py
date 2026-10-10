#!/usr/bin/env python3
"""CPU metadata/source preparation only. Print fresh leaf commands; never execute them."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ENGINE_PLAN=ROOT/'strata/flash-next/current-ple-prompt35-engine-build-plan-v1.json'
ENGINE_PLAN_SHA='82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7'
FOUNDATION=ROOT/'strata/flash-next/original-gguf-reference-foundation-plan-v1.json'
FOUNDATION_SHA='7d5217d844b411b427e4c6c6df2b4837b66970937c044389b40270346804bd2b'
SOURCE_RELS=['include/strata/kernels/gdn.hpp','include/strata/kernels/fused_gdn.hpp','include/strata/kernels/verify_kernels.hpp','sycl/include/strata/sycl_queue.hpp','sycl/src/kernels/cuda/gdn.dp.cpp','sycl/src/kernels/cuda/fused_gdn.dp.cpp','sycl/src/kernels/cuda/verify_kernels.dp.cpp','sycl/src/kernels/gdn_parity.cpp','sycl/src/core/verify.cpp','sycl/CMakeLists.txt']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(1<<20),b''):h.update(data)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,msg):
 if not ok:raise ValueError(msg)
def geometry(inventory,foundation):
 require(inventory['inventory_complete'] is True and not inventory['errors'],'Locked header inventory incomplete')
 m=inventory['files'][0]['metadata'];wanted={'qwen4exp.ssm.state_size':128,'qwen4exp.ssm.group_count':16,'qwen4exp.ssm.time_step_rank':48,'qwen4exp.ssm.conv_kernel':4,'qwen4exp.ssm.inner_size':6144}
 require(all(m[k]['value']==v for k,v in wanted.items()),'Locked actual GDN metadata differs')
 require(all(foundation['geometry'][k]==v for k,v in wanted.items() if k in foundation['geometry']),'Frozen foundation geometry differs')
 tensor=[t for f in inventory['files'] for t in f['tensors'] if t['name']=='blk.0.ssm_conv1d.weight']
 require(len(tensor)==1 and tensor[0]['shape_ggml_order']==[4,10240] and tensor[0]['type']=='F32','Actual locked conv geometry differs')
 return {'S':128,'h_k':16,'h_v':48,'conv_channels':10240,'conv_kernel':4,'state_dtype':'F32','state_bytes':3145728,'history_bytes':122880,'derived_only_from_locked_header_metadata':True}
def prepare(engine):
 engine=Path(engine).resolve();require(sha(ENGINE_PLAN)==ENGINE_PLAN_SHA and sha(FOUNDATION)==FOUNDATION_SHA,'Frozen SDK/foundation source plan changed')
 foundation=read(FOUNDATION);ip=Path(foundation['inventory']);require(sha(ip)==foundation['inventory_sha256'],'Locked header inventory changed')
 inventory=read(ip);lock=ROOT/'strata/flash-next/model-lock.json';require(sha(lock)==foundation['lock_sha256'],'Original model lock changed')
 require(inventory['source_revision']==read(lock)['revision'] and inventory['source_repo']==read(lock)['repo'],'Locked inventory publisher identity differs')
 geom=geometry(inventory,foundation);sdk=read(ENGINE_PLAN);r=read(engine/'receipt.json')
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
 ninja=(engine/'build/build.ninja').read_text();require('build gdn_parity:' in ninja and '-fsycl-default-sub-group-size=32' in ninja and '-cl-fp32-correctly-rounded-divide-sqrt' in ninja,'Current consumed SYCL compile/link recipe differs')
 source=ROOT/'strata/flash-next/gdn_state_transition35_v1.cpp'
 compile_cmd=['icpx','-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-I/sdk/source/sycl/include','-I/sdk/source/include','/leaf/gdn_state_transition35_v1.cpp','/sdk/build/libstrata_kernels.a','/sdk/build/libstrata_core.a','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/out/gdn_state_transition35_v1']
 return {'schema':1,'status':'CPU_SOURCE_PREPARED; no compile or GPU run','engine_root':str(engine),'engine_receipt_sha256':sha(engine/'receipt.json'),'engine_plan_sha256':ENGINE_PLAN_SHA,'image':r['image'],'geometry':geom,'model_lock_sha256':sha(lock),'header_inventory_path':str(ip),'header_inventory_sha256':sha(ip),'leaf_source':str(source),'leaf_source_sha256':sha(source),'producer_sha256':sha(Path(__file__)),'collector_source_sha256':sha(ROOT/'strata/flash-next/collect_gdn_state_transition35_v1.py'),'current_SDK_dependencies':files,'actual_eight_SDK_ELFs':r['binary_sha256'],'compile_argv_inside_pinned_image':compile_cmd,'run_argv_inside_pinned_image':['/out/gdn_state_transition35_v1','--output','/results/raw-new'],'environment':{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','STRATA_GDN_SPLIT':'0','SYCL_UR_TRACE':'2'},'required_absent_environment':['STRATA_VERIFY_EAGER'],'cases':['T2 forward preserves zeroF32 recurrent/history','accepted0/1/2 commit matches T1 sequence fullstate/history','accepted0/1/2 followed by T1 carry matches fullstate/history/output','modified persistentstate negative changes carry output'],'expected_raw_comparisons':27,'scope':'synthetic current SYCL component state-transition parity only; independent FP64 from own synthetic inputs diagnostic without tolerance','model_math_qualified':False,'real_row3_GDN_inputs_observed':False,'backend_patch_required':False,'root_owned_execution_requirements':['bin/gpu-run --card N plus matching workload device pin','strict per-card and compiled P2P0 pre/post health under parent pair lease','fresh leaf compile in pinned current SDK baseimage','owned compile/run container clean terminal and removed','fresh all4 publisher hashes after terminal and posthealth with both known source pages bracketing','actual USM logicalfree audit and no kernel faults','recheck all current SDK/leaf/source/archive/ELF hashes before compile and after run'],'archive_association_limit':'Current static archive bytes are freshly pinned here, but original SDK receipt binds eight target ELFs rather than archives; no retroactive original-build archive hash claim'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Output must be new');j=prepare(a.engine_root);a.output.write_text(json.dumps(j,indent=2)+'\n',encoding='ascii');print(json.dumps({'plan':str(a.output),'GPU_executed':False,'compile_executed':False}))
if __name__=='__main__':main()
