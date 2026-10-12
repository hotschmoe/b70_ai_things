"""Root-only NoDri object/operator recipe; no compiler, GPU or model execution."""
import argparse,hashlib,json,shlex
from pathlib import Path
import prepare_hc_t2_caller_v1 as candidate
import source52_model_sdk_binding_v1 as sdk_binding
HERE=Path(__file__).resolve().parent;PLAN=HERE/'hc-t2-operator-compile-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(sdk,receipt,out):
 plan=json.loads(PLAN.read_bytes());sdk=Path(sdk);out=Path(out)
 for p,h in plan['files'].items():
  if sha(HERE.parents[1]/p)!=h:raise ValueError('Exact HC operator source changed '+p)
 sdk_binding.configure(sdk,receipt);actual=sdk_binding.binding()
 ninja=(sdk/'build/build.ninja').read_text();needle='build CMakeFiles/strata_kernels.dir/src/kernels/hc_native_composition.cpp.o:'
 if ninja.count(needle)!=1:raise ValueError('Actual HC production object recipe required')
 block=ninja.split(needle,1)[1].split('\n\n',1)[0];fields={line.strip().split(' = ',1)[0]:line.strip().split(' = ',1)[1]for line in block.splitlines()if ' = 'in line};flags=shlex.split(fields['FLAGS']);defines=[]
 for value in fields['DEFINES'].split():
  if value.startswith('-DSTRATA_VERSION='):defines.append('-DSTRATA_VERSION="0.1.41"')
  else:defines.append(value)
 if '/opt/intel/oneapi/compiler/2026.1/bin/icpx $DEFINES $INCLUDES $FLAGS'not in(sdk/'build/CMakeFiles/rules.ninja').read_text() or '0.1.41'not in fields['DEFINES']:raise ValueError('Actual SDK52 compiler/version rule required')
 if '-fp-model=precise'not in flags or '-fsycl'not in flags or '-fsycl-default-sub-group-size=32'not in flags:raise ValueError('Actual precise SYCL/SG32 recipe required')
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical compile input directory required')
 out.mkdir();sources=out/'source';candidate.prepare(sources);(out/'output').mkdir();fixture=out/'fixture.cpp';fixture.write_bytes((HERE.parents[1]/'kernels/hc_t2_dead_bank_fixture_v1.cpp').read_bytes())
 includes=['-I/candidate/sycl/include','-I/sdk/source/sycl/include','-I/sdk/source/include','-I/sdk/source/third_party/ggml','-I/sdk/source/sycl/src/kernels/cuda'];compiler='/opt/intel/oneapi/compiler/2026.1/bin/icpx';commands=[]
 for source,name in [('/candidate/sycl/src/kernels/hc_native_projection.cpp','hc_projection.t2.o'),('/candidate/sycl/src/kernels/hc_native_composition.cpp','hc_composition.t2.o'),('/candidate/sycl/src/core/verify.cpp','verify.t2.o'),('/fixture.cpp','fixture.o')]:
  commands.append({'name':name,'argv':[compiler,*flags,*defines,*includes,'-c',source,'-o','/out/'+name]})
 link=[compiler,*flags,'/out/fixture.o','/out/hc_projection.t2.o','/out/hc_composition.t2.o','-Wl,--start-group','/out/libstrata_kernels.without_hc.a','/sdk/build/libstrata_core.a','-Wl,--end-group','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-qmkl=sequential','-o','/out/hc-t2-operator']
 archive={'original':'/sdk/build/libstrata_kernels.a','copy':'/out/libstrata_kernels.without_hc.a','remove_members':['hc_native_projection.cpp.o','hc_native_composition.cpp.o'],'before_counts':[1,1],'after_counts':[0,0],'copy_command':['cp','/sdk/build/libstrata_kernels.a','/out/libstrata_kernels.without_hc.a'],'remove_command':['ar','d','/out/libstrata_kernels.without_hc.a','hc_native_projection.cpp.o','hc_native_composition.cpp.o'],'original_never_modified':True,'verify_object_compiled_but_not_linked_in_leaf':True}
 record={'schema':1,'source_plan_sha256':sha(PLAN),'candidate_proposal_sha256':sha(HERE/'hc-t2-caller-source-proposal-v1.json'),'actual52':actual,'mounts':{'SDK_RO':str(sdk),'candidate_RO':str(sources),'fixture_RO':str(fixture),'output_RW':str(out/'output')},'compiler_image':'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','compiler_identity_from_actual_rules':True,'production_ninja_sha256':sha(sdk/'build/build.ninja'),'original_archive_sha256':{name:sha(sdk/'build'/name)for name in ('libstrata_kernels.a','libstrata_core.a')},'commands':commands,'archive_filter':archive,'link_argv':link,'source_object_sha256':{name:sha(sources/name)for name in candidate.TARGETS},'fixture_sha256':sha(fixture),'bounds':{'NoDri':True,'network_none':True,'cpus':2,'memory_gib':4},'leaf_runtime':{'card':0,'ZE_AFFINITY_MASK':'0','model_mounted_or_read':False,'require_normal_exit_owned_EOF_removal_health_kernel':True,'require_exact_marker':'HC_T2_OPERATOR_PASS pending=2 replays=4 fields=7 borrowed_tail_exact=1 model_qualified=0 speed_qualified=0'},'scope':'Whole-model G1 T2 caller is first prototype; leaf validates full HC arithmetic/bank. Paired/staged caller extension and actual model still required','objects_and_ELF_receipts_required_after_root_compile':True,'compile_or_GPU_executed_by_preparer':False}
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
