"""Source58 semantic bridge object plus original loaded reset ELF; root builds only."""
import argparse,hashlib,json,shlex
from pathlib import Path
import source58_model_sdk_binding_v1 as sdk_binding
HERE=Path(__file__).resolve().parent;PLAN=HERE/'original-mtp53-joint-checkpoint-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(sdk,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for name,want in plan['files'].items():
  if sha(HERE.parents[1]/name)!=want:raise ValueError('Serving/reset source changed '+name)
 sdk=Path(sdk);out=Path(out);sdk_binding.configure(sdk,receipt);proof=sdk_binding.binding()
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical compile input root required')
 ninja=(sdk/'build/build.ninja').read_text();needle='build CMakeFiles/strata_kernels.dir/src/kernels/hc_native_composition.cpp.o:'
 if ninja.count(needle)!=1:raise ValueError('Unique actual58 production compile rule required')
 block=ninja.split(needle,1)[1].split('\n\n',1)[0];fields={l.strip().split(' = ',1)[0]:l.strip().split(' = ',1)[1]for l in block.splitlines()if ' = 'in l};flags=shlex.split(fields['FLAGS'])
 if '-fp-model=precise'not in flags or '-fsycl'not in flags or '-fsycl-default-sub-group-size=32'not in flags:raise ValueError('Precise actual58 SYCL recipe required')
 defines=[('-DSTRATA_VERSION="0.1.41"'if v.startswith('-DSTRATA_VERSION=')else v)for v in fields['DEFINES'].split()]
 if '0.1.41'not in fields['DEFINES']:raise ValueError('Actual58 original version definition changed')
 out.mkdir();overlay=out/'overlay';overlay.mkdir();(out/'output').mkdir()
 # Exact inherited original53 headers, superseded ONLY by explicit new entries.
 step=json.loads((HERE/'original-publisher-mtp53-step-proposal-v2.json').read_bytes());pair=json.loads((HERE/'original-publisher-mtp53-two-stage-proposal-v6.json').read_bytes())
 sources=dict(step['overlay_sources']);sources.update(pair['new_overlay_sources']);sources.update(plan['overlay_sources'])
 expected=dict(step['own_source_sha256']);expected.update(pair['own_source_sha256']);expected.update(plan['files'])
 for source,target in sources.items():
  p=HERE.parents[1]/source
  if sha(p)!=expected[source]:raise ValueError('Exact inherited original header changed '+source)
  t=overlay/target;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(p.read_bytes())
 compiler='/opt/intel/oneapi/compiler/2026.1/bin/icpx';common=[compiler,*flags,*defines,'-I/overlay/sycl/include','-I/sdk/source/sycl/include','-I/sdk/source/include','-I/sdk/source/third_party/ggml','-I/sdk/source/sycl/src/kernels/cuda']
 bridge=[*common,'-c','/overlay/sycl/src/kernels/original_mtp53_joint_semantic.cpp','-o','/out/original-mtp53-joint-semantic.o']
 archives=['libstrata_engine.a','libstrata_prefill.a','libstrata_spec.a','libstrata_kernels.a','libstrata_core.a','libstrata_kernels_cpu.a','ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']
 cpu=[*common,'/overlay/sycl/src/kernels/original_mtp53_joint_cpu.cpp','-Wl,--start-group',*['/sdk/build/'+a for a in archives],'-Wl,--end-group','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-qmkl=sequential','-o','/out/original-mtp53-joint-cpu']
 leaf=[*common,'/overlay/sycl/src/kernels/original_mtp53_checkpoint_continuation.cpp','-Wl,--start-group',*['/sdk/build/'+a for a in archives],'-Wl,--end-group','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-qmkl=sequential','-o','/out/original-mtp53-checkpoint-continuation']
 record={'schema':1,'source_plan_sha256':sha(PLAN),'actual58':proof,'compiler_image':'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','compile_commands':[{'name':'original-mtp53-joint-semantic.o','argv':bridge},{'name':'original-mtp53-joint-cpu','argv':cpu},{'name':'original-mtp53-checkpoint-continuation','argv':leaf}],'mounts':{'SDK_RO':str(sdk),'overlay_RO':str(overlay),'output_RW':str(out/'output')},'overlay_sha256':{str(p.relative_to(overlay)):sha(p)for p in overlay.rglob('*')if p.is_file()},'original_archive_sha256':{a:sha(sdk/'build'/a)for a in archives},'production_ninja_sha256':sha(sdk/'build/build.ninja'),'NoDri_required':True,'No_network_required':True,'compile_executed':False,'GPU_executed':False,'MAIN_executed':False,'scope':'Source58 compound paired/original cache native semantic object and bounded CPU ownership executable; no model/GPU/API promotion'}
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
