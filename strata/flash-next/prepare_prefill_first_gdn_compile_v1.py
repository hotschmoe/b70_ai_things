"""Root-owned actual58 target-only first-row observer compile inputs, no numerical authority."""
import argparse,json,hashlib,shlex
from pathlib import Path
import source58_model_sdk_binding_v1 as sdk_binding
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'prefill-first-gdn-observer-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(a):
 plan=json.loads(PLAN.read_bytes())
 for n,w in plan['files'].items():
  if sha(ROOT/n)!=w:raise ValueError('First GDN source changed '+n)
 sdk=a.sdk;sdk_binding.configure(sdk,a.sdk_receipt_sha256);proof=sdk_binding.binding()
 for n,w in plan['actual58_source_sha256'].items():
  if sha(sdk/'source'/n)!=w:raise ValueError('Actual58 source changed '+n)
 out=a.output
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical first GDN compile root required')
 out.mkdir();overlay=out/'overlay';overlay.mkdir();(out/'output').mkdir()
 original=(sdk/'source/sycl/src/prefill/prefill.cpp').read_text()
 body='#include "strata/core/prefill_first_gdn_observer.hpp"\n'+original
 needle='                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);'
 hook=plan['exact_hook']
 if body.count(needle)!=1:raise ValueError('Unique actual first GDN seam required')
 body=body.replace(needle,hook+needle)
 if hashlib.sha256(body.encode('ascii')).hexdigest()!=plan['constructed_prefill_sha256']:raise ValueError('Constructed Prefill source changed')
 destinations={'sycl/src/prefill/prefill.cpp':body.encode('ascii'),'sycl/include/strata/core/prefill_first_gdn_observer.hpp':(HERE/'original-mtp53/prefill_first_gdn_observer_v1.hpp').read_bytes()}
 # Original relative quoted includes resolve identically in the new source overlay.
 for n in ('src/prefill/mmq_resident_sort.hpp','src/prefill/wmma_gemm.h'):destinations[n]=(sdk/'source'/n).read_bytes()
 for n,v in destinations.items():p=overlay/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
 ninja=(sdk/'build/build.ninja').read_text();needle='build CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o:'
 if ninja.count(needle)!=1:raise ValueError('Unique production Prefill rule required')
 block=ninja.split(needle,1)[1].split('\n\n',1)[0];fields={v.strip().split(' = ',1)[0]:v.strip().split(' = ',1)[1]for v in block.splitlines()if ' = 'in v};flags=shlex.split(fields['FLAGS']);defs=shlex.split(fields['DEFINES'])
 if '-fsycl'not in flags or '-fp-model=precise'not in flags or '-fsycl-default-sub-group-size=32'not in flags:raise ValueError('Actual precise SYCL flags required')
 compiler='/opt/intel/oneapi/compiler/2026.1/bin/icpx';common=[compiler,*flags,*defs,'-I/overlay/sycl/include','-I/sdk/source/sycl/include','-I/sdk/source/include','-I/sdk/source/third_party/ggml','-I/sdk/source/sycl/src/kernels/cuda']
 main=[*common,'-MMD','-MF','/out/main.d','-c','/sdk/source/sycl/src/program/generate.cpp','-o','/out/first-gdn-main.o']
 pf=[*common,'-MMD','-MF','/out/prefill.d','-c','/overlay/sycl/src/prefill/prefill.cpp','-o','/out/first-gdn-prefill.o']
 archives=['libstrata_engine.a','libstrata_prefill.a','libstrata_spec.a','libstrata_kernels.a','libstrata_core.a','libstrata_kernels_cpu.a','ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']
 link=[compiler,*flags,'/out/first-gdn-main.o','/out/first-gdn-prefill.o','-Wl,--start-group',*['/out/libstrata_prefill.without_prefill.a'if n=='libstrata_prefill.a'else'/sdk/build/'+n for n in archives],'-Wl,--end-group','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-qmkl=sequential','-o','/out/first-gdn-target']
 record={'schema':1,'source_plan_sha256':sha(PLAN),'actual58':proof,'compiler_image':'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','compile_commands':[{'name':n,'argv':v}for n,v in [('first-gdn-main.o',main),('first-gdn-prefill.o',pf),('first-gdn-target',link)]],'mounts':{'SDK_RO':str(sdk),'overlay_RO':str(overlay),'output_RW':str(out/'output')},'overlay_sha256':{n:sha(overlay/n)for n in destinations},'original_archive_sha256':{n:sha(sdk/'build'/n)for n in archives},'archive_filters':[{'original':'/sdk/build/libstrata_prefill.a','copy':'/out/libstrata_prefill.without_prefill.a','remove_members':['prefill.cpp.o'],'original_never_modified':True}],'NoDri_required':True,'No_network_required':True,'GPU_executed':False,'model_payload_read':False,'source_only':True}
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);print(json.dumps(prepare(p.parse_args()),ensure_ascii=True))
