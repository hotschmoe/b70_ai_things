"""Root-only recipe preparation; no compiler, GPU, GGUF or model reader execution."""
import argparse,hashlib,json,re,shlex
from pathlib import Path
import source52_model_sdk_binding_v1 as sdk
HERE=Path(__file__).resolve().parent;PLAN=HERE/'original-publisher-mtp53-step-proposal-v1.json'
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def prepare(root,receipt,out):
 sdk.configure(root,receipt);proof=sdk.binding();plan=json.loads(PLAN.read_bytes());root=Path(root);out=Path(out)
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical compile input directory required')
 for path,want in plan['own_source_sha256'].items():
  p=HERE.parents[1]/path
  if hashlib.sha256(p.read_bytes()).hexdigest()!=want:raise ValueError('Exact source53 source changed '+path)
 ninja=(root/'build/build.ninja').read_text();needle='build CMakeFiles/strata_kernels.dir/src/kernels/hc_native_composition.cpp.o:'
 if ninja.count(needle)!=1:raise ValueError('Unique actual SDK52 object flags required')
 block=ninja.split(needle,1)[1].split('\n\n',1)[0];fields={l.strip().split(' = ',1)[0]:l.strip().split(' = ',1)[1]for l in block.splitlines()if ' = 'in l};flags=shlex.split(fields['FLAGS'])
 expected='-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra'
 if fields['FLAGS']!=expected:raise ValueError('Exact actual SDK52 precise object flags required')
 if '/opt/intel/oneapi/compiler/2026.1/bin/icpx $DEFINES $INCLUDES $FLAGS'not in(root/'build/CMakeFiles/rules.ninja').read_text():raise ValueError('Exact actual compiler command required')
 link=ninja.split('build strata: ',1)[1].split('\n\n',1)[0];lf=next(l.split(' = ',1)[1]for l in link.splitlines()if l.strip().startswith('LINK_FLAGS = '));linkflags=shlex.split(lf)
 if lf!='-fsycl -Xsycl-target-backend=spir64 -cl-fp32-correctly-rounded-divide-sqrt -fsycl-device-code-split=per_kernel -qmkl=sequential':raise ValueError('Exact production device-link flags required')
 defines=[]
 for value in fields['DEFINES'].split():
  if value.startswith('-DSTRATA_VERSION='):defines.append('-DSTRATA_VERSION="'+re.search(r'[0-9]+\.[0-9]+\.[0-9]+',value).group(0)+'"')
  else:defines.append(value)
 out.mkdir();overlay=out/'overlay';overlay.mkdir()
 for source,target in plan['overlay_sources'].items():
  targetpath=overlay/target;targetpath.parent.mkdir(parents=True,exist_ok=True);targetpath.write_bytes((HERE.parents[1]/source).read_bytes())
 archives=['libstrata_engine.a','libstrata_prefill.a','libstrata_spec.a','libstrata_kernels.a','libstrata_core.a','libstrata_kernels_cpu.a','ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']
 commands=[]
 for name,source in [('original-mtp53-step','sycl/src/kernels/original_mtp53_draft_step_loaded_fixture.cpp'),('original-mtp53-front','sycl/src/kernels/original_mtp53_eh_hc_loaded_fixture_v2.cpp'),('original-mtp53-f32','sycl/src/kernels/original_mtp53_primitive_fixture_v2.cpp')]:
  argv=['/opt/intel/oneapi/compiler/2026.1/bin/icpx',*flags,*defines,'-I/overlay/sycl/include','-I/sdk/source/sycl/include','-I/sdk/source/include','-I/sdk/source/third_party/ggml','/overlay/'+source,'-Wl,--start-group',*['/sdk/build/'+a for a in archives],'-Wl,--end-group','/usr/lib/x86_64-linux-gnu/libze_loader.so',*linkflags,'-o','/out/'+name];commands.append({'name':name,'argv':argv})
 record={'schema':1,'actual52':proof,'source53_proposal_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'object_fields':fields,'production_link_flags':lf,'compile_commands':commands,'mounts':{'SDK_RO':str(root),'overlay_RO':str(overlay),'output_RW':str(out/'output')},'compiler_image':plan['compiler_image'],'NoDri_required':True,'No_network_required':True,'compiler_or_GPU_executed':False,'current_original_MTP_identity_qualified':False}
 (out/'output').mkdir();write(out/'compile-recipe.json',record);return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
