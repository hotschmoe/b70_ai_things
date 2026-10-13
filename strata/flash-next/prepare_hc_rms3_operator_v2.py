"""Source-only third explicit sqrt/divide oracle; immutable RMS3 V1 retained."""
import argparse,json
from pathlib import Path
import prepare_hc_rms3_operator_v1 as prior
ROOT=prior.ROOT;HERE=prior.HERE;SDK59=prior.SDK59;SDK61=prior.SDK61;COMPOSITION_SHA=prior.COMPOSITION_SHA
require=prior.require;sha=prior.sha
PLAN=HERE/'hc-rms3-explicit-sqrt-divide-source-plan-v2.json'
def source_binding():
 before=sha(PLAN)
 for n,w in json.loads(PLAN.read_text())['files'].items():require(sha(ROOT/n)==w,'Explicit sqrt/divide oracle source changed '+n)
 require(sha(PLAN)==before,'Explicit oracle plan changed');return before
def extract(text):
 base=prior.extract(text);function=base[base.index('static void baseline'):base.index('static void observed')];old='        const float rs=sycl::rsqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps);';new='        const float argument=reduce32(item.get_sub_group(),sum)/float(N)+a.eps;\n        const float rs=1.0f/sycl::sqrt(argument);'
 require(function.count(old)==1,'Exact original normalization expression required');variant=function.replace('static void baseline','static void explicit_sqrt_divide').replace('q.parallel_for<HcNorm>','q.parallel_for<HcNormSqrtDivide>').replace(old,new)
 require(variant.replace('static void explicit_sqrt_divide','static void baseline').replace('q.parallel_for<HcNormSqrtDivide>','q.parallel_for<HcNorm>').replace(new,old)==function,'Exact deliberate expression-only undo failed')
 return base+'\nclass HcNormSqrtDivide;\n'+variant

def source_prepare(out):
 binding=source_binding();require(not out.exists(),'New V2 oracle source root required');texts=[]
 for sdk in(SDK59,SDK61):
  path=sdk/'source/sycl/src/kernels/hc_native_composition.cpp';require(sha(path)==COMPOSITION_SHA,'Actual59/61 norm source changed');texts.append(path.read_text(encoding='ascii'))
 require(texts[0]==texts[1],'Actual59/61 norm sources differ');out.mkdir();(out/'hc_rms3_generated_v2.inc').write_text(extract(texts[0]),encoding='ascii');(out/'fixture.cpp').write_bytes((ROOT/'kernels/hc_rms3_operator_fixture_v2.cpp').read_bytes());(out/'hc_native_composition.hpp').write_bytes((SDK61/'source/sycl/include/strata/kernels/hc_native_composition.hpp').read_bytes());require(source_binding()==binding,'Closing V2 source changed')
 argv=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-I/inputs','/inputs/fixture.cpp','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/out/hc-rms3'];r={'schema':2,'source_plan_sha256':binding,'scope':'Isolated explicit sqrt-F32 then reciprocal-F32 candidate; no production MAIN arithmetic change','compiler_argv':argv,'original59_61_composition_sha256':COMPOSITION_SHA,'files':{p.name:sha(p)for p in out.iterdir()if p.is_file()},'observer_default_OFF':True,'explicit_sqrt_divide_default_OFF':True,'original_public_layout_changed':False,'compiled_or_GPU_executed':False,'native_rsqrt_emulated':False};(out/'compile-recipe.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);source_prepare(p.parse_args().output)
