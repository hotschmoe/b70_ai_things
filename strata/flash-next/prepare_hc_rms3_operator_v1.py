"""Source-only extraction and root-only original prefix1 input export for HC oracle."""
import argparse,hashlib,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SDK59=Path('/mnt/vm_8tb/b70/build/strata-native59-incremental58-v1-20261012T215430Z-tmxlm889')
SDK61=Path('/mnt/vm_8tb/b70/build/strata-native61-incremental60-v1-20261012T233538Z-lrau0o0c')
COMPOSITION_SHA='c63ce71dee3572daca974f03587db54ade1d9ec842176bd83bb81dc7d22bb83f'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_binding():
 plan=HERE/'hc-rms3-isolated-operator-source-plan-v1.json';before=sha(plan)
 for n,w in json.loads(plan.read_text())['files'].items():require(sha(ROOT/n)==w,'RMS oracle source changed '+n)
 require(sha(plan)==before,'RMS oracle plan changed');return before

def extract(text):
 require(hashlib.sha256(text.encode('ascii')).hexdigest()==COMPOSITION_SHA,'Exact59/61 HC composition required')
 start=text.index('    q.parallel_for<HcNorm>');end=text.index('    // All projection descriptors',start);body=text[start:end].rstrip()
 rs='        const float rs=sycl::rsqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps);'
 require(body.count(rs)==1 and body.count('q.parallel_for<HcNorm>')==1,'Unique actual HcNorm launch/body required')
 observed=body.replace('q.parallel_for<HcNorm>','q.parallel_for<HcRmsObserved>').replace(rs,'        const float square_sum=reduce32(item.get_sub_group(),sum);\n        const float argument=square_sum/float(N)+a.eps;\n        const float rs=sycl::rsqrt(argument);\n        if (!lane) { observed_sum[tc]=square_sum; observed_argument[tc]=argument; }')
 require(observed.replace('q.parallel_for<HcRmsObserved>','q.parallel_for<HcNorm>').replace('        const float square_sum=reduce32(item.get_sub_group(),sum);\n        const float argument=square_sum/float(N)+a.eps;\n        const float rs=sycl::rsqrt(argument);\n        if (!lane) { observed_sum[tc]=square_sum; observed_argument[tc]=argument; }',rs)==body,'Exact observed source undo failed')
 reduce=text[text.index('float reduce32('):text.index('float sigmoid(')]
 pre='using strata::kernels::HcNativeArgs;\nclass HcNorm;class HcRmsObserved;\n'+reduce
 signature='(sycl::queue& q,const HcNativeArgs& a,float* observed_sum,float* observed_argument) {\n    constexpr size_t D=10240,N=2560,H=4;const size_t T=1;const float* R=a.R;\n'
 return pre+'static void baseline'+signature+body+'\n}\nstatic void observed'+signature+observed+'\n}\n'
def source_prepare(out):
 binding=source_binding();require(not out.exists(),'New operator compile input root required');texts=[]
 for sdk in(SDK59,SDK61):
  p=sdk/'source/sycl/src/kernels/hc_native_composition.cpp';require(sha(p)==COMPOSITION_SHA,'HC actual source changed');texts.append(p.read_text(encoding='ascii'))
 require(texts[0]==texts[1],'59/61 HC kernels differ');generated=extract(texts[0]);out.mkdir();(out/'hc_rms3_generated_v1.inc').write_text(generated,encoding='ascii');(out/'fixture.cpp').write_bytes((ROOT/'kernels/hc_rms3_operator_fixture_v1.cpp').read_bytes());header=SDK61/'source/sycl/include/strata/kernels/hc_native_composition.hpp';(out/'hc_native_composition.hpp').write_bytes(header.read_bytes())
 flags=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-I/inputs','/inputs/fixture.cpp','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/out/hc-rms3']
 require(source_binding()==binding,'Closing source preparation changed');recipe={'schema':1,'source_plan_sha256':binding,'scope':'Isolated actual-source prefix1 HC normalization; not actual MAIN intermediate authority','compiler_argv':flags,'original59_61_composition_sha256':COMPOSITION_SHA,'files':{p.name:sha(p)for p in out.iterdir()if p.is_file()},'observer_default_OFF':True,'original_public_layout_changed':False,'no_whole48_engine_build':True,'compiled_or_GPU_executed':False};(out/'compile-recipe.json').write_text(json.dumps(recipe,indent=2)+'\n',encoding='ascii');return recipe

def export_original(a):
 # This root-only branch reads one40KiB ORIGINAL norm tensor, never author-run.
 import numpy as np
 import run_current59_first_gdn_reference_v1 as first
 from run_source44_num38_original_reference_v2 import closing_identity
 binding=source_binding();require(not a.output.exists(),'New original operator input root required');a.output.mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'scope':'Independent original first residual/norm for isolatedHC arithmetic oracle','source_plan_sha256':binding,'model_original_payload_read_bytes':40960,'native_values_used_as_primary_inputs':False};pins={}
 try:
  require(sha(a.model_identity)==a.identity_sha256,'Caller fresh identity required');own=json.loads((a.own_root/'report.json').read_text());require(sha(a.own_root/'report.json')=='620a4ec5f3c8a1bf053afe8529658eb6e02a01a792d1ea403b0013cc6c32a504'and own['passed']is True and own['captured_inputs_used']is False,'Exact independently executed first own state required')
  original=first.BoundOriginalFull48Rows(HERE/'original-gguf-reference-foundation-plan-v1.json',a.model_identity);norm=np.asarray(original.rows('blk.0.hc_attn_norm.weight',[0]),dtype='<f4');require(norm.shape==(1,10240)and np.isfinite(norm).all(),'Original HC norm shape differs');(a.output/'norm.f32').write_bytes(norm.tobytes())
  for name in('residual_input','attn_hc_normalized','HC_square_sum','HC_rms_argument','HC_rms_scale'):
   p=a.own_root/(name+'.f32');want=own['arrays'][name]['sha256'];require(sha(p)==want and own['artifacts'][p.name]==want,'Own recorded HC artifact differs');pins[p]=want;(a.output/('own-'+name+'.f32')).write_bytes(p.read_bytes())
  native_root=a.own_root.parent/'source59-mode0-fidelity-v3-run-v1';public=a.own_root.parent/'source59-mode0-fidelity-v3-recovered-component-v1.json'
  targets,targetpins=first.targets(native_root,'2efcaa63076d6421ea3476e29aea62c44e38809d2199c42ca23d6a54ae947bc8',public,'0df16e374eb0fbbd5883292475ac369c4e6fb5e5ae7a2d406d0fad7e04431d41');pins.update(targetpins);native_xn=targets[0]['fields']['attn_hc_normalized']
  require(all(np.array_equal(t['fields']['attn_hc_normalized'],native_xn)for t in targets),'Native repeat normalized mismatch');(a.output/'target-attn_hc_normalized.f32').write_bytes(native_xn.astype('<f4').tobytes());report['target_arrays_are_comparison_only']=True
  report['original_norm_only']=True
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  try:
   require(source_binding()==binding,'Closing original export source changed')
   for p,w in pins.items():require(sha(p)==w,'Saved own input changed')
   require(sha(a.model_identity)==a.identity_sha256,'Identity changed')
  except BaseException as exc:report['errors'].append('closing-source: '+str(exc))
  try:report['postCPU_full4']=closing_identity(a.output,time.time())
  except BaseException as exc:report['errors'].append('fresh-close: '+str(exc))
  report.update(passed=not report['errors'],finished_epoch=time.time(),artifacts={p.name:sha(p)for p in a.output.iterdir()if p.is_file()});(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii')
 require(report['passed'],'Original HC input export failed');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--export-original',action='store_true');p.add_argument('--own-root',type=Path);p.add_argument('--model-identity',type=Path);p.add_argument('--identity-sha256');a=p.parse_args()
 if a.export_original:require(a.own_root and a.model_identity and a.identity_sha256,'Explicit root-only original export arguments required');export_original(a)
 else:source_prepare(a.output)
if __name__=='__main__':main()
