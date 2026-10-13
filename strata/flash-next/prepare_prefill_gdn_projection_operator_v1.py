"""Root source preparation, no compiler/device; original exports are a separate intake."""
import argparse,hashlib,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'strata/flash-next'
SDK=Path('/mnt/vm_8tb/b70/build/strata-native62-incremental61-v1-20261013T020818Z-wcsa7gom')
RECEIPT_SHA='613694ec6fda4b0136a090406ec5f8c40070e31d967a9af726d308bab4ca48db'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 require(p.is_file()and not p.is_symlink(),'Regular source required '+str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def function(text,name):
 mark='__dpct_inline__ '+name;begin=text.index(mark);body=text.index('{',begin);depth=1;end=body+1
 while depth:
  if text[end]=='{':depth+=1
  elif text[end]=='}':depth-=1
  end+=1
 return text[begin:end]+'\n'
def extract(text):
 pieces=[function(text,'float warp_sum'),function(text,'float sigm'),function(text,'uint16_t hf')]
 post=function(text,'void gdn_out_norm_kernel')
 old='    y16[(size_t) t * ld16 + (size_t) head * S + col] = hf(v);'
 require(post.count(old)==1,'Exact original post store required')
 patch=(HERE/'patches/prefill-gdn-decoder-projections-v1.patch').read_text(encoding='ascii');require(patch.count('+template<bool StoreF32=false>')==1 and patch.count('+    if constexpr(StoreF32)y[at]=v; // already normalized+sigmoid-gated F32, no new arithmetic')==1,'Exact candidate post specialization required');patched='template<bool StoreF32>\n'+post.replace(old,'    if constexpr(StoreF32)y[at]=v;\n'+old)
 require(patched.removeprefix('template<bool StoreF32>\n').replace('    if constexpr(StoreF32)y[at]=v;\n','')==post,'Post observer exact undo failed')
 return '#include <dpct/dpct.hpp>\nnamespace pf_post {\nconstexpr int S=128,HV=48;\n'+''.join(pieces)+patched+'''template<bool StoreF32>void launch(sycl::queue&q,const float*z,const float*gamma,float eps,float*y,uint16_t*y16,int T){
 q.parallel_for<class PostStoreFixtureV1<StoreF32>>(sycl::nd_range<3>(sycl::range<3>(1,48,T)*sycl::range<3>(1,1,128),sycl::range<3>(1,1,128)),[=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{gdn_out_norm_kernel<StoreF32>(z,gamma,eps,y,y16,6144);});
}\n}\n'''.replace('template<bool StoreF32>void launch','template<bool>class PostStoreFixtureV1;\ntemplate<bool StoreF32>void launch').replace('class PostStoreFixtureV1<StoreF32>','PostStoreFixtureV1<StoreF32>')
def commands(text):
 lines=text.splitlines();compile_lines=[x for x in lines if 'verify_parity.cpp.o -c 'in x];link_lines=[x for x in lines if ' -o verify_parity 'in x]
 require(len(compile_lines)==len(link_lines)==1,'Exact actual verifier compiler/link recipe required');c=shlex.split(compile_lines[0]);i=c.index('-MD');c=c[:i]+['-I/inputs','-o','/out/fixture.o','-c','/inputs/fixture.cpp'];l=shlex.split(link_lines[0].removeprefix(': && ').removesuffix(' && :'));l=[('/out/fixture.o'if x=='CMakeFiles/verify_parity.dir/src/kernels/verify_parity.cpp.o'else'/out/pf-gdn'if x=='verify_parity'else'/sdk/build/'+x if x in ('libstrata_kernels.a','libstrata_core.a')else x)for x in l];require('-std=c++20'in c and'-fp-model=precise'in c and'-cl-fp32-correctly-rounded-divide-sqrt'in l,'Production precise compile/device-link flags required');return c,l

def prepare(output):
 require(sha(SDK/'receipt.json')==RECEIPT_SHA,'Actual62 receipt changed');receipt=json.loads((SDK/'receipt.json').read_text());require(receipt['build_rc']==0 and receipt['devices_exposed']is False,'Actual completed NoDri SDK required');require(not output.exists(),'New prepared source root required')
 post=SDK/'source/sycl/src/prefill/kernels.dp.cpp';cmdfile=SDK/'build/incremental-parent-commands.txt';compile_argv,link_argv=commands(cmdfile.read_text(encoding='ascii'));helper=HERE/'original-mtp53/prefill_gdn_decoder_projection_v1.hpp';fixture=ROOT/'kernels/prefill_gdn_decoder_projection_fixture_v1.cpp';output.mkdir();(output/'fixture.cpp').write_bytes(fixture.read_bytes());(output/helper.name).write_bytes(helper.read_bytes());(output/'prefill_gdn_post_store_generated_v1.inc').write_text(extract(post.read_text(encoding='ascii')),encoding='ascii')
 archives={k:v for k,v in(receipt['reused_archives_sha256']|receipt['updated_archive_sha256']).items()if k in('libstrata_kernels.a','libstrata_core.a')};require(len(archives)==2,'Actual SDK archive metadata required')
 sourcepins={str(p):sha(p)for p in (Path(__file__).resolve(),fixture,helper,HERE/'patches/prefill-gdn-decoder-projections-v1.patch',HERE/'prefill-gdn-decoder-projections-source-plan-v1.json',post,cmdfile,SDK/'receipt.json',SDK/'build/compiler-version.txt',SDK/'build/packages.txt')};r={'schema':1,'image':receipt['image'],'SDK':str(SDK),'SDK_receipt_sha256':RECEIPT_SHA,'compile_argv':compile_argv,'link_argv':link_argv,'files':{p.name:sha(p)for p in output.iterdir()},'source_pins':sourcepins,'archives_sha256':archives,'original_weight_payloads_read':False,'compiled_or_GPU_executed':False,'post_store_scope':'Exact source-derived post half output preservation, no decoder conv/recurrence parity','projection_scope':'Original packed Q80/BF16 weights with synthetic independent F32 operands, T1/7/8/9 versus decoder T1'};(output/'compile-recipe.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
