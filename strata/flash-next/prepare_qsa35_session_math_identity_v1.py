"""Pure source overlay: default QSA3 retained; actual layer35 and disk math v8."""
import argparse,difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'strata/flash-next'
SDK=Path('/mnt/vm_8tb/b70/build/strata-native62-incremental61-v1-20261013T020818Z-wcsa7gom')
RECEIPT_SHA='613694ec6fda4b0136a090406ec5f8c40070e31d967a9af726d308bab4ca48db'
OBSERVER='sycl/include/strata/core/layer3_qsa_target_observer.hpp'
IDENTITY='sycl/include/strata/core/api_state_identity.hpp'
VERIFY='sycl/src/core/verify.cpp'
PLAN=HERE/'qsa35-session-math-identity-source-plan-v1.json'
def source_binding():
 before=sha(PLAN);r=json.loads(PLAN.read_text())
 for n,h in r['files'].items():require(sha(ROOT/n)==h,'Overlay source changed '+n)
 for n,h in r['actual_source_files'].items():require(sha(Path(n))==h,'Actual parent source/dependencies changed '+n)
 require(sha(PLAN)==before,'Overlay plan changed');return before
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 require(p.is_file()and not p.is_symlink(),'Regular source required '+str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(text,old,new,label):
 require(text.count(old)==1,'Exact single source anchor required '+label);return text.replace(old,new,1)
def identity(text):
 text=replace(text,'sycl-state-contract-v7','sycl-state-contract-v8','identity version')
 return replace(text,'"STRATA_STAGE_MIRRORS","STRATA_STAGE_MIRROR_SEGMENT_MIB","SYCL_PROGRAM_COMPILE_OPTIONS"','"STRATA_STAGE_MIRRORS","STRATA_STAGE_MIRROR_SEGMENT_MIB","SYCL_PROGRAM_COMPILE_OPTIONS",\n        "STRATA_HC_EXPLICIT_SQRT_DIVIDE","STRATA_PF_GDN_DECODER_PROJ"','actual math switches')
def observer(text):
 # Selector is diagnostic, read only when the original observer is enabled.
 text=replace(text,'bool enabled=false;const char*dir=nullptr;const char*binding=nullptr;','bool enabled=false;int layer=3;const char*dir=nullptr;const char*binding=nullptr;','private selector')
 text=replace(text,'if(!enabled)return;\n  dir=','if(!enabled)return;\n  const char* selected=std::getenv("STRATA_QSA3_TARGET_LAYER");if(selected){if(!std::strcmp(selected,"3"))layer=3;else if(!std::strcmp(selected,"35"))layer=35;else throw std::invalid_argument("QSA target layer must be absent/3/35");}\n  dir=','strict selector parsing')
 text=replace(text,'owner_(lb<=3&&3<le)','owner_(lb<=settings().layer&&settings().layer<le)','actual source owner')
 text=replace(text,'(owner_&&stage!=0)','(owner_&&settings().layer==3&&stage!=0)','layer35 may belong to stage1')
 text=replace(text,'fd_=::open(settings().dir','if(settings().layer==35&&!((stage==0&&lb==0&&le==24)||(stage==1&&lb==24&&le==48)))throw std::invalid_argument("QSA35 requires actual24/24 pair stage bounds");\n  fd_=::open(settings().dir','strict actual pair ownership')
 # Legacy source geometry remains exactly2048; late35 is separately bounded4096.
 text=replace(text,'st.max_cells!=2048','st.max_cells!=(settings().layer==3?2048:4096)','explicit state capacity')
 start=' void begin(const char*route,int64_t p0,int rows){';a=text.index(start);b=text.index('\n void dump()',a);old=text[a:b]
 needle='if(!r.active||r.resume!=0||r.ids.size()!=4||r.ordinal<1||r.ordinal>4||std::getenv("STRATA_VERIFY_EAGER")||!((p0==0&&rows==2&&!std::strcmp(route,"prompt_verifier"))||(p0==2&&rows==1&&!std::strcmp(route,"prompt_verifier"))||(p0==3&&rows==1&&!std::strcmp(route,"verifier"))))throw std::invalid_argument("QSA3 exact fresh prefix4/2,1,1 windows required");for(int i=0;i<4;++i)if(r.ids[i]!=ids[i])throw std::invalid_argument("QSA3 actual accepted IDs differ");'
 new='if(settings().layer==3){'+needle+'}else{if(!r.active||r.resume!=0||r.ids.size()!=1||r.ids[0]!=248045||r.ordinal<1||r.ordinal>4||std::getenv("STRATA_VERIFY_EAGER")||p0!=0||rows!=1||std::strcmp(route,"verifier"))throw std::invalid_argument("QSA35 exact fresh literal248045/MAX1 verifier window required");}'
 text=text[:a]+replace(old,needle,new,'bounded independent literal35 corpus')+text[b:]
 text=replace(text,'\\"layer\\":3,','\\"layer\\":%d,','actual recorded layer')
 text=replace(text,'r.ordinal,stage_,lb_,le_,owner_?','r.ordinal,stage_,lb_,le_,settings().layer,owner_?','actual metadata selector')
 text=replace(text,'std::fprintf(meta,"],\\"internal_norm_argument_or_rsqrt_observed', 'std::fprintf(meta,"]");if(settings().layer==35)std::fprintf(meta,",\\"qsa_ordinal_global\\":%d,\\"qsa_ordinal_local\\":%d",settings().layer/4,(settings().layer-lb_)/4);std::fprintf(meta,",\\"internal_norm_argument_or_rsqrt_observed','late35 state ordinal metadata')
 return text

def verifier(text):return replace(text,'&&l==3;','&&l==qsa3_target::settings().layer;','actual existing hook selection')
def prepare(output):
 source=source_binding();require(sha(SDK/'receipt.json')==RECEIPT_SHA,'Exact actual62 source parent required');require(not output.exists(),'New source overlay root required');output.mkdir();transforms={OBSERVER:observer,IDENTITY:identity,VERIFY:verifier};files={};diff=[]
 for name,transform in transforms.items():
  p=SDK/'source'/name;before=p.read_text(encoding='ascii');after=transform(before);out=output/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(after,encoding='ascii');files[name]={'before_sha256':sha(p),'after_sha256':sha(out)};diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
 patch=output/'qsa35-session-math-identity-v1.patch';patch.write_text(''.join(diff),encoding='ascii');r={'schema':1,'source_plan_sha256':source,'parent_SDK':str(SDK),'parent_receipt_sha256':RECEIPT_SHA,'files':files,'patch_sha256':sha(patch),'default_observer_layer':3,'optional_observer_layer':35,'layer35_corpus':[248045],'layer35_context':4096,'layer35_MAX1':True,'legacy_layer3_corpus_and_context_preserved':True,'new_session_identity_version':'sycl-state-contract-v8','new_math_switches':['STRATA_HC_EXPLICIT_SQRT_DIVIDE','STRATA_PF_GDN_DECODER_PROJ'],'kernel_arithmetic_changed':False,'Verifier_public_layout_changed':False,'GPU_queries_or_waits_added':False,'compiled_or_GPU_executed':False,'fresh_consumers':['CMakeFiles/strata_engine.dir/src/core/verify.cpp.o','CMakeFiles/strata.dir/src/program/generate.cpp.o'],'fresh_engine_archive_required':True,'all8_fresh_links_required':True};require(source_binding()==source,'Closing overlay source changed');(output/'proposal.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
