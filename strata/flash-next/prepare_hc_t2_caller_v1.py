"""Pure copied-source construction: bounded T2 HC caller with existing dead scratch."""
import argparse,difflib,hashlib,json,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source')
TARGETS=['sycl/src/core/verify.cpp','sycl/src/kernels/hc_native_composition.cpp','sycl/include/strata/kernels/hc_native_composition.hpp','sycl/src/kernels/hc_native_projection.cpp','sycl/include/strata/kernels/hc_native_projection.hpp']
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Unique actual52 HC seam required '+a[:80])
 return s.replace(a,b,1)
def construct():
 files={p:(SDK/p).read_text()for p in TARGETS}
 with tempfile.TemporaryDirectory()as d:
  root=Path(d)
  for p in TARGETS:
   out=root/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(files[p])
  subprocess.run(['git','apply',str(HERE/'patches/candidate-default-off-hc-q8-token-reuse-v1.patch')],cwd=root,check=True,capture_output=True)
  for p in TARGETS:files[p]=(root/p).read_text()
 p='sycl/src/kernels/hc_native_composition.cpp';s=files[p]
 s=once(s,'bool hc_native_read_f32(const HcNativeArgs& a, void *stream) {','static bool hc_native_read_impl_t2(const HcNativeArgs& a, void *stream,bool reuse) {')
 for role,k,m,src,cap,dst,ocap in [('down','D','L','a.xn','T*D','a.lo','T*L'),('up','L','D','a.lo','T*L','a.gate','T*D')]:
  old=f'hc_q8_0_project_f32(a.w.{role},a.w.{role}_bytes,{src},{cap},{dst},{ocap},{k},{m},a.tokens,stream)'
  new=f'(reuse ? hc_q8_0_project_f32_token_reuse(a.w.{role},a.w.{role}_bytes,{src},{cap},{dst},{ocap},{k},{m},a.tokens,2,stream) : {old})'
  s=once(s,old,new)
 s=once(s,'bool hc_native_write_f32(', 'bool hc_native_read_f32(const HcNativeArgs& a,void* stream){return hc_native_read_impl_t2(a,stream,false);}\nbool hc_native_read_f32_t2(const HcNativeArgs& a,void* stream){return a.tokens==2 && hc_native_read_impl_t2(a,stream,true);}\nbool hc_native_write_f32(')
 files[p]=s
 p='sycl/include/strata/kernels/hc_native_composition.hpp';s=files[p];s=once(s,'bool hc_native_read_f32(const HcNativeArgs& a, void *stream);','bool hc_native_read_f32(const HcNativeArgs& a, void *stream);\n// Explicit two-row caller path; same layout/math, qualified tile2 projection source.\nbool hc_native_read_f32_t2(const HcNativeArgs& a,void* stream);');files[p]=s
 p='sycl/src/core/verify.cpp';s=files[p]
 anchor='''            if (native_hc_requested()) {
                // Rt(t) is explicitly R_+t*HC*N.'''
 branch='''            if (native_hc_requested()) {
                static const bool t2_hc=[](){const char* v=std::getenv("STRATA_VERIFY_HC_T2_REUSE");return v && std::strcmp(v,"1")==0;}();
                // parts_ is dead before current post and after prior post's combine on ordered cs.
                // No batch/G2/pipeline/observer/remote alias is admitted to this bank.
                const bool pair=t2_hc && T==2 && G==1 && tb==0 && te==2 && !batch_rec_ && !batch_observe_ &&
                    !prefix30_prompt_ && !fidelity_first_ && !layer0_first_ && !remote_opt_ &&
                    hits_.stage_mirror && device_plan_ && !ar_on() && next_==nullptr && lb_==0 && le_==g.n_layers && ext_stream_==nullptr && cs==cs_ && cs_->is_in_order() &&
                    parts_ && K>=HC && max_t_>=2;
                if(pair){
                    if(apply && !native_hc_write_rows(Rt(0),bo_,inj_prev,2,cs,err)){native_gr_ok=false;return false;}
                    HcNativeArgs a;a.eps=EPS;a.tokens=2;
                    if(!native_hc_bind(wn[half],wd[half],wu[half],wi[half],a,err)){native_gr_ok=false;return false;}
                    a.R=Rt(0);a.R_floats=size_t(2*HC*N);
                    a.xn=xn_;a.xn_floats=size_t(2*HC*N);a.lo=lo_;a.lo_floats=size_t(2*g.hc_lr);
                    a.gate=parts_;a.gate_floats=size_t(2*HC*N);a.rs=rs_;a.rs_floats=size_t(2*HC);
                    a.mixed=mixed_;a.mixed_floats=size_t(2*N);a.inject_out=inj_out;a.inject_out_floats=size_t(2*HC);
                    if(!strata::kernels::hc_native_read_f32_t2(a,cs)){err="native HC T2 descriptor/queue refused";native_gr_ok=false;return false;}
                    return false; // Same no-QFUSE contract as the per-row native route.
                }
                // Rt(t) is explicitly R_+t*HC*N.'''
 s=once(s,anchor,branch);files[p]=s
 return files

def prepare(out):
 plan=json.loads((HERE/'hc-t2-caller-source-proposal-v1.json').read_bytes())
 for p,h in plan['original_sha256'].items():
  if hashlib.sha256((SDK/p).read_bytes()).hexdigest()!=h:raise ValueError('Original52 HC source changed')
 for p,h in plan['own_source_sha256'].items():
  if hashlib.sha256((HERE.parents[1]/p).read_bytes()).hexdigest()!=h:raise ValueError('HC candidate source changed')
 files=construct();out=Path(out)
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical copied source required')
 out.mkdir()
 for p,s in files.items():
  if hashlib.sha256(s.encode('ascii')).hexdigest()!=plan['constructed_sha256'][p]:raise ValueError('Exact HC construction changed')
  target=out/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s,encoding='ascii')
 record={'source_proposal_sha256':hashlib.sha256((HERE/'hc-t2-caller-source-proposal-v1.json').read_bytes()).hexdigest(),'copied_sources':plan['constructed_sha256'],'compile_executed':False,'GPU_executed':False,'model_payload_read':False}
 (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output),ensure_ascii=True))
