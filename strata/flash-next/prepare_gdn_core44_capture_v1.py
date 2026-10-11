"""Fresh source44 output-only NUM38 proposal; never changes actual SDK43."""
import copy,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T120045Z-kfiohqmn')
PARENT=HERE/'native43-engine-build-proposal-v1.json'
HEADER='include/strata/kernels/verify_kernels.hpp'
KERNEL='sycl/src/kernels/cuda/verify_kernels.dp.cpp'
CONTRACT='sycl/include/strata/core/layer0_numerical_contract.hpp'
OBSERVER='sycl/include/strata/core/layer0_numerical_observer.hpp'
VERIFY='sycl/src/core/verify.cpp'
NAMES=(HEADER,KERNEL,CONTRACT,OBSERVER,VERIFY)
FIELDS=(('gdn_core_scaled','LE_F32[48,128]',24576),('gdn_rms_square_sum','LE_F32[48]',192),('gdn_rms_argument','LE_F32[48]',192),('gdn_rms_scale','LE_F32[48]',192),('gdn_rsqrt128','LE_F32[1]',4))
DECL=r'''
// Source44 output-only buffers owned by the existing NUM38 snapshot.
struct GdnCoreCapture44 {
 float* core=nullptr; float* square_sum=nullptr; float* argument=nullptr;
 float* scale=nullptr; float* inverse_s=nullptr; uint32_t* error=nullptr;
};
void gdn_step_norm_observed44(float* state,const float* h,int conv_channels,
 const float* gate,const float* beta,const float* z,const float* gamma,
 float eps,float* y,int h_k,int h_v,int n_tok,const int32_t* n_keep,
 void* stream,GdnCoreCapture44 observed);
'''
DEVICE_TARGETS=r'''
 strata::kernels::GdnCoreCapture44 gdn_core_targets44(sycl::queue&q){
  affinity(q);if(!settings().enabled||!data_||building_<0)throw std::invalid_argument("GDN44 targets outside owned graph construction");
  auto target=[&](const char*name,std::size_t bytes){auto i=index(name);copy_extent(i,bytes);rosters_[building_][i]=true;return reinterpret_cast<float*>(data_+offset(i));};
  return {target("gdn_core_scaled",24576),target("gdn_rms_square_sum",192),target("gdn_rms_argument",192),target("gdn_rms_scale",192),target("gdn_rsqrt128",4),reinterpret_cast<std::uint32_t*>(data_+frame_bytes()+56)};
 }
'''
LAUNCH=r'''
void gdn_step_norm_observed44(float* state,const float* h,int conv_channels,
 const float* gate,const float* beta,const float* z,const float* gamma,
 float eps,float* y,int h_k,int h_v,int n_tok,const int32_t* n_keep,
 void* stream,GdnCoreCapture44 observed){
 if(!state||!h||!gate||!beta||!z||!gamma||!y||!stream||!n_keep||h_k!=16||h_v!=48||conv_channels!=10240||n_tok!=1||!observed.core||!observed.square_sum||!observed.argument||!observed.scale||!observed.inverse_s||!observed.error)
  throw std::invalid_argument("GDN44 requires complete owned T1 NUM38 outputs");
 const dpct::dim3 grid((unsigned)h_v),block(S,RG);
 auto props=sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
 strata::q_of(stream)->parallel_for<dpct_kernel_name<class gdn_core44_observed_kernel>>(
  sycl::nd_range<3>(grid*block,block),props,[=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
   gdn_step_norm_multi_kernel<false,false,true>(state,h,conv_channels,gate,beta,z,gamma,eps,y,h_k,h_v,n_tok,n_keep,0,nullptr,observed);
  });
 check("gdn_step_norm_observed44");
}
'''
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def once(text,old,new):
 require(text.count(old)==1,'Exact single source anchor required '+old[:60]);return text.replace(old,new,1)
def originals():return {n:(SDK/'source'/n).read_bytes()for n in NAMES}
def changes():
 old=originals();out={}
 h=old[HEADER].decode();h=once(h,'inline constexpr int kVerifyMaxT = 8;','inline constexpr int kVerifyMaxT = 8;\n'+DECL);out[HEADER]=h.encode()
 c=old[CONTRACT].decode();c=once(c,'std::array<Field,33> fields','std::array<Field,38> fields');line=' {"residual_after_ffn","LE_F32[4,2560]",40960}\n';addition=',\n'.join(' {"%s","%s",%d}'%r for r in FIELDS);c=once(c,line,line.rstrip('\n')+',\n'+addition+'\n');out[CONTRACT]=c.encode()
 o=old[OBSERVER].decode();o=once(o,'#include "strata/kernels/layer0_producer_hooks.hpp"','#include "strata/kernels/layer0_producer_hooks.hpp"\n#include "strata/kernels/verify_kernels.hpp"');o=once(o,' void begin_roster(int key){',DEVICE_TARGETS+'\n void begin_roster(int key){');o=o.replace('exact33 producer','exact38 producer').replace('exact33 producer copies','exact38 producer copies');o=once(o,'\\"raw_fused_hidden_observed\\":false,','\\"numerical_source_generation\\":44,\\"gdn_core_output_observed\\":true,\\"raw_fused_hidden_observed\\":false,');out[OBSERVER]=o.encode()
 k=old[KERNEL].decode();k=once(k,'#include <cmath>','#include <cmath>\n#include <stdexcept>');start=k.index('template <bool ALL_OUT, bool Q>\n/*\nDPCT1110:');end=k.index('\ntemplate',start+25)
 region=k[start:end]
 region=once(region,'template <bool ALL_OUT, bool Q>','template <bool ALL_OUT, bool Q, bool CAPTURE44=false>')
 region=once(region,'    GdnQ81 *__restrict__ xq) {','    GdnQ81 *__restrict__ xq,GdnCoreCapture44 observed={}) {')
 point='    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);'
 guard='\n    if constexpr(CAPTURE44){if(n!=1||T!=1||t_out_begin!=0){if(head==0&&col==0&&rg==0)*observed.error|=4096u;return;}}'
 region=once(region,point,point+guard)
 core='''            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] +
                  red_o[3][col]) *
                 sycl::rsqrt((float)S);'''
 captured='''            if constexpr(CAPTURE44){
             const float inverse_s=sycl::rsqrt((float)S);
             oc=(red_o[0][col]+red_o[1][col]+red_o[2][col]+red_o[3][col])*inverse_s;
             observed.core[head*S+col]=oc;
             if(head==0&&col==0)observed.inverse_s[0]=inverse_s;
            }else{
'''+core+'\n            }'
 region=once(region,core,captured)
 rms='            const float scale = sycl::rsqrt(ss / (float)S + eps);'
 # Preserve the original false-specialization expression literally. ON binds
 # the actual expression result once and publishes the same value to raw output.
 rms_new='''            float scale;
            if constexpr(CAPTURE44){const float argument=ss/(float)S+eps;scale=sycl::rsqrt(argument);
             if(col==0){observed.square_sum[head]=ss;observed.argument[head]=argument;observed.scale[head]=scale;}
            }else{scale=sycl::rsqrt(ss / (float)S + eps);}'''
 region=once(region,rms,rms_new);k=k[:start]+region+k[end:];k=once(k,'    check("gdn_step_norm_multi");\n}\n','    check("gdn_step_norm_multi");\n}\n'+LAUNCH);out[KERNEL]=k.encode()
 v=old[VERIFY].decode();call='''                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, self_commit ? one_ : nullptr, cs, tb, g_qfuse() ? (void*) xq_ : nullptr);'''
 branch='''                } else {
                 if(l0_capture)gdn_step_norm_observed44(state,hb,(int)C,gate,beta,z_,(const float*)wnm->data,EPS,y_,(int)HK,(int)HV,te,self_commit?one_:nullptr,cs,layer0_snapshot_->gdn_core_targets44(*cs));
                 else
'''+call.split('                } else\n',1)[1]+'\n                }'
 v=once(v,call,branch);out[VERIFY]=v.encode();return out
def patch_text():
 old=originals();new=changes();return ''.join(''.join(difflib.unified_diff(old[n].decode().splitlines(True),new[n].decode().splitlines(True),fromfile='a/'+n,tofile='b/'+n))for n in NAMES)
def proposal(patch_path):
 parent=json.loads(PARENT.read_bytes());new=copy.deepcopy(parent);new['patches'].append({'path':str(patch_path.relative_to(ROOT)),'sha256':sha(patch_path.read_bytes())});new['source_generation']=44;new['generation']='source44 output-only first-GDN core/RMS NUM38 diagnostic; default NUM10 OFF'
 for n,b in changes().items():
  if n not in new['overlay_files']:new['overlay_files'].append(n)
  new['expected_patched_source_sha256'][n]=sha(b)
 new.update(source44_parent_plan_path=str(PARENT.relative_to(ROOT)),source44_parent_plan_sha256=sha(PARENT.read_bytes()),source44_before_patch_inputs={n:sha(b)for n,b in originals().items()},source43_runtime_proof_transferred=False,source44_core_capture={'fields':list(FIELDS),'added_raw_bytes_per_frame':sum(r[2]for r in FIELDS),'field_count':38,'only_existing_NUM10_ON':True,'additional_dump_readbacks_per_frame':5,'additional_queries_or_barriers':0,'full_model_math_qualified':False,'actual_capture_qualified':False},abi_rebuild_contract={'required_all_eight_targets':True,'source_generation':44,'patched_sources':len(new['overlay_files']),'fresh_public_header_oracle_required':True})
 return new
if __name__=='__main__':print(patch_text(),end='')
