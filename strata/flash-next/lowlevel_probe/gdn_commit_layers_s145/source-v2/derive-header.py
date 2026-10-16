from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
S=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/verify_kernels.dp.cpp')
s=S.read_text(encoding='ascii');assert hashlib.sha256(s.encode()).hexdigest()=='5faf60ccd6e8d0665281253374bd97d4d874deb86f0aad479ce94f8565b2673e'
def body(name):
 at=s.index(name+'(');start=s.rfind('__dpct_inline__',0,at);opening=s.index('{',at);depth=1;i=opening+1
 while depth:
  if s[i]=='{':depth+=1
  if s[i]=='}':depth-=1
  i+=1
 return s[start:i]
conv=body('gdn_conv_commit_kernel');state=body('gdn_step_commit_kernel')
# Exact current SYCL bodies, only function names changed. No FP/CUDA rewrite.
c=conv.replace('gdn_conv_commit_kernel(','s145_conv_body(',1);g=state.replace('gdn_step_commit_kernel(','s145_state_body(',1)
assert c.replace('s145_conv_body(','gdn_conv_commit_kernel(',1)==conv
assert g.replace('s145_state_body(','gdn_step_commit_kernel(',1)==state
prefix='''// S145 SOURCE ONLY: exact current SYCL bodies with unique private names.
#pragma once
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <stdexcept>
namespace s145 {
constexpr int S=128,RG=4,RPG=S/RG;
'''
wrapper='''
// Host adapter, fixed real geometry. One call/owner submits two kernel roles.
inline void commit_layers(sycl::queue&q,float*state,const float*qkv,const float*h,
                          const float*gate,const float*beta,const int32_t*keep) {
 constexpr int C=10240,HV=48,HK=16,L=36,MT=8;
 constexpr int64_t STRIDE=int64_t(S)*HV*S+int64_t(C)*3,CONV=int64_t(S)*HV*S;
 q.parallel_for<class s145_conv_layers>(
  sycl::nd_range<3>(sycl::range<3>(1,L,(C+255)/256)*sycl::range<3>(1,1,256),sycl::range<3>(1,1,256)),
  [=](sycl::nd_item<3>item){const int64_t l=item.get_group(1);
   s145_conv_body(state+l*STRIDE+CONV,qkv+l*MT*C,C,keep);
  });
 q.parallel_for<class s145_state_layers>(
  sycl::nd_range<3>(sycl::range<3>(L,4,HV)*sycl::range<3>(1,RG,32),sycl::range<3>(1,RG,32)),
  [=](sycl::nd_item<3>item){const int64_t l=item.get_group(0);
   s145_state_body(state+l*STRIDE,h+l*MT*C,C,gate+l*MT*HV,beta+l*MT*HV,HK,HV,keep);
  });
}
} // namespace s145
'''
(P/'private_gdn.hpp').write_text(prefix+c+'\n\n'+g+'\n'+wrapper,encoding='ascii')
pins={'source':str(S),'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'exact_function_spans':{}}
for name,raw,new in [('gdn_conv_commit_kernel',conv,c),('gdn_step_commit_kernel',state,g)]:
 pins['exact_function_spans'][name]={'original_bytes':len(raw),'original_sha256':hashlib.sha256(raw.encode()).hexdigest(),'renamed_bytes':len(new),'renamed_sha256':hashlib.sha256(new.encode()).hexdigest(),'inverse_name_substitution_exact':True}
pins['private_header_sha256']=hashlib.sha256((P/'private_gdn.hpp').read_bytes()).hexdigest()
(P/'source-projection.json').write_text(json.dumps(pins,indent=2)+'\n',encoding='ascii')
