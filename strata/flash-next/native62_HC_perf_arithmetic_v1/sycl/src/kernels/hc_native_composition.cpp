// F32 contract composition over exact-source projection primitives.
#include "strata/kernels/hc_native_composition.hpp"
#include "strata/kernels/hc_native_projection.hpp"
#include "strata/sycl_queue.hpp"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
namespace strata::kernels {
namespace {
float reduce32(sycl::sub_group group, float x) {
    for (unsigned mask=16; mask; mask>>=1) x += sycl::permute_group_by_xor(group,x,mask);
    return x;
}
float sigmoid(float x) { return 1.0f/(1.0f+sycl::exp(-x)); }
// Independent explicit two-operation RMS contract, default OFF.
int hc_explicit_sqrt_divide() {
    static const int mode=[] {const char* v=std::getenv("STRATA_HC_EXPLICIT_SQRT_DIVIDE");
        if(!v || std::strcmp(v,"0")==0)return 0;
        if(std::strcmp(v,"1")==0)return 1;
        return -1;}();
    return mode;
}
class HcPending; class HcNorm; class HcSilu; class HcMix; class HcStandaloneWrite;
// New prefill selection only; old <=8 entry points never read this flag.
int hc_prefill_projection_tile() {
    static const int tile=[] {const char* v=std::getenv("STRATA_HC_PREFILL_TILE4");
        if(!v || std::strcmp(v,"0")==0)return 2;
        if(std::strcmp(v,"1")==0)return 4;
        return -1;}();
    return tile;
}
}
static bool hc_read_impl(const HcNativeArgs& a, void *stream,bool prefill) {
    const int explicit_sqrt_divide=hc_explicit_sqrt_divide();
    if(explicit_sqrt_divide<0)return false; // Refuse before any HC queue work.
    const int tile=prefill?hc_prefill_projection_tile():2;
    if(tile<0)return false; // Refuse before any HC kernel or queue work.
    if (prefill ? !hc_native_prefill_descriptor_ok(a) : !hc_native_descriptor_ok(a)) return false;
    auto& q = *strata::q_of(stream);
    if (!q.has_property<sycl::property::queue::in_order>()) return false;
    constexpr size_t D=10240,N=2560,H=4,L=320;
    const size_t T=size_t(a.tokens);
    // Separate write avoids reading partially updated residuals during RMS.
    // FMA matches the pending-write reference. R_out may equal R exactly.
    if (a.apply) q.parallel_for<HcPending>(sycl::range<1>(T*D), [=](sycl::id<1> index) {
        const size_t i=index[0],t=i/D,c=(i%D)/N,d=i%N;
        const float w=2.0f*sigmoid(a.inj_prev[t*H+c]/float(H));
        a.R_out[i]=sycl::fma(a.bo_prev[t*N+d],w,a.R[i]);
    });
    const float *R=a.apply?a.R_out:a.R;
    q.parallel_for<HcNorm>(sycl::nd_range<1>(sycl::range<1>(T*H*32),sycl::range<1>(32)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const size_t tc=item.get_group_linear_id(),t=tc/H,c=tc%H,lane=item.get_local_linear_id();
        float sum=0;
        for (size_t d=lane;d<N;d+=32) { const float r=R[t*D+c*N+d]; sum=sycl::fma(r,r,sum); }
        const float rs=explicit_sqrt_divide
            ? 1.0f/sycl::sqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps)
            : sycl::rsqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps);
        if (!lane) a.rs[tc]=rs;
        for (size_t d=lane;d<N;d+=32) { const size_t i=c*N+d; a.xn[t*D+i]=(R[t*D+i]*a.w.norm[i])*rs; }
    });
    // All projection descriptors are implied by the complete preflight above.
    if (!(prefill ? hc_q8_0_project_f32_token_reuse(a.w.down,a.w.down_bytes,a.xn,T*D,a.lo,T*L,D,L,a.tokens,tile,stream)
                  : hc_q8_0_project_f32(a.w.down,a.w.down_bytes,a.xn,T*D,a.lo,T*L,D,L,a.tokens,stream)))
        throw std::logic_error("native HC down rejected after descriptor preflight");
    q.parallel_for<HcSilu>(sycl::range<1>(T*L), [=](sycl::id<1> i) {
        const float x=a.lo[i[0]]/float(H); a.lo[i[0]]=x*sigmoid(x);
    });
    if (!(prefill ? hc_q8_0_project_f32_token_reuse(a.w.up,a.w.up_bytes,a.lo,T*L,a.gate,T*D,L,D,a.tokens,tile,stream)
                  : hc_q8_0_project_f32(a.w.up,a.w.up_bytes,a.lo,T*L,a.gate,T*D,L,D,a.tokens,stream)))
        throw std::logic_error("native HC up rejected after descriptor preflight");
    if (a.w.inject && !(prefill ? hc_f32_project_f32_prefill(a.w.inject,a.w.inject_floats,a.xn,T*D,a.inject_out,T*H,D,H,a.tokens,stream)
            : hc_f32_project_f32(a.w.inject,a.w.inject_floats,a.xn,T*D,a.inject_out,T*H,D,H,a.tokens,stream)))
        throw std::logic_error("native HC injection rejected after descriptor preflight");
    q.parallel_for<HcMix>(sycl::range<1>(T*N), [=](sycl::id<1> index) {
        const size_t td=index[0],t=td/N,d=td%N;
        float sum=0;
        for (size_t c=0;c<H;++c) { const size_t i=t*D+c*N+d; sum+=a.xn[i]*sigmoid(a.gate[i]); }
        a.mixed[td]=sum/float(H);
    });
    return true;
}
static bool hc_write_impl(float* R,size_t R_floats,const float* bo,size_t bo_floats,
                        const float* inject,size_t inject_floats,int tokens,void* stream,bool prefill) {
    if(hc_explicit_sqrt_divide()<0)return false; // Same invalid-option refusal.
    if(prefill && hc_prefill_projection_tile()<0)return false;
    if (!R || !bo || !inject || (prefill ? !(tokens>=1 && tokens<=512 && R_floats>=size_t(tokens)*10240 && bo_floats>=size_t(tokens)*2560 && inject_floats>=size_t(tokens)*4)
        : !hc_native_write_shape_ok(R_floats,bo_floats,inject_floats,tokens))) return false;
    auto& q=*strata::q_of(stream);
    if (!q.has_property<sycl::property::queue::in_order>()) return false;
    // Same kernel name/body as the pending-write launch cannot be reused with
    // another lambda; a separate type keeps SYCL kernel identity unambiguous.
    q.parallel_for<HcStandaloneWrite>(sycl::range<1>(size_t(tokens)*10240),[=](sycl::id<1> index) {
        const size_t i=index[0],t=i/10240,c=(i%10240)/2560,d=i%2560;
        const float w=2.0f*sigmoid(inject[t*4+c]/4.0f);
        R[i]=sycl::fma(bo[t*2560+d],w,R[i]);
    });
    return true;
}

bool hc_native_read_f32(const HcNativeArgs& a,void* stream) {return hc_read_impl(a,stream,false);}
bool hc_native_prefill_read_f32(const HcNativeArgs& a,void* stream) {return hc_read_impl(a,stream,true);}
bool hc_native_write_f32(float* R,size_t nr,const float* bo,size_t nb,const float* inject,size_t ni,int tokens,void* stream) {
    return hc_write_impl(R,nr,bo,nb,inject,ni,tokens,stream,false);
}
bool hc_native_prefill_write_f32(float* R,size_t nr,const float* bo,size_t nb,const float* inject,size_t ni,int tokens,void* stream) {
    return hc_write_impl(R,nr,bo,nb,inject,ni,tokens,stream,true);
}

}
