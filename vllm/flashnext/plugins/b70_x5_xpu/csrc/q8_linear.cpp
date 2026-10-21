// X5: aligned Q8_0 payload, DP4A, weight reuse across up to eight rows.
// GGUF codes/scales are only relaid out. Activation quantization matches
// b70_gguf kreg.hpp b70_q8_quant, including FP32 scale and rint/clamp.
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

namespace {
inline float bf(uint16_t v) { return sycl::bit_cast<float>(uint32_t(v) << 16); }
inline int dp4(int a, int b, int c) {
    auto va = sycl::vec<int,1>(a).template as<sycl::vec<int8_t,4>>().template convert<int>();
    auto vb = sycl::vec<int,1>(b).template as<sycl::vec<int8_t,4>>().template convert<int>();
    return c + va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
}
inline int dot4(sycl::int4 a, sycl::int4 b, int c) {
    c=dp4(a.x(),b.x(),c); c=dp4(a.y(),b.y(),c);
    c=dp4(a.z(),b.z(),c); return dp4(a.w(),b.w(),c);
}
class quant;
template<int M, bool BF> class matvec;

void quantize(sycl::queue& q, const uint16_t* x, int8_t* codes, float* scales, int count) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<quant>(sycl::nd_range<1>(((size_t)count+63)/64*64,64), [=](sycl::nd_item<1> it) {
            int i=int(it.get_global_id(0)); if(i>=count) return;
            float v[32], mx=0.f;
#pragma unroll
            for(int j=0;j<32;++j) { v[j]=bf(x[(size_t)i*32+j]); mx=sycl::fmax(mx,sycl::fabs(v[j])); }
            float inv=mx>0.f?127.f/mx:0.f;
            scales[i]=mx*(1.f/127.f);
#pragma unroll
            for(int j=0;j<32;++j) {
                int z=int(sycl::rint(v[j]*inv));
                codes[(size_t)i*32+j]=int8_t(sycl::clamp(z,-127,127));
            }
        });
    });
}

template<int M, bool BF>
void launch(sycl::queue& q, const int8_t* a, const float* ad, const int8_t* w,
            const sycl::half* wd, void* output, int N, int K) {
    constexpr int R=4, L=64;
    int nb=K/32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<matvec<M,BF>>(sycl::nd_range<1>(((size_t)N+R-1)/R*L,L),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg=it.get_sub_group();
                int lane=int(sg.get_local_linear_id());
                int row=int(it.get_group(0))*R+int(it.get_local_id(0))/16;
                float sum[M];
#pragma unroll
                for(int m=0;m<M;++m) sum[m]=0.f;
                if(row<N) {
                    for(int b=lane;b<nb;b+=16) {
                        const int8_t* p=w+(size_t)row*K+(size_t)b*32;
                        sycl::int4 w0=*reinterpret_cast<const sycl::int4*>(p);
                        sycl::int4 w1=*reinterpret_cast<const sycl::int4*>(p+16);
                        float ds=float(wd[(size_t)row*nb+b]);
#pragma unroll
                        for(int m=0;m<M;++m) {
                            const int8_t* z=a+(size_t)m*K+(size_t)b*32;
                            sycl::int4 a0=*reinterpret_cast<const sycl::int4*>(z);
                            sycl::int4 a1=*reinterpret_cast<const sycl::int4*>(z+16);
                            int d=dot4(w0,a0,0); d=dot4(w1,a1,d);
                            sum[m] += (ds*ad[(size_t)m*nb+b])*float(d);
                        }
                    }
                }
#pragma unroll
                for(int m=0;m<M;++m) {
                    float v=sum[m];
#pragma unroll
                    for(int delta=8;delta;delta>>=1) v+=sycl::permute_group_by_xor(sg,v,delta);
                    if(lane==0 && row<N) {
                        if constexpr(BF) static_cast<sycl::ext::oneapi::bfloat16*>(output)[(size_t)m*N+row]=sycl::ext::oneapi::bfloat16(v);
                        else static_cast<float*>(output)[(size_t)m*N+row]=v;
                    }
                }
            });
    });
}

template<bool BF>
void dispatch(sycl::queue& q,const int8_t* a,const float* ad,const int8_t* w,
              const sycl::half* wd,void* y,int M,int N,int K) {
#define CASE(m) case m: launch<m,BF>(q,a,ad,w,wd,y,N,K); break
    switch(M) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8); }
#undef CASE
}

at::Tensor q8_linear(at::Tensor x,at::Tensor w,at::Tensor s,bool fp32) {
    TORCH_CHECK(x.device().type()==at::kXPU && w.device()==x.device() && s.device()==x.device(),"X5 Q8 devices");
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kChar && s.scalar_type()==at::kHalf,"X5 Q8 dtypes");
    TORCH_CHECK(x.dim()==2 && w.dim()==2 && s.dim()==2 && x.is_contiguous() && w.is_contiguous() && s.is_contiguous(),"X5 Q8 packed 2D");
    int M=x.size(0),K=x.size(1),N=w.size(0);
    TORCH_CHECK(M>=1 && M<=8 && K%32==0 && w.size(1)==K && s.size(0)==N && s.size(1)==K/32,"X5 Q8 shape");
    auto a=at::empty({M,K},x.options().dtype(at::kChar));
    auto ad=at::empty({M,K/32},x.options().dtype(at::kFloat));
    auto y=at::empty({M,N},x.options().dtype(fp32?at::kFloat:at::kBFloat16));
    auto& q=c10::xpu::getCurrentXPUStream().queue();
    quantize(q,reinterpret_cast<uint16_t*>(x.data_ptr()),a.data_ptr<int8_t>(),ad.data_ptr<float>(),M*(K/32));
    if(fp32) dispatch<false>(q,a.data_ptr<int8_t>(),ad.data_ptr<float>(),w.data_ptr<int8_t>(),reinterpret_cast<sycl::half*>(s.data_ptr()),y.data_ptr(),M,N,K);
    else dispatch<true>(q,a.data_ptr<int8_t>(),ad.data_ptr<float>(),w.data_ptr<int8_t>(),reinterpret_cast<sycl::half*>(s.data_ptr()),y.data_ptr(),M,N,K);
    return y;
}
}
TORCH_LIBRARY(b70_x5,m) { m.def("q8_linear(Tensor x, Tensor codes, Tensor scales, bool fp32=False) -> Tensor"); }
TORCH_LIBRARY_IMPL(b70_x5,XPU,m) { m.impl("q8_linear",q8_linear); }
