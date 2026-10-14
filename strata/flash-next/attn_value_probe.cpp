// Compare two separately compiled qsa_decode_attn implementations on identical INT8 pools.
// Compile the variant with qsa_decode_attn_{batch,step,scratch_floats} renamed to value_*.
// Device selection belongs to the leased launch command. No model weights required.
#include <sycl/sycl.hpp>
#include "strata/kernels/qsa_decode_attn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace strata::kernels {
void value_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*,
                 int64_t, const QsaShapes&, float*, float*, int64_t, void*);
}

template<class T> T* upload(const std::vector<T>& h, sycl::queue& q) {
    T* d = sycl::malloc_device<T>(h.size(), q);
    if (!d) throw std::bad_alloc();
    q.memcpy(d, h.data(), h.size()*sizeof(T)).wait();
    return d;
}
int main() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}});
    std::printf("device=%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Case { int queries, width, cap, page, masked; };
    const Case cases[] = {{1,1,64,4,0}, {1,63,64,4,0}, {1,64,64,4,0},
        {1,65,128,4,0}, {1,65,128,4,1}, {1,2051,2112,4,0}, {4,2051,2112,4,0},
        {32,2051,2112,4,0}, {32,2051,2112,512,0}, {32,1999,2112,4,1},
        {12,11,64,4,0}, {32,0,128,4,0}};
    for (unsigned seed : {7u, 31u, 109u}) for (const auto& c : cases) {
        std::mt19937 rng(seed);
        auto s = k::qsa_real_shapes(); s.page_size = c.page;
        const int pages = (8192+c.page-1)/c.page, cells = pages*c.page;
        std::vector<int32_t> table(pages), ids(c.queries*c.cap), steps(c.queries*k::kStepCount);
        for (int p=0;p<pages;++p) table[p]=p;
        std::shuffle(table.begin(),table.end(),rng);
        if (c.masked) for (int p=0;p<pages;p+=7) table[p]=-1;
        std::vector<float> hq(size_t(c.queries)*24*256);
        std::vector<int8_t> hk(size_t(cells)*2*256), hv(hk.size());
        std::vector<uint16_t> hs(size_t(cells)*2*4);
        for (auto& x : hq) x=(int(rng()%1025)-512)/512.0f;
        for (auto& x : hk) x=int(rng()%255)-127;
        for (auto& x : hv) x=int(rng()%255)-127;
        for (auto& x : hs) x=uint16_t(0x1800+(rng()%2048));
        for (auto& x : ids) x=rng()%cells;
        for (int i=0;i<c.queries;++i) steps[i*k::kStepCount+k::kStepWidth]=c.width;
        auto* dq=upload(hq,q); auto* dk=upload(hk,q); auto* dv=upload(hv,q);
        auto* ds=upload(hs,q); auto* dt=upload(table,q); auto* di=upload(ids,q); auto* dst=upload(steps,q);
        k::QsaAttnPools pools; pools.k_q=dk; pools.v_q=dv; pools.k_scale=ds; pools.v_scale=ds; pools.page_table=dt;
        const size_t nf=k::qsa_decode_attn_scratch_floats(c.cap,s)*c.queries, no=hq.size();
        auto* a=sycl::malloc_device<float>(nf,q); auto* b=sycl::malloc_device<float>(nf,q);
        auto* ya=sycl::malloc_device<float>(no,q); auto* yb=sycl::malloc_device<float>(no,q);
        // Initialize padding and unused partials identically, making all scratch bytes comparable.
        q.fill(a,0.0f,nf); q.fill(b,0.0f,nf); q.wait();
        auto run=[&](bool variant) {
            if (variant) k::value_batch(dq,pools,di,dst,c.cap,s,b,yb,c.queries,&q);
            else k::qsa_decode_attn_batch(dq,pools,di,dst,c.cap,s,a,ya,c.queries,&q);
            q.wait_and_throw();
        };
        run(false); run(true);
        std::vector<float> ha(no),hb(no),sa(nf),sb(nf);
        q.memcpy(ha.data(),ya,no*4);q.memcpy(hb.data(),yb,no*4);
        q.memcpy(sa.data(),a,nf*4);q.memcpy(sb.data(),b,nf*4);q.wait();
        size_t mismatch=0; double maxerr=0;
        for (size_t i=0;i<no;++i) {
            if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) return 3;
            mismatch += std::memcmp(&ha[i],&hb[i],4)!=0;
            maxerr=std::max(maxerr,double(std::abs(ha[i]-hb[i])));
        }
        const bool scratch_equal=std::memcmp(sa.data(),sb.data(),nf*4)==0;
        if (mismatch || !scratch_equal) {
            std::printf("FAIL seed=%u nq=%d width=%d page=%d masked=%d outputs=%zu maxerr=%.9g scratch_equal=%d\n",
                        seed,c.queries,c.width,c.page,c.masked,mismatch,maxerr,scratch_equal); return 4;
        }
        // Independent double-precision dense softmax for the small one-query boundary cases.
        double oracle_error=0;
        if (c.queries==1 && c.width<=65) {
            for (int h=0;h<24;++h) {
                std::vector<double> scores(c.width,-INFINITY); double mx=-INFINITY,denom=0;
                for (int i=0;i<c.width;++i) {
                    int cell=ids[i],page=table[cell/c.page]; if(page<0) continue;
                    size_t row=(size_t(page)*2+h/12)*c.page+cell%c.page;
                    double dot=0;
                    for(int d=0;d<256;++d) {
                        float sc=float(sycl::bit_cast<sycl::half>(hs[row*4+d/64]));
                        dot+=double(hq[h*256+d])*(float(hk[row*256+d])*sc);
                    }
                    scores[i]=dot/16;mx=std::max(mx,scores[i]);
                }
                for(double z:scores) if(std::isfinite(z)) denom+=std::exp(z-mx);
                for(int d=0;d<256;++d) {
                    double expected=0;
                    for(int i=0;i<c.width;++i) {
                        int cell=ids[i],page=table[cell/c.page];if(page<0)continue;
                        size_t row=(size_t(page)*2+h/12)*c.page+cell%c.page;
                        float sc=float(sycl::bit_cast<sycl::half>(hs[row*4+d/64]));
                        expected+=std::exp(scores[i]-mx)*double(float(hv[row*256+d])*sc);
                    }
                    if(denom) expected/=denom;
                    oracle_error=std::max(oracle_error,std::abs(expected-ha[h*256+d]));
                }
            }
            if(oracle_error>2e-5) {std::printf("FAIL oracle %.9g\n",oracle_error);return 5;}
        }
        std::vector<double> times[2];
        for(int rep=0;rep<24;++rep) for(int arm=0;arm<2;++arm) {
            int v=arm^(rep&1); auto start=std::chrono::steady_clock::now();run(v);
            double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            if(rep>=4) times[v].push_back(us);
        }
        for(auto& t:times)std::sort(t.begin(),t.end());
        std::printf("PASS seed=%u nq=%d width=%d cap=%d page=%d masked=%d scratch=exact output=exact oracle_max=%.9g base_us=%.3f value_us=%.3f\n",
            seed,c.queries,c.width,c.cap,c.page,c.masked,oracle_error,times[0][10],times[1][10]);std::fflush(stdout);
        for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)ds,(void*)dt,(void*)di,(void*)dst,(void*)a,(void*)b,(void*)ya,(void*)yb})sycl::free(p,q);
    }
    std::puts("ALL 36 CASES PASS");
}
