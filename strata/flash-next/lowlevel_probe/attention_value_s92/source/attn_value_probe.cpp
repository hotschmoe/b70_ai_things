// Compare two separately compiled qsa_decode_attn implementations on identical INT8 pools.
// Compile the variant with qsa_decode_attn_{batch,step,scratch_floats} renamed to value_*.
// Device selection belongs to the leased launch command. No model weights required.
#include <sycl/sycl.hpp>
#include "strata/kernels/qsa_decode_attn.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
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

// Existing S13 allocation sites, with 4096-byte aligned-prefix guards and readonly witnesses.
struct Allocation { uint8_t* raw; void* data; size_t bytes; std::vector<uint8_t> expected; };
static std::vector<Allocation> allocations;
constexpr size_t GUARD = 4096;
template<class T> T* allocate(size_t n, sycl::queue& q) {
    const size_t bytes=n*sizeof(T);
    auto* raw=sycl::malloc_device<uint8_t>(bytes+2*GUARD,q);
    if(!raw) throw std::bad_alloc();
    q.memset(raw,0xd7,bytes+2*GUARD).wait_and_throw();
    void* data=raw+GUARD;if(reinterpret_cast<uintptr_t>(data)%64)throw std::runtime_error("64-byte alignment");allocations.push_back({raw,data,bytes,{}});
    return static_cast<T*>(data);
}
template<class T> void put_readonly(T* d,const std::vector<T>& h,sycl::queue& q) {
    for(auto& a:allocations) if(a.data==d) {
        if(a.bytes!=h.size()*sizeof(T))throw std::runtime_error("input size");
        const auto* b=reinterpret_cast<const uint8_t*>(h.data());a.expected.assign(b,b+a.bytes);
        q.memcpy(d,h.data(),a.bytes).wait_and_throw();return;
    }
    throw std::runtime_error("input allocation");
}
template<class T> T* upload(const std::vector<T>& h,sycl::queue& q) {
    T* d=allocate<T>(h.size(),q);put_readonly(d,h,q);return d;
}
static void memory(sycl::queue& q) {
    for(const auto& a:allocations) {
        std::array<uint8_t,2*GUARD> g;
        q.memcpy(g.data(),a.raw,GUARD);q.memcpy(g.data()+GUARD,a.raw+GUARD+a.bytes,GUARD).wait_and_throw();
        if(!std::all_of(g.begin(),g.end(),[](uint8_t x){return x==0xd7;}))throw std::runtime_error("allocation guard");
        if(!a.expected.empty()) {
            std::vector<uint8_t> got(a.bytes);q.memcpy(got.data(),a.data,a.bytes).wait_and_throw();
            if(got!=a.expected)throw std::runtime_error("readonly input");
        }
    }
}
static void release(sycl::queue& q) {
    q.wait_and_throw();for(const auto& a:allocations)sycl::free(a.raw,q);allocations.clear();
}
int main() try {
    if(std::getenv("STRATA_ATTN_LANECELL"))throw std::runtime_error("STRATA_ATTN_LANECELL must be absent: value-pipeline default path");
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
        auto* a=allocate<float>(nf,q); auto* b=allocate<float>(nf,q);
        auto* ya=allocate<float>(no,q); auto* yb=allocate<float>(no,q);
        // Initialize padding and unused partials identically, making all scratch bytes comparable.
        auto reset=[&]{q.fill(a,0.0f,nf);q.fill(b,0.0f,nf);q.memset(ya,0xff,no*4);q.memset(yb,0xff,no*4);q.wait_and_throw();};
        reset();
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
        memory(q);
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
        const auto initial_out=ha,initial_scratch=sa;
        auto snapshot=[&]{
            q.memcpy(ha.data(),ya,no*4);q.memcpy(hb.data(),yb,no*4);
            q.memcpy(sa.data(),a,nf*4);q.memcpy(sb.data(),b,nf*4);q.wait_and_throw();
            if(std::memcmp(ha.data(),hb.data(),no*4)||std::memcmp(sa.data(),sb.data(),nf*4))throw std::runtime_error("strict full output/scratch");
            for(float v:ha)if(!std::isfinite(v))throw std::runtime_error("nonfinite output");
            for(float v:sa)if(!std::isfinite(v))throw std::runtime_error("nonfinite scratch");
            memory(q);
        };
        snapshot();
        reset();run(false);run(true);snapshot();
        if(std::memcmp(ha.data(),initial_out.data(),no*4)||std::memcmp(sa.data(),initial_scratch.data(),nf*4))throw std::runtime_error("own repeat");
        auto changed=hq;for(auto& v:changed)v=std::bit_cast<float>(std::bit_cast<uint32_t>(v)^0x80000000u);
        put_readonly(dq,changed,q);reset();run(false);run(true);snapshot();
        if(c.width>0&&!std::memcmp(ha.data(),initial_out.data(),no*4)&&!std::memcmp(sa.data(),initial_scratch.data(),nf*4))throw std::runtime_error("changed query unobservable");
        const auto changed_out=ha,changed_scratch=sa;
        reset();run(false);run(true);snapshot();
        if(std::memcmp(ha.data(),changed_out.data(),no*4)||std::memcmp(sa.data(),changed_scratch.data(),nf*4))throw std::runtime_error("own changed repeat");
        put_readonly(dq,hq,q);reset();run(false);run(true);snapshot();
        if(std::memcmp(ha.data(),initial_out.data(),no*4)||std::memcmp(sa.data(),initial_scratch.data(),nf*4))throw std::runtime_error("restored initial");
        std::vector<double> times[2];
        for(int rep=0;rep<24;++rep) for(int arm=0;arm<2;++arm) {
            int v=arm^(rep&1); auto start=std::chrono::steady_clock::now();run(v);
            double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            if(rep>=4) {times[v].push_back(us);std::printf("TIMING seed=%u nq=%d width=%d cap=%d page=%d masked=%d round=%d arm=%d host_completed_us=%.6f\n",seed,c.queries,c.width,c.cap,c.page,c.masked,rep-4,v,us);}
        }
        snapshot();
        if(std::memcmp(ha.data(),initial_out.data(),no*4)||std::memcmp(sa.data(),initial_scratch.data(),nf*4))throw std::runtime_error("timed final initial");
        for(auto& t:times)std::sort(t.begin(),t.end());
        std::printf("PASS seed=%u nq=%d width=%d cap=%d page=%d masked=%d scratch=exact output=exact oracle_max=%.9g base_us=%.3f candidate_us=%.3f\n",
            seed,c.queries,c.width,c.cap,c.page,c.masked,oracle_error,times[0][10],times[1][10]);std::fflush(stdout);
        release(q);
    }
    std::puts("ALL 36 CASES PASS");
}
catch(const std::exception& e) { std::fprintf(stderr,"S92 FAIL: %s\n",e.what());return 6; }
