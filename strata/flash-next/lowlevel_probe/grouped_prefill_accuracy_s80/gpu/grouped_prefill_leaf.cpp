// S80 new accuracy contract; S42 strict failure remains unchanged.
#define DPCT_PROFILING_ENABLED
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <oneapi/mkl.hpp>
#include <memory>
#include <type_traits>
#include <sstream>
#include <limits>
#include <cfenv>
#include <bit>
#include <xmmintrin.h>
#include "reference_bound.hpp"

namespace pf = strata::prefill;
namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int64_t N = 2560, FF = 640, CAP = 4096, ROWS = 1748;
constexpr size_t GUARD = 256;
static void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
struct Buffer {
    sycl::queue& q; size_t bytes; uint8_t* base; const char* name;
    Buffer(sycl::queue& queue, size_t n, const char* tag) : q(queue), bytes(n), name(tag) {
        base = sycl::malloc_device<uint8_t>(bytes + 2 * GUARD, q);
        require(base != nullptr, "device allocation failed");
        q.memset(base, 0xD7, bytes + 2 * GUARD);
    }
    void* data() const { return base + GUARD; }
    template<class T> T* as() const { return reinterpret_cast<T*>(base + GUARD); }
    void check() const {
        std::array<uint8_t, GUARD> a{}, b{};
        q.memcpy(a.data(), base, GUARD);
        q.memcpy(b.data(), base + GUARD + bytes, GUARD).wait_and_throw();
        for (size_t i = 0; i < GUARD; ++i)
            if (a[i] != 0xD7 || b[i] != 0xD7) throw std::runtime_error(std::string("guard: ") + name);
    }
    ~Buffer() { try { q.wait_and_throw(); sycl::free(base, q); } catch (...) {} }
};
static void poison_tail(const Buffer& b, size_t from, uint8_t poison) {
    std::vector<uint8_t> h(1 << 20);
    for (size_t at = from; at < b.bytes; at += h.size()) {
        const size_t n = std::min(h.size(), b.bytes - at);
        b.q.memcpy(h.data(), b.as<uint8_t>() + at, n).wait_and_throw();
        for (size_t i = 0; i < n; ++i) require(h[i] == poison, "unused-capacity overwrite");
    }
}
struct GateFailure : std::runtime_error { using std::runtime_error::runtime_error; };
static void gate(bool b,const char* m){if(!b)throw GateFailure(m);}
using Bytes=std::vector<uint8_t>; namespace fs=std::filesystem;
static_assert(std::endian::native==std::endian::little && sizeof(float)==4 && sizeof(double)==8);
template<class T> static std::vector<T> load(const fs::path& p,size_t n){
    std::ifstream f(p,std::ios::binary);require(bool(f),"open input");std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T)));
    require(size_t(f.gcount())==n*sizeof(T),"short input");require(f.peek()==std::char_traits<char>::eof(),"long input");return v;
}
static void save(const fs::path& p,const void* data,size_t n){
    require(!fs::exists(p),"refuse overwrite");std::ofstream f(p,std::ios::binary);require(bool(f),"open export");
    f.write(static_cast<const char*>(data),std::streamsize(n));f.close();require(bool(f),"close export");
}
static Bytes get(const Buffer& b,size_t n){require(n<=b.bytes,"readback bounds");Bytes v(n);b.q.memcpy(v.data(),b.data(),n).wait_and_throw();return v;}
static void readonly(const Buffer& b,const void* data,size_t n){
    require(n<=b.bytes,"readonly bounds");Bytes v(std::min(n,size_t(1<<20)));
    for(size_t at=0;at<n;at+=v.size()){const size_t z=std::min(v.size(),n-at);b.q.memcpy(v.data(),b.as<uint8_t>()+at,z).wait_and_throw();gate(!std::memcmp(v.data(),static_cast<const uint8_t*>(data)+at,z),"readonly bytes changed");}
}
static void equal(const Bytes& a,const Bytes& b){gate(a==b,"own producer full bytes mismatch");}
struct Expert {int layer,expert;cpu::NativeFmt f;size_t blob;int64_t offset,ne;};
struct Frame {Bytes gu,h,down;};
struct Drain {sycl::queue& q;~Drain(){try{q.wait_and_throw();}catch(...){}}};
static const char* armname(int a){return a?"grouped":"scalar";}
static int run(int argc,char**argv){
    require(argc==4,"usage: grouped_prefill_leaf FIXTURES CPU_REFERENCE NEW_OUT");
    require(std::fegetround()==FE_TONEAREST,"host RNE required");require((_mm_getcsr()&0x8040)==0,"host FTZ/DAZ must be off");
    const fs::path fixtures(argv[1]),refs(argv[2]),out(argv[3]);require(fs::create_directory(out),"new output required");
    auto blobs=load<uint8_t>(fixtures/"weights.bin",28160000);
    auto source=load<uint16_t>(fixtures/"source.fp16",size_t(CAP*N));
    auto ids_host=load<int32_t>(fixtures/"source_ids.i32",size_t(ROWS));
    auto expected_x=load<uint16_t>(fixtures/"actual_gathered.fp16",size_t(ROWS*N));
    auto changed_x=load<uint16_t>(refs/"actual-gathered-changed.fp16",expected_x.size());
    for(size_t i=0;i<expected_x.size();++i)require(changed_x[i]==uint16_t(expected_x[i]^0x8000),"changed reference sign flip");
    auto cpu_gu=load<uint16_t>(refs/"gu-planes.fp16",size_t(16*1280*N));
    auto ideal=load<double>(refs/"gu-T64.f64",size_t(ROWS*1280));
    auto bound=load<double>(refs/"gu-bound.f64",ideal.size());
    for(size_t i=0;i<ideal.size();++i)require(std::isfinite(ideal[i])&&std::isfinite(bound[i])&&bound[i]>=0,"reference finite/nonnegative");
    for(uint16_t x:cpu_gu)require((x&0x7c00)!=0x7c00 && ((x&0x7c00)!=0 || (x&0x3ff)==0),"GU half must be normal/zero");
    for(uint16_t x:expected_x)require((x&0x7c00)!=0x7c00 && ((x&0x7c00)!=0 || (x&0x3ff)==0),"input half must be normal/zero");
    std::ifstream ts(fixtures/"fixtures.tsv");require(bool(ts),"open fixtures");std::string line;std::getline(ts,line);
    require(line=="index\tlayer\texpert\tgu_type\tdown_type\tne\trow_offset\tblob_offset\tblob_bytes\tup_offset\tdown_offset","TSV header");
    std::vector<Expert> experts;int64_t end=0,max_ne=0;std::map<std::pair<int,int>,int> formats;
    while(std::getline(ts,line)){
        std::istringstream in(line);int idx,layer,e,gt,dt;int64_t ne,off;size_t pos,bytes,up,down;
        require(bool(in>>idx>>layer>>e>>gt>>dt>>ne>>off>>pos>>bytes>>up>>down),"fixture row");in>>std::ws;require(in.eof(),"extra fixture field");
        require(idx==int(experts.size())&&idx<16&&layer>=0&&layer<48&&e>=0&&e<512,"fixture identity");
        cpu::NativeFmt f;std::string err;require(cpu::native_fmt(gt,dt,N,FF,f,err),"native format");
        require(bytes==f.bytes&&up==f.up_off&&down==f.down_off&&pos<=blobs.size()&&bytes<=blobs.size()-pos,"payload bounds/format");
        require(ne>0&&ne<=CAP&&off==end&&ne<=ROWS-end,"route bounds/contiguity");
        experts.push_back({layer,e,f,pos,off,ne});end+=ne;max_ne=std::max(max_ne,ne);++formats[{gt,dt}];
    }
    require(experts.size()==16&&end==ROWS&&max_ne==641&&experts[0].ne==641&&experts[1].ne==1&&formats.size()==10,"exact compact wave");
    for(int gt:{16,17,18,21,22})for(int dt:{20,42})require(formats.count({gt,dt}),"format coverage");
    for(int64_t row=0;row<ROWS;++row){require(ids_host[size_t(row)]>=0&&ids_host[size_t(row)]<CAP,"gather ID");require(!std::memcmp(expected_x.data()+row*N,source.data()+int64_t(ids_host[size_t(row)])*N,size_t(N*2)),"saved gather mapping");}
    require(std::getenv("STRATA_SYCL_GEMM_STAGE")==nullptr&&std::getenv("MKL_BLAS_COMPUTE_MODE")==nullptr,"unset staging/alternate mode env");
    require(dpct::device_count()==1,"one visible GPU");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);
    require(dev.has(sycl::aspect::ext_intel_pci_address),"PCI unavailable");auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});require(pci=="0000:44:00.0","physical card1 required");
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,{sycl::property::queue::in_order()});
    std::printf("{\"type\":\"identity\",\"pci\":\"%s\",\"experts\":16,\"rows\":1748,\"production_eager_only\":true,\"graph_capture_required\":false,\"no_graph_attempt\":true}\n",pci.c_str());
    const size_t gu_bytes=size_t(CAP*1280)*4,h_bytes=size_t(CAP*FF)*2,out_bytes=size_t(CAP*N)*4;
    const size_t gu_used=size_t(ROWS*1280)*4,h_used=size_t(ROWS*FF)*2,out_used=size_t(ROWS*N)*4;
    constexpr size_t dqgu_bytes=size_t(1280*N)*2,dqd_bytes=size_t(N*FF)*2;constexpr uint8_t poison=0xa5;
    Buffer weights(q,blobs.size(),"weights"),input(q,source.size()*2,"source"),ids(q,ids_host.size()*4,"ids");
    Buffer scalar_out(q,out_bytes,"scalar_alias"),group_out(q,out_bytes,"group_alias");
    Buffer sg(q,gu_bytes,"scalar_GU_scratch"),sh(q,h_bytes,"scalar_H_scratch"),saved_gu(q,gu_bytes,"scalar_GU_saved"),saved_h(q,h_bytes,"scalar_H_saved");
    Buffer gg(q,gu_bytes,"group_GU"),gh(q,h_bytes,"group_H"),hc(q,h_bytes,"producer_H_check");
    std::vector<std::unique_ptr<Buffer>> dqgu,dqd;
    for(size_t i=0;i<16;++i){dqgu.emplace_back(new Buffer(q,dqgu_bytes,"GU_plane"));dqd.emplace_back(new Buffer(q,dqd_bytes,"down_plane"));}
    std::vector<Buffer*> all{&weights,&input,&ids,&scalar_out,&group_out,&sg,&sh,&saved_gu,&saved_h,&gg,&gh,&hc};
    for(auto& v:dqgu)all.push_back(v.get());for(auto& v:dqd)all.push_back(v.get());
    struct Group {oneapi::mkl::transpose ta[16],tb[16];int64_t m[16],n[16],k[16],lda[16],ldb[16],ldc[16],count[16];float alpha[16],beta[16];const sycl::half* a[16];const sycl::half* b[16];float* c[16];};
    struct Metadata {sycl::queue& q;uint8_t* raw;Group* p;
        explicit Metadata(sycl::queue& z):q(z),raw(static_cast<uint8_t*>(sycl::aligned_alloc_shared(GUARD,2*sizeof(Group)+2*GUARD,z))),p(nullptr){require(raw,"metadata allocation");std::memset(raw,0xd7,2*sizeof(Group)+2*GUARD);p=reinterpret_cast<Group*>(raw+GUARD);std::construct_at(p);std::construct_at(p+1);}
        ~Metadata(){try{q.wait_and_throw();sycl::free(raw,q);}catch(...){}}
        void check(){for(size_t i=0;i<GUARD;++i)gate(raw[i]==0xd7&&raw[GUARD+2*sizeof(Group)+i]==0xd7,"metadata guard");}
    } meta(q);
    Bytes metadata_expected(2*sizeof(Group));
    // Caller reset() drained q before every full() call. No descriptor is
    // modified while vendor work may still read it; no wait is added in timing.
    auto fill_metadata=[&](){
        std::memset(meta.p,0,2*sizeof(Group));
        for(size_t i=0;i<16;++i){const auto&e=experts[i];Group& gu=meta.p[0];Group& d=meta.p[1];
            for(Group* z:{&gu,&d}){z->ta[i]=oneapi::mkl::transpose::trans;z->tb[i]=oneapi::mkl::transpose::nontrans;z->n[i]=e.ne;z->count[i]=1;z->alpha[i]=1;z->beta[i]=0;}
            gu.m[i]=1280;gu.k[i]=N;gu.lda[i]=N;gu.ldb[i]=N;gu.ldc[i]=1280;gu.a[i]=dqgu[i]->as<sycl::half>();gu.b[i]=group_out.as<sycl::half>()+e.offset*N;gu.c[i]=gg.as<float>()+e.offset*1280;
            d.m[i]=N;d.k[i]=FF;d.lda[i]=FF;d.ldb[i]=FF;d.ldc[i]=N;d.a[i]=dqd[i]->as<sycl::half>();d.b[i]=gh.as<sycl::half>()+e.offset*FF;d.c[i]=group_out.as<float>()+e.offset*N;
        }std::memcpy(metadata_expected.data(),meta.p,metadata_expected.size());
    };
    fill_metadata();pf::Gemm gemm;std::string err;require(gemm.init(&q,1<<20,err),"production Gemm init");Drain drain{q};
    q.memcpy(weights.data(),blobs.data(),blobs.size());q.memcpy(ids.data(),ids_host.data(),ids_host.size()*4);
    Bytes actual_gu(size_t(16)*dqgu_bytes),actual_down(size_t(16)*dqd_bytes);bool planes_ready=false;int changed=0;
    auto set_input=[&](int c){q.wait_and_throw();changed=c;std::vector<uint16_t> v=source;if(c)for(auto& h:v)h^=0x8000;q.memcpy(input.data(),v.data(),v.size()*2).wait_and_throw();};
    auto reset=[&](){q.wait_and_throw();for(Buffer* z:{&scalar_out,&group_out,&sg,&sh,&saved_gu,&saved_h,&gg,&gh,&hc})q.memset(z->data(),poison,z->bytes);q.wait_and_throw();};
    auto dq=[&](size_t ei,size_t slot){const auto&e=experts[ei];const uint8_t*w=weights.as<uint8_t>()+e.blob;strata::kernels::iq_dequant_expert_f16(e.f.gu_type,e.f.d_type,w,w+e.f.up_off,w+e.f.down_off,FF,N,dqgu[slot]->as<uint16_t>(),dqd[slot]->as<uint16_t>(),&q);};
    auto batch=[&](Group& z){return oneapi::mkl::blas::column_major::gemm_batch(q,z.ta,z.tb,z.m,z.n,z.k,z.alpha,z.a,z.lda,z.b,z.ldb,z.beta,z.c,z.ldc,int64_t(16),z.count,oneapi::mkl::blas::compute_mode::unset);};
    auto full=[&](int arm,bool observe,bool collect_planes){
        double metadata_ms=0;
        if(!arm){pf::gather_rows16(input.as<uint16_t>(),ids.as<int32_t>(),scalar_out.as<uint16_t>(),ROWS,N,&q);
            if(observe){auto x=get(scalar_out,size_t(ROWS*N)*2);gate(!std::memcmp(x.data(),(changed?changed_x:expected_x).data(),x.size()),"scalar gather/sign flip");}
            for(size_t z=16;z>0;--z){const size_t i=z-1;const auto&e=experts[i];dq(i,0);
                if(collect_planes){q.memcpy(actual_gu.data()+i*dqgu_bytes,dqgu[0]->data(),dqgu_bytes);q.memcpy(actual_down.data()+i*dqd_bytes,dqd[0]->data(),dqd_bytes);}
                gemm.f16(scalar_out.as<uint16_t>()+e.offset*N,dqgu[0]->as<uint16_t>(),sg.as<float>(),e.ne,1280,N);
                pf::swiglu_interleaved(sg.as<float>(),sh.as<uint16_t>(),e.ne,&q);
                gemm.f16(sh.as<uint16_t>(),dqd[0]->as<uint16_t>(),scalar_out.as<float>()+e.offset*N,e.ne,N,FF);
                if(observe){q.memcpy(saved_gu.as<float>()+e.offset*1280,sg.data(),size_t(e.ne*1280)*4);q.memcpy(saved_h.as<uint16_t>()+e.offset*FF,sh.data(),size_t(e.ne*FF)*2);}
            }
        }else{
            const auto a=Clock::now();fill_metadata();metadata_ms=ms(a,Clock::now());
            pf::gather_rows16(input.as<uint16_t>(),ids.as<int32_t>(),group_out.as<uint16_t>(),ROWS,N,&q);
            if(observe){auto x=get(group_out,size_t(ROWS*N)*2);gate(!std::memcmp(x.data(),(changed?changed_x:expected_x).data(),x.size()),"group gather/sign flip");}
            for(size_t i=0;i<16;++i)dq(i,i);
            const auto gu_event=batch(meta.p[0]);q.ext_oneapi_submit_barrier({gu_event});
            pf::swiglu_interleaved(gg.as<float>(),gh.as<uint16_t>(),ROWS,&q);
            const auto down_event=batch(meta.p[1]);q.ext_oneapi_submit_barrier({down_event});
        }return metadata_ms;
    };
    auto memory=[&](bool all_planes){q.wait_and_throw();for(Buffer*z:all)z->check();meta.check();gate(!std::memcmp(meta.p,metadata_expected.data(),metadata_expected.size()),"metadata mutated");
        readonly(weights,blobs.data(),blobs.size());readonly(ids,ids_host.data(),ids_host.size()*4);std::vector<uint16_t> v=source;if(changed)for(auto&h:v)h^=0x8000;readonly(input,v.data(),v.size()*2);
        poison_tail(scalar_out,out_used,poison);poison_tail(group_out,out_used,poison);poison_tail(saved_gu,gu_used,poison);poison_tail(saved_h,h_used,poison);poison_tail(gg,gu_used,poison);poison_tail(gh,h_used,poison);poison_tail(hc,h_used,poison);poison_tail(sg,size_t(max_ne*1280)*4,poison);poison_tail(sh,size_t(max_ne*FF)*2,poison);
        if(all_planes){for(size_t i=0;i<16;++i){readonly(*dqgu[i],actual_gu.data()+i*dqgu_bytes,dqgu_bytes);readonly(*dqd[i],actual_down.data()+i*dqd_bytes,dqd_bytes);}}
    };
    auto frame=[&](int arm){q.wait_and_throw();Frame f{get(arm?gg:saved_gu,gu_used),get(arm?gh:saved_h,h_used),get(arm?group_out:scalar_out,out_used)};
        for(const Bytes*z:{&f.gu,&f.down})for(size_t i=0;i<z->size();i+=4){float x;std::memcpy(&x,z->data()+i,4);gate(std::isfinite(x),"nonfinite FP32 stage");}
        for(size_t i=0;i<f.h.size();i+=2){uint16_t h;std::memcpy(&h,f.h.data()+i,2);gate((h&0x7c00)!=0x7c00,"nonfinite H");}return f;};
    auto same=[&](const Frame&a,const Frame&b){equal(a.gu,b.gu);equal(a.h,b.h);equal(a.down,b.down);};
    auto export_frame=[&](const Frame&f,const std::string&tag){save(out/(tag+"-gu.f32"),f.gu.data(),f.gu.size());save(out/(tag+"-h.fp16"),f.h.data(),f.h.size());save(out/(tag+"-down.f32"),f.down.data(),f.down.size());};
    auto accuracy=[&](const Frame&f,int arm){double worst=0,maxabs=0,ss=0;size_t fail=0;
        for(size_t i=0;i<ideal.size();++i){float y;std::memcpy(&y,f.gu.data()+i*4,4);const double target=changed?-ideal[i]:ideal[i];const double e=std::abs(double(y)-target);const double upper=s80ref::residual_upper(double(y),target);
            if(!s80ref::accepts(double(y),target,bound[i]))++fail;if(bound[i]>0)worst=std::max(worst,upper/bound[i]);maxabs=std::max(maxabs,e);ss+=e*e;}
        std::printf("{\"type\":\"GU_accuracy\",\"arm\":\"%s\",\"changed\":%d,\"values\":%zu,\"failures\":%zu,\"max_abs\":%.17g,\"rms\":%.17g,\"worst_bound_ratio\":%.17g}\n",armname(arm),changed,ideal.size(),fail,maxabs,std::sqrt(ss/double(ideal.size())),worst);gate(fail==0,"full GU reference target failed");
    };
    auto producer=[&](int arm,const Frame&f){Buffer&gu=arm?gg:saved_gu;
        q.memset(hc.data(),poison,hc.bytes);pf::swiglu_interleaved(gu.as<float>(),hc.as<uint16_t>(),ROWS,&q);equal(get(hc,h_used),f.h);
        q.memset(hc.data(),poison,hc.bytes);for(const auto&e:experts)pf::swiglu_interleaved(gu.as<float>()+e.offset*1280,hc.as<uint16_t>()+e.offset*FF,e.ne,&q);equal(get(hc,h_used),f.h);readonly(gu,f.gu.data(),f.gu.size());hc.check();poison_tail(hc,h_used,poison);
    };
    Frame own[2][2];
    for(int c:{0,1}){set_input(c);
        for(int arm:{0,1}){reset();full(arm,true,!planes_ready&&!arm);auto f=frame(arm);export_frame(f,std::string(armname(arm))+(c?"-b":"-a"));
            if(!planes_ready&&!arm){save(out/"actual-gu-planes.fp16",actual_gu.data(),actual_gu.size());save(out/"actual-down-planes.fp16",actual_down.data(),actual_down.size());gate(!std::memcmp(actual_gu.data(),cpu_gu.data(),actual_gu.size()),"full GPU/CPU GU plane identity");planes_ready=true;}
            memory(arm==1);if(!arm){readonly(*dqgu[0],actual_gu.data(),dqgu_bytes);readonly(*dqd[0],actual_down.data(),dqd_bytes);}
            accuracy(f,arm);producer(arm,f);own[c][arm]=std::move(f);
            if(c==0&&arm==0){equal(own[0][0].gu,load<uint8_t>(fixtures/"reference_GU.f32",gu_used));equal(own[0][0].h,load<uint8_t>(fixtures/"reference_H.fp16",h_used));equal(own[0][0].down,load<uint8_t>(fixtures/"reference_down.f32",out_used));}
        }
        for(int arm:{0,1}){reset();full(arm,true,false);same(frame(arm),own[c][arm]);memory(true);}
    }
    gate(own[0][0].gu!=own[1][0].gu&&own[0][1].gu!=own[1][1].gu,"both changed GU observable");
    set_input(0);for(int arm:{0,1}){reset();full(arm,true,false);same(frame(arm),own[0][arm]);memory(true);}
    std::printf("{\"type\":\"correctness_complete\",\"full_GU_targets\":4,\"GPU_CPU_GU_bytes\":104857600,\"GPU_down_crossalgorithm_bytes\":52428800,\"own_eager_repeat_changed_restore\":true,\"same_GU_SiLU_producer\":true,\"ideal_H_down_accuracy_claim\":false,\"graph_scope\":\"not_required_by_actual_eager_prefill_no_attempt\"}\n");
    auto timed_check=[&](int arm){q.wait_and_throw();equal(get(arm?group_out:scalar_out,out_used),own[0][arm].down);
        if(arm){equal(get(gg,gu_used),own[0][arm].gu);equal(get(gh,h_used),own[0][arm].h);}else{
            const size_t g=size_t(experts[0].ne*1280)*4,h=size_t(experts[0].ne*FF)*2;
            equal(get(sg,g),Bytes(own[0][0].gu.begin(),own[0][0].gu.begin()+g));equal(get(sh,h),Bytes(own[0][0].h.begin(),own[0][0].h.begin()+h));}
        memory(true);
    };
    for(int w=0;w<3;++w)for(int arm:{0,1}){reset();full(arm,false,false);timed_check(arm);}
    for(int pair=0;pair<4;++pair)for(int pos=0;pos<2;++pos){const int arm=(pair==0||pair==3)?pos:1-pos;reset();
        const auto a=Clock::now();const double md=full(arm,false,false);const auto b=Clock::now();q.wait_and_throw();const auto c=Clock::now();timed_check(arm);
        if(pair==3){auto actual=get(arm?group_out:scalar_out,out_used);save(out/(std::string("timed-final-")+armname(arm)+"-down.f32"),actual.data(),actual.size());}
        std::printf("{\"type\":\"timing\",\"pair\":%d,\"position\":%d,\"arm\":\"%s\",\"host_submit_ms\":%.9f,\"completed_wall_ms\":%.9f,\"metadata_fill_ms\":%.9f,\"metadata_included\":true,\"timed_stage_copies\":false,\"own_final_down_full_bytes\":true,\"scalar_GU_H_timed_scope\":\"last_expert_scratch_only_no_stale_saved_buffers\"}\n",pair,pos,armname(arm),ms(a,b),ms(a,c),md);
    }
    const char* result="{\"status\":\"PASS_FULL_GU_TARGET_AND_EAGER_PIPELINE_COHERENCE_ONLY\",\"old_S42_strict_status\":\"FAILED_UNCHANGED\",\"reference_contract\":\"S77_ROOT_SCOPED_S80\",\"paired_samples\":4,\"ideal_H_down_accuracy\":false,\"serving_qualified\":false}\n";
    save(out/"result.json",result,std::strlen(result));q.wait_and_throw();return 0;
}
int main(int argc,char**argv){setvbuf(stdout,nullptr,_IONBF,0);try{return run(argc,argv);}catch(const GateFailure&e){std::fprintf(stderr,"S80 STRICT_OR_REFERENCE_FAILURE: %s\n",e.what());return 5;}catch(const std::exception&e){std::fprintf(stderr,"S80 RUNTIME_OR_IO_FAILURE: %s\n",e.what());return 2;}}
