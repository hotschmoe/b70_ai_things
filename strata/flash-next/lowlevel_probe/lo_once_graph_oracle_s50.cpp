// S50 v2: capture 32 reads per timed graph. Relink regenerates AOT device images.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
namespace strata::kernels { bool s48_set_lo_once(bool); } // private probe TU only
namespace K = strata::kernels;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int N=2560, HC=4, D=N*HC, LR=320, TM=K::kFusedGrMaxT;
constexpr size_t G=256;
static void need(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
static uint16_t bf16(float x) { uint32_t u; std::memcpy(&u,&x,4); return uint16_t((u+0x7fff+((u>>16)&1))>>16); }
static double us(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::micro>(b-a).count();}
struct Buffer {
    sycl::queue& q; size_t bytes; uint8_t* raw; const char* name;
    Buffer(sycl::queue& queue,size_t size,const char* label):q(queue),bytes(size),name(label) {
        raw=sycl::malloc_device<uint8_t>(bytes+2*G,q); need(raw!=nullptr,std::string("allocation ")+name); reset();
    }
    template<class T> T* ptr() const {return reinterpret_cast<T*>(raw+G);}
    void reset() {q.memset(raw,0xd7,bytes+2*G);q.memset(raw+G,0xff,bytes);}
    template<class T> void put(const std::vector<T>& h) {need(h.size()*sizeof(T)==bytes,"upload size");q.memcpy(raw+G,h.data(),bytes);}
    std::vector<uint8_t> get() const {std::vector<uint8_t> h(bytes);q.memcpy(h.data(),raw+G,bytes).wait_and_throw();return h;}
    void guards() const {
        uint8_t h[2*G];q.memcpy(h,raw,G);q.memcpy(h+G,raw+G+bytes,G).wait_and_throw();
        for(auto v:h)need(v==0xd7,std::string("guard ")+name);
    }
    ~Buffer(){try{q.wait_and_throw();sycl::free(raw,q);}catch(...){}}
};
struct Graph {
    sycl::queue& q;
    dpct::experimental::command_graph_ptr graph=nullptr;
    dpct::experimental::command_graph_exec_ptr exec=nullptr;
    ~Graph(){try{q.wait_and_throw();delete exec;delete graph;}catch(...){}}
};
static void finite(const std::vector<uint8_t>& h,size_t elements,const char* name) {
    for(size_t i=0;i<elements;++i){float x;std::memcpy(&x,h.data()+4*i,4);need(std::isfinite(x),std::string("nonfinite ")+name);}
}
static void poison_after(const std::vector<uint8_t>& h,size_t bytes,const char* name) {
    for(size_t i=bytes;i<h.size();++i)need(h[i]==0xff,std::string("tail/untaken write ")+name);
}
static bool same(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){return a==b;}
static void exact(const Buffer& b,const void* expected) {auto h=b.get();need(!std::memcmp(h.data(),expected,h.size()),std::string("input modified ")+b.name);}
static void env_is(const char* name,const char* wanted) {const char* v=std::getenv(name);need(v&&std::string(v)==wanted,std::string("require ")+name+"="+wanted);}
static int run(int argc,char** argv) {
    need(argc==2||argc==3,"usage: lo_once_graph_oracle NEW_OUTPUT_DIR [reads_per_graph=32]");
    const int iters=argc==3?std::atoi(argv[2]):32;need(iters==32,"exactly 32 captured reads per batch graph");
    env_is("STRATA_GR_LARGE_GRF","0");
    env_is("STRATA_GR_MORE_GROUPS","0");env_is("STRATA_GR_DOWN_SLICED","1");env_is("STRATA_GR_UP_COLS","16");
    env_is("STRATA_GR_UP_ILP","0");env_is("STRATA_GR_V3","0");env_is("STRATA_GR_NORM_SPLIT","1");
    need(!std::getenv("STRATA_HC_Q8")&&!std::getenv("STRATA_QFUSE"),"Q8/QFUSE must be absent");
    const fs::path output=argv[1];need(fs::create_directory(output),"output must be new, parent must exist");
    need(dpct::device_count()==1,"require one exposed device");
    const sycl::device dev=dpct::dev_mgr::instance().get_device(0);
    need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI address unavailable");
    std::string pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});
    need(pci=="0000:44:00.0","require physical card1 PCI0000:44:00.0");
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,
        sycl::property_list{sycl::property::queue::in_order(),sycl::property::queue::enable_profiling()});
    need(q.is_in_order()&&K::fused_gr_supported(N,HC,LR),"unsupported queue/geometry");
    // Exact current fixture's seed, distribution, scales and BF16 conversion.
    std::mt19937 rng(7);std::normal_distribution<float> nd(0.f,1.f);
    auto fill=[&](std::vector<float>& v,float scale){for(auto& x:v)x=scale*nd(rng);};
    std::vector<float> R((size_t)TM*D),bo((size_t)TM*N),inj((size_t)TM*HC),wn(D);
    fill(R,1.f);fill(bo,.5f);fill(inj,1.f);fill(wn,.3f);for(auto& x:wn)x+=1.f;
    std::vector<uint16_t> wd((size_t)LR*D),wu((size_t)D*LR),wi((size_t)HC*D);
    for(auto& x:wd)x=bf16(.02f*nd(rng));for(auto& x:wu)x=bf16(.05f*nd(rng));for(auto& x:wi)x=bf16(.02f*nd(rng));
    auto R2=R,bo2=bo,inj2=inj;
    // Changed-input recipe from current gr_parity.cpp550..553.
    for(size_t i=0;i<R2.size();++i)R2[i]=-.7f*R2[i]+.001f*float(i%17);
    for(auto& x:bo2)x=-.4f*x+.02f;for(auto& x:inj2)x+=.3f;
    Buffer r(q,R.size()*4,"R"),b(q,bo.size()*4,"bo"),ip(q,inj.size()*4,"inj_prev"),norm(q,wn.size()*4,"norm"),
        down(q,wd.size()*2,"w_down"),up(q,wu.size()*2,"w_up"),iw(q,wi.size()*2,"w_inject"),
        xn(q,size_t(TM)*D*4,"xn"),lo(q,size_t(TM)*LR*4,"lo"),rs(q,size_t(TM)*HC*4,"rs"),
        io(q,size_t(TM)*HC*4,"inject_out"),mix(q,size_t(TM)*N*4,"mixed");
    std::vector<Buffer*> buffers{&r,&b,&ip,&norm,&down,&up,&iw,&xn,&lo,&rs,&io,&mix};
    norm.put(wn);down.put(wd);up.put(wu);iw.put(wi);q.wait_and_throw();
    int strict_cases=0;
    for(int T:{6})for(int apply:{1})for(int inject:{1}) {
        int variant=0;std::map<std::string,std::vector<uint8_t>> baseline;
        const std::string stem="T"+std::to_string(T)+"-a"+std::to_string(apply)+"-i"+std::to_string(inject);
        K::FusedGrArgs args[TM]{};
        for(int t=0;t<T;++t){auto& a=args[t];a.R=r.ptr<float>()+size_t(t)*D;a.R_out=r.ptr<float>()+size_t(t)*D;a.apply=apply;
            a.bo_prev=b.ptr<float>()+size_t(t)*N;a.inj_prev=ip.ptr<float>()+size_t(t)*HC;a.w_norm=norm.ptr<float>();
            a.w_down=down.ptr<uint16_t>();a.w_up=up.ptr<uint16_t>();a.w_inject=inject?iw.ptr<uint16_t>():nullptr;
            a.lo=lo.ptr<float>()+size_t(t)*LR;a.rs=rs.ptr<float>()+size_t(t)*HC;
            a.inject_out=io.ptr<float>()+size_t(t)*HC;a.mixed=mix.ptr<float>()+size_t(t)*N;}
        auto reset=[&](bool changed){q.wait_and_throw();for(auto* p:buffers)p->guards();r.reset();b.reset();ip.reset();xn.reset();lo.reset();rs.reset();io.reset();mix.reset();
            r.put(changed?R2:R);b.put(changed?bo2:bo);ip.put(changed?inj2:inj);q.wait_and_throw();};
        auto call=[&]{K::fused_gr_read_multi(args,T,xn.ptr<float>(),&q);};
        auto snapshot=[&](const std::string& phase,bool changed){
            q.wait_and_throw();for(auto* p:buffers)p->guards();
            const auto& input=changed?R2:R;auto hr=r.get(),hl=lo.get(),hs=rs.get(),hi=io.get(),hm=mix.get(),hx=xn.get();
            finite(hr,size_t(T)*D,"R");finite(hl,size_t(T)*LR,"lo");finite(hs,size_t(T)*HC,"rs");
            finite(hm,size_t(T)*N,"mixed");finite(hx,size_t(T)*D,"xn");if(inject)finite(hi,size_t(T)*HC,"inject_out");
            need(!std::memcmp(hr.data()+size_t(T)*D*4,input.data()+size_t(T)*D,(TM-T)*D*4),"inactive R tail modified");
            if(!apply)need(!std::memcmp(hr.data(),input.data(),input.size()*4),"R unexpectedly modified");
            else need(std::memcmp(hr.data(),input.data(),size_t(T)*D*4)!=0,"active R not applied");
            poison_after(hl,size_t(T)*LR*4,"lo");poison_after(hs,size_t(T)*HC*4,"rs");
            poison_after(hi,inject?size_t(T)*HC*4:0,"inject_out");poison_after(hm,size_t(T)*N*4,"mixed");poison_after(hx,size_t(T)*D*4,"xn");
            exact(b,(changed?bo2:bo).data());exact(ip,(changed?inj2:inj).data());
            std::vector<uint8_t> joined;for(const auto* h:{&hr,&hl,&hs,&hi,&hm,&hx})joined.insert(joined.end(),h->begin(),h->end());
            std::ofstream file(output/(stem+"-mode"+std::to_string(variant)+"-"+phase+".bin"),std::ios::binary);file.write(reinterpret_cast<const char*>(joined.data()),joined.size());need(bool(file),"snapshot write failed");
            std::printf("{\"type\":\"snapshot\",\"T\":%d,\"apply\":%d,\"inject\":%d,\"phase\":\"%s\",\"variant\":%d,\"bytes\":%zu,\"guards\":true,\"finites\":true,\"tails\":true}\n",T,apply,inject,phase.c_str(),variant,joined.size());
            return joined;};
        auto select=[&](int v){q.wait_and_throw();need(K::s48_set_lo_once(v==1)==(v==1),"private selector readback");variant=v;};
        auto checked_snapshot=[&](const std::string& phase,bool changed){auto data=snapshot(phase,changed);
            if(variant==0)baseline[phase]=data;else need(data==baseline.at(phase),"STRICT cross-variant "+stem+" "+phase);return data;};
        // Both graphs use the same buffers/scratch and are submitted sequentially on q.
        // Capture mode selects a concrete device kernel. No captured node reads the host selector.
        Graph graphs[2]{{q},{q}}, batches[2]{{q},{q}};
        std::vector<uint8_t> own_direct[2][2];
        for(int v:{0,1}){select(v);reset(false);call();own_direct[v][0]=checked_snapshot("direct-a",false);
            reset(true);call();own_direct[v][1]=checked_snapshot("direct-b",true);
            reset(false);dpct::experimental::begin_recording(&q);call();
            dpct::experimental::end_recording(&q,&graphs[v].graph);
            need(graphs[v].graph!=nullptr,"graph end returned null");
            graphs[v].graph->print_graph((output/("mode"+std::to_string(v)+"-single.dot")).string(),true);
            graphs[v].exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graphs[v].graph->finalize());
            reset(false);dpct::experimental::begin_recording(&q);
            for(int i=0;i<iters;++i)call();
            dpct::experimental::end_recording(&q,&batches[v].graph);
            need(batches[v].graph!=nullptr,"batch graph end returned null");
            batches[v].graph->print_graph((output/("mode"+std::to_string(v)+"-batch32.dot")).string(),true);
            batches[v].exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(batches[v].graph->finalize());
        }
        // Finalized graph nodes bind their captured kernels (source/API property).
        // Bit-identical outputs cannot identify the captured variant. Keep host selector fixed.
        select(0);
        for(int v:{0,1}){variant=v;
            reset(false);q.ext_oneapi_graph(*graphs[v].exec).wait_and_throw();
            auto ga=checked_snapshot("graph-a",false);need(ga==own_direct[v][0],"own graph initial mismatch");
            reset(true);q.ext_oneapi_graph(*graphs[v].exec).wait_and_throw();
            auto gb=checked_snapshot("graph-b",true);need(gb==own_direct[v][1],"own graph changed mismatch");
            constexpr size_t mix_at=size_t(TM)*(D+LR+HC+HC)*4;
            need(std::memcmp(ga.data()+mix_at,gb.data()+mix_at,size_t(T)*N*4)!=0,"changed fixture not observable in mixed");
        }
        ++strict_cases;
        exact(norm,wn.data());exact(down,wd.data());exact(up,wu.data());exact(iw,wi.data());
        // Validate a single 32-read graph from initial state before warm/timing.
        // External analyzer additionally requires exact S48 warm-r0 bytes per variant.
        for(int v:{0,1}){variant=v;reset(false);q.ext_oneapi_graph(*batches[v].exec).wait_and_throw();checked_snapshot("batch-a",false);}
        constexpr int warmup=4;
        for(int v:{0,1})for(int i=0;i<warmup;++i){variant=v;reset(false);q.ext_oneapi_graph(*batches[v].exec).wait_and_throw();}
        // One graph submission per timed batch; selector fixed0, no host kernel dispatch.
        constexpr int orders[8][2]={{0,1},{1,0},{1,0},{0,1},{0,1},{1,0},{1,0},{0,1}};
        for(int round=0;round<8;++round){std::vector<uint8_t> pair[2];
            for(int position=0;position<2;++position){const int v=orders[round][position];variant=v;
                reset(false);auto begin=q.ext_oneapi_submit_barrier();const auto t0=Clock::now();
                auto batch=q.ext_oneapi_graph(*batches[v].exec);
                const auto t1=Clock::now();auto end=q.ext_oneapi_submit_barrier();q.wait_and_throw();const auto t2=Clock::now();
                std::string device="null", graph_device="null";
                try{const auto a=begin.get_profiling_info<sycl::info::event_profiling::command_end>();const auto z=end.get_profiling_info<sycl::info::event_profiling::command_end>();if(z>=a)device=std::to_string(double(z-a)/1000.0);}catch(const sycl::exception&){}
                // Installed queue.hpp returns an event. Profiling support is runtime-conditional;
                // no fallback queue/properties or schedule changes if timestamps unavailable.
                try{const auto a=batch.get_profiling_info<sycl::info::event_profiling::command_start>();const auto z=batch.get_profiling_info<sycl::info::event_profiling::command_end>();if(z>=a)graph_device=std::to_string(double(z-a)/1000.0);}catch(const sycl::exception&){}
                pair[v]=snapshot("batch-warm-r"+std::to_string(round),false);
                exact(norm,wn.data());exact(down,wd.data());exact(up,wu.data());exact(iw,wi.data());
                std::printf("{\"type\":\"paired_batch_graph_timing\",\"variant\":%d,\"round\":%d,\"position\":%d,\"captured_reads\":%d,\"graph_submissions\":1,\"warmup_batch_replays_per_variant\":%d,\"enqueue_us\":%.3f,\"after_enqueue_to_drained_us\":%.3f,\"complete_us\":%.3f,\"same_queue_barrier_span_us\":%s,\"graph_event_span_us\":%s,\"begin_waited_before_host_t0\":false,\"serving_qualified\":false,\"external_profiler_status\":\"root_identity_required\"}\n",v,round,position,iters,warmup,us(t0,t1),us(t1,t2),us(t0,t2),device.c_str(),graph_device.c_str());
            }
            need(pair[0]==pair[1],"STRICT timed-final paired batch graph output mismatch");
        }
        q.wait_and_throw(); // graphs die before shared buffers; no queue work survives scope
    }

    exact(norm,wn.data());exact(down,wd.data());exact(up,wu.data());exact(iw,wi.data());
    q.wait_and_throw();need(K::s48_set_lo_once(false)==false,"final selector reset");std::printf("{\"status\":\"PASS_S50_STRICT_GRAPH_PROBE_ONLY\",\"strict_cases\":1,\"strict_snapshot_pairs\":5,\"paired_warm_rounds\":8,\"calls\":%d,\"serving_qualified\":false,\"speed_qualified\":false,\"device_pci\":\"%s\"}\n",iters,pci.c_str());return 0;
}
int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);try{return run(argc,argv);}catch(const std::exception& e){std::fprintf(stderr,"lo_once_graph_oracle FAIL: %s\n",e.what());return 5;}}
