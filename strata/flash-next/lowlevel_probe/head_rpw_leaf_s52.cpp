// S52: public production Q5_K API, exact vocabulary head, no private kernel TU.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace K=strata::kernels;
namespace fs=std::filesystem;
using Bytes=std::vector<uint8_t>;
using Clock=std::chrono::steady_clock;
constexpr int N=2560,M=248320,CMAX=6,BATCH=20,WARM=3,ROUNDS=3;
constexpr size_t WB=437043200,WOFF=11024256,G=4096;
constexpr uint64_t WFNV=0xda8746c3dcc2469eull;
struct Strict:std::runtime_error {using std::runtime_error::runtime_error;};
static void need(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
static void strict(bool x,const std::string& m){if(!x)throw Strict(m);}
static double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
static void write(const fs::path& p,const void* v,size_t n){std::ofstream f(p,std::ios::binary|std::ios::trunc);need(bool(f),"open export "+p.string());f.write(static_cast<const char*>(v),n);need(bool(f),"write export "+p.string());f.close();need(bool(f),"close export "+p.string());}
static void text(const fs::path& p,const std::string& s){write(p,s.data(),s.size());}
static Bytes read(const fs::path& p,size_t n){std::ifstream f(p,std::ios::binary);need(bool(f),"open reference "+p.string());Bytes v(n);f.read(reinterpret_cast<char*>(v.data()),n);need(size_t(f.gcount())==n,"short reference "+p.string());need(f.peek()==std::char_traits<char>::eof(),"long reference "+p.string());f.clear();f.close();need(bool(f),"close reference "+p.string());return v;}
static uint64_t fnv(const Bytes& b){uint64_t h=0xcbf29ce484222325ull;for(auto v:b){h^=v;h*=0x100000001b3ull;}return h;}
static void finite(const Bytes& b){need(b.size()%4==0,"float byte count");for(size_t i=0;i<b.size();i+=4){float x;std::memcpy(&x,b.data()+i,4);strict(std::isfinite(x),"nonfinite output");}}
static void strict_equal(const Bytes& a,const Bytes& b,const std::string& label){strict(a==b,"STRICT "+label);}
struct Buffer {
    sycl::queue& q;size_t bytes;uint8_t* raw=nullptr;const char* name;
    Buffer(sycl::queue& s,size_t n,const char* label):q(s),bytes(n),name(label){raw=sycl::malloc_device<uint8_t>(bytes+2*G,q);need(raw!=nullptr,std::string("allocate ")+name);q.memset(raw,0xd7,bytes+2*G);poison();}
    template<class T=uint8_t>T* ptr()const{return reinterpret_cast<T*>(raw+G);}
    void poison(){q.memset(raw+G,0xff,bytes);}
    void put(const void* p,size_t n){need(n==bytes,std::string("upload size ")+name);q.memcpy(raw+G,p,n);}
    Bytes get()const{Bytes b(bytes);q.memcpy(b.data(),raw+G,bytes).wait_and_throw();return b;}
    void guards()const{uint8_t h[2*G];q.memcpy(h,raw,G);q.memcpy(h+G,raw+G+bytes,G).wait_and_throw();for(auto v:h)strict(v==0xd7,std::string("guard ")+name);}
    void readonly(const void* h)const{
        constexpr size_t CHUNK=8*1024*1024;Bytes b(std::min(bytes,CHUNK));
        for(size_t off=0;off<bytes;off+=CHUNK){size_t n=std::min(bytes-off,CHUNK);q.memcpy(b.data(),raw+G+off,n).wait_and_throw();strict(!std::memcmp(b.data(),static_cast<const uint8_t*>(h)+off,n),std::string("readonly ")+name);}
    }
    ~Buffer(){if(raw){try{q.wait_and_throw();sycl::free(raw,q);}catch(...){}}}
};
struct Graph {
    sycl::queue& q;dpct::experimental::command_graph_ptr graph=nullptr;
    dpct::experimental::command_graph_exec_ptr exec=nullptr;
    ~Graph(){try{q.wait_and_throw();delete exec;delete graph;}catch(...){}}
    template<class F>void capture(F call,const fs::path& dot){q.wait_and_throw();dpct::experimental::begin_recording(&q);call();dpct::experimental::end_recording(&q,&graph);need(graph!=nullptr,"null graph");graph->print_graph(dot.string(),true);need(fs::is_regular_file(dot)&&fs::file_size(dot)>0,"missing graph DOT");exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());}
    void run(){q.ext_oneapi_graph(*exec).wait_and_throw();}
};
static void argmax_equal(const Bytes& a,const Bytes& b,int C){for(int c=0;c<C;++c){int ia=0,ib=0;float va=-INFINITY,vb=-INFINITY;for(int i=0;i<M;++i){float x,y;std::memcpy(&x,a.data()+4*(size_t(c)*M+i),4);std::memcpy(&y,b.data()+4*(size_t(c)*M+i),4);if(x>va){va=x;ia=i;}if(y>vb){vb=y;ib=i;}}strict(ia==ib,"argmax mismatch");}}
static void reference(const fs::path& ref,const std::string& file,const Bytes& b){if(!ref.empty())strict_equal(b,read(ref/file,b.size()),"cross-process "+file);}
static int run(const fs::path& output,const char* weightfile,const fs::path& ref){
    const char* wide=std::getenv("STRATA_MMVQ_WIDE_K"),*rpw=std::getenv("STRATA_MMVQ_WIDE_K_RPW");
    need(wide&&std::string(wide)=="1","require STRATA_MMVQ_WIDE_K=1");need(rpw&&(std::string(rpw)=="1"||std::string(rpw)=="2"),"require production RPW env 1/2");
    need(!std::getenv("STRATA_WIDE_K_RPW"),"misspelled RPW env must be absent");
    need(K::native_mmvq_weight_bytes(13,N,M)==WB,"weight geometry");
    need(K::native_q8_1_bytes(N,CMAX)==17280,"Q8 geometry");
    // Weight SHA256 is checked by the window driver on this exact read-only mount.
    std::ifstream f(weightfile,std::ios::binary);need(bool(f),"open real head shard");f.seekg(WOFF);need(bool(f),"seek real head");Bytes weights(WB);f.read(reinterpret_cast<char*>(weights.data()),WB);need(size_t(f.gcount())==WB,"short head slice");f.close();need(bool(f),"close head shard");need(fnv(weights)==WFNV,"real head FNV identity");
    need(dpct::device_count()==1,"require exactly one visible device");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);
    need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI identity unavailable");std::string pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});need(pci=="0000:0b:00.0","require physical card0 BDF0000:0b:00.0");
    std::printf("{\"type\":\"identity\",\"rpw\":%s,\"N\":%d,\"M\":%d,\"weight_bytes\":%zu,\"weight_fnv\":\"%016llx\",\"pci\":\"%s\",\"driver\":\"%s\",\"device\":\"%s\"}\n",rpw,N,M,WB,(unsigned long long)WFNV,pci.c_str(),dev.get_info<sycl::info::device::driver_version>().c_str(),dev.get_info<sycl::info::device::name>().c_str());
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,sycl::property_list{sycl::property::queue::in_order(),sycl::property::queue::enable_profiling()});
    std::mt19937 rng(7);std::normal_distribution<float> nd(0.f,1.f);std::vector<float> a(size_t(CMAX)*N),b(a.size());for(auto& x:a)x=nd(rng);for(size_t i=0;i<a.size();++i){b[i]=-.7f*a[i]+.001f*float(i%17);need(std::isfinite(a[i])&&std::isfinite(b[i])&&std::abs(a[i])<64&&std::abs(b[i])<64,"finite bounded quantizer input");}
    Buffer w(q,WB,"weights"),x(q,a.size()*4,"float input"),xq(q,K::native_q8_1_bytes(N,CMAX),"Q8"),y(q,size_t(CMAX)*M*4,"logits");w.put(weights.data(),WB);q.wait_and_throw();
    std::vector<Buffer*> all{&w,&x,&xq,&y};
    for(int C:{2,6}){
        const std::string stem="C"+std::to_string(C);const size_t active=size_t(C)*M*4,qbytes=K::native_q8_1_bytes(N,C);
        Bytes direct[2],directq[2];Graph single{q},batch{q};
        auto reset=[&](int changed){q.wait_and_throw();for(auto* p:all)p->guards();x.put((changed?b:a).data(),x.bytes);xq.poison();y.poison();q.wait_and_throw();};
        auto quantize=[&]{if(C==2){K::native_quantize_q8_1(x.ptr<float>(),xq.ptr(),N,1,&q);K::native_quantize_q8_1(x.ptr<float>()+N,xq.ptr()+K::native_q8_1_bytes(N,1),N,1,&q);}else K::native_quantize_q8_1(x.ptr<float>(),xq.ptr(),N,C,&q);};
        auto call=[&]{quantize();K::native_q5_k_mmvq(w.ptr(),xq.ptr(),y.ptr<float>(),N,M,C,&q);};
        auto snapshot=[&](const std::string& phase,int changed,bool save,bool checkweights){
            q.wait_and_throw();auto h=y.get(),hq=xq.get();
            if(save){write(output/(stem+"-"+phase+".bin"),h.data(),active);write(output/(stem+"-"+phase+"-q8.bin"),hq.data(),qbytes);}
            try{for(auto* p:all)p->guards();x.readonly((changed?b:a).data());if(checkweights)w.readonly(weights.data());
                for(size_t i=active;i<h.size();++i)strict(h[i]==0xff,"logits inactive tail modified");for(size_t i=qbytes;i<hq.size();++i)strict(hq[i]==0xff,"Q8 inactive tail modified");h.resize(active);hq.resize(qbytes);finite(h);
            }catch(...){if(!save){write(output/(stem+"-"+phase+"-failure.bin"),h.data(),h.size());write(output/(stem+"-"+phase+"-failure-q8.bin"),hq.data(),hq.size());}throw;}
            std::printf("{\"type\":\"checked_snapshot\",\"C\":%d,\"phase\":\"%s\",\"changed\":%d,\"guards\":true,\"tails\":true,\"finite\":true,\"input_readonly\":true,\"weight_readonly\":%s}\n",C,phase.c_str(),changed,checkweights?"true":"false");return std::make_pair(h,hq);
        };
        for(int changed:{0,1}){const std::string label=changed?"b":"a";reset(changed);const auto& in=changed?b:a;write(output/(stem+"-input-"+label+".bin"),in.data(),in.size()*4);Bytes input(in.size()*4);std::memcpy(input.data(),in.data(),input.size());reference(ref,stem+"-input-"+label+".bin",input);
            quantize();q.wait_and_throw();auto pristine=xq.get();pristine.resize(qbytes);
            write(output/(stem+"-quantized-"+label+".bin"),pristine.data(),pristine.size());
            K::native_q5_k_mmvq(w.ptr(),xq.ptr(),y.ptr<float>(),N,M,C,&q);
            auto s=snapshot("eager-"+label,changed,true,true);strict_equal(s.second,pristine,"Q8 readonly after eager head");direct[changed]=std::move(s.first);directq[changed]=std::move(pristine);
            reference(ref,stem+"-eager-"+label+"-q8.bin",directq[changed]);reference(ref,stem+"-eager-"+label+".bin",direct[changed]);}
        reset(0);single.capture(call,output/(stem+"-single.dot"));reset(0);batch.capture([&]{for(int i=0;i<BATCH;++i)call();},output/(stem+"-batch20.dot"));
        for(int changed:{0,1}){const std::string label=changed?"b":"a";reset(changed);single.run();auto s=snapshot("graph-"+label,changed,true,true);strict_equal(s.second,directq[changed],"own graph Q8");strict_equal(s.first,direct[changed],"own graph logits");argmax_equal(s.first,direct[changed],C);reference(ref,stem+"-graph-"+label+".bin",s.first);reference(ref,stem+"-graph-"+label+"-q8.bin",s.second);}
        strict(direct[0]!=direct[1],"changed input not observable");
        auto checked=[&](const std::string& phase,bool checkweights){auto s=snapshot(phase,0,false,checkweights);if(s.first!=direct[0]||s.second!=directq[0]){write(output/(stem+"-"+phase+"-failure.bin"),s.first.data(),s.first.size());write(output/(stem+"-"+phase+"-failure-q8.bin"),s.second.data(),s.second.size());}strict_equal(s.first,direct[0],phase+" logits");strict_equal(s.second,directq[0],phase+" Q8");};
        reset(0);batch.run();checked("batch-check",true);
        for(int i=0;i<WARM;++i){reset(0);batch.run();} // untimed, fixed current input and independent output writes
        for(int round=0;round<ROUNDS;++round){reset(0);auto begin=q.ext_oneapi_submit_barrier();begin.wait_and_throw();const auto t0=Clock::now();q.ext_oneapi_graph(*batch.exec);const auto t1=Clock::now();auto end=q.ext_oneapi_submit_barrier();end.wait_and_throw();const auto t2=Clock::now();
            checked("timed-r"+std::to_string(round),false);
            const auto tbegin=begin.get_profiling_info<sycl::info::event_profiling::command_end>(),tend=end.get_profiling_info<sycl::info::event_profiling::command_end>();need(tend>=tbegin,"timestamp order");
            std::printf("{\"type\":\"batch_timing\",\"rpw\":%s,\"C\":%d,\"round\":%d,\"captured_calls\":%d,\"warm_batch_replays\":%d,\"graph_submissions\":1,\"enqueue_us\":%.3f,\"complete_us\":%.3f,\"barrier_span_us\":%.3f,\"begin_waited_before_host_t0\":true,\"serving_qualified\":false}\n",rpw,C,round,BATCH,WARM,us(t0,t1),us(t0,t2),double(tend-tbegin)/1000.0);
        }
        // Full immutable weight comparison and guards after timing, outside measurement.
        checked("final",true);
    }
    q.wait_and_throw();text(output/"result.json","{\"status\":\"PASS\",\"scope\":\"S52_PRIMITIVE_ONLY\",\"rpw\":"+std::string(rpw)+",\"own_graph_cases\":4,\"timing_batches\":6,\"cross_process_reference\":"+(ref.empty()?std::string("false"):std::string("true"))+",\"serving_qualified\":false}\n");return 0;
}
int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);fs::path output;bool created=false;try{need(argc==4,"usage: head_rpw_leaf NEW_DIR REAL_SHARD REFERENCE_DIR_OR_DASH");output=argv[1];need(fs::create_directory(output),"new output directory required");created=true;const fs::path ref=std::string(argv[3])=="-"?fs::path{}:fs::path(argv[3]);return run(output,argv[2],ref);}catch(const std::exception& e){int rc=dynamic_cast<const Strict*>(&e)?5:2;std::fprintf(stderr,"S52 %s: %s\n",rc==5?"STRICT_FAILURE":"RUNTIME_OR_IO_FAILURE",e.what());if(created){try{text(output/"result.json","{\"status\":\"FAIL\",\"exit\":"+std::to_string(rc)+",\"kind\":\""+(rc==5?std::string("strict"):std::string("runtime_or_io"))+"\"}\n");}catch(const std::exception& io){std::fprintf(stderr,"S52 failure-result IO: %s\n",io.what());return 2;}}return rc;}}
