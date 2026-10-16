// SOURCE ONLY S126: Q5 C2 full head, runtime packed ones operand only.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "private_q5_runtime_ones.hpp"
#include <array>
#include <memory>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <exception>
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
constexpr int C=2,BATCH=20,WARM=3,ROUNDS=2;
static int runtime_ones=0;
static bool cleanup_error=false;
constexpr size_t G=4096,TAIL=512;
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
    sycl::queue& q;dpct::experimental::command_graph_exec_ptr exec=nullptr;
    ~Graph(){try{q.wait_and_throw();}catch(...){cleanup_error=true;}delete exec;}
    template<class F>void capture(F call,const fs::path& dot){
        q.wait_and_throw();need(exec==nullptr,"capture once");
        dpct::experimental::command_graph_ptr graph=nullptr;bool begun=false,ended=false;
        try{
            begun=true;dpct::experimental::begin_recording(&q);call();
            dpct::experimental::end_recording(&q,&graph);ended=true;
            need(graph!=nullptr,"null graph");graph->print_graph(dot.string(),true);
            need(fs::is_regular_file(dot)&&fs::file_size(dot)>0,"missing graph DOT");
            exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
            delete graph;graph=nullptr;q.wait_and_throw();
        }catch(...){const auto original=std::current_exception();
            if(begun&&!ended){try{if(graph)graph->end_recording();else dpct::experimental::end_recording(&q,&graph);}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture balance failed\n");}}
            try{q.wait_and_throw();}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture drain failed\n");}
            delete graph;std::rethrow_exception(original);
        }
    }
    void run(){q.ext_oneapi_graph(*exec).wait_and_throw();}
};
// Validation-only output; never a producer or a captured projection dependency.
static void raw_gate(sycl::queue& q,Buffer& xq,const Bytes& expected,size_t qbytes,int word,const std::string& label){
    need(qbytes%36==0,"rawsum packet geometry");const size_t count=(qbytes/36)*2;
    Buffer sums(q,count*sizeof(int)+TAIL,"validation raw16 sums");
    s126::raw16_sums(xq.ptr(),sums.ptr<int>(),int(qbytes/36),word,q);q.wait_and_throw();
    sums.guards();xq.guards();auto actual=sums.get();strict_equal(xq.get(),expected,label+" rawgate Q8 readonly");
    for(size_t k=count*4;k<actual.size();++k)strict(actual[k]==0xff,"rawgate tail");
    for(size_t k=0;k<count;++k){int64_t total=0;for(int j=0;j<16;++j){
        const uint8_t byte=expected[(k/2)*36+4+(k%2)*16+j];
        const int qv=byte<128?int(byte):int(byte)-256;
        const unsigned ob=(uint32_t(word)>>(8*(j%4)))&255u;const int ov=ob<128?int(ob):int(ob)-256;
        total+=int64_t(qv)*ov;
    }int value;std::memcpy(&value,actual.data()+4*k,4);strict(value==total,label+" exact INT64 raw16 gate");}
    std::printf("{\"type\":\"raw16_gate\",\"label\":\"%s\",\"word\":%u,\"cells\":%zu,\"INT64_exact\":true,\"readonly_guards\":true}\n",label.c_str(),uint32_t(word),count);
}
static void raw_extremes(sycl::queue& q){
    Bytes input(4*36+TAIL,0xff);for(size_t i=0;i<4*36;++i)input[i]=0;
    for(int packet=0;packet<4;++packet)for(int j=0;j<32;++j){const int v=packet==0?-128:packet==1?127:packet==2?(j%2?-128:127):(j%4==0?1:0);input[packet*36+4+j]=uint8_t(v);}
    Buffer qx(q,input.size(),"raw validation packets");qx.put(input.data(),input.size());q.wait_and_throw();
    raw_gate(q,qx,input,4*36,runtime_ones,"extremes-ones");
    // Same validation kernel with a different payload, not an alternative projection arm.
    raw_gate(q,qx,input,4*36,runtime_ones^3,"payload-negative-control");
}
struct Shape {const char* tag;const char* tensor;int N,M;size_t offset,bytes;uint64_t weight_fnv;};
constexpr Shape SHAPES[]={
 {"head","output.weight",2560,248320,11024256,437043200,0xda8746c3dcc2469eull}
};
static const char* armname(int arm){need(arm>=0&&arm<=2,"bad arm");return arm==0?"public":arm==1?"original":"candidate";}
struct State {
    sycl::queue& q;Shape d;fs::path dir;Bytes weights;std::vector<float> a,b;Bytes savedq[2],anchor[2];
    size_t active,qbytes;Buffer w,x,xq,y;
    Graph pub,orig,cand,orig20,cand20;
    Bytes expected[2],expectedq[2];int current=-1;
    State(sycl::queue& queue,Shape shape,const fs::path& output,const char* shard,const fs::path& fixtures):
      q(queue),d(shape),dir(output/d.tag),weights(d.bytes),a(size_t(C)*d.N),b(a.size()),
      active(size_t(C)*d.M*4),qbytes(K::native_q8_1_bytes(d.N,C)),
      w(q,d.bytes,"weights"),x(q,a.size()*4,"float input"),xq(q,qbytes+TAIL,"Q8"),y(q,active+TAIL,"output"),
      pub{q},orig{q},cand{q},orig20{q},cand20{q} {
        need(fs::create_directory(dir),"new shape directory required");
        need(K::native_mmvq_weight_bytes(13,d.N,d.M)==d.bytes,"type13 weight geometry");
        need(qbytes==size_t(C)*(d.N/32)*36,"canonical Q8 geometry");
        std::ifstream f(shard,std::ios::binary);need(bool(f),"open actual model");
        f.seekg(d.offset);need(bool(f),"seek actual tensor");f.read(reinterpret_cast<char*>(weights.data()),weights.size());
        need(size_t(f.gcount())==weights.size(),"short actual tensor");f.close();need(bool(f),"close actual model");
        need(fnv(weights)==d.weight_fnv,"actual tensor FNV mismatch");
        for(int changed:{0,1}){
            const std::string label=changed?"b":"a";
            auto input=read(fixtures/("C2-input-"+label+".bin"),size_t(6)*d.N*4);
            auto& dst=changed?b:a;std::memcpy(dst.data(),input.data(),dst.size()*4);finite(input);
            auto qinput=read(fixtures/("C2-eager-"+label+"-q8.bin"),qbytes);
            savedq[changed]=Bytes(qbytes+TAIL,0xff);std::copy(qinput.begin(),qinput.end(),savedq[changed].begin());
            anchor[changed]=read(fixtures/("C2-eager-"+label+".bin"),active);finite(anchor[changed]);
        }
        w.put(weights.data(),weights.size());q.wait_and_throw();
        std::printf("{\"type\":\"shape_identity\",\"shape\":\"%s\",\"tensor\":\"%s\",\"N\":%d,\"M\":%d,\"C\":2,\"weight_bytes\":%zu,\"offset\":%zu,\"fnv\":\"%016llx\"}\n",d.tag,d.tensor,d.N,d.M,d.bytes,d.offset,(unsigned long long)d.weight_fnv);
    }
    void guards(){for(Buffer* p:{&w,&x,&xq,&y})p->guards();}
    void output_reset(){q.wait_and_throw();guards();y.poison();q.wait_and_throw();}
    void input(int changed){
        need(changed==0||changed==1,"input selector");q.wait_and_throw();guards();
        const auto& v=changed?b:a;x.put(v.data(),x.bytes);xq.poison();y.poison();
        xq.put(savedq[changed].data(),xq.bytes);q.wait_and_throw();
        auto pristine=xq.get();strict_equal(pristine,savedq[changed],"saved canonical Q8 upload/tail");
        expectedq[changed]=savedq[changed];
        current=changed;
    }
    void call(int arm){
        if(arm==0)K::native_mmvq(13,w.ptr(),xq.ptr(),y.ptr<float>(),d.N,d.M,C,&q);
        else if(arm==1)s126::launch_original(w.ptr(),xq.ptr(),y.ptr<float>(),d.N,d.M,q);
        else if(arm==2)s126::launch_candidate(w.ptr(),xq.ptr(),y.ptr<float>(),d.N,d.M,runtime_ones,q);
        else need(false,"bad projection arm");
    }
    Bytes snapshot(const std::string& phase,bool save){
        q.wait_and_throw();auto out=y.get();auto quant=xq.get();
        if(save){write(dir/(phase+".bin"),out.data(),active);write(dir/(phase+"-q8.bin"),quant.data(),qbytes);}
        try {
            guards();x.readonly((current?b:a).data());
            strict_equal(quant,expectedq[current],phase+" Q8/read-only/tail");
            for(size_t i=active;i<out.size();++i)strict(out[i]==0xff,phase+" output tail");
            out.resize(active);finite(out);
        }catch(...){if(!save){write(dir/(phase+"-failure.bin"),out.data(),out.size());write(dir/(phase+"-failure-q8.bin"),quant.data(),quant.size());}throw;}
        std::printf("{\"type\":\"checked_snapshot\",\"shape\":\"%s\",\"phase\":\"%s\",\"changed\":%d,\"guards\":true,\"tail\":true,\"finite\":true,\"input_readonly\":true,\"weight_guards_checked\":true,\"weight_payload_verified\":false,\"Q8_readonly\":true}\n",d.tag,phase.c_str(),current);
        return out;
    }
    void weight_payload_check(const char* boundary){
        q.wait_and_throw();w.guards();w.readonly(weights.data());
        std::printf("{\"type\":\"weight_payload_check\",\"shape\":\"%s\",\"boundary\":\"%s\",\"bytes\":%zu,\"weight_guards_checked\":true,\"weight_payload_verified\":true}\n",d.tag,boundary,weights.size());
    }
    void equal_snapshot(const std::string& phase,bool save){
        auto out=snapshot(phase,save);
        if(out!=expected[current])write(dir/(phase+"-mismatch.bin"),out.data(),out.size());
        strict_equal(out,expected[current],std::string(d.tag)+" "+phase+" full output bytes");
    }
    void correctness(){
        for(int changed:{0,1}){
            input(changed);const std::string label=changed?"b":"a";
            write(dir/("input-"+label+".bin"),(changed?b:a).data(),x.bytes);
            write(dir/("canonical-"+label+"-q8.bin"),expectedq[changed].data(),qbytes);
            raw_gate(q,xq,expectedq[changed],qbytes,runtime_ones,"canonical-"+label);
            output_reset();call(0);expected[changed]=snapshot("eager-public-"+label,true);
            strict_equal(expected[changed],anchor[changed],"saved S52 RPW1 public byte anchor");
            // The private-original/public byte anchor precedes every candidate gate.
            output_reset();call(1);equal_snapshot("eager-original-"+label,true);
            output_reset();call(2);equal_snapshot("eager-candidate-"+label,true);
            for(int arm:{1,2}){output_reset();call(arm);equal_snapshot(std::string("repeat-")+armname(arm)+"-"+label,true);}
        }
        strict(expected[0]!=expected[1],"changed input not observable in output");
        strict(expectedq[0]!=expectedq[1],"changed canonical Q8 not observable");
        input(0);
        for(int arm:{0,1,2}){output_reset();call(arm);equal_snapshot(std::string("restore-eager-")+armname(arm),true);}
        pub.capture([&]{call(0);},dir/"single-public.dot");
        orig.capture([&]{call(1);},dir/"single-original.dot");
        cand.capture([&]{call(2);},dir/"single-candidate.dot");
        orig20.capture([&]{for(int i=0;i<BATCH;++i)call(1);},dir/"batch20-original.dot");
        cand20.capture([&]{for(int i=0;i<BATCH;++i)call(2);},dir/"batch20-candidate.dot");
        for(int changed:{0,1}){
            input(changed);const std::string label=changed?"b":"a";
            for(auto item:{std::make_pair(&pub,"public"),std::make_pair(&orig,"original"),std::make_pair(&cand,"candidate")}){
                output_reset();item.first->run();equal_snapshot(std::string("graph-")+item.second+"-"+label,true);
                output_reset();item.first->run();equal_snapshot(std::string("graph-repeat-")+item.second+"-"+label,true);
            }
            for(auto item:{std::make_pair(&orig20,"original"),std::make_pair(&cand20,"candidate")}){
                output_reset();item.first->run();equal_snapshot(std::string("batch20-")+item.second+"-"+label,true);
            }
        }
        input(0);
        for(auto item:{std::make_pair(&pub,"public"),std::make_pair(&orig,"original"),std::make_pair(&cand,"candidate")}){
            output_reset();item.first->run();equal_snapshot(std::string("restore-graph-")+item.second,true);
        }
        for(auto item:{std::make_pair(&orig20,"original"),std::make_pair(&cand20,"candidate")}){
            output_reset();item.first->run();equal_snapshot(std::string("restore-batch20-")+item.second,true);
        }
        std::printf("{\"type\":\"correctness_pass\",\"shape\":\"%s\",\"public_original_bit_anchor\":true,\"candidate_full_bytes\":true,\"initial_changed\":true,\"single_batch20_graphs\":true}\n",d.tag);
    }
    void warm(){
        input(0);
        for(int i=0;i<WARM;++i)for(int arm:{1,2}){
            output_reset();(arm==1?orig20:cand20).run();equal_snapshot("warm-"+std::to_string(i)+"-"+armname(arm),false);
        }
    }
    void timing(){
        need(current==0,"timing initial input");
        for(int round=0;round<ROUNDS;++round){
            const std::array<int,4> order=round%2?std::array<int,4>{2,1,1,2}:std::array<int,4>{1,2,2,1};
            double complete[3]{},span[3]{};
            for(int slot=0;slot<4;++slot){
                const int arm=order[slot];output_reset();
                auto begin=q.ext_oneapi_submit_barrier();begin.wait_and_throw();
                const auto t0=Clock::now();q.ext_oneapi_graph(*(arm==1?orig20.exec:cand20.exec));const auto t1=Clock::now();
                auto end=q.ext_oneapi_submit_barrier();end.wait_and_throw();const auto t2=Clock::now();
                const auto tb=begin.get_profiling_info<sycl::info::event_profiling::command_end>();
                const auto te=end.get_profiling_info<sycl::info::event_profiling::command_end>();need(te>=tb,"timestamp order");
                equal_snapshot("timed-r"+std::to_string(round)+"-s"+std::to_string(slot)+"-"+armname(arm),false);
                const double cu=us(t0,t2),sp=double(te-tb)/1000.0;
                complete[arm]+=cu/2;span[arm]+=sp/2;
                std::printf("{\"type\":\"timing_sample\",\"shape\":\"%s\",\"round\":%d,\"slot\":%d,\"arm\":\"%s\",\"order\":\"%s\",\"captured_projection_calls\":20,\"quantization_in_timed_graph\":false,\"enqueue_us\":%.6f,\"complete_us\":%.6f,\"barrier_span_us\":%.6f,\"checked_full_bytes\":true}\n",d.tag,round,slot,armname(arm),round%2?"COOC":"OCCO",us(t0,t1),cu,sp);
            }
            std::printf("{\"type\":\"paired_mean\",\"shape\":\"%s\",\"round\":%d,\"original_complete_us_per_call\":%.6f,\"candidate_complete_us_per_call\":%.6f,\"original_span_us_per_call\":%.6f,\"candidate_span_us_per_call\":%.6f,\"observational_only\":true}\n",d.tag,round,complete[1]/BATCH,complete[2]/BATCH,span[1]/BATCH,span[2]/BATCH);
        }
        equal_snapshot("final",true);
    }
};
static int run(const fs::path& out,const char* shard,const fs::path& fixtures){
    auto env=[](const char* k,const char* v){const char* e=std::getenv(k);need(e&&std::string(e)==v,std::string("require ")+k+"="+v);};
    env("STRATA_MMVQ_WIDE_K","1");env("STRATA_MMVQ_WIDE_K_RPW","1");env("STRATA_HEAD_Q5K_RPW2","0");
    need(K::native_mmvq_multi_exact(),"public exact mode required");
    need(dpct::device_count()==1,"require exactly one visible device");
    const sycl::device dev=dpct::dev_mgr::instance().get_device(0);
    need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI identity unavailable");
    std::string pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});
    need(pci=="0000:0b:00.0","require physical card0 BDF0000:0b:00.0");
    std::printf("{\"type\":\"identity\",\"pci\":\"%s\",\"visible_devices\":1,\"driver\":\"%s\",\"device\":\"%s\",\"public_type\":13,\"C\":2,\"runtime_ones\":%d,\"timing_scope\":\"projection only; canonical Q8 outside graphs\"}\n",pci.c_str(),dev.get_info<sycl::info::device::driver_version>().c_str(),dev.get_info<sycl::info::device::name>().c_str(),runtime_ones);
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,
        sycl::property_list{sycl::property::queue::in_order(),sycl::property::queue::enable_profiling()});
    raw_extremes(q);
    std::vector<std::unique_ptr<State>> states;
    for(const auto& shape:SHAPES){states.emplace_back(std::make_unique<State>(q,shape,out,shard,fixtures));states.back()->correctness();}
    // All shapes' correctness gates and one full weight check pass before warming/timing.
    for(auto& state:states)state->weight_payload_check("after-correctness-before-timing");
    for(auto& state:states){state->warm();state->timing();}
    for(auto& state:states)state->weight_payload_check("after-timing-final");
    q.wait_and_throw();states.clear();q.wait_and_throw();need(!cleanup_error,"graph cleanup failure");
    text(out/"result.json","{\"status\":\"PASS\",\"scope\":\"S126_ONE_HEAD_RUNTIME_ONES_ONLY\",\"shapes\":1,\"paired_rounds_per_shape\":2,\"samples_per_round\":4,\"calls_per_graph\":20,\"public_original_bit_anchor\":true,\"candidate_full_bytes\":true,\"serving_qualified\":false,\"speed_qualified\":false}\n");return 0;
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IONBF,0);fs::path out;bool created=false;
    try{need(argc==5,"usage: q5_runtime_ones NEW_DIR REAL_SHARD S52_A1_SNAPSHOTS ONES_HEX");
        char* end=nullptr;const unsigned long word=std::strtoul(argv[4],&end,16);need(end&&*end==0&&word==0x01010101ul,"ones operand must be01010101");runtime_ones=int(word);out=argv[1];need(fs::create_directory(out),"new output directory required");created=true;return run(out,argv[2],argv[3]);}
    catch(const std::exception& e){const int rc=cleanup_error?2:(dynamic_cast<const Strict*>(&e)?5:2);
        std::fprintf(stderr,"S126 %s: %s\n",rc==5?"STRICT_FAILURE":"RUNTIME_OR_IO_FAILURE",e.what());
        if(created)try{text(out/"result.json","{\"status\":\"FAIL\",\"exit\":"+std::to_string(rc)+",\"kind\":\""+(rc==5?std::string("strict"):std::string("runtime_or_io"))+"\"}\n");}
        catch(const std::exception& io){std::fprintf(stderr,"S126 failure-result IO: %s\n",io.what());return 2;}return rc;}
}
