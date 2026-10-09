// Actual segmented source/tier probe; bounded subset, never a full serving gate.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/gguf_expert_source.hpp"
#include "strata/core/stage_mirror_budget.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"
#include "ggml.h"
#include "stage_mirror_q8_reference.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace cpu=strata::kernels::cpu;
namespace sycl_exp=sycl::ext::oneapi::experimental;
namespace {
constexpr int T=3,H=2560,FF=640,NE=512;
constexpr size_t guard=128;constexpr uint8_t sentinel=0xa5;
void need(bool b,const std::string& s){if(!b) throw std::runtime_error(s);}
std::string quoted(const std::string& s){std::string out="\"";char buf[8];for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=char(c);}else if(c<32||c>=127){std::snprintf(buf,sizeof(buf),"\\u%04x",c);out+=buf;}else out+=char(c);}return out+'"';}
std::string address(const void* p){char text[40];std::snprintf(text,sizeof(text),"%p",p);return quoted(text);}
std::string sha(const void* p,size_t n){unsigned char hash[32];unsigned count=0;auto ctx=std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(),EVP_MD_CTX_free);need(ctx && EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1 && EVP_DigestUpdate(ctx.get(),p,n)==1 && EVP_DigestFinal_ex(ctx.get(),hash,&count)==1 && count==32,"SHA failure");std::string out;char b[3];for(unsigned char v:hash){std::snprintf(b,sizeof(b),"%02x",v);out+=b;}return out;}
uint64_t fnv(const uint8_t* p,size_t n){uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
void mark(const char* event,const std::string& stage){std::fprintf(stderr,"UPLOAD_USM {\"event\":%s,\"stage\":%s%s}\n",quoted(event).c_str(),quoted(stage).c_str(),std::string(event)=="destroy_end" ? ",\"owning_destructor_returned\":true":"");std::fflush(stderr);}
void register_owner(const std::shared_ptr<strata::core::StageExpertMirror>& owner,const std::string& stage,int device){auto native=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(owner->q.get_context());auto emit=[&](const void* p,uint64_t bytes,const std::string& role){std::fprintf(stderr,"UPLOAD_USM {\"event\":\"owner_register\",\"stage\":%s,\"device\":%d,\"ze_context\":%s,\"pointer\":%s,\"bytes\":%llu,\"role\":%s}\n",quoted(stage).c_str(),device,address(native).c_str(),address(p).c_str(),(unsigned long long)bytes,quoted(role).c_str());};for(size_t i=0;i<owner->segments.size();++i) emit(owner->segments[i].base,owner->segments[i].bytes,"segment"+std::to_string(i));emit(owner->table,48*NE*8,"table");std::fflush(stderr);}
struct Buffer {
    sycl::queue& q;uint8_t* base=nullptr;size_t bytes;
    Buffer(sycl::queue& queue,size_t n,const void* original=nullptr):q(queue),bytes(n){base=sycl::malloc_device<uint8_t>(n+2*guard,q);need(base,"device allocation failed");std::vector<uint8_t> image(n+2*guard,sentinel);if(original) std::memcpy(image.data()+guard,original,n);q.memcpy(base,image.data(),image.size()).wait_and_throw();}
    ~Buffer(){if(base) sycl::free(base,q);}
    uint8_t* ptr(){return base+guard;}
    template<class U>U* as(){return reinterpret_cast<U*>(ptr());}
    std::vector<uint8_t> read(){std::vector<uint8_t> image(bytes+2*guard);q.memcpy(image.data(),base,image.size()).wait_and_throw();return image;}
    void check(const void* original=nullptr){auto image=read();need(std::all_of(image.begin(),image.begin()+guard,[](uint8_t b){return b==sentinel;}) && std::all_of(image.end()-guard,image.end(),[](uint8_t b){return b==sentinel;}),"device buffer guard changed");if(original) need(std::memcmp(image.data()+guard,original,bytes)==0,"immutable device buffer changed");}
    Buffer(const Buffer&)=delete;
};
struct Expert {int layer,expert;cpu::NativeFmt fmt;std::vector<uint8_t> original;std::string digest;};
std::vector<uint8_t> original_blob(const std::vector<std::unique_ptr<strata::GgufFile>>& files,int layer,int expert,const cpu::NativeFmt& fmt){std::vector<uint8_t> out(fmt.bytes);const char* roles[]={"gate","up","down"};const size_t off[]={0,fmt.up_off,fmt.down_off},sizes[]={fmt.up_off,fmt.up_off,fmt.bytes-fmt.down_off};for(int r=0;r<3;++r){const strata::TensorInfo* tensor=nullptr;const strata::GgufFile* file=nullptr;const std::string name="blk."+std::to_string(layer)+".ffn_"+roles[r]+"_exps.weight";for(const auto& f:files) if(auto* t=f->find(name)){need(!tensor,"duplicate original tensor");tensor=t;file=f.get();}need(tensor && tensor->shape==std::vector<uint64_t>{uint64_t(r==2 ? FF:H),uint64_t(r==2 ? H:FF),NE} && int(tensor->type)==(r==2 ? fmt.d_type:fmt.gu_type),"original expert tensor identity/shape/type");std::memcpy(out.data()+off[r],file->tensor_data(*tensor)+size_t(expert)*sizes[r],sizes[r]);}return out;}
struct References {std::vector<float> input,cpu_output,float_output,G,U,D,q_gate,q_up,q_hidden;std::vector<uint8_t> input_q8;};
References references(const Expert& e){References refs;refs.input.resize(T*H);for(size_t i=0;i<refs.input.size();++i) refs.input[i]=float(int((i*31+e.layer*17+e.expert*13)%251)-125)*0.007813f+0.000123f;refs.cpu_output.resize(T*H);refs.float_output.resize(T*H);
    const auto* gu=ggml_get_type_traits(ggml_type(e.fmt.gu_type));const auto* down=ggml_get_type_traits(ggml_type(e.fmt.d_type));need(gu && gu->to_float && down && down->to_float,"original dequantizer absent");refs.G.resize(FF*H);refs.U.resize(FF*H);refs.D.resize(H*FF);auto& G=refs.G;auto& U=refs.U;auto& D=refs.D;for(int r=0;r<FF;++r){gu->to_float(e.original.data()+size_t(r)*e.fmt.gu_row,G.data()+size_t(r)*H,H);gu->to_float(e.original.data()+e.fmt.up_off+size_t(r)*e.fmt.gu_row,U.data()+size_t(r)*H,H);}for(int r=0;r<H;++r) down->to_float(e.original.data()+e.fmt.down_off+size_t(r)*e.fmt.d_row,D.data()+size_t(r)*FF,FF);
    for(int t=0;t<T;++t){std::vector<float> hidden(FF);for(int r=0;r<FF;++r){double g=0,u=0;for(int c=0;c<H;++c){g+=double(G[size_t(r)*H+c])*refs.input[size_t(t)*H+c];u+=double(U[size_t(r)*H+c])*refs.input[size_t(t)*H+c];}hidden[r]=float(g/(1+std::exp(-g))*u);}for(int r=0;r<H;++r){double sum=0;for(int c=0;c<FF;++c) sum+=double(D[size_t(r)*FF+c])*hidden[c];refs.float_output[size_t(t)*H+r]=float(sum);}}
    std::vector<std::vector<uint8_t>> acts(T,std::vector<uint8_t>(cpu::kNativeActBytes)),hq(T,std::vector<uint8_t>(cpu::kNativeHBytes));std::vector<std::vector<float>> hidden(T,std::vector<float>(FF));const void* act[T];const void* quant_hidden[T];float* ff[T];float* outputs[T];for(int t=0;t<T;++t){cpu::native_quant_act(e.fmt,refs.input.data()+t*H,acts[t].data());act[t]=acts[t].data();ff[t]=hidden[t].data();outputs[t]=refs.cpu_output.data()+t*H;}cpu::native_gu_rows(e.fmt,e.original.data(),act,T,ff,0,FF);for(int t=0;t<T;++t){cpu::native_quant_h(e.fmt,hidden[t].data(),hq[t].data());quant_hidden[t]=hq[t].data();}cpu::native_down_rows(e.fmt,e.original.data(),quant_hidden,T,outputs,0,H);
    refs.input_q8=mirror_reference::q8_1(refs.input);refs.q_gate.resize(T*FF);refs.q_up.resize(T*FF);refs.q_hidden.resize(T*FF);
    for(int t=0;t<T;++t) for(int r=0;r<FF;++r){const uint8_t* xq=refs.input_q8.data()+size_t(t)*H/32*36;double g=mirror_reference::dot(G.data()+size_t(r)*H,e.original.data()+size_t(r)*e.fmt.gu_row,e.fmt.gu_type,xq,H);double u=mirror_reference::dot(U.data()+size_t(r)*H,e.original.data()+e.fmt.up_off+size_t(r)*e.fmt.gu_row,e.fmt.gu_type,xq,H);refs.q_gate[size_t(t)*FF+r]=float(g);refs.q_up[size_t(t)*FF+r]=float(u);refs.q_hidden[size_t(t)*FF+r]=float(g/(1+std::exp(-g))*u);}
    return refs;}
double l1_relative(const std::vector<float>& values,const std::vector<float>& ref){double numerator=0,denominator=0;for(size_t i=0;i<values.size();++i){need(std::isfinite(values[i]) && std::isfinite(ref[i]),"nonfinite expert arithmetic");numerator+=std::abs(double(values[i])-ref[i]);denominator+=std::abs(double(ref[i]));}return numerator/std::max(1e-30,denominator);}
class MirrorBytesFromActualPlan;
struct Probe {
    std::shared_ptr<strata::core::StageExpertMirror> owner;sycl::queue q;Expert e;References refs;
    std::vector<int32_t> residency,ids;
    Buffer res,ids_device,plan,errors,cache,input,quant,scratch,bytes,output;
    std::unique_ptr<sycl_exp::command_graph<sycl_exp::graph_state::executable>> graph;
    Probe(std::shared_ptr<strata::core::StageExpertMirror> o,const Expert& expert):owner(o),q(o->q),e(expert),refs(references(e)),residency(48*NE,-1),ids(T,e.expert),
        res(q,residency.size()*4,residency.data()),ids_device(q,ids.size()*4,ids.data()),plan(q,512),errors(q,4),cache(q,e.original.size(),e.original.data()),input(q,refs.input.size()*4,refs.input.data()),quant(q,size_t(T)*H/32*36),scratch(q,strata::kernels::native_expert_scratch_bytes(T,FF)),bytes(q,e.original.size()),output(q,T*H*4) {}
    void submit(bool mirrored){
        q.memset(errors.ptr(),0,4);
        const auto* layer=mirrored ? owner->layer_table(e.layer):nullptr;
        strata::kernels::resident_plan_bound(ids_device.as<int32_t>(),T,1,res.as<int32_t>()+e.layer*NE,NE,cache.ptr(),nullptr,e.original.size(),plan.as<int32_t>(),T,nullptr,0,&q,layer,errors.as<uint32_t>());
        // The consumer dereferences the pointer emitted by actual resident_plan.
        int32_t* pl=plan.as<int32_t>();uint32_t* err=errors.as<uint32_t>();uint8_t* destination=bytes.ptr();size_t n=e.original.size();constexpr size_t ptr_offset=((4+(T+1)+2*T)+1)&~size_t(1);
        q.parallel_for<MirrorBytesFromActualPlan>(sycl::range<1>(n),[=](sycl::id<1> index){if(pl[0]==1 && *err==0){auto* pointers=reinterpret_cast<unsigned long long*>(pl+ptr_offset);destination[index[0]]=reinterpret_cast<const uint8_t*>(pointers[0])[index[0]];}});
        strata::kernels::quantize_q8_1_rows(input.as<float>(),T,H,quant.ptr(),&q);
        auto layout=strata::kernels::native_expert_layout(e.fmt.gu_type,e.fmt.d_type,H,FF);
        strata::kernels::native_expert_grouped(layout,reinterpret_cast<unsigned long long*>(pl+ptr_offset),pl+4,pl,pl+4+(T+1),pl+4+(T+1)+T,1,T,quant.ptr(),scratch.ptr(),output.as<float>(),&q);
    }
    std::vector<float> result(){auto image=output.read();std::vector<float> out(T*H);std::memcpy(out.data(),image.data()+guard,out.size()*4);return out;}
    void check(){for(auto* b:{&res,&ids_device,&plan,&errors,&cache,&input,&quant,&scratch,&bytes,&output}) b->check();res.check(residency.data());ids_device.check(ids.data());cache.check(e.original.data());input.check(refs.input.data());uint32_t err=9;q.memcpy(&err,errors.ptr(),4).wait_and_throw();need(err==0,"actual resident plan rejected selected expert");auto image=bytes.read();need(std::memcmp(image.data()+guard,e.original.data(),e.original.size())==0 && sha(image.data()+guard,e.original.size())==e.digest,"actual GPU-consumed pointer bytes/sourceSHA mismatch");}
    std::vector<float> scratch_values(size_t byte_offset,size_t floats){auto image=scratch.read();need(byte_offset+floats*4<=scratch.bytes,"scratch reference extent");std::vector<float> values(floats);std::memcpy(values.data(),image.data()+guard+byte_offset,floats*4);return values;}
    std::vector<mirror_reference::Metric> arithmetic(){
        auto xq=quant.read();need(std::memcmp(xq.data()+guard,refs.input_q8.data(),refs.input_q8.size())==0,"independent input Q8_1 rounding/scale/sum/codes differ");
        const size_t fa=(size_t(T)*FF*4+255)&~size_t(255);auto gate=scratch_values(0,T*FF),up=scratch_values(fa,T*FF),hidden=scratch_values(2*fa,T*FF);
        auto hidden_q8=mirror_reference::q8_1(hidden);auto image=scratch.read();need(std::memcmp(image.data()+guard+3*fa,hidden_q8.data(),hidden_q8.size())==0,"independent hidden Q8_1 rounding/scale/sum/codes differ");
        std::vector<float> down_reference(T*H);for(int t=0;t<T;++t) for(int r=0;r<H;++r) down_reference[size_t(t)*H+r]=float(mirror_reference::dot(refs.D.data()+size_t(r)*FF,e.original.data()+e.fmt.down_off+size_t(r)*e.fmt.d_row,e.fmt.d_type,hidden_q8.data()+size_t(t)*FF/32*36,FF));
        std::vector<mirror_reference::Metric> metrics={mirror_reference::metric(gate,refs.q_gate),mirror_reference::metric(up,refs.q_up),mirror_reference::metric(hidden,refs.q_hidden),mirror_reference::metric(result(),down_reference)};
        for(const auto& metric:metrics) need(metric.finite && metric.nmse<=1e-6 && metric.linf<=1e-4,"strict quantized-stage implementation numeric oracle failed");
        return metrics;
    }
    void capture(){dpct::experimental::begin_recording(&q);submit(true);dpct::experimental::command_graph_ptr recorded=nullptr;dpct::experimental::end_recording(&q,&recorded);need(recorded!=nullptr,"graph end returned no captured graph");graph=std::make_unique<sycl_exp::command_graph<sycl_exp::graph_state::executable>>(recorded->finalize());delete recorded;}
    void replay(){need(bool(graph),"captured graph absent");q.ext_oneapi_graph(*graph).wait_and_throw();}
    std::vector<float> resident(){residency[size_t(e.layer)*NE+e.expert]=0;q.memcpy(res.ptr(),residency.data(),residency.size()*4).wait_and_throw();submit(false);q.wait_and_throw();check();auto value=result();residency[size_t(e.layer)*NE+e.expert]=-1;q.memcpy(res.ptr(),residency.data(),residency.size()*4).wait_and_throw();return value;}
    void missing_negative(){auto negative=ids;std::fill(negative.begin(),negative.end(),511);q.memcpy(ids_device.ptr(),negative.data(),negative.size()*4).wait_and_throw();q.memset(errors.ptr(),0,4);strata::kernels::resident_plan_bound(ids_device.as<int32_t>(),T,1,res.as<int32_t>()+e.layer*NE,NE,cache.ptr(),nullptr,e.original.size(),plan.as<int32_t>(),T,nullptr,0,&q,owner->layer_table(e.layer),errors.as<uint32_t>());q.wait_and_throw();uint32_t err=0;int32_t counts[3];q.memcpy(&err,errors.ptr(),4).wait_and_throw();q.memcpy(counts,plan.ptr(),12).wait_and_throw();need(err==1 && counts[0]==0 && counts[1]==0 && counts[2]==0,"missing mirror entry silently fell back");q.memcpy(ids_device.ptr(),ids.data(),ids.size()*4).wait_and_throw();}
};
}
int main(int argc,char** argv){
    try{
        std::string pack,output,mode;std::vector<std::string> paths;
        for(int i=1;i<argc;++i){std::string arg=argv[i];need(i+1<argc,"missing argument");std::string value=argv[++i];if(arg=="--pack") pack=value;else if(arg=="--output") output=value;else if(arg=="--mode") mode=value;else if(arg=="--shard") paths.push_back(value);else throw std::runtime_error("unknown argument");}
        need(!pack.empty() && !output.empty() && paths.size()==4 && (mode=="single" || mode=="pair"),"pack/output/fourshards/mode required");need(!fs::exists(output),"new output required");std::ofstream report(output);need(bool(report),"cannot open report");report<<std::setprecision(17);report<<"{\"schema\":1,\"scope\":\"bounded actual segmented expert source/tier/arithmetic subset; full serving unqualified\",\"stages\":[";report.flush();
        std::string error;need(cpu::expert_layout_load(pack,48,NE,error),error);const auto& layout=cpu::expert_layout();std::vector<std::unique_ptr<strata::GgufFile>> files;for(const auto& path:paths) files.emplace_back(std::make_unique<strata::GgufFile>(path));
        std::vector<Expert> experts;for(int layer:{0,2,11,32,47}) for(int expert:{0,17}){auto fmt=layout.fmt[size_t(layer)];auto original=original_blob(files,layer,expert,fmt);std::string digest=sha(original.data(),original.size());experts.push_back({layer,expert,fmt,std::move(original),digest});}
        dpct::select_device(0);auto& q0=dpct::get_in_order_queue();
        // Real partial read failure occurs after segment allocations, without ever
        // exposing or submitting invalid source bytes to a GPU kernel.
        fs::path bad=fs::path(output).parent_path()/"negative-source";need(!fs::exists(bad),"new negative source directory required");fs::create_directory(bad);std::string broken=fs::path(paths[1]).filename();for(const auto& path:paths){fs::path dst=bad/fs::path(path).filename();if(dst.filename()==broken){std::ofstream empty(dst);}else fs::create_symlink(path,dst);}
        {strata::core::GgufExpertSource failing;need(failing.open((bad/fs::path(paths[0]).filename()).string(),48,NE,error),error);auto failed=failing.mirror_stage({{0,0},{0,17},{2,0},{2,17}},64ull<<20,1,0,32,q0,error,4ull<<20);need(!failed && failing.mirrored_bytes()==0 && !failing.pinned(0,0),"partial source read failure published mirror state");failing.close();}fs::remove_all(bad);
        strata::core::StageMirrorBudget budget;const uint64_t reserves[]={0};need(strata::core::StageMirrorBudget::make(64ull<<20,64ull<<20,reserves,1,0,budget),"fixture budget creation");need(!budget.covers_all({64ull<<20,1}),"global shortage negative accepted");
        strata::core::GgufExpertSource source;need(source.open(paths[0],48,NE,error),error);auto invalid=source.mirror_stage({{0,0}},16ull<<20,1,0,32,q0,error,1ull<<20);need(!invalid && source.mirrored_bytes()==0,"too-small segment negative accepted");
        std::vector<std::shared_ptr<strata::core::StageExpertMirror>> owners;std::vector<std::unique_ptr<Probe>> probes;std::set<std::pair<int,int>> formats;size_t owner_records=0;
        for(int stage=0;stage<2;++stage){int device=mode=="pair" ? stage:0;dpct::select_device(device);auto& queue=dpct::get_in_order_queue();int lo=stage ? 32:0,hi=stage ? 48:32;std::vector<std::pair<int64_t,int64_t>> pairs;uint64_t required=0;for(const auto& e:experts) if(e.layer>=lo && e.layer<hi){pairs.push_back({e.layer,e.expert});required+=(e.original.size()+255)/256*256;}
            auto owner=source.mirror_stage(pairs,budget.remaining(),1,lo,hi,queue,error,4ull<<20);need(owner && owner->segments.size()>1 && owner->bytes==required && budget.commit(owner->bytes),error);owners.push_back(owner);std::string id="mirror-"+std::to_string(lo)+":"+std::to_string(hi)+":"+std::to_string(device);register_owner(owner,id,device);owner_records+=owner->segments.size()+1;
            std::vector<unsigned long long> table(48*NE);queue.memcpy(table.data(),owner->table,table.size()*8).wait_and_throw();std::set<size_t> wanted;for(const auto& pair:pairs) wanted.insert(size_t(pair.first*NE+pair.second));for(size_t i=0;i<table.size();++i) need(bool(table[i])==bool(wanted.count(i)),"foreign/unselected pointer-table entry present");need(!owner->layer_table(stage ? 0:47),"foreign-stage layer lookup accepted");
            if(stage) report<<',';report<<"{\"lo\":"<<lo<<",\"hi\":"<<hi<<",\"device\":"<<device<<",\"segments\":"<<owner->segments.size()<<",\"bytes\":"<<owner->bytes<<",\"experts\":"<<pairs.size()<<"}";report.flush();
            for(const auto& e:experts) if(e.layer>=lo && e.layer<hi){need(source.pinned(e.layer,e.expert),"selected expert not pinned");std::vector<uint8_t> got(e.original.size());need(source.read_into(e.layer,e.expert,got.data(),got.size()) && got==e.original,"mirrored host read differs from originalGGUF");need(std::memcmp(source.blob(e.layer,e.expert),e.original.data(),got.size())==0,"mirrored blob differs from source");need(owner->source_fnv64[size_t(e.layer*NE+e.expert)]==fnv(e.original.data(),e.original.size()),"original source checksum differs");auto probe=std::make_unique<Probe>(owner,e);formats.insert({e.fmt.gu_type,e.fmt.d_type});probe->missing_negative();probe->capture();probe->replay();probe->check();auto mirrored=probe->result();probe->arithmetic();auto resident=probe->resident();need(std::memcmp(mirrored.data(),resident.data(),mirrored.size()*4)==0,"mirrored versus resident expert arithmetic bytes differ");need(l1_relative(mirrored,probe->refs.float_output)<=3e-2 && l1_relative(probe->refs.cpu_output,probe->refs.float_output)<=3e-2 && l1_relative(mirrored,probe->refs.cpu_output)<=3e-2,"upstream preregistered expert numeric oracle failed");probe->replay();probe->check();need(probe->result()==mirrored,"captured expert repeat differs");probes.push_back(std::move(probe));}
        }
        report<<"],\"expert_rows\":[";
        for(size_t index=0;index<probes.size();++index){auto& p=*probes[index];auto metrics=p.arithmetic();auto out=p.result();if(index) report<<',';report<<"{\"layer\":"<<p.e.layer<<",\"expert\":"<<p.e.expert<<",\"gu_type\":"<<p.e.fmt.gu_type<<",\"down_type\":"<<p.e.fmt.d_type<<",\"source_sha256\":"<<quoted(p.e.digest)<<",\"raw_bytes\":"<<p.e.original.size()<<",\"gpu_vs_float_l1_relative\":"<<l1_relative(out,p.refs.float_output)<<",\"cpu_vs_float_l1_relative\":"<<l1_relative(p.refs.cpu_output,p.refs.float_output)<<",\"gpu_vs_native_cpu_l1_relative\":"<<l1_relative(out,p.refs.cpu_output)<<",\"input_q8_bytes_exact\":true,\"hidden_q8_bytes_exact\":true,\"implementation_metrics\":[";for(size_t stage=0;stage<metrics.size();++stage){if(stage) report<<',';report<<"{\"stage\":"<<quoted(stage==0?"gate":stage==1?"up":stage==2?"silu_hidden":"down_matched_q8")<<",\"nmse\":"<<metrics[stage].nmse<<",\"normalized_linf\":"<<metrics[stage].linf<<'}';}report<<"]}";}report.flush();
        need(formats==std::set<std::pair<int,int>>{{12,7},{12,8},{13,8}},"selected original mixed quant layout coverage");need(source.storage_reads()==experts.size() && source.mirror_host_reads()==2*experts.size(),"storage versus mirror read counters differ");uint64_t storage=source.storage_reads(),host_reads=source.mirror_host_reads();need(source.mirrored_bytes()==budget.used,"global/source byte counter mismatch");
        // Check actual bind/context association without any invalid device read.
        for(size_t stage=0;stage<owners.size();++stage){auto& owner=owners[stage];auto& p=*probes[stage==0 ? 0:6];need(owner->bind(p.res.as<int32_t>(),48,NE,p.q,error),error);need(!owner->bind(p.ids_device.as<int32_t>(),48,NE,p.q,error),"different residency binding accepted");if(mode=="pair") need(!owner->matches(p.res.as<int32_t>(),48,NE,owners[1-stage]->q),"foreign context/device binding accepted");}
        source.close();owners.clear();
        for(auto& p:probes){auto before=p->result();p->replay();p->check();need(p->result()==before,"graph lost owner/source bytes after sourceclose");}
        // Retire every graph/probe before the final shared mirror owner release.
        std::map<std::string,std::shared_ptr<strata::core::StageExpertMirror>> retiring;for(auto& p:probes){auto o=p->owner;std::string id="mirror-"+std::to_string(o->lb)+":"+std::to_string(o->le)+":"+std::to_string(mode=="pair" && o->lb==32 ? 1:0);retiring[id]=o;p->q.wait_and_throw();p->graph.reset();}probes.clear();
        for(auto& pair:retiring){mark("destroy_begin",pair.first);pair.second->release();pair.second.reset();mark("destroy_end",pair.first);}retiring.clear();
        report<<"],\"expert_cases\":10,\"tokens_per_expert\":3,\"graph_replays_after_source_close\":10,\"mixed_quant_layouts\":3,\"mirrored_padded_bytes\":"<<budget.used<<",\"source_storage_reads\":"<<storage<<",\"mirror_host_reads\":"<<host_reads<<",\"registered_owner_records\":"<<owner_records<<",\"partial_read_failure_cleanup_returned\":true,\"missing_entry_plan_negative_checks\":10,\"source_and_arithmetic_passed\":true,\"all_owners_release_returned\":true,\"full_serving_qualified\":false}\n";report.flush();return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"MIRROR_ORACLE FAIL: %s\n",e.what());return 2;}
}
