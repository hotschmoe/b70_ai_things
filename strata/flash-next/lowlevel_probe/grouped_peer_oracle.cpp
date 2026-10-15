#define DPCT_PROFILING_ENABLED
#include "strata/artifact/gguf_reader.hpp"
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

namespace pf = strata::prefill;
namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int64_t N = 2560, FF = 640, C = 12288, K = 10;
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
struct Layer { int number; cpu::NativeFmt f; const strata::TensorInfo* t[3]; };
struct Expert { int layer, expert; cpu::NativeFmt f; size_t blob; int64_t offset, ne; };
struct Drain { sycl::queue& q; ~Drain() { try { q.wait_and_throw(); } catch (...) {} } };
struct Timing { double enqueue, drain, complete, device_span; };
struct GateFailure : std::runtime_error { using std::runtime_error::runtime_error; };


// First 16 experts from the unchanged S35 mean-80 fixture.
static int run(int argc, char** argv) {
    require((argc == 4 || argc == 5) && std::string(argv[2]) == "--out", "usage: grouped_peer_oracle shard1.gguf --out EXISTING_DIR [--timed]");
    const bool timed = argc == 5 && std::string(argv[4]) == "--timed";
    require(argc == 4 || timed, "unknown option");
    const std::filesystem::path out(argv[3]);
    require(std::filesystem::is_directory(out), "output directory must exist");
    require(std::filesystem::is_empty(out), "output directory must be fresh and empty");
    const int mean = 80;
    const int selected = 363;
    const int64_t T = 4096;
    strata::GgufFile gguf(argv[1]);
    std::map<std::pair<int,int>, Layer> formats;
    for (int l = 0; l < 48; ++l) {
        Layer a{}; a.number = l;
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& t : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (t.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") a.t[r] = &t;
        require(a.t[0] && a.t[1] && a.t[2], "expert tensor missing from shard1");
        std::string err;
        require(a.t[0]->type == a.t[1]->type, "gate/up format mismatch");
        for (int r = 0; r < 3; ++r) {
            const auto& dims = a.t[r]->shape;
            require(dims.size() == 3 && dims[2] == 512, "bad expert tensor rank/count");
            require(dims[0] == uint64_t(r == 2 ? FF : N) && dims[1] == uint64_t(r == 2 ? N : FF), "bad expert tensor shape");
        }
        if (!cpu::native_fmt(int(a.t[0]->type), int(a.t[2]->type), N, FF, a.f, err)) throw std::runtime_error(err);
        formats.emplace(std::make_pair(a.f.gu_type, a.f.d_type), a);
    }
    require(formats.size() == 10, "expected ten real format pairs");
    for (int gt : {16,17,18,21,22}) for (int dt : {20,42})
        require(formats.count({gt,dt}) == 1, "expected real quant format pair absent");
    std::vector<Layer> layers;
    for (const auto& a : formats) layers.push_back(a.second);
    std::vector<int64_t> sizes(selected, 1);
    const int64_t fixture_rows = int64_t(selected) * mean;
    sizes[0] = 8 * mean + 1; sizes[1] = 1;
    int64_t left = fixture_rows - sizes[0] - 1 - (selected - 2);
    int64_t weight_sum = 0;
    for (int j = 2; j < selected; ++j) weight_sum += 1 + (j * 17) % 7;
    int64_t assigned = 0;
    for (int j = 2; j < selected; ++j) {
        const int64_t add = left * (1 + (j * 17) % 7) / weight_sum;
        sizes[j] += add; assigned += add;
    }
    for (int j = 2; assigned < left; ++j, ++assigned) ++sizes[j];
    require(*std::max_element(sizes.begin(), sizes.end()) == sizes[0] && sizes[1] == 1, "skew fixture failed");
    std::vector<Expert> experts;
    std::vector<uint8_t> blobs;
    int64_t offset = 0, max_ne = 0;
    for (int j = 0; j < 16; ++j) {
        const auto& a = layers[j % layers.size()];
        const int e = (j * 17 + 7) % 512;
        const size_t pos = (blobs.size() + 255) & ~size_t(255);
        blobs.resize(pos + a.f.bytes, 0xC3);
        std::memcpy(blobs.data() + pos, gguf.tensor_data(*a.t[0]) + size_t(e) * a.f.up_off, a.f.up_off);
        std::memcpy(blobs.data() + pos + a.f.up_off, gguf.tensor_data(*a.t[1]) + size_t(e) * a.f.up_off, a.f.up_off);
        std::memcpy(blobs.data() + pos + a.f.down_off, gguf.tensor_data(*a.t[2]) + size_t(e) * (a.f.bytes - a.f.down_off), a.f.bytes - a.f.down_off);
        experts.push_back({a.number, e, a.f, pos, offset, sizes[j]});
        offset += sizes[j]; max_ne = std::max(max_ne, sizes[j]);
        std::printf("{\"expert\":%d,\"layer\":%d,\"physical_expert\":%d,\"gu\":%d,\"down\":%d,\"offset\":%lld,\"ne\":%lld}\n", j, a.number, e, a.f.gu_type, a.f.d_type, (long long) experts.back().offset, (long long) sizes[j]);
    }
    const int64_t rows = offset;
    require(rows > 0 && rows <= T && max_ne <= T && sizes[0] == 641 && sizes[1] == 1, "compact wave capacity/fixture failed");
    std::map<std::pair<int,int>,int> used_formats;
    for (const auto& e : experts) ++used_formats[{e.f.gu_type,e.f.d_type}];
    require(used_formats.size() == 10, "compact wave must use ten actual format pairs");
    require(dpct::device_count() == 1, "need exactly one exposed GPU");
    const sycl::device dev = dpct::dev_mgr::instance().get_device(0);
    require(dev.has(sycl::aspect::ext_intel_pci_address), "PCI address unavailable");
    std::string pci = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    require(pci == "0000:44:00.0", "require physical card1 PCI0000:44:00.0");
    require(std::getenv("STRATA_SYCL_GEMM_STAGE") == nullptr, "do not force scalar-only operand staging");
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(), dev,
                  {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    require(q.is_in_order(), "in-order queue required");
    std::vector<uint16_t> source(size_t(T * N));
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = uint16_t(0x2800 + (i * 13 + i / N * 17) % 1024) | uint16_t((i % 3 == 0) ? 0x8000 : 0);
    std::vector<int32_t> src(size_t(rows), 0);
    for (int64_t i = 0; i < rows; ++i) src[size_t(i)] = int32_t((i * 37 + i / 7 * 13 + 17) % T);
    std::vector<uint16_t> actual_x(size_t(rows*N));
    constexpr size_t dqgu_bytes = size_t(1280 * N) * 2, dqd_bytes = size_t(N * FF) * 2;
    const size_t gu_bytes = size_t(T * 1280) * 4, h_bytes = size_t(T * FF) * 2;
    const size_t out_bytes = size_t(T * N) * 4;
    const size_t gu_used = size_t(rows * 1280) * 4, h_used = size_t(rows * FF) * 2, out_used = size_t(rows * N) * 4;
    Buffer weights(q, blobs.size(), "weights"), input(q, source.size() * 2, "source"), ids(q, src.size() * 4, "ids");
    Buffer refout(q,out_bytes,"scalar_down"), gout(q,out_bytes,"group_down_alias");
    Buffer sg(q,gu_bytes,"scalar_GU_scratch"), sh(q,h_bytes,"scalar_H_scratch");
    Buffer rg(q,gu_bytes,"scalar_GU_saved"), rh(q,h_bytes,"scalar_H_saved");
    Buffer gg(q,gu_bytes,"group_GU"), gh(q,h_bytes,"group_H");
    // One 16-slot arena, with an independent guarded allocation for each matrix.
    std::vector<std::unique_ptr<Buffer>> dqgu, dqd;
    for (size_t i=0;i<experts.size();++i) {
        dqgu.emplace_back(new Buffer(q,dqgu_bytes,"dq_gu_slot"));
        dqd.emplace_back(new Buffer(q,dqd_bytes,"dq_down_slot"));
    }
    std::vector<Buffer*> guards{&weights,&input,&ids,&refout,&gout,&sg,&sh,&rg,&rh,&gg,&gh};
    for (auto& b:dqgu) guards.push_back(b.get());
    for (auto& b:dqd) guards.push_back(b.get());
    pf::Gemm gemm; std::string err;
    if (!gemm.init(&q,1<<20,err)) throw std::runtime_error(err);
    // All metadata is USM-shared, same context, immutable from here through completion.
    // Separate records for GU and down; no mutation of outstanding descriptors.
    struct Group {
        oneapi::mkl::transpose ta[16],tb[16];
        int64_t m[16],n[16],k[16],lda[16],ldb[16],ldc[16],count[16];
        float alpha[16],beta[16];
        const sycl::half* a[16]; const sycl::half* b[16]; float* c[16];
    };
    struct Metadata {
        sycl::queue& q; uint8_t* base; Group* p;
        explicit Metadata(sycl::queue& queue):q(queue),base(static_cast<uint8_t*>(sycl::aligned_alloc_shared(GUARD,2*sizeof(Group)+2*GUARD,q))),p(nullptr) {
            require(base!=nullptr,"metadata allocation failed");
            std::memset(base,0xD7,2*sizeof(Group)+2*GUARD);
            p=reinterpret_cast<Group*>(base+GUARD); std::construct_at(p); std::construct_at(p+1);
        }
        void check() const {
            q.wait_and_throw();
            for(size_t i=0;i<GUARD;++i)require(base[i]==0xD7 && base[GUARD+2*sizeof(Group)+i]==0xD7,"metadata guard corruption");
        }
        ~Metadata(){try{q.wait_and_throw();sycl::free(base,q);}catch(...){}}
    } metadata(q);
    Group& gu=metadata.p[0]; Group& down=metadata.p[1];
    const auto metadata_start=Clock::now();
    for(size_t i=0;i<experts.size();++i){
        const auto& e=experts[i];
        for(Group* a:{&gu,&down}){
            a->ta[i]=oneapi::mkl::transpose::trans; a->tb[i]=oneapi::mkl::transpose::nontrans;
            a->n[i]=e.ne; a->count[i]=1; a->alpha[i]=1.0f; a->beta[i]=0.0f;
        }
        gu.m[i]=1280; gu.k[i]=N; gu.lda[i]=N; gu.ldb[i]=N; gu.ldc[i]=1280;
        gu.a[i]=reinterpret_cast<const sycl::half*>(dqgu[i]->data());
        gu.b[i]=reinterpret_cast<const sycl::half*>(gout.as<uint16_t>()+e.offset*N);
        gu.c[i]=gg.as<float>()+e.offset*1280;
        down.m[i]=N; down.k[i]=FF; down.lda[i]=FF; down.ldb[i]=FF; down.ldc[i]=N;
        down.a[i]=reinterpret_cast<const sycl::half*>(dqd[i]->data());
        down.b[i]=reinterpret_cast<const sycl::half*>(gh.as<uint16_t>()+e.offset*FF);
        down.c[i]=gout.as<float>()+e.offset*N;
    }
    const auto metadata_end=Clock::now();
    std::array<uint8_t,2*sizeof(Group)> metadata_snapshot{};
    std::memcpy(metadata_snapshot.data(),metadata.p,metadata_snapshot.size());
    Drain drain{q};
    q.memcpy(weights.data(),blobs.data(),blobs.size());
    q.memcpy(input.data(),source.data(),source.size()*2);
    q.memcpy(ids.data(),src.data(),src.size()*4);
    auto dq=[&](size_t ei,size_t slot){
        const auto& e=experts[ei];const auto* b=weights.as<uint8_t>()+e.blob;
        strata::kernels::iq_dequant_expert_f16(e.f.gu_type,e.f.d_type,b,b+e.f.up_off,b+e.f.down_off,
                                              FF,N,dqgu[slot]->as<uint16_t>(),dqd[slot]->as<uint16_t>(),&q);
    };
    auto gather=[&](Buffer& dest){pf::gather_rows16(input.as<uint16_t>(),ids.as<int32_t>(),dest.as<uint16_t>(),rows,N,&q);};
    auto scalar=[&](bool save){
        gather(refout);
        if(save)q.memcpy(actual_x.data(),refout.data(),actual_x.size()*2);
        for(size_t z=experts.size();z>0;--z){
            const auto& e=experts[z-1];dq(z-1,0);
            gemm.f16(refout.as<uint16_t>()+e.offset*N,dqgu[0]->as<uint16_t>(),sg.as<float>(),e.ne,1280,N);
            pf::swiglu_interleaved(sg.as<float>(),sh.as<uint16_t>(),e.ne,&q);
            gemm.f16(sh.as<uint16_t>(),dqd[0]->as<uint16_t>(),refout.as<float>()+e.offset*N,e.ne,N,FF);
            if(save){
                q.memcpy(rg.as<float>()+e.offset*1280,sg.data(),size_t(e.ne*1280)*4);
                q.memcpy(rh.as<uint16_t>()+e.offset*FF,sh.data(),size_t(e.ne*FF)*2);
            }
        }
    };
    auto batch=[&](Group& a){
        return oneapi::mkl::blas::column_major::gemm_batch(q,a.ta,a.tb,a.m,a.n,a.k,a.alpha,a.a,a.lda,
                a.b,a.ldb,a.beta,a.c,a.ldc,int64_t(experts.size()),a.count,oneapi::mkl::blas::compute_mode::unset);
    };
    auto group_prepare=[&](){gather(gout);for(size_t i=0;i<experts.size();++i)dq(i,i);};
    auto group_full=[&](){
        group_prepare();
        const auto gu_done=batch(gu);q.ext_oneapi_submit_barrier({gu_done});
        pf::swiglu_interleaved(gg.as<float>(),gh.as<uint16_t>(),rows,&q);
        const auto down_done=batch(down);q.ext_oneapi_submit_barrier({down_done});
    };
    const uint8_t poison=0xA5;
    for(Buffer* b:{&refout,&gout,&sg,&sh,&rg,&rh,&gg,&gh})q.memset(b->data(),poison,b->bytes);
    scalar(true);
    std::vector<float> refgu(size_t(rows*1280)),refdown(size_t(rows*N));
    std::vector<uint16_t> refh(size_t(rows*FF));
    q.memcpy(refgu.data(),rg.data(),gu_used);q.memcpy(refh.data(),rh.data(),h_used);
    q.memcpy(refdown.data(),refout.data(),out_used).wait_and_throw();
    std::vector<float> candgu,canddown,last_scalar_down;std::vector<uint16_t> candh;
    bool timing_executed=false;
    auto close_file=[](std::ofstream& f){
        f.flush();require(bool(f),"export flush failed");
        f.close();require(!f.fail(),"export close failed");
    };
    auto write=[&](const char* name,const void* data,size_t bytes){
        const auto path=out/name;
        require(!std::filesystem::exists(path),"refuse export overwrite");
        std::ofstream f(path,std::ios::binary);require(bool(f),"export open failed");
        f.write(static_cast<const char*>(data),std::streamsize(bytes));require(bool(f),"export write failed");
        close_file(f);
    };
    auto text=[&](const char* name,const std::string& value){write(name,value.data(),value.size());};
    auto dump=[&](const char* status,const char* stage,size_t first,const char* category,const std::string& reason){
        // Compact exact bytes on PASS as well as the first failed gate.
        write("weights.bin",blobs.data(),blobs.size());write("source.fp16",source.data(),source.size()*2);
        write("source_ids.i32",src.data(),src.size()*4);write("actual_gathered.fp16",actual_x.data(),actual_x.size()*2);
        write("reference_GU.f32",refgu.data(),gu_used);write("reference_H.fp16",refh.data(),h_used);
        write("reference_down.f32",refdown.data(),out_used);
        if(!candgu.empty())write("candidate_GU.f32",candgu.data(),candgu.size()*4);
        if(!candh.empty())write("candidate_H.fp16",candh.data(),candh.size()*2);
        if(!canddown.empty())write("candidate_down.f32",canddown.data(),canddown.size()*4);
        if(!last_scalar_down.empty())write("timed_scalar_down.f32",last_scalar_down.data(),last_scalar_down.size()*4);
        write("metadata_expected.bin",metadata_snapshot.data(),metadata_snapshot.size());
        write("metadata_actual.bin",metadata.p,metadata_snapshot.size());
        std::string fixtures="index\tlayer\texpert\tgu_type\tdown_type\tne\trow_offset\tblob_offset\tblob_bytes\tup_offset\tdown_offset\n";
        for(size_t i=0;i<experts.size();++i){const auto&e=experts[i];
            fixtures+=std::to_string(i)+"\t"+std::to_string(e.layer)+"\t"+std::to_string(e.expert)+"\t"+
                std::to_string(e.f.gu_type)+"\t"+std::to_string(e.f.d_type)+"\t"+std::to_string(e.ne)+"\t"+
                std::to_string(e.offset)+"\t"+std::to_string(e.blob)+"\t"+std::to_string(e.f.bytes)+"\t"+
                std::to_string(e.f.up_off)+"\t"+std::to_string(e.f.down_off)+"\n";
        }
        text("fixtures.tsv",fixtures);text("model-path.txt",std::string(argv[1])+"\n");
        text("reason.txt",reason+"\n");
        const bool pass=std::strcmp(status,"PASS")==0;
        const std::string result=std::string("{\"status\":\"")+status+"\",\"category\":\""+category+
            "\",\"stage\":\""+stage+"\",\"first_element\":"+std::to_string(first)+
            ",\"native_exit_code\":"+(pass?"0":"5")+",\"timing_run\":"+(timing_executed?"true":"false")+
            ",\"rows\":"+std::to_string(rows)+",\"numerical_policy\":\"GU_H_down_bitwise\"}\n";
        text("result.json",result);
        std::fprintf(pass?stdout:stderr,"%s stage=%s first=%zu category=%s; exports=%s\n",status,stage,first,category,out.c_str());
    };
    auto finite=[&](const auto& a,const char* what){
        for(size_t i=0;i<a.size();++i){
            float v;
            if constexpr(std::is_same_v<typename std::decay_t<decltype(a)>::value_type,uint16_t>){
                sycl::half h;std::memcpy(&h,&a[i],2);v=float(h);
            }else v=float(a[i]);
            if(!std::isfinite(v)){dump("FAIL",what,i,"NONFINITE","nonfinite stage output");throw GateFailure("nonfinite stage output");}
        }
    };
    auto compare=[&](const auto& a,const auto& b,const char* stage){
        require(a.size()==b.size(),"comparison shape mismatch");
        for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(a[i]))){
            dump("FAIL",stage,i,"NUMERICAL_MISMATCH","strict bit mismatch");return false;
        }
        return true;
    };
    auto check=[&](const char* stage){
        try{
            for(const auto* b:guards)b->check();
            metadata.check();
            require(std::memcmp(metadata_snapshot.data(),metadata.p,metadata_snapshot.size())==0,"immutable metadata changed");
            poison_tail(rg,gu_used,poison);poison_tail(rh,h_used,poison);poison_tail(refout,out_used,poison);
            poison_tail(gg,gu_used,poison);poison_tail(gh,h_used,poison);
            poison_tail(gout,out_used,poison);
            poison_tail(sg,size_t(max_ne*1280)*4,poison);
            poison_tail(sh,size_t(max_ne*FF)*2,poison);
        }catch(const sycl::exception&){throw;}
        catch(const std::exception& e){dump("FAIL",stage,0,"MEMORY_INTEGRITY",e.what());throw GateFailure("memory integrity failure");}
    };
    auto immutable=[&](const char* stage){
        std::vector<uint8_t> scratch(1<<20);
        auto one=[&](const Buffer& buffer,const void* expected,const char* name){
            const auto* bytes=static_cast<const uint8_t*>(expected);
            for(size_t at=0;at<buffer.bytes;at+=scratch.size()){
                const size_t n=std::min(scratch.size(),buffer.bytes-at);
                q.memcpy(scratch.data(),buffer.as<uint8_t>()+at,n).wait_and_throw();
                if(std::memcmp(scratch.data(),bytes+at,n)){
                    // Preserve actual corrupted bytes independently of expected fixture bytes.
                    const std::string filename=std::string("device_")+name+".bin";
                    std::ofstream f(out/filename,std::ios::binary);require(bool(f),"device export open failed");
                    for(size_t pos=0;pos<buffer.bytes;pos+=scratch.size()){
                        const size_t count=std::min(scratch.size(),buffer.bytes-pos);
                        q.memcpy(scratch.data(),buffer.as<uint8_t>()+pos,count).wait_and_throw();
                        f.write(reinterpret_cast<const char*>(scratch.data()),std::streamsize(count));require(bool(f),"device export write failed");
                    }
                    close_file(f);
                    dump("FAIL",stage,at,"IMMUTABLE_INPUT_CHANGED",name);throw GateFailure("immutable input changed");
                }
            }
        };
        one(weights,blobs.data(),"weights");one(input,source.data(),"source");one(ids,src.data(),"ids");
    };
    auto gathered_check=[&](const char* stage){
        for(int64_t r=0;r<rows;++r)if(std::memcmp(actual_x.data()+r*N,source.data()+int64_t(src[size_t(r)])*N,size_t(N)*2)){
            dump("FAIL",stage,size_t(r),"GATHER_INPUT_CHANGED","actual gathered FP16 row differs");throw GateFailure("gather byte mismatch");
        }
    };
    check("after_scalar_guards");immutable("after_scalar_immutable");gathered_check("scalar_gather");
    // First numerical gate: compact GU only. No candidate SiLU/down or timing if this differs.
    group_prepare();
    q.memcpy(actual_x.data(),gout.data(),actual_x.size()*2).wait_and_throw();
    check("group_prepare_guards");gathered_check("group_gather");
    finite(refgu,"reference_GU");finite(refh,"reference_H");finite(refdown,"reference_down");
    const auto first_gu=batch(gu);q.ext_oneapi_submit_barrier({first_gu});
    candgu.resize(refgu.size());q.memcpy(candgu.data(),gg.data(),gu_used).wait_and_throw();
    check("GU_guards");immutable("after_GU_immutable");
    finite(candgu,"GU");if(!compare(refgu,candgu,"GU"))return 5;
    pf::swiglu_interleaved(gg.as<float>(),gh.as<uint16_t>(),rows,&q);
    candh.resize(refh.size());q.memcpy(candh.data(),gh.data(),h_used).wait_and_throw();
    check("H_guards");immutable("after_H_immutable");
    finite(candh,"H");if(!compare(refh,candh,"H"))return 5;
    const auto first_down=batch(down);q.ext_oneapi_submit_barrier({first_down});
    canddown.resize(refdown.size());q.memcpy(canddown.data(),gout.data(),out_used).wait_and_throw();
    check("down_guards");immutable("after_down_immutable");
    finite(canddown,"down");if(!compare(refdown,canddown,"down"))return 5;
    const size_t device_bytes=blobs.size()+source.size()*2+src.size()*4+2*out_bytes+3*gu_bytes+3*h_bytes+16*(dqgu_bytes+dqd_bytes);
    std::printf("{\"numerical_gate\":\"PASS\",\"experts\":16,\"format_pairs_used\":10,\"rows\":%lld,\"max_ne\":%lld,\"explicit_device_bytes\":%zu,\"USM_metadata_bytes\":%zu,\"GU_H_down_bitwise\":true,\"guards\":true}\n",(long long)rows,(long long)max_ne,device_bytes,2*sizeof(Group));
    std::printf("{\"metadata_fill_ms\":%.6f,\"timing_metadata_policy\":\"fixed immutable descriptors; fill reported separately\"}\n",ms(metadata_start,metadata_end));
    if(timed){
        timing_executed=true;
        // Correctness observations above are outside timing. One completion wait per arm only.
        scalar(false);q.wait_and_throw();group_full();q.wait_and_throw();
        for(int repeat=0;repeat<3;++repeat)for(int position=0;position<2;++position){
            const bool grouped=((repeat+position)&1)!=0;
            const auto a=Clock::now();if(grouped)group_full();else scalar(false);const auto b=Clock::now();
            q.wait_and_throw();const auto c=Clock::now();
            std::printf("{\"repeat\":%d,\"position\":%d,\"arm\":\"%s\",\"host_submit_ms\":%.6f,\"completed_wall_ms\":%.6f}\n",repeat,position,grouped?"grouped":"scalar",ms(a,b),ms(a,c));
        }
        // Read final completed repeat outputs only after all timed windows close.
        last_scalar_down.resize(refdown.size());
        q.memcpy(candgu.data(),gg.data(),gu_used);q.memcpy(candh.data(),gh.data(),h_used);
        q.memcpy(canddown.data(),gout.data(),out_used);
        q.memcpy(last_scalar_down.data(),refout.data(),out_used).wait_and_throw();
        check("timed_guards");immutable("after_timing_immutable");
        finite(candgu,"timed_GU");if(!compare(refgu,candgu,"timed_GU"))return 5;
        finite(candh,"timed_H");if(!compare(refh,candh,"timed_H"))return 5;
        finite(canddown,"timed_down");if(!compare(refdown,canddown,"timed_down"))return 5;
        finite(last_scalar_down,"timed_scalar_down");if(!compare(refdown,last_scalar_down,"timed_scalar_down"))return 5;
    }
    dump("PASS","all_stages",0,"BITWISE_PASS","GU/H/down, finites, guards, tails and immutable inputs passed");
    q.wait_and_throw();return 0;
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try{return run(argc,argv);}
    catch(const GateFailure&e){std::fprintf(stderr,"GATE_FAILURE exit=5 %s\n",e.what());return 5;}
    catch(const std::exception&e){std::fprintf(stderr,"RUNTIME_EXCEPTION exit=2 %s\n",e.what());return 2;}
}
