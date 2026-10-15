// Actual production PleTable/PleReader; existing tiny-GGUF writer and format helpers.
#define main inherited_ple_reader_test_main
#include "/mnt/vm_8tb/b70/build/strata-ab/src-ep/src/ngram/ple_reader_test.cpp"
#undef main
#include <iostream>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {
struct Case { const char* name; std::vector<int> pre, want; uint64_t candidate, baseline; bool pending = false; };
std::vector<uint32_t> keys(const std::vector<int>& ids) {
    std::vector<uint32_t> r;
    for (int id : ids) for (int h=0; h<k::PLE_N_HEADS; ++h) r.push_back((uint32_t)(id*32+h));
    return r;
}
uint64_t requests(k::PleTable& t) {
    const std::string s=t.io_report(); unsigned long long n=0;
    if (std::sscanf(s.c_str(), "ple io: %llu rows", &n)!=1) throw std::runtime_error("unparseable production io_report");
    return (uint64_t)n;
}
}
int main(int argc, char** argv) try {
    if (argc<2 || argc>3) throw std::runtime_error("usage: oracle EMPTY_ON_DISK_DIR [--baseline]");
    const bool baseline=argc==3 && std::string(argv[2])=="--baseline";
    if (argc==3 && !baseline) throw std::runtime_error("unknown option");
    const std::filesystem::path dir=argv[1];
    if (!std::filesystem::exists(dir) || !std::filesystem::is_empty(dir)) throw std::runtime_error("output directory must exist and be empty");
    const k::PleFormatInfo* fmt=nullptr;
    for (int i=0;i<k::ple_format_count();++i) if (std::string(k::ple_formats()[i].name)=="IQ4_NL") fmt=&k::ple_formats()[i];
    if (!fmt) throw std::runtime_error("missing production IQ4_NL format");
    const std::string path=write_format_table(dir.string(),*fmt,4096);
    if (path.empty()) throw std::runtime_error("fixture write failed");
    // Keep every synthetic half scale finite; preserve the original helper's distinct quant bytes.
    const size_t payload=4096*fmt->row_bytes;
    const size_t header=std::filesystem::file_size(path)-payload;
    std::fstream table(path,std::ios::binary|std::ios::in|std::ios::out);
    if (!table) throw std::runtime_error("fixture reopen failed");
    for (uint32_t row=0;row<4096;++row) for (int block=0;block<k::PLE_HEAD_DIM/32;++block) {
        const uint16_t h=uint16_t(0x3800+row%31);
        const char bytes[2]={char(h&255),char(h>>8)};
        table.seekp(header+row*fmt->row_bytes+block*18);table.write(bytes,2);
        if (!table) throw std::runtime_error("finite fixture scale write failed");
    }
    table.close();if (!table) throw std::runtime_error("finite fixture close failed");
    std::vector<int> seventeen; for (int i=0;i<17;++i) seventeen.push_back(i);
    const std::vector<Case> cases={
        {"ordered",{0,1},{0,1},0,0},
        {"permutation",{0,1,2,3,4,5},{3,4,5,0,1,2},0,96},
        {"subset",{0,1,2,3,4,5},{4,1},0,32},
        {"duplicates",{0},{0,0,0},0,48},
        {"duplicate_prefetch",{0,0,1},{0,1},0,0},
        {"missing",{0,1},{0,2},32,32},
        {"extra",{0,1,2},{1},0,16},
        {"zero",{0},{},0,0},
        {"gather_over16",{0,1},seventeen,272,272},
        {"prefetch_overflow",seventeen,{16},0,0},
        {"pending_guard",{1},{1},0,0,true}
    };
    // Produce independent mapped results, then close all mappings before any Direct open.
    std::vector<std::vector<float>> refs(cases.size());
    k::PleTable mapped; k::PleIoOptions mio; mio.mode=k::PleIo::Mmap;
    std::string err;
    if (!mapped.open(path,err,mio)) throw std::runtime_error(err);
    for (size_t c=0;c<cases.size();++c) {
        const auto r=keys(cases[c].want); refs[c].resize(cases[c].want.size()*k::PLE_N_HEADS*k::PLE_HEAD_DIM);
        if (!mapped.gather_batch(r.data(),cases[c].want.size(),refs[c].data(),err)) throw std::runtime_error(err);
    }
    mapped.close();
    for (const auto& ref:refs) for (float value:ref)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite mapped reference");
    const size_t token_bytes=k::PLE_N_HEADS*k::PLE_HEAD_DIM*sizeof(float);
    for (int i=0;i<6;++i) for (int j=i+1;j<6;++j)
        if (!std::memcmp(refs[1].data()+i*k::PLE_N_HEADS*k::PLE_HEAD_DIM,
                         refs[1].data()+j*k::PLE_N_HEADS*k::PLE_HEAD_DIM,token_bytes))
            throw std::runtime_error("permutation fixture tokens must be distinguishable");
    std::ofstream log(dir/"results.tsv");
    if (!log) throw std::runtime_error("results open failed");
    log<<"case\tcache_rows\tio_thread\trequests_delta\tbytes_delta\twall_us\n";
    size_t count=0;
    for (uint64_t cache : {uint64_t(0),uint64_t(256)}) for (bool worker : {false,true}) for (size_t c=0;c<cases.size();++c) {
        const auto& x=cases[c]; k::PleTable direct; k::PleIoOptions io;
        io.mode=k::PleIo::Direct; io.cache_rows=cache; io.io_thread=worker; io.max_inflight=8; io.keepalive_ms=0;
        if (!direct.open(path,err,io)) throw std::runtime_error(err);
        const auto pr=keys(x.pre), wr=keys(x.want);
        if (x.pending) { const auto first=keys({0}); if (!direct.issue(first.data())) throw std::runtime_error("issue failed"); }
        const uint64_t before_prefetch=requests(direct);
        for (size_t t=0;t<x.pre.size();++t) direct.prefetch_rows(pr.data()+t*k::PLE_N_HEADS);
        const std::set<int> unique_pre(x.pre.begin(),x.pre.end());
        if (requests(direct)-before_prefetch!=unique_pre.size()*k::PLE_N_HEADS)
            throw std::runtime_error("duplicate prefetch suppression changed");
        const uint64_t rq=requests(direct), by=direct.bytes_read();
        const float sentinel=-12345.5f; std::vector<float> out(refs[c].size()+2,sentinel);
        const auto start=std::chrono::steady_clock::now();
        const bool ok=direct.gather_batch(wr.data(),x.want.size(),out.data()+1,err);
        const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        const uint64_t rd=requests(direct)-rq, bd=direct.bytes_read()-by;
        if (x.pending) {
            if (ok || err!="PleTable::gather_batch while a token is in flight" || rd || bd) throw std::runtime_error("pending guard changed");
            for (float value:out) if(value!=sentinel) throw std::runtime_error("pending guard wrote output");
            std::vector<float> one(k::PLE_N_HEADS*k::PLE_HEAD_DIM);
            if (!direct.collect(one.data(),err)) throw std::runtime_error(err);
        } else {
            if (!ok) throw std::runtime_error(err);
            if (rd!=(baseline?x.baseline:x.candidate)) throw std::runtime_error(std::string(x.name)+": wrong production request delta "+std::to_string(rd));
            if (bd!=x.want.size()*k::PLE_N_HEADS*fmt->row_bytes) throw std::runtime_error("bytes_read delta differs");
            if (!refs[c].empty() && std::memcmp(out.data()+1,refs[c].data(),refs[c].size()*sizeof(float))) throw std::runtime_error("mapped byte comparison failed");
        }
        if (out.front()!=sentinel || out.back()!=sentinel) throw std::runtime_error("output guards failed");
        direct.close();
        log<<x.name<<'\t'<<cache<<'\t'<<worker<<'\t'<<rd<<'\t'<<bd<<'\t'<<us<<'\n';
        if (!log) throw std::runtime_error("results write failed");
        ++count;
    }
    log.close(); if (!log) throw std::runtime_error("results close failed");
    std::cout<<"PASS cases="<<count<<" mode="<<(baseline?"baseline":"candidate")<<" no GPU; timing is diagnostic only\n";
    return 0;
} catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 2; }
