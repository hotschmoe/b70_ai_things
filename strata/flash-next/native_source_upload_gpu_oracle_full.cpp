// Actual original-source upload oracle. No model forward, experts, KV or sampling.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/native_hc_dispatch.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <openssl/evp.h>
#include <sstream>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
uint64_t fnv(const uint8_t* p,size_t n) { uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h; }
std::string quote(const std::string& s) {std::string out="\"";char b[8];for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=char(c);}else if(c<32||c>=127){std::snprintf(b,sizeof(b),"\\u%04x",unsigned(c));out+=b;}else out+=char(c);}return out+'"';}
int layer(const std::string& name) {return name.rfind("blk.",0)==0?std::atoi(name.c_str()+4):-1;}
bool hc(const std::string& name) {return name.rfind("output_hc_",0)==0||(name.rfind("blk.",0)==0&&(name.find(".hc_attn_")!=std::string::npos||name.find(".hc_ffn_")!=std::string::npos));}
bool ple(const std::string& n) {return n=="blk.1.ple_key.weight"||n=="blk.1.ple_value.weight"||n=="blk.1.ple_conv1d.weight";}
struct Stage {int lo=0,hi=0,device=0;size_t scratch_bytes=0;strata::core::WeightTable table;std::unique_ptr<strata::core::NativeDense> owner;sycl::queue* q=nullptr;std::set<const void*> allocations;const void* scratch=nullptr;bool destroyed=false;};
struct Source {std::unique_ptr<strata::GgufFile> gguf;};
struct Entry {std::string name,shard;uint32_t type=0;uint64_t offset=0,bytes=0,ne0=0,ne1=0,hash=0;std::string sha;};
struct Expected {std::string shard,sha;unsigned type=0;uint64_t offset=0,bytes=0;};
std::map<std::string,Expected> expected_roster;
const strata::TensorInfo* find(const std::vector<Source>& files,const std::string& name,const strata::GgufFile*& file) {const strata::TensorInfo* found=nullptr;for(auto& f:files)if(auto* t=f.gguf->find(name)){require(!found,"duplicate source tensor "+name);found=t;file=f.gguf.get();}require(found!=nullptr,"missing source tensor "+name);return found;}
void owned(const void* p,Stage& stage,const std::string& name) {require(p!=nullptr,"missing pointer "+name);require(sycl::get_pointer_type(p,stage.q->get_context())==sycl::usm::alloc::device,"not device USM "+name);require(sycl::get_pointer_device(p,stage.q->get_context())==stage.q->get_device(),"foreign device ownership "+name);}
Entry check(const std::vector<Source>& files,const std::string& name,const void* pointer,uint64_t bytes,int type,uint64_t offset,uint64_t hash,uint64_t ne0,uint64_t ne1,const std::string& shard,Stage& stage) {
    const strata::GgufFile* file=nullptr;const auto* t=find(files,name,file);
    require(bytes>0&&bytes<=32u*1024u*1024u,"source row bound "+name);
    require(t->type==uint32_t(type)&&file->path()==shard&&file->data_start()+t->offset==offset,"source location/type mismatch "+name);
    require(t->shape.size()>=1&&t->shape.size()<=2&&t->shape[0]==ne0&&(t->shape.size()==1?ne1==1:t->shape[1]==ne1),"source dimensions mismatch "+name);
    const uint64_t expected=type==8?ne0*ne1/32*34:ne0*ne1*4;
    require((type==0||type==8)&&bytes==expected&&offset<=file->file_size()&&bytes<=file->file_size()-offset,"source byte count/bounds mismatch "+name);
    const auto e=expected_roster.find(name);require(e!=expected_roster.end(),"source not preregistered "+name);
    require(file->path().substr(file->path().find_last_of('/')+1)==e->second.shard&&type==int(e->second.type)&&bytes==e->second.bytes&&offset==e->second.offset,"preregistered source location/extent mismatch "+name);
    const auto* original=file->tensor_data(*t);require(fnv(original,size_t(bytes))==hash,"source FNV metadata mismatch "+name);
    owned(pointer,stage,name);require(stage.allocations.insert(pointer).second,"source allocation alias "+name);
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(),EVP_MD_CTX_free);require(bool(digest)&&EVP_DigestInit_ex(digest.get(),EVP_sha256(),nullptr)==1,"SHA initialization failed");
    std::vector<uint8_t> copy(std::min<uint64_t>(bytes,4u*1024u*1024u));
    for(uint64_t at=0;at<bytes;at+=copy.size()) {const size_t n=size_t(std::min<uint64_t>(copy.size(),bytes-at));stage.q->memcpy(copy.data(),static_cast<const uint8_t*>(pointer)+at,n).wait_and_throw();require(std::memcmp(copy.data(),original+at,n)==0,"GPU/source byte mismatch "+name+" at "+std::to_string(at));require(EVP_DigestUpdate(digest.get(),copy.data(),n)==1,"SHA update failed");}
    unsigned char raw_hash[EVP_MAX_MD_SIZE];unsigned hash_n=0;require(EVP_DigestFinal_ex(digest.get(),raw_hash,&hash_n)==1&&hash_n==32,"SHA final failed");std::string hex;char hb[3];for(unsigned i=0;i<hash_n;++i){std::snprintf(hb,sizeof(hb),"%02x",raw_hash[i]);hex+=hb;}require(hex==e->second.sha,"GPU SHA differs from preregistered original source "+name);
    // A deliberate host-byte negative control must detect mismatched source data.
    copy[0]^=1;require(copy[0]!=(original+((bytes-1)/copy.size())*copy.size())[0],"byte oracle negative control failed");
    return {name,shard,uint32_t(type),offset,bytes,ne0,ne1,hash,hex};
}
std::string stage_id(const Stage& stage){return std::to_string(stage.lo)+":"+std::to_string(stage.hi)+":"+std::to_string(stage.device);}
std::string address(const void* p){char b[40];std::snprintf(b,sizeof(b),"%p",p);return quote(b);}
void register_owners(Stage& stage){
    const auto native=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(stage.q->get_context());
    const auto emit=[&](const void* p,uint64_t bytes,const char* role,const std::string& name){
        std::fprintf(stderr,"UPLOAD_USM {\"event\":\"owner_register\",\"stage\":%s,\"device\":%d,\"ze_context\":%s,\"pointer\":%s,\"bytes\":%llu,\"role\":%s,\"name\":%s}\n",stage_id(stage).empty()?"null":quote(stage_id(stage)).c_str(),stage.device,address(native).c_str(),address(p).c_str(),(unsigned long long)bytes,quote(role).c_str(),quote(name).c_str());
    };
    for(const auto& [name,ref]:stage.table.all()){
        if(ref.hc_source_type>=0)emit(ref.hc_source_type==8?static_cast<const void*>(ref.hc_source_q8):ref.hc_source_f32,ref.hc_source_bytes,"HC",name);
        if(ref.ple_source_type>=0)emit(ref.ple_source_type==8?static_cast<const void*>(ref.ple_source_q8):ref.ple_source_f32,ref.ple_source_bytes,"PLE",name);
        if(ref.native_data)emit(ref.native_data,strata::kernels::native_mmvq_weight_bytes(ref.native_type,int(ref.ne0),int(ref.ne1)),"ordinary",name);
    }
    emit(stage.scratch,stage.scratch_bytes,"scratch","native_q8_1");
    std::fflush(stderr);
}
void destroy(Stage& stage) {
    if(stage.destroyed||!stage.owner)return;
    dpct::select_device(stage.device);stage.q->wait_and_throw();
    std::fprintf(stderr,"UPLOAD_USM {\"event\":\"destroy_begin\",\"stage\":%s}\n",quote(stage_id(stage)).c_str());std::fflush(stderr);
    stage.owner.reset();stage.q->wait_and_throw();
    std::fprintf(stderr,"UPLOAD_USM {\"event\":\"destroy_end\",\"stage\":%s,\"owning_destructor_returned\":true}\n",quote(stage_id(stage)).c_str());std::fflush(stderr);
    for(const auto* p:stage.allocations){const auto type=sycl::get_pointer_type(p,stage.q->get_context());std::fprintf(stderr,"UPLOAD_USM {\"event\":\"postfree_query\",\"stage\":%s,\"pointer\":%s,\"sycl_type_code\":%d}\n",quote(stage_id(stage)).c_str(),address(p).c_str(),int(type));}
    // Postfree classification is observed only: local controls show stale type even after raw zeMemFree.
    std::fprintf(stderr,"UPLOAD_USM {\"event\":\"probe_begin\",\"stage\":%s}\n",quote(stage_id(stage)).c_str());std::fflush(stderr);
    auto* probe=sycl::malloc_device<uint8_t>(65536,*stage.q);require(probe!=nullptr,"post-destructor probe allocation failed");stage.q->memset(probe,0xa5,65536).wait_and_throw();uint8_t observed=0;stage.q->memcpy(&observed,probe,1).wait_and_throw();sycl::free(probe,*stage.q);stage.q->wait_and_throw();require(observed==0xa5,"post-destructor probe byte mismatch");
    std::fprintf(stderr,"UPLOAD_USM {\"event\":\"probe_end\",\"stage\":%s}\n",quote(stage_id(stage)).c_str());std::fflush(stderr);stage.destroyed=true;
}
}

int main(int argc,char** argv) {
    std::string pack,output,roster,bounds="explicit";std::vector<std::string> shards;std::vector<std::unique_ptr<Stage>> stages;std::map<int,std::set<const void*>> all_allocations;
    std::ofstream report;bool first=true;unsigned total_hc=0,total_ple=0;uint64_t total_bytes=0;
    try {
        for(int i=1;i<argc;++i) {const std::string a=argv[i];require(i+1<argc,"missing argument "+a);const std::string value=argv[++i];if(a=="--pack")pack=value;else if(a=="--output")output=value;else if(a=="--source-roster")roster=value;else if(a=="--bounds")bounds=value;else if(a=="--shard")shards.push_back(value);else if(a=="--stage") {auto s=std::make_unique<Stage>();char tail=0;require(std::sscanf(value.c_str(),"%d:%d:%d%c",&s->lo,&s->hi,&s->device,&tail)==3,"invalid stage LO:HI:DEVICE");require(s->lo>=0&&s->hi>s->lo&&s->hi<=48&&s->hi-s->lo<=48&&s->device>=0&&s->device<=1,"stage bound");stages.push_back(std::move(s));}else throw std::runtime_error("unknown option "+a);}
        require(!pack.empty()&&!output.empty()&&!roster.empty()&&shards.size()==4&&!stages.empty()&&stages.size()<=2,"pack/output/selected four shards/1..2 stages required");
        std::ifstream roster_file(roster);require(bool(roster_file),"source roster open failed");std::string roster_line;while(std::getline(roster_file,roster_line)){if(roster_line.empty()||roster_line[0]=='#')continue;std::istringstream row(roster_line);std::string name;Expected e;require(bool(row>>name>>e.shard>>e.type>>e.offset>>e.bytes>>e.sha)&&e.sha.size()==64,"malformed source roster");require(expected_roster.emplace(name,e).second,"duplicate source roster name");}require(expected_roster.size()==390,"exact387HC+3PLE preregistered roster required");
        const char* layers=std::getenv("UR_ENABLE_LAYERS");const char* log=std::getenv("UR_LOG_TRACING");require(layers&&std::strstr(layers,"UR_LAYER_TRACING")&&log&&std::strstr(log,"level:info"),"explicit UR logical-free tracing required");
        require(std::getenv("STRATA_SYCL_NATIVE_HC")&&std::strcmp(std::getenv("STRATA_SYCL_NATIVE_HC"),"1")==0,"strict native HC flag required");
        require(!strata::kernels::native_q8_0_packed_enabled(),"packed secondary allocations must be disabled for this census");
        const int report_fd=::open(output.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);require(report_fd>=0,"new output path required");::close(report_fd);report.open(output,std::ios::out|std::ios::app);require(bool(report),"output open failed");report<<"{\"schema\":2,\"coverage\":\"whole390\",\"bounds\":"<<quote(bounds)<<",\"scope\":\"actual HC/PLE original-source uploads only; no inference\",\"stages\":[";report.flush();
        std::vector<Source> files;for(const auto& path:shards){require(path.find("UD-Q4_K_XL-")!=std::string::npos,"unexpected artifact shard");files.push_back({std::make_unique<strata::GgufFile>(path)});}
        // Actual packed index loader with all canonical bytes skipped: retain complete metadata only.
        std::set<std::string> skip;std::ifstream index(pack+"/index.txt");require(bool(index),"index open failed");std::string line;while(std::getline(index,line)){if(line.empty()||line[0]=='#')continue;const auto end=line.find(' ');require(end!=std::string::npos,"malformed index line");skip.insert(line.substr(0,end));}
        for(auto& handle:stages) {
            auto& stage=*handle;dpct::select_device(stage.device);stage.q=&dpct::get_in_order_queue();require(stage.q->has_property<sycl::property::queue::in_order>(),"unordered owner queue");
            std::string err;uint64_t pool=1;require(strata::core::WeightTable::pool_bytes(pack,pool,err,&skip),err);require(pool==0,"canonical metadata-only arena unexpectedly nonzero");
            require(stage.table.load(pack,nullptr,0,err,&skip),err);for(const auto& [name,ref]:stage.table.all())require(!ref.data&&!ref.resident,"canonical bytes unexpectedly loaded "+name);
            strata::core::NativeDense::set_layer_range(stage.lo,stage.hi);stage.owner=std::make_unique<strata::core::NativeDense>();
            require(bounds=="explicit"||bounds=="static","bounds must be explicit or static");
            require(bounds=="explicit"?stage.owner->load(shards,stage.table,err,true,stage.lo,stage.hi):stage.owner->load(shards,stage.table,err,true),err);stage.q->wait_and_throw();
            std::vector<Entry> rows;unsigned nhc=0,nple=0,nordinary=0;uint64_t bytes=0;
            for(const auto& [name,ref]:stage.table.all()) {
                const int l=layer(name);const bool local=l>=stage.lo&&l<stage.hi;const bool head=name.rfind("output_hc_",0)==0;
                const bool expected_hc=hc(name)&&(head?stage.hi==48:local);const bool expected_ple=ple(name)&&local;
                require((ref.hc_source_type>=0)==expected_hc,"foreign/missing HC metadata "+name);require((ref.ple_source_type>=0)==expected_ple,"foreign/missing PLE metadata "+name);
                if(expected_hc){rows.push_back(check(files,name,ref.hc_source_type==8?static_cast<const void*>(ref.hc_source_q8):ref.hc_source_f32,ref.hc_source_bytes,ref.hc_source_type,ref.hc_source_offset,ref.hc_source_fnv64,ref.hc_source_ne0,ref.hc_source_ne1,ref.hc_source_shard,stage));++nhc;bytes+=ref.hc_source_bytes;}
                else require(!ref.hc_source_q8&&!ref.hc_source_f32&&!ref.hc_source_bytes&&ref.hc_source_shard.empty(),"foreign HC pointer/image "+name);
                if(expected_ple){rows.push_back(check(files,name,ref.ple_source_type==8?static_cast<const void*>(ref.ple_source_q8):ref.ple_source_f32,ref.ple_source_bytes,ref.ple_source_type,ref.ple_source_offset,ref.ple_source_fnv64,ref.ple_source_ne0,ref.ple_source_ne1,ref.ple_source_shard,stage));++nple;bytes+=ref.ple_source_bytes;}
                else require(!ref.ple_source_q8&&!ref.ple_source_f32&&!ref.ple_source_bytes&&ref.ple_source_shard.empty(),"foreign PLE pointer/image "+name);
                if(ref.native_data) {require(local,"foreign ordinary projection "+name);owned(ref.native_data,stage,name);require(stage.allocations.insert(ref.native_data).second,"ordinary allocation alias "+name);bytes+=strata::kernels::native_mmvq_weight_bytes(ref.native_type,int(ref.ne0),int(ref.ne1));++nordinary;stage.scratch_bytes=std::max(stage.scratch_bytes,strata::kernels::native_q8_1_bytes(int(ref.ne0)));owned(ref.native_q8_1,stage,"shared scratch");if(!stage.scratch)stage.scratch=ref.native_q8_1;else require(stage.scratch==ref.native_q8_1,"multiple shared scratch images");}
                require(!ref.hc_q8,"legacy HC image unexpectedly allocated");
            }
            // Exercise the actual shared model dispatch binder without submitting HC math.
            auto bind=[&](const std::string& prefix,bool injection) {
                const auto* wn=stage.table.find(prefix+"norm.weight");const auto* wd=stage.table.find(prefix+"down.weight");const auto* wu=stage.table.find(prefix+"up.weight");const auto* wi=injection?stage.table.find(prefix+"inject.weight"):nullptr;
                strata::kernels::HcNativeArgs args;std::string why;
                require(strata::core::native_hc_bind(wn,wd,wu,wi,args,why),why);
                require(args.w.norm==wn->hc_source_f32&&args.w.down==wd->hc_source_q8&&args.w.up==wu->hc_source_q8&&(!wi||args.w.inject==wi->hc_source_f32),"dispatch bound compatibility rather than original source");
                auto bad=*wd;bad.hc_source_type=-1;
                require(!strata::core::native_hc_bind(wn,&bad,wu,wi,args,why),"invalid source type accepted by actual binder");
            };
            for(int l=stage.lo;l<stage.hi;++l)for(const char* half:{"hc_attn_","hc_ffn_"})bind("blk."+std::to_string(l)+"."+half,true);
            if(stage.hi==48)bind("output_hc_",false);
            if(nple) {
                const auto* key=stage.table.find("blk.1.ple_key.weight");const auto* value=stage.table.find("blk.1.ple_value.weight");const auto* conv=stage.table.find("blk.1.ple_conv1d.weight");
                strata::kernels::PleWeights w;w.source_exact=true;w.key_source_q8=key->ple_source_q8;w.key_source_bytes=key->ple_source_bytes;w.value_source_q8=value->ple_source_q8;w.value_source_bytes=value->ple_source_bytes;w.conv_source_f32=conv->ple_source_f32;w.conv_source_floats=conv->ple_source_bytes/4;
                require(strata::kernels::ple_exact_sources_ok(w),"actual PLE consumed-source descriptor rejected");--w.value_source_bytes;require(!strata::kernels::ple_exact_sources_ok(w),"bad PLE source byte extent accepted");
            }
            require(nhc==unsigned((stage.hi-stage.lo)*8+(stage.hi==48?3:0)),"HC count incomplete");require(nple==unsigned(stage.lo<=1&&stage.hi>1?3:0),"PLE count incomplete");require(nordinary>0&&stage.scratch,"ordinary loader path unexercised");
            // Accounting must include HC, PLE and ordinary native images, never hide missing PLE bytes.
            for(const auto* p:stage.allocations)require(all_allocations[stage.device].insert(p).second,"same-device cross-stage source allocation alias");
            require(all_allocations[stage.device].insert(stage.scratch).second,"same-device cross-stage scratch allocation alias");
            register_owners(stage);
            const bool bytes_ok=stage.owner->weight_bytes()==bytes;
            if(!first)report<<',';first=false;report<<"{\"lo\":"<<stage.lo<<",\"hi\":"<<stage.hi<<",\"device\":"<<stage.device<<",\"device_name\":"<<quote(stage.q->get_device().get_info<sycl::info::device::name>())<<",\"hc_images\":"<<nhc<<",\"ple_images\":"<<nple<<",\"ordinary_images\":"<<nordinary<<",\"unique_allocations\":"<<stage.allocations.size()<<",\"legacy_tensor_count\":"<<stage.owner->tensor_count()<<",\"expected_image_bytes\":"<<bytes<<",\"reported_weight_bytes\":"<<stage.owner->weight_bytes()<<",\"accounting_equal\":"<<(bytes_ok?"true":"false")<<",\"source_rows\":[";
            bool row_first=true;for(const auto& row:rows){if(!row_first)report<<',';row_first=false;report<<"{\"name\":"<<quote(row.name)<<",\"shard\":"<<quote(row.shard)<<",\"type\":"<<row.type<<",\"ne0\":"<<row.ne0<<",\"ne1\":"<<row.ne1<<",\"absolute_offset\":"<<row.offset<<",\"bytes\":"<<row.bytes<<",\"fnv64\":"<<quote(std::to_string(row.hash))<<",\"gpu_sha256\":"<<quote(row.sha)<<",\"gpu_byte_equal\":true}";}report<<"]}";report.flush();
            total_hc+=nhc;total_ple+=nple;total_bytes+=bytes;require(total_bytes<=8ull*1024*1024*1024,"total allocation census bound");require(bytes_ok,"native weight byte accounting omitted an allocated image");
        }
        require(total_hc==387&&total_ple==3,"whole coverage incomplete or duplicated; require387HC+3PLE across stages");
        // Foreign ownership negatives query metadata only; never submit an invalid cross-device read.
        unsigned foreign_negatives=0;
        for(auto& from:stages)for(auto& to:stages)if(from->device!=to->device) {
            bool rejected=false;try{owned(*from->allocations.begin(),*to,"foreign-device negative");}catch(const std::exception&){rejected=true;}
            require(rejected,"foreign-device pointer accepted");++foreign_negatives;
        }
        // Both card owners remain live until every source image is verified, then free on each owning queue.
        for(auto& stage:stages)destroy(*stage);
        report<<"],\"foreign_device_negative_checks\":"<<foreign_negatives<<",\"hc_images\":"<<total_hc<<",\"ple_images\":"<<total_ple<<",\"expected_image_bytes\":"<<total_bytes<<",\"all_owners_destructor_returned\":true,\"source_and_probe_passed\":true,\"logical_free_trace_required\":true,\"passed\":false}\n";std::puts("UPLOAD_ORACLE_FULL SOURCE_PROBE_COMPLETE; UR logical-free parser and parent health gates required");return 0;
    } catch(const std::exception& error) {
        for(auto& stage:stages)try{destroy(*stage);}catch(const std::exception& cleanup){std::fprintf(stderr,"UPLOAD_ORACLE cleanup_failed %s\n",cleanup.what());}
        if(report){report<<"],\"passed\":false,\"error\":"<<quote(error.what())<<"}\n";report.flush();}
        std::fprintf(stderr,"UPLOAD_ORACLE FAIL %s\n",error.what());return 1;
    }
}
