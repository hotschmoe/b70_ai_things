// Bounded allocator control: only pointer metadata is queried after free.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include "strata/sycl_queue.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

namespace {
void check(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
std::string esc(const std::string& s){std::string r="\"";char b[8];for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=char(c);}else if(c<32||c>=127){std::snprintf(b,sizeof(b),"\\u%04x",unsigned(c));r+=b;}else r+=char(c);}return r+'"';}
std::string ptr(const void* p){char b[40];std::snprintf(b,sizeof(b),"%p",p);return esc(b);}
const char* kind(sycl::usm::alloc t){return t==sycl::usm::alloc::device?"device":t==sycl::usm::alloc::host?"host":t==sycl::usm::alloc::shared?"shared":"unknown";}
struct Query {std::string type,error;int ze_props_rc=-1,ze_range_rc=-1;unsigned ze_type=0;uint64_t ze_id=0;size_t page=0,size=0;void* base=nullptr;bool pointer_device_matches=false;};
Query query(const void* address,sycl::queue& q){
    Query r;
    try {auto t=sycl::get_pointer_type(address,q.get_context());r.type=kind(t);if(t==sycl::usm::alloc::device||t==sycl::usm::alloc::shared)r.pointer_device_matches=sycl::get_pointer_device(address,q.get_context())==q.get_device();}
    catch(const std::exception& e){r.type="exception";r.error=e.what();}
    const auto context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    ze_memory_allocation_properties_t properties{};properties.stype=ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t device=nullptr;
    r.ze_props_rc=int(zeMemGetAllocProperties(context,address,&properties,&device));
    if(r.ze_props_rc==int(ZE_RESULT_SUCCESS)){r.ze_type=unsigned(properties.type);r.ze_id=properties.id;r.page=properties.pageSize;}
    r.ze_range_rc=int(zeMemGetAddressRange(context,address,&r.base,&r.size));
    return r;
}
void write_query(std::ostream& out,const Query& q){out<<"{\"sycl_type\":"<<esc(q.type)<<",\"sycl_error\":"<<esc(q.error)<<",\"pointer_device_matches\":"<<(q.pointer_device_matches?"true":"false")<<",\"ze_properties_rc\":"<<q.ze_props_rc<<",\"ze_type\":"<<q.ze_type<<",\"ze_allocation_id\":"<<q.ze_id<<",\"ze_page_size\":"<<q.page<<",\"ze_range_rc\":"<<q.ze_range_rc<<",\"ze_base\":"<<ptr(q.base)<<",\"ze_bytes\":"<<q.size<<'}';}
void environment(std::ostream& out){out<<'{';bool first=true;for(const char* key:{"UR_L0_DISABLE_USM_ALLOCATOR","SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR","UR_L0_USM_ALLOCATOR","UR_L0_USM_ALLOCATOR_TRACE","UR_L0_LEAKS_DEBUG","UR_ENABLE_LAYERS","ZE_AFFINITY_MASK","ONEAPI_DEVICE_SELECTOR","SYCL_CACHE_PERSISTENT","STRATA_ARENA_ALIAS_CHECK"}){if(!first)out<<',';first=false;const char* value=std::getenv(key);out<<esc(key)<<':'<<(value?esc(value):"null");}out<<'}';}
}
int main(int argc,char** argv){
    int device=0,rounds=2;std::string path;std::ofstream out;void* active=nullptr;int active_path=0;sycl::queue* queue=nullptr;
    try{
        for(int i=1;i<argc;++i){const std::string key=argv[i];check(i+1<argc,"missing argument");const std::string value=argv[++i];if(key=="--output")path=value;else if(key=="--device")device=std::stoi(value);else if(key=="--rounds")rounds=std::stoi(value);else throw std::runtime_error("unknown argument");}
        check(!path.empty()&&device>=0&&device<=1&&rounds>=1&&rounds<=3,"new output/device0..1/rounds1..3 required");
        const int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);check(fd>=0,"new output path required");::close(fd);out.open(path,std::ios::app);check(bool(out),"output failed");
        dpct::select_device(device);queue=&dpct::get_in_order_queue();auto& q=*queue;check(q.has_property<sycl::property::queue::in_order>(),"matching in-order default queue required");check(q.get_backend()==sycl::backend::ext_oneapi_level_zero,"LevelZero backend required");
        out<<"{\"schema\":1,\"scope\":\"bounded allocator metadata semantics; never dereference freed address\",\"device\":"<<device<<",\"device_name\":"<<esc(q.get_device().get_info<sycl::info::device::name>())<<",\"environment\":";environment(out);out<<",\"runtime_maps\":[";
        std::ifstream maps("/proc/self/maps");std::string line;bool first=true;while(std::getline(maps,line))if(line.find("libur_")!=std::string::npos||line.find("libsycl")!=std::string::npos||line.find("libze_loader")!=std::string::npos){if(!first)out<<',';first=false;out<<esc(line);}out<<"],\"cases\":[";out.flush();first=true;
        unsigned persistent=0,reused=0,completed=0;
        for(int round=0;round<rounds;++round)for(int path_id:{0,1,2})for(size_t bytes:{size_t(8192),size_t(40960),size_t(163840),size_t(3481600),size_t(6963200),size_t(27852800),size_t(33554432)}){
            const auto allocate=[&]()->void* {
                if(path_id==1)return strata::malloc_device_guarded(bytes,q,"USM free query control");
                if(path_id==0)return sycl::malloc_device(bytes,q);
                ze_device_mem_alloc_desc_t descriptor{};descriptor.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;void* p=nullptr;
                const auto context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());const auto dev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
                check(zeMemAllocDevice(context,&descriptor,bytes,0,dev,&p)==ZE_RESULT_SUCCESS,"raw LevelZero alloc failed");return p;
            };
            const auto release=[&](void* p){if(path_id<2)sycl::free(p,q);else check(zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()),p)==ZE_RESULT_SUCCESS,"raw LevelZero free failed");};
            active_path=path_id;active=allocate();check(active!=nullptr,"allocation returned null");const void* old=active;
            auto before=query(old,q);check(before.type=="device"&&before.pointer_device_matches,"allocated pointer not owned device USM");
            const size_t probe=std::min<size_t>(bytes,65536);q.memset(active,0xa5,probe).wait_and_throw();q.memset(static_cast<uint8_t*>(active)+bytes-1,0x5a,1).wait_and_throw();uint8_t first_byte=0,last_byte=0;q.memcpy(&first_byte,active,1).wait_and_throw();q.memcpy(&last_byte,static_cast<uint8_t*>(active)+bytes-1,1).wait_and_throw();check(first_byte==0xa5&&last_byte==0x5a,"live source probe wrong");
            q.wait_and_throw();release(active);active=nullptr;q.wait_and_throw();
            // Metadata queries only. No memcpy, fill, kernel or host load targets old now.
            auto after=query(old,q);
            active=allocate();check(active!=nullptr,"fresh allocation returned null");auto fresh=query(active,q);check(fresh.type=="device"&&fresh.pointer_device_matches,"fresh pointer not owned");const bool same_address=active==old;
            q.memset(active,0x3c,probe).wait_and_throw();q.memset(static_cast<uint8_t*>(active)+bytes-1,0xc3,1).wait_and_throw();q.memcpy(&first_byte,active,1).wait_and_throw();q.memcpy(&last_byte,static_cast<uint8_t*>(active)+bytes-1,1).wait_and_throw();check(first_byte==0x3c&&last_byte==0xc3,"fresh probe wrong");
            const void* fresh_old=active;q.wait_and_throw();release(active);active=nullptr;q.wait_and_throw();auto after_fresh=query(fresh_old,q);
            if(after.type=="device")++persistent;if(same_address)++reused;++completed;
            if(!first)out<<',';first=false;out<<"{\"round\":"<<round<<",\"path\":"<<esc(path_id==1?"strata_malloc_device_guarded":path_id==0?"sycl_malloc_device":"raw_zeMemAllocDevice")<<",\"requested_bytes\":"<<bytes<<",\"old_address\":"<<ptr(old)<<",\"before_free\":";write_query(out,before);out<<",\"owning_free_returned\":true,\"after_free_before_reallocation\":";write_query(out,after);out<<",\"fresh_address_reused\":"<<(same_address?"true":"false")<<",\"fresh_allocation\":";write_query(out,fresh);out<<",\"fresh_probe_passed\":true,\"fresh_free_returned\":true,\"after_fresh_free\":";write_query(out,after_fresh);out<<'}';out.flush();
        }
        out<<"],\"completed_cases\":"<<completed<<",\"postfree_device_query_count\":"<<persistent<<",\"fresh_address_reuse_count\":"<<reused<<",\"allocator_operations_passed\":true,\"oracle_gate_change_authorized\":false}\n";
        std::printf("USM_FREE_CONTROL completed=%u postfree_device=%u reuse=%u; postfree type is observation, not pass/fail\n",completed,persistent,reused);return 0;
    }catch(const std::exception& e){if(active&&queue)try{queue->wait_and_throw();if(active_path<2)sycl::free(active,*queue);else check(zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue->get_context()),active)==ZE_RESULT_SUCCESS,"cleanup raw free failed");queue->wait_and_throw();}catch(const std::exception& cleanup){std::fprintf(stderr,"USM_FREE_CONTROL cleanup %s\n",cleanup.what());}if(out){out<<"],\"allocator_operations_passed\":false,\"error\":"<<esc(e.what())<<"}\n";out.flush();}std::fprintf(stderr,"USM_FREE_CONTROL FAIL %s\n",e.what());return 1;}
}
