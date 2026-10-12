#pragma once
// Host-only opt-in proof. No Verifier layout change, device command, wait or query.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>
namespace strata::core::hc_t2_route_proof {
inline bool active(){static const bool on=[](){const char* p=std::getenv("STRATA_HC_T2_ROUTE_PROOF");return p&&std::strcmp(p,"1")==0;}();return on;}
inline int enabled(){const char* p=std::getenv("STRATA_VERIFY_HC_T2_REUSE");return p&&std::strcmp(p,"1")==0;}
struct Row {const void* owner=nullptr;const void* graph=nullptr;int device=0,lb=0,le=0,on=0,eligible=0,selected=0;uint64_t submitted=0,completed=0;int64_t first=-1,last=-1;};
struct Store {std::mutex mutex;std::array<Row,32> rows{};bool failed=false;};
inline Store& store(){static Store s;return s;}
inline Row& slot(Store& s,const void* owner,const void* graph){for(auto& r:s.rows)if(r.owner==owner&&r.graph==graph)return r;for(auto& r:s.rows)if(!r.owner){r.owner=owner;r.graph=graph;return r;}s.failed=true;return s.rows.back();}
struct Record {const void* owner;bool on;int device,lb,le,eligible=0,selected=0;Record(const void* p,int T,int d,int a,int b):owner(p),on(active()&&T==2),device(d),lb(a),le(b){}
 void note(bool eligible_here,bool selected_here){if(on){eligible+=int(eligible_here);selected+=int(selected_here);}}
 void finish(){if(!on)return;auto& s=store();std::lock_guard<std::mutex> lock(s.mutex);auto& r=slot(s,owner,nullptr);r.device=device;r.lb=lb;r.le=le;r.on=enabled();r.eligible=eligible;r.selected=selected;}
};
inline void seal(const void* owner,const void* graph,int T){if(!active()||T!=2)return;auto& s=store();std::lock_guard<std::mutex> lock(s.mutex);Row* pending=nullptr;for(auto& r:s.rows)if(r.owner==owner&&!r.graph)pending=&r;if(!pending||!graph){s.failed=true;return;}for(const auto& old:s.rows)if(old.owner==owner&&old.graph==graph){s.failed=true;return;}auto& r=slot(s,owner,graph);r=*pending;r.graph=graph;*pending=Row{};
 std::fprintf(stderr,"HC_T2_ROUTE_CAPTURE pid=%ld owner=%p graph=%p device=%d lb=%d le=%d enabled=%d eligible=%d selected=%d graph_kind=SYCL_host_identity\n",long(getpid()),owner,graph,r.device,r.lb,r.le,r.on,r.eligible,r.selected);}
inline bool submit(const void* owner,const void* graph,int T,int64_t pos){if(!active()||T!=2)return false;auto& s=store();std::lock_guard<std::mutex> lock(s.mutex);for(auto& r:s.rows)if(r.owner==owner&&r.graph==graph&&graph){++r.submitted;if(r.first<0)r.first=pos;r.last=pos;return true;}s.failed=true;return false;}
inline void complete(const void* owner,const void* graph,bool tracked){if(!tracked)return;auto& s=store();std::lock_guard<std::mutex> lock(s.mutex);for(auto& r:s.rows)if(r.owner==owner&&r.graph==graph){++r.completed;return;}s.failed=true;}
inline void close(){if(!active())return;auto& s=store();std::lock_guard<std::mutex> lock(s.mutex);for(const auto& r:s.rows)if(r.owner&&(!r.graph||r.submitted!=r.completed))s.failed=true;for(const auto& r:s.rows)if(r.owner&&r.graph)std::fprintf(stderr,"HC_T2_ROUTE_CLOSE pid=%ld owner=%p graph=%p device=%d lb=%d le=%d enabled=%d eligible=%d selected=%d submitted=%llu completed=%llu first_pos=%lld last_pos=%lld graph_kind=SYCL_host_identity\n",long(getpid()),r.owner,r.graph,r.device,r.lb,r.le,r.on,r.eligible,r.selected,(unsigned long long)r.submitted,(unsigned long long)r.completed,(long long)r.first,(long long)r.last);std::fprintf(stderr,"HC_T2_ROUTE_FINAL pid=%ld failed=%d GPU_queries_added=0 GPU_waits_added=0\n",long(getpid()),int(s.failed));}
}
