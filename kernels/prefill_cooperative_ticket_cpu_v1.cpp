#include "control_v1.hpp"
#include <future>
#include <iostream>
#include <memory>
using namespace strata::prefill::cooperative;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(){
    // Real shared control methods and real worker/owner threads, no GPU/model.
    std::atomic<int> ready{0};std::atomic<bool> stop{false};
    auto waiting=std::async(std::launch::async,[&]{return wait_ready(ready,stop);});
    stop.store(true,std::memory_order_release);require(waiting.get()==ReadyResult::cancelled,"blocked readiness did not cancel");
    ready.store(-1,std::memory_order_release);require(wait_ready(ready,stop)==ReadyResult::failed,"copy failure hidden by cancellation");
    require(ready.load()==-1,"failure readiness manufactured");
    auto ticket=std::make_shared<Ticket>();const auto owner=std::this_thread::get_id();
    auto worker=std::async(std::launch::async,[&]{return ticket->yield({7,0,1,24,1536,512,0},[](const Point&)->Result{throw std::runtime_error("worker executed owner callback");});});
    while(!ticket->pump([&](const Point& p){require(std::this_thread::get_id()==owner&&p.request==7&&p.device==1&&p.layer==24,"owner/point changed");return Result{false,"injected real callback failure",0};}))std::this_thread::yield();
    require(!worker.get().ok,"failed ACK accepted");
    auto closed=std::make_shared<Ticket>();auto parked=std::async(std::launch::async,[&]{return closed->yield({8,0,1,25,1536,512,0},[](const Point&){return Result{};});});
    while(!closed->pending())std::this_thread::yield();
    closed->close("owner cancelled");require(!parked.get().ok,"parked worker not unblocked by close");
    std::cout<<"PASS owner-only callback, failure-priority, real cancellation, parked-worker join; no GPU/model\n";
}
