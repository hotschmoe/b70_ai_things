// Same SYCL command_graph/UR path; no model or captured operands.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <cstdio>
#include <stdexcept>
using namespace sycl::ext::oneapi::experimental;
class PausedRegularFixtureA;
class PausedRegularFixtureB;
int main(){
  sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
  int* values=sycl::malloc_shared<int>(2,q);if(!values)throw std::bad_alloc();
  values[0]=values[1]=0;
  try{
    {
      command_graph graph{q.get_context(),q.get_device()};graph.begin_recording(q);
      q.single_task<PausedRegularFixtureA>([=]{values[0]+=1;});
      q.single_task<PausedRegularFixtureB>([=]{values[1]=2*values[0];});
      graph.end_recording(q);auto executable=graph.finalize();
      q.ext_oneapi_graph(executable).wait_and_throw();
      if(values[0]!=1||values[1]!=2)throw std::runtime_error("warmup data mismatch");
      std::printf("PAUSED_LIST_WARM_DONE values=1,2\n");std::fflush(stdout);
      if(std::getchar()!='R')throw std::runtime_error("actual owner resume ACK missing");
      q.ext_oneapi_graph(executable).wait_and_throw();
      if(values[0]!=2||values[1]!=4)throw std::runtime_error("selected data mismatch");
      std::printf("PAUSED_LIST_SELECTED_DONE values=2,4\n");std::fflush(stdout);
      q.wait_and_throw();
    }
    sycl::free(values,q);std::printf("PAUSED_LIST_RETIRED graph=1 allocation=1\n");
  }catch(...){q.wait_and_throw();sycl::free(values,q);throw;}
  return 0;
}
