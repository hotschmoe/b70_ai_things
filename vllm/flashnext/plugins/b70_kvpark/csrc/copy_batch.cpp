// Direct KV copies for parking.
//
// swap_blocks_batch stages every H2D through a CPU memcpy and waits each
// pageable D2H. This op submits memcpy on one in-order queue and does not
// stage. The queue is an immediate Level Zero command list on the first
// copy-only group (Battlemage group 1). If that group cannot be opened,
// copies fall back to the current compute stream and engine_ordinal()
// returns -1.

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include <level_zero/ze_api.h>

namespace {

struct ZeFns {
  decltype(&zeDeviceGetCommandQueueGroupProperties) get_groups = nullptr;
  decltype(&zeCommandListCreateImmediate) create_imm = nullptr;
};

bool load_ze(ZeFns* fns) {
  void* handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    handle = dlopen("libze_loader.so.1", RTLD_NOW);
  }
  if (handle == nullptr) {
    return false;
  }
  fns->get_groups = reinterpret_cast<decltype(fns->get_groups)>(
      dlsym(handle, "zeDeviceGetCommandQueueGroupProperties"));
  fns->create_imm = reinterpret_cast<decltype(fns->create_imm)>(
      dlsym(handle, "zeCommandListCreateImmediate"));
  return fns->get_groups != nullptr && fns->create_imm != nullptr;
}

struct Engine {
  sycl::queue* queue = nullptr;
  int ordinal = -1;
};

Engine& engine() {
  static Engine eng;
  static std::once_flag once;
  std::call_once(once, [] {
    sycl::queue& probe = c10::xpu::getCurrentXPUStream().queue();
    if (probe.get_device().get_backend() != sycl::backend::ext_oneapi_level_zero) {
      std::fprintf(stderr, "b70_kvpark: not Level Zero; copies use the compute stream\n");
      return;
    }
    ZeFns fns;
    if (!load_ze(&fns)) {
      std::fprintf(stderr, "b70_kvpark: Level Zero loader missing; compute stream\n");
      return;
    }
    auto context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(probe.get_context());
    auto device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(probe.get_device());
    uint32_t ngroups = 0;
    if (fns.get_groups(device, &ngroups, nullptr) != ZE_RESULT_SUCCESS || ngroups == 0) {
      std::fprintf(stderr, "b70_kvpark: no queue groups; compute stream\n");
      return;
    }
    std::vector<ze_command_queue_group_properties_t> groups(ngroups);
    for (auto& group : groups) {
      std::memset(&group, 0, sizeof(group));
      group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }
    if (fns.get_groups(device, &ngroups, groups.data()) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "b70_kvpark: queue group query failed; compute stream\n");
      return;
    }
    int ordinal = -1;
    for (uint32_t i = 0; i < ngroups; ++i) {
      const bool copy = (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0;
      const bool compute = (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0;
      if (copy && !compute && groups[i].numQueues > 0) {
        ordinal = static_cast<int>(i);
        break;
      }
    }
    if (ordinal < 0) {
      std::fprintf(stderr, "b70_kvpark: no copy-only group; compute stream\n");
      return;
    }
    ze_command_queue_desc_t desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    desc.ordinal = static_cast<uint32_t>(ordinal);
    desc.index = 0;
    desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t list = nullptr;
    if (fns.create_imm(context, device, &desc, &list) != ZE_RESULT_SUCCESS || list == nullptr) {
      std::fprintf(stderr, "b70_kvpark: immediate copy list failed; compute stream\n");
      return;
    }
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> input(
        list,
        probe.get_device(),
        sycl::ext::oneapi::level_zero::ownership::keep,
        sycl::property_list{sycl::property::queue::in_order{}});
    eng.queue = new sycl::queue(
        sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(input, probe.get_context()));
    eng.ordinal = ordinal;
    std::fprintf(stderr, "b70_kvpark: copy queue group %d\n", ordinal);
  });
  return eng;
}

sycl::queue& copy_queue() {
  Engine& eng = engine();
  if (eng.queue != nullptr) {
    return *eng.queue;
  }
  return c10::xpu::getCurrentXPUStream().queue();
}

// The torch transfer stream and the copy engine are different queues.
// A host wait is not required: the copy waits for the current stream,
// and the current stream waits for the copy. Same-queue fallback is a no-op.
void order_copy_after_current(sycl::queue& copy) {
  sycl::queue& current = c10::xpu::getCurrentXPUStream().queue();
  if (&current == &copy) {
    return;
  }
  sycl::event ready = current.ext_oneapi_submit_barrier();
  copy.ext_oneapi_submit_barrier(std::vector<sycl::event>{ready});
}

void order_current_after_copy(sycl::queue& copy, const sycl::event& done) {
  sycl::queue& current = c10::xpu::getCurrentXPUStream().queue();
  if (&current == &copy) {
    return;
  }
  current.ext_oneapi_submit_barrier(std::vector<sycl::event>{done});
}

const char* alloc_name(sycl::usm::alloc kind) {
  switch (kind) {
    case sycl::usm::alloc::host:
      return "host";
    case sycl::usm::alloc::device:
      return "device";
    case sycl::usm::alloc::shared:
      return "shared";
    case sycl::usm::alloc::unknown:
      return "unknown";
    default:
      return "other";
  }
}

int64_t alloc_host(int64_t nbytes) {
  TORCH_CHECK(nbytes > 0, "nbytes");
  const at::DeviceGuard guard(at::Device(at::kXPU, c10::xpu::current_device()));
  sycl::queue& probe = c10::xpu::getCurrentXPUStream().queue();
  void* ptr = sycl::malloc_host(static_cast<size_t>(nbytes), probe);
  TORCH_CHECK(ptr != nullptr, "sycl::malloc_host failed");
  auto kind = sycl::get_pointer_type(ptr, probe.get_context());
  std::fprintf(
      stderr, "b70_kvpark: malloc_host %lld bytes type=%s\n",
      static_cast<long long>(nbytes), alloc_name(kind));
  TORCH_CHECK(kind == sycl::usm::alloc::host, "malloc_host did not return USM host");
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
}

void free_host(int64_t ptr) {
  if (ptr == 0) {
    return;
  }
  const at::DeviceGuard guard(at::Device(at::kXPU, c10::xpu::current_device()));
  sycl::queue& probe = c10::xpu::getCurrentXPUStream().queue();
  sycl::free(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)), probe);
}

int64_t copy_batch(
    const torch::Tensor& src,
    const torch::Tensor& dst,
    const torch::Tensor& sizes,
    bool sync) {
  TORCH_CHECK(src.is_cpu() && dst.is_cpu() && sizes.is_cpu(), "pointer tensors are CPU");
  TORCH_CHECK(
      src.scalar_type() == torch::kLong && dst.scalar_type() == torch::kLong &&
          sizes.scalar_type() == torch::kLong,
      "pointer tensors are int64");
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && sizes.is_contiguous(), "contiguous");
  TORCH_CHECK(src.numel() == dst.numel() && src.numel() == sizes.numel(), "equal counts");
  const at::DeviceGuard guard(at::Device(at::kXPU, c10::xpu::current_device()));
  sycl::queue& q = copy_queue();
  order_copy_after_current(q);
  const int64_t* src_p = src.data_ptr<int64_t>();
  const int64_t* dst_p = dst.data_ptr<int64_t>();
  const int64_t* sz_p = sizes.data_ptr<int64_t>();
  int64_t bytes = 0;
  sycl::event last;
  bool any = false;
  for (int64_t i = 0; i < src.numel(); ++i) {
    const size_t n = static_cast<size_t>(sz_p[i]);
    if (n == 0) {
      continue;
    }
    last = q.memcpy(
        reinterpret_cast<void*>(static_cast<uintptr_t>(dst_p[i])),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(src_p[i])),
        n);
    bytes += static_cast<int64_t>(n);
    any = true;
  }
  if (any) {
    order_current_after_copy(q, last);
  }
  if (sync && any) {
    last.wait();
  }
  return bytes;
}

int64_t engine_ordinal() {
  const at::DeviceGuard guard(at::Device(at::kXPU, c10::xpu::current_device()));
  return engine().ordinal;
}

// 1 when this binary orders the copy queue against the current stream.
// Older builds do not export this op.
int64_t ordered() {
  return 1;
}

}  // namespace

TORCH_LIBRARY(b70_kvpark, m) {
  m.def("alloc_host(int nbytes) -> int");
  m.def("free_host(int ptr) -> ()");
  m.def("copy_batch(Tensor src, Tensor dst, Tensor sizes, bool sync) -> int");
  m.def("engine_ordinal() -> int");
  m.def("ordered() -> int");
}

TORCH_LIBRARY_IMPL(b70_kvpark, CompositeExplicitAutograd, m) {
  m.impl("alloc_host", alloc_host);
  m.impl("free_host", free_host);
  m.impl("copy_batch", copy_batch);
  m.impl("engine_ordinal", engine_ordinal);
  m.impl("ordered", ordered);
}
