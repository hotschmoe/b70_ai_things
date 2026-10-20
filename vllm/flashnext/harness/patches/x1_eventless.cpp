// SPDX-License-Identifier: BSD-3-Clause
// Adapted from PyTorch XPUGraph.cpp at 08187d9e0fba026dc8217405802ab5381dc88d90.
// Process-local interposition; rebuild against the installed Torch headers.
#include <ATen/xpu/XPUGraph.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/XPUGraphsUtils.h>
#include <c10/core/DeviceGuard.h>
#include <atomic>
#include <cstdio>
#include <unistd.h>

static std::atomic<uint64_t> replay_count{0};

extern "C" uint64_t b70_x1_eventless_replay_count() {
    return replay_count.load(std::memory_order_relaxed);
}

namespace at::xpu {
void XPUGraphImpl::replay() {
    TORCH_CHECK(capture_ended_, "XPUGraph replay requires completed capture");
    if (!has_graph_exec_) {
        TORCH_INTERNAL_ASSERT(keep_graph_);
        instantiate();
    }
    c10::OptionalDeviceGuard guard{capture_stream_.device()};
    for (auto& [state, increment] : captured_generator_states_) {
        // XPUGeneratorState helpers are hidden in this Torch build. Mirror
        // replay_prologue/increase using their public state fields.
        TORCH_CHECK(currentStreamCaptureStatus() == CaptureStatus::Executing,
                    "Cannot prepare replay during capture");
        if (increment) {
            TORCH_INTERNAL_ASSERT(!state->capturing_);
            TORCH_INTERNAL_ASSERT(state->philox_offset_per_thread_ % 4 == 0);
            state->seed_extragraph_.fill_(int64_t(state->seed_));
            state->offset_extragraph_.fill_(
                int64_t(state->philox_offset_per_thread_));
            state->philox_offset_per_thread_ += ((increment + 3) / 4) * 4;
        }
    }
    auto& queue = at::xpu::getCurrentXPUStream().queue();
    sycl::ext::oneapi::experimental::execute_graph(queue, *graph_exec_);
    if (replay_count.fetch_add(1, std::memory_order_relaxed) == 0) {
        std::fprintf(stderr, "X1_EVENTLESS_REPLAY_ACTIVE pid=%d\n", int(getpid()));
    }
}
} // namespace at::xpu
