# Exact source55 T512 Q8 projection direction control

CONFIG -> Actual55 SDK041742Z-stwovdcr/receiptc9d197af; unchanged existing
hc_q8_0_project_f32_token_reuse kernel. Compare whole-T512 tile2 against tile4
separately for down K10240/M320 and up K320/M10240. This is not an eight-row
slice benchmark, a tile sweep, an engine change, or a new arithmetic route.
Current coalesced HC uses tile2 for both directions; down2/up4 is only a future
candidate if actual T512 evidence supports it. Prior T8 results cannot decide
T512 behavior. Natural source55 and all previous fixtures remain immutable.

COMMAND -> Root compile kernels/hc_q8_t512_direction_fixture_v1.cpp against
actual55 projection source/header using unchanged production O3/c++20 precise,
subgroup32, per-kernel split and correctly-rounded divide/sqrt flags. The
compiler argv is in hc-q8-t512-direction-fixture-source-plan-v1.json. Reuse
root's original protected NoDri compiler/controller and original card1 owned
operator/health/kernel/normal-removal route, with full /dev/dri exposure and
ZE_AFFINITY_MASK=1. Pass B70_HC_EXPECTED_PCI=0000:44:00.0. The fixture requires
a LevelZero GPU, supported PCI-query aspect, and exact actual BDF before any
USM or kernel submission. Do not repeat the single-visible card1 node plus
global-mask1 combination that previously crashed in libze.
The author has not compiled or touched any GPU.

RESULT -> Source-only fixture, actual operator/timing pending. Two cases only.
Each direction uses deterministic dense original Q8_0 blocks and finite F32
inputs, initializes a whole-T512 tile2 reference, checks both tile outputs
bitwise, warms both three rounds, then records nine alternating sequential
2/4 host-submit-plus-original-queue-wait samples. Output readback and bitwise
comparison occur after the timing interval. No performance threshold or device
execution-time claim exists. Both tiles also record/replay a graph after input
changes and compare against independently rerun tile2. No model payload.

VERDICT -> Decide direction-specific launch tiling only after root obtains
actual bitwise, graph, timing and health evidence. This fixture changes no
native arithmetic or model path. Original decoded Q8 scale/code, lane FMA
column order and XOR32 reduction are exactly the existing source55 kernel.
No GEMM, quant conversion, global scratch, custom wait, or captured operand.
Host submit+wait is measured, not physical memory bandwidth or device-only
latency. The reported sizes are literal buffer payloads, not measured traffic.

| Direction | Tokens | Q8 weights bytes | F32 input bytes | F32 output bytes |
| --- | ---: | ---: | ---: | ---: |
| down10240->320 | 512 | 3481600 | 20971520 | 655360 |
| up320->10240 | 512 | 3481600 | 655360 | 20971520 |

Owner storage: four USM allocations and at most two heap graph/executable
pairs per direction. Host weight/input backing vectors outlive owner scope.
Original queue drains before every graph deletion and USM free. Failed drain
retains storage and exits86; failed free exits87. Readback failure exits86
before releasing its temporary host buffers. Recording-attempt ownership is
retained even if graph startup throws. Each direction completes before the
next creates its own buffers. No full model qualification transfers.

PCI query API reference: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_device_info.asciidoc . Actual compiler/API and device support remain root compile/runtime checks.
