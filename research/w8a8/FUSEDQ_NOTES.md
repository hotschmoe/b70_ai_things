# FUSEDQ -- fuse per-token int8 activation quant into the W8A8 GEMM op (plan B1)

Target: kill the capture-persistent activation-quant hotspot in the custom W8A8
int8 path. `dynamic_per_token_int8_quant` cost ~101 us on a [1, 17408] down_proj
activation (ideal <1 us), ~35% of down_proj layer time, and PERSISTS under graph
capture (it is compute, not launch). Authority: `docs/kernel/23_b70_gemv_gemm_roofline.md`.

Known failure NOT repeated: swapping the decomposed quant for a STANDALONE opaque
custom op regressed 19% under capture (inductor could no longer fuse it). So the
fix folds the quant INSIDE the existing custom GEMM op boundary (one opaque node
that does BOTH), plus a cheap parallel-launch fix to the standalone kernel.

## Root cause of the 101 us (confirmed by reading the kernel)

`csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp` already reduced the per-row
absmax with a WORK-GROUP `sycl::reduce_over_group` -- but it launched a **32-wide
work-group (one sub-group) per row**. At M=1 decode that is a single 32-lane
work-group for the whole op: 32 lanes each stride ~544 elements of K=17408, and
with only 32 in-flight lanes global-memory latency is NOT hidden -> ~100 us
(matches the doc's "~15 us launch floor + ~5 ns/element serial" fit; it was
latency-bound, not literally 1-work-item). The reduction algorithm was already
correct; the launch geometry starved it.

## Design (two parts, both in the ONE shared kernel source)

1. **Widen the standalone quant launch** (cheap insurance; helps every caller
   incl. W4A8 which shares the act-quant). Same kernel, launched with a
   multi-sub-group work-group (up to 512 lanes/row, ~32 elements/lane) so far
   more loads are in flight per row. The group reduce still yields an identical
   absmax to every lane -> **bit-identical numerics** to the 32-lane launch.

2. **Fused op `int8_gemm_w8a8_fusedq`** (the real B1 win under capture). Takes an
   UNQUANTIZED f16/bf16 activation, runs the (now-parallel) per-token int8 quant
   SYCL kernel, then the SAME oneDNN s8s8 matmul as `int8_gemm_w8a8` -- both
   submitted on the SAME in-order XPU stream inside ONE torch op. Result: the
   quant is no longer a separate node in the captured graph (no extra graph node,
   no inductor-fusion loss), and it uses the parallel quant. Ordering is safe with
   no explicit sync: `vllm::xpu::vllmGetQueue()` (quant) and oneDNN's
   `GpuStreamManager::get_stream()` (matmul) both derive from
   `c10::xpu::getCurrentXPUStream()` -- the same in-order queue.

The python routing (`XPUInt8ScaledMMLinearKernel.apply_weights`) calls the fused
op when `B70_FUSEDQ=1` (default) and the op is present; `B70_FUSEDQ=0` restores
the two-step baseline for A/B. A `register_fake` for the fused op keeps XPU graph
capture / dynamo tracing consistent (output shape `A.shape[:-1] + (N,)`).

## Files changed

Kernel source (patched tree `/mnt/vm_8tb/b70/vllm-xpu-kernels-w8a8`, mirrored into
repo `kernels/`):
- `csrc/xpu/sycl/int8_quant_common.hpp`  NEW -- shared parallel quant kernel +
  `launch_dynamic_per_token_int8_quant()` launcher (`choose_int8_quant_local()`
  sizes the work-group). Mirrored to `kernels/int8_quant_common.hpp`.
- `csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp`  now includes the shared
  header and calls the launcher (same output contract).
- `csrc/xpu/onednn/onednn_matmul.cpp`  adds `int8_gemm_w8a8_fusedq(...)` (includes
  the shared quant header; reuses `check_and_create_output_tensor` +
  `oneDNN::dnnl_matmul_w8a8_int8`).
- `csrc/xpu/ops.h`  declares `int8_gemm_w8a8_fusedq`.
- `csrc/xpu/torch_bindings.cpp`  defines + impls
  `int8_gemm_w8a8_fusedq(Tensor A, Tensor B, Tensor B_scale, Tensor? bias,
  ScalarType? out_dtype) -> Tensor`.

vLLM-side routing (repo source of truth; the image bakes it):
- `vllm/contrib/vllm_int8_xpu/xpu_int8.py`  `B70_FUSEDQ` flag routes
  `apply_weights` through the fused op; adds `_fake_int8_gemm_fusedq` register_fake.

Build + test:
- `vllm/build_v0240_int8gdn_fusedq_so.sh`  NEW build (output to the NEW dir
  `/mnt/vm_8tb/b70/w8a8_kernel_v0240_fusedq/`; production `w8a8_kernel_v0240/`
  untouched).
- `vllm/test_fusedq.py`  GPU correctness+perf test.

## Build command (no GPU)

    bash vllm/build_v0240_int8gdn_fusedq_so.sh
    # -> /mnt/vm_8tb/b70/w8a8_kernel_v0240_fusedq/{_xpu_C.abi3.so, libgdn_attn_kernels_xe_2.so}
    # log: /mnt/vm_8tb/b70/build24/build_int8gdn_fusedq.log

## GPU validation (orchestrator)

### 1. Correctness + micro-perf (no serve)

    ROOT=/mnt/vm_8tb/b70
    PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
    ./bin/gpu-run --card 0 docker run --rm --device /dev/dri \
      -v $ROOT/w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so:$PKGD/_xpu_C.abi3.so:ro \
      -v $ROOT/w8a8_kernel_v0240_fusedq/libgdn_attn_kernels_xe_2.so:$PKGD/libgdn_attn_kernels_xe_2.so:ro \
      -v /mnt/vm_8tb/github/b70_ai_things/vllm/test_fusedq.py:/opt/test_fusedq.py:ro \
      -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
      --entrypoint bash vllm-xpu-env:int8g-v0240 -lc \
      'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; python /opt/test_fusedq.py'

PASS: max abs diff ~0 on every shape; standalone quant M=1 K=17408 well under the
old ~101 us; fused time <= quant+gemm sum.

### 2. Serve A/B (mount the new .so over the shelf entry)

The shelf serve already reads `GDN_SO` / `GDN_LIB`
(`rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh`). Point them at the fused build and
mount the updated `xpu_int8.py` over the baked one (same path the image bakes to):

    cd /mnt/vm_8tb/github/b70_ai_things
    GDN_SO=/mnt/vm_8tb/b70/w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so \
    GDN_LIB=/mnt/vm_8tb/b70/w8a8_kernel_v0240_fusedq/libgdn_attn_kernels_xe_2.so \
    ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh

To also swap the routing without rebaking, add a mount of the repo's
`vllm/contrib/vllm_int8_xpu/xpu_int8.py` over
`/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py`
(and the `/workspace/vllm` copy if present). A/B with `B70_FUSEDQ=0` vs `1`
(inject via `B70_EXTRA_ENV="B70_FUSEDQ=1"`). Gate coherence with
`vllm/gate_concurrent_coherence.py` and measure with `perf_probe.py`.
Landing rule (AGENTS.md): only ship if MEASURED faster-or-equal AND coherent
(sweep-gated). The captured end-to-end win is expected to be modest per the doc's
14B sim (the linear path is ~part of the step, and capture already removes the
LAUNCH part) -- the point is removing the capture-PERSISTENT compute, so measure
the captured (CGMODE=PIECEWISE) decode, not eager.

## Rollback

- Production is untouched: the shelf still defaults `GDN_SO`/`GDN_LIB` to
  `/mnt/vm_8tb/b70/w8a8_kernel_v0240/` and the baked image's `xpu_int8.py`.
- To disable the fused route in a fusedq-mounted serve: `B70_FUSEDQ=0`.
- To fully revert: do not mount the fusedq `.so` / `xpu_int8.py`; nothing else
  changed. The `w8a8_kernel_v0240_fusedq/` dir and `xpu_int8.py` flag are additive.
