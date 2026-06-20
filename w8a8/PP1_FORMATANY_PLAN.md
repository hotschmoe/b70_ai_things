# PP-1: format_tag::any weight + cached reorder -- ready-to-execute implementation plan

GOAL: let oneDNN pick its VNNI/blocked DPAS weight layout for the int8 W8A8 GEMM (instead of the plain
row-major `s8::blocked:ab` we pin today). Targets the measured gaps: prefill 66-81% -> 90%+ of 367 TOPS,
and the wide-n decode 50% -> 85%+ BW (coalesced weight reads). Mirror IPEX `QMatmul.h` (L222-335).
Codex recipe verified. All GPU runs via gpu-run (CLAUDE.md).

## The two code changes (host: /mnt/vm_8tb/b70/vllm-xpu-kernels/csrc/xpu/onednn/)

### Change 1 -- onednn_ext.h:795, weight md -> format_tag::any (ONLY for the s8s8 int8 path)
The `matmul_primitive_create_and_cache<Ts>` template builds `wei_md = memory::desc({k,n}, wei_dt,
wei_strides)`. Make it `format_tag::any` so the pd is free to choose the packed layout:
```
  // gate on the joint dtype so we DON'T change w4a8/fp8 paths (which have their own packing):
  bool is_s8s8 = (Ts == joint_dtypes_t::s8_s8_f16 || Ts == joint_dtypes_t::s8_s8_bf16);
  auto wei_md = is_s8s8 ? memory::desc({k, n}, wei_dt, memory::format_tag::any)
                        : memory::desc({k, n}, wei_dt, wei_strides);
```
Keep src_md/dst_md as-is (src is per-call activations -> not reusable; dst stays plain).

### Change 2 -- int8_gemm_w8a8.h, reorder the user weight ONCE into the chosen layout, cache by ptr
After `matmul_primitive_create_and_cache(...)` returns the `primitive_ext& matmul_ext`:
```
  // query the layout the pd chose (blocked/VNNI if change-1 took effect)
  auto chosen_wei_md = matmul_ext.weights_desc();      // const_dnnl_memory_desc_t (already exposed, line 578)
  // plain user weight md ({k,n}, s8, NT strides) -- what mat2 currently is
  static thread_local std::unordered_map<void*, dnnl::memory> g_wei_reorder_cache;  // key = mat2.data_ptr()
  void* wkey = mat2.data_ptr();
  void* wei_exec_handle = wkey;                         // default: pass plain (no reorder needed)
  if (memory::desc(chosen_wei_md) != plain_wei_md) {    // pd wants a different layout
    auto it = g_wei_reorder_cache.find(wkey);
    if (it == g_wei_reorder_cache.end()) {
      auto plain_mem = make_onednn_memory(plain_wei_md, engine, wkey);
      dnnl::memory blocked_mem(memory::desc(chosen_wei_md), engine);   // owns a fresh buffer
      dnnl::reorder(plain_mem, blocked_mem).execute(strm, plain_mem, blocked_mem);
      strm.wait();
      it = g_wei_reorder_cache.emplace(wkey, std::move(blocked_mem)).first;
    }
    wei_exec_handle = it->second.get_data_handle();
  }
  // then in arg_handles: emplace_back(DNNL_ARG_WEIGHTS, wei_exec_handle)  // instead of mat2.data_ptr()
```
NB: `primitive_ext` already exposes `weights_desc()` (onednn_ext.h:578) and `make_weight()` (the execute path
calls `make_memory(weights_desc(), engine, handle)` -- so once the handle points at the reordered buffer,
the existing execute path Just Works with the blocked desc).

## Cache lifetime / correctness
- Key by `mat2.data_ptr()` -- vLLM's weight tensor is stable after `process_weights_after_loading` (the
  pointer does not move). The reorder is a ONE-TIME cost per weight, amortized to zero (read every token).
- The reordered `dnnl::memory` OWNS its buffer (allocated on the engine) -> survives in the static cache.
  ~15.3 GiB of weights -> the reordered copy is the SAME size (it replaces, conceptually, but we keep BOTH
  the plain (vLLM-owned) and the blocked (our cache) -- so +15.3 GiB VRAM. ACCEPTABLE for 14B on 32 GB
  (15.3 plain + 15.3 blocked + KV ~ 30 GiB tight). MITIGATION if OOM: free the plain weight after reorder
  (replace_parameter in process_weights), OR do the reorder in process_weights_after_loading (Python) and
  drop the plain copy -- the cleaner long-term design (do this if the C++ cache OOMs).
- ALTERNATIVE (cleaner, no double VRAM): move the reorder to `process_weights_after_loading` in xpu_int8.py:
  call a new `_xpu_C.int8_reorder_weight(w_q)` op that returns the blocked weight, replace_parameter, and the
  GEMM just uses it. More plumbing but no double-resident weight. Decide after measuring the C++-cache version.

## Build + validate
1. Edit the two files on the host. `scripts/44_build_int8_kernel.sh` (CPU, ~1-2 min, ccache). NB agent B: if
   touching the fused ops later, flip BASIC_KERNELS_ENABLED ON -- not needed for this onednn-only change.
2. Re-run `w8a8/20_microbench_int8_gemm.sh` (mount the rebuilt .so) -> PREFILL % of 367 TOPS should rise from
   66-81% toward 85-90%; DECODE wide-n (4096x11008) from 50% up. Also `VERBOSE=1` -> the weights md should now
   show a blocked tag (NOT `ab`) and a one-time `reorder` line at warmup (NOT per-call).
3. Correctness: a fingerprint A/B (baked vs rebuilt) on identical inputs (like w4a8/22_validate_b1.sh) -- the
   GEMM output must match (reorder is layout-only, numerically identical).
4. If prefill still <60% after this -> escalate to PP-2 (hand DPAS GEMM, doc 10 c.1). If >=85% -> PP-1 is the win.

## Why this is the right first move (the principle)
Hand the library `format_tag::any`, not a fixed stride: the JIT GEMM generator gates 2D-block prefetch on
`isPacked(B.layout)` and DPAS consumes a crosspacked systolic B. A plain B forces a worse access path or a
per-call repack. One offline reorder converts the every-token weight read into the layout the XMX array wants.
