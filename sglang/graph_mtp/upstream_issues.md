# Upstream issue drafts (sglang breakable-cuda-graph + XPU) -- found 2026-07-02 on dual Arc B70

Repro context: sglang 0.5.6.post3.dev6841+g09ca4fc96 fork, torch 2.12 xpu, Qwen3.6-27B hybrid GDN,
NEXTN spec (steps=10, draft=11), TP=2, --cuda-graph-backend-decode breakable.

## 1. BreakableCUDAGraphCapture.__exit__ double-ends the current segment during exception unwinding

If any eager_on_graph break function raises, the exception unwinds into __exit__, which calls
_end_current_segment() on a segment that the wrapper already ended -> capture_end() on a dead
graph. On XPU this is a near-NULL-deref SEGFAULT inside sycl modifiable_command_graph::
end_recording() that MASKS the real exception (we chased 4 phantom segfaults before finding it).
On CUDA it may error differently but the pattern is the same. Fix: track segment open/closed state
and make __exit__ tolerant on the exception path (or re-raise the original error first).

## 2. weak_ref_tensor import hard-raises on XPU (breakable backend unusable)

sglang/srt/compilation/weak_ref_tensor.py raises NotImplementedError at import for non-CUDA/NPU.
The breakable backend imports it lazily at the FIRST break -> combined with (1), an opaque
segfault. sgl-kernel-xpu #251 (2026-06-29) adds the XPU from_blob impl; the python module should
gate on availability and/or fall back to strong refs (correct, just holds pool memory).

## 3. BreakableCudaGraphBackend output handling breaks on spec decode

(a) _slice_output/_copy_output_to_buffer raise TypeError on LogitsProcessorOutput (the verify
    forward's return type). Needs dataclass-field recursion.
(b) capture_one slices stored outputs with shape_key.size, which is BS for the decode runner --
    but spec forwards emit bs*num_tokens_per_bs rows. Verify logits get truncated (e.g. 11x for
    NEXTN draft=11): "shape [1, 11] is invalid for input of size 2" in eagle_sample at replay.
    The slice length must be bs * num_tokens_per_bs.

## 4. (torch-xpu / oneCCL, informational) collectives inside XPUGraph capture

- oneCCL all_reduce RECORDS into a SYCL graph but deadlocks at replay (host-staged half never
  re-executes); all_gather HANGS capture outright. Any XPU graph-capture design must treat
  collectives as graph boundaries (or use a device-side recordable collective).
- One oneDNN primitive ("could not execute a primitive") fails under recording at bs=1/M=11 in
  the Qwen3.6 W8A8 stack while M=22/33/44 record fine -- not yet bisected.

## 5. (fork) breakable decode does not break at attention for non-deepseek/nemotron models

radix_attention.py only routes through breakable_unified_attention_with_output on the
extend/tc_piecewise path; decode/target_verify attention is captured directly. Models whose
attention kernels are not graph-capturable (intel_xpu XMX mha: sycl work_group_scratch_memory not
supported with SYCL Graph) need per-model eager_on_graph wraps or a generic decode-path break.
