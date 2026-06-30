# ZML TP=2 all-reduce vs our vLLM/sglang push-all-reduce -- relevance review

How our hand-rolled vLLM/sglang custom all-reduce relates to ZML's TP path on the dual B70,
and what can (and cannot) be leveraged. Companion to `ZML_W8A8.md` (M5 is the TP=2 W8A8 step).
Source review: `docs/P2P_GPU.md`, `docs/20260625_bcs_wedge_rootcause.md`,
`vllm/contrib/vllm_push_allreduce/`, `scripts/101-118*`, `zml/REVIEW_intel_arch.md`,
`docs/patch_applicability_matrix.md`, and the ZML clone `/mnt/vm_8tb/b70/zml`.

## A. What our vLLM/sglang push-all-reduce does (and why)

- Why: oneCCL on this cross-die fabric is slow (host-staged ~1.16 GB/s) or wedge-prone (P2P-on
  DEVICE_LOSTs in the multiproc serve). The only fast cross-die primitive is a POSTED PEER WRITE.
- Measured primitive asymmetry (P2P_GPU.md J.2): peer PUSH (posted write) 11.3 GB/s vs peer PULL
  (read) 3.24 GB/s -- PCIe writes are posted/fire-and-forget, reads are non-posted round-trips.
- Algorithm (2-rank): each rank PUSHES its buffer into the peer's scratch (one posted write, both
  directions concurrent), then both LOCAL-reduce mine+peer_scratch. Never reduce-into-peer (that
  pays read+write = 2.36 GB/s). The push is a 4-byte-word copy (element-wise bf16 writes don't
  coalesce: 0.83 vs ~10 GB/s). Cross-process via L0 IPC (zeMemGetIpcHandle + SCM_RIGHTS fd).
- Sync: a device spin-flag barrier HANGS on Xe2 (peer writes only visible at kernel completion).
  Prefill uses a host sense-reversing barrier (~1-2 us). Decode (graph-capturable) uses a
  cross-device L0 EVENT waited on by the OTHER card's COMMAND STREAMER (HW semaphore, not EU spin);
  IPC event pool must span both devices; submit on torch's current capture stream per call.
- Wins (27B-W8A8 TP=2 GRAPH+MTP): prefill TTFT 3.8x, throughput +80-126%, decode +8-10%, all at
  CCL_TOPO_P2P_ACCESS=0 (no wedge). Shipped as the shelf default PUSH_AR=1 PUSH_AR_GRAPH=1.

## B. How ZML does TP all-reduce instead

- IN-GRAPH collective: ZML emits a StableHLO `all_reduce` MLIR op in the SAME compiled graph as the
  math (`zml/ops.zig:142`, reducer block `:96-116`, replica groups, channel handle). For dense TP
  you don't call it -- you tag dims to mesh axes and Shardy/SPMD inserts+schedules+overlaps the
  collectives (`module.zig:579-620` use_spmd_partitioning, `Sharding.zig:48-52` Shardy default).
  The qwen3_5 dense TP layout is already standard: q/k/v .dout=.model, o_proj row-parallel, MLP
  gate/up intermediate-sharded, down_proj row-parallel (`qwen3_5/model.zig:441-443,495-498`).
  => The graph-break-around-an-opaque-collective problem that drove all of section A DOES NOT EXIST
  in ZML; the compiler owns both matmul and collective and overlaps them.
- BUT it routes through the SAME oneCCL/Level-Zero stack: the oneAPI PJRT plugin bundles oneCCL
  2022.0 (`platforms/oneapi/oneapi.bzl:14`, `libpjrt_oneapi.BUILD.bazel:23-24`), and
  `oneapi.zig:30-34` sets CCL_* env at load. So the StableHLO all_reduce lowers to a oneCCL
  all-reduce over L0 -- same transport, same hardware, same wedge risk.
- LIVE BUG (`oneapi.zig:33`): CCL_TOPO_P2P_ACCESS defaults from the wrong env var (CCL_ATL_TRANSPORT,
  = "ofi") -> garbage. ALWAYS export CCL_TOPO_P2P_ACCESS explicitly (the ZML serve scripts do).
- Topology difference that matters: ZML is SINGLE-PROCESS multi-device via PJRT (one binary drives
  both cards) = the H.12 raw-microbench topology that WORKED with P2P=1 at 9.7 GB/s, NOT the H.13
  vLLM multiproc-worker path that wedged. So ZML *might* tolerate P2P=1 -- unproven, validate attended.

## C. Verdict: do NOT port the kernel; DO port the knowledge

- (i) PJRT custom-call to inject our push kernel as a collective: NO. The collective is StableHLO IR
  lowered inside a prebuilt hermetic plugin; a custom-call is a per-device opaque op, not a
  cross-device collective, so SPMD won't insert/overlap it -- you'd lose ZML's structural advantage.
  patch_applicability_matrix.md:78 says the surgery is N/A *because its absence is the advantage*.
- (ii) Faster algorithm/env: the only real oneCCL lever is CCL_TOPO_P2P_ACCESS=1 (8.4x BW, 9.7 GB/s)
  and ZML's single-process path is where P2P=1 actually worked. CCL_ENABLE_SYCL_KERNELS capturability
  is MOOT (ZML disables XLA command buffers for oneAPI; no graph break to fix).
- (iii) Moot because XLA solves the graph-break differently: YES. ZML gets capturable/overlapped
  collectives for free; the only BW gap to our 11 GB/s push is closed (partway) by P2P=1's 9.7 GB/s,
  which is the one realistic in-ZML experiment.

## D. Recommendations for the ZML W8A8 TP=2 (M5) attempt

- Env: CCL_TOPO_P2P_ACCESS=0 (neutralize the oneapi.zig:33 garbage AND avoid the wedge first),
  ZE_FLAT_DEVICE_HIERARCHY=FLAT, ONEAPI_DEVICE_SELECTOR=level_zero:gpu. Pin GuC firmware 70.54.0
  before any TP=2 VRAM-heavy load (the real BCS-wedge fix, backend-independent). Watch the
  bazel-under-gpu-run flock leak (bazelisk shutdown at end; verify fuser gpu.lock.{0,1} empty).
- Discipline: gpu-run (both cards); pre/post xpu-health; B70_AUTO_RESET=1 but recovery is reboot-only;
  never SIGKILL mid-collective; ONE TP=2 experiment at a time; add a 2-rank "Paris" coherence probe
  (xpu-health's single-card probe does NOT catch collective-state corruption).
- Validate first with //examples/sharding, then Llama-3.2-1B TP=2, then the 27B.
- Measure: in-graph all-reduce cost at the hidden=5120 reduce shapes -- decode [1,5120]/[B,5120]
  (latency-bound) and prefill [2048,5120] (BW-bound); A/B P2P=0 vs P2P=1; compare vs sglang ~25 t/s.
  Expect P2P to be a prefill/TTFT win, not a decode win.

### The W8A8 down_proj SHARDED-REDUCE wrinkle (verify, do not assume)

down_proj is row-parallel: contracting/intermediate axis is .model-SHARDED. The per-token
activation quant `act_scale = max(|x|, axis=intermediate)/127` therefore reduces over a SHARDED axis.

- Our `QuantizedLinear.forward` (common_quant.zig) computes act_scale PER SHARD, quantizes, does the
  i8 partial dot, and DEQUANTS (x acc * wscale * act_scale) BEFORE returning. Shardy then inserts the
  row-parallel all_reduce(SUM) on the dequantized bf16/f32 output. This is NUMERICALLY CORRECT:
  per-shard dequant before the cross-shard sum = sum_shard sum_local (x_shard * w_full) = full dot.
  It is per-(token,shard) quant -- FINER groups than single-card per-token, so equal-or-better
  accuracy, NOT bit-identical to single-card. It needs only ONE sum-collective at bf16/f32 (== the
  bf16 down_proj collective cost; no extra max-allreduce).
- Shardy CANNOT legally sink the sum past the per-shard `x act_scale` multiply (operand differs per
  shard), so it is forced into the correct ordering. But VERIFY on GPU: (1) dump the compiled
  StableHLO and confirm the down_proj region has an `add`-reducer all_reduce and NO unexpected
  `maximum`-reducer all_reduce; (2) numeric parity of TP=2 W8A8 down_proj vs a single-card reference.
- HAZARD to watch: any partitioner choice that all-reduces the int32 accumulator BEFORE the per-shard
  scale would be WRONG. Keep the dequant unambiguously before the reduction (it already is).
