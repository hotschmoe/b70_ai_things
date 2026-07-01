# zml INT8 performance handoff -- fused GEMM/GEMV, MTP amortization, P2P comms (new session)

Paste this as the opening prompt for a fresh, DEDICATED session. It is self-contained: a new
session with no prior context can execute from here. Scope is deliberately large and deep -- the
goal is to milk the Intel Arc Pro B70 (Battlemage, oneAPI) for every bit of int8 throughput:
profile each data step, disassemble instructions, hand-tune, hand-write Zig and (where it pays)
hand-write GPU assembly / low-level XeTLA/DPAS.

---

## 0. Mission

Make the zml (oneAPI PJRT over XLA/StableHLO) W8A8 serve of qwen3.6-27b ("qwen3_5" model) FAST on
dual B70 TP=2. Profiling this session found decode is ~79% GDN (Gated-DeltaNet linear-attention) and
~21% int8 full-attention, so the priorities are:

1. **GDN is the PRIMARY lever (~79% of decode).** Optimize the 48 bf16 GDN layers: int8 their
   projections (via the fused w8a16 matmul below) to halve weight bandwidth, AND replace the naive
   per-step f32 `stablehlo.while` delta-rule scan with a fused/chunked scan kernel (the compute-bound
   part that also blocks MTP-verify amortization).
2. **Fused w8a16 decode matmul** (both the GDN projections and the 21% int8 full-attn layers): s8
   weight read + f16/bf16 XMX, epilogue dequant, NO bf16 weight materialization (not the oneDNN
   int8-GEMV trap it is today).
3. **Prefill saturates int8 XMX** (s8xs8->s32 DPAS; already ~2.1-2.6x bf16 at large M -- verify it
   holds and push it).
4. **MTP/spec-decode then amortizes** the already-working 98.8%-accept verify into ~1.9x -- for FREE
   once (1)+(2) land (the MTP code is DONE and byte-exact; it just needs the verify to stop costing
   ~Kv x a decode, which is mostly the GDN f32 scan running Kv sequential steps).
5. **TP=2 collectives are int8-aware / cheaper** (P2P GPU comms): reduce int8 partials, not bf16;
   evaluate P2P BW; keep the box wedge-free.

Target: close 13.7 t/s -> ~22-30+ t/s decode (sglang W8A8+fused+MTP on the SAME box hits ~23.8 t/s
= 2.6x bf16; that is the existence proof the hardware can do it).

## 1. Where things stand (what's DONE, on branch `zml-w8a8-optimize`, pushed)

- **W8A8 serve works, TP=2, coherent, 13.7 t/s, wedge-free.** M0-M5 + act-quant dedup done. The
  int8 math is `Tensor.dotAcc(.i32, ...)` = genuine `stablehlo.dot_general(s8, s8)` with
  `preferred_element_type=s32` (in `zml/tensor.zig`; `dotAcc`/`dotGeneralAcc`). q/k/v/gate/up run
  full W8A8; o_proj/down_proj run WEIGHT-ONLY int8 (see below); GDN/vision/lm_head/embed stay bf16.
- **MTP/NEXTN spec-decode is COMPLETE + byte-exact** (this session, 2026-07-01). Draft head 96.2%
  accept (measure mode); full drive loop (draft -> s=Kv verify -> accept/commit -> GDN rollback)
  produces output BYTE-IDENTICAL to greedy MTP-off, 98.8% accept, avg 1.99 tok/verify, no wedge.
  Files: `qwen3_5/model.zig` (MtpHead/MtpDraft/MtpPrefill/GdnSnapshot/SliceHidden + GDN
  forwardVerify), `qwen3_5/inference.zig` (MtpExes: mtp prefill/draft/verify/snapshot/slice exes),
  `qwen3_5/session.zig` (Mtp state + runDecodeMtp). Run: `zml/run_w8a8_mtp_drive_tp2_gpu.sh`
  (ZML_MTP=1 drive / ZML_MTP_MEASURE=1 observe). Design + status: `zml/ZML_MTP_PLAN.md`.
- **THE BLOCKER (measured this session):** MTP gives NO speedup (10.5-12.7 t/s <= 13.7). Per-iter
  drive timers: draft 4.2ms | GDN snapshot 0.7ms | **VERIFY 143.7ms** | commit 6.0ms. The s=Kv=2
  verify costs ~2x a single s=1 decode -> it does NOT amortize. Root cause: decode is COMPUTE-bound
  at small M, not bandwidth-bound. Evidence: the MTP draft reads the 2.5 GiB bf16 lm_head in 4.2ms
  (~600 GB/s -- bandwidth is fast), yet a 64-layer int8 decode is ~73ms = ~240 GB/s effective
  (~5x below bandwidth). int8 full-attn layers are ~1.14ms at M=1 and ~2.2ms at M=2 (LINEAR in M).

- **SURPRISE (per-layer-type profiling, ZML_PROFILE_LAYERS=1, this session -- READ THIS):** decode
  time is DOMINATED BY THE GDN, not the int8 matmuls. Split: **full-attn (int8 x16 layers) = 21%
  (2423ms) vs GDN linear-attn (bf16 x48 layers) = 79% (9073ms)** over a full generation. So the
  16 int8 full-attention layers are only ~1/5 of decode; the 48 Gated-DeltaNet layers (bf16
  projections nn.Linear + f32 sequential recurrent scan via stablehlo.while + conv1d) are ~4/5. The
  fused int8 GEMM is therefore a SECONDARY (~21%) lever; the GDN is the PRIMARY (~79%) one. The MTP
  verify likely fails to amortize mostly because the GDN f32 scan runs Kv sequential steps (compute-
  bound, linear in s) while the int8 full-attn GEMV also scales with M. PROFILE THE GDN INTERNALS
  FIRST (bf16 projections -- bandwidth-bound, should amortize -- vs the f32 recurrent scan -- does
  not) before deciding where to spend kernel effort.

## 2. What we learned from `kernels/` (the sglang/vLLM fused int8 ops -- the blueprint)

`kernels/` is the SHARED custom-kernel source (oneDNN-based) that sglang+vLLM compile per backend.
Read all of: `kernels/README.md`, `kernels/int8_gemm_w8a8.h`, `kernels/int8_gemm_w8a16.h`,
`kernels/int8_gemm_kernel.patch`. Key design (these are the target zml must match):

- **w8a8 (prefill), `dnnl_matmul_w8a8_int8`:** s8 src x s8 weight -> f16/bf16. oneDNN accumulates
  s32 internally (native s8s8s32 on Battlemage XMX) and applies per-token SRC scale (mask
  (1<<0)|(1<<1), group {1,k}) + per-channel WEIGHT scale (mask 1<<1) in the epilogue. This is the
  FAST large-M path. zml's `dotAcc(.i32)` already reaches this regime at large M; confirm parity.

- **w8a16 (decode), `dnnl_matmul_w8a16_int8` -- THE ONE zml IS MISSING:** f16/bf16 activation x s8
  weight -> f16/bf16, in ONE fused launch. It does NOT quantize the activation. It passes the s8
  weight directly and sets `pattr.set_scales(DNNL_ARG_WEIGHTS, ...)` (per-channel / per-block /
  per-tensor) so oneDNN dequantizes the weight in the matmul epilogue, plus
  `pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*allow_downconvert=*/true)` (bf16 variant for
  bf16 act) so the compute runs on the f16/bf16 XMX/DPAS path. Result: reads only the int8 weight
  (half the bytes vs bf16) AND uses fast XMX -> weight-bandwidth-bound, fast decode. The primitive
  is created-and-cached keyed by (dtypes, trans, bias, m, n, k, lda/ldb/ldc, dev, scale-group).
  The patch adds the `f16_int8` / `bf16_int8` joint-dtype mappings + dispatch to
  `vllm-xpu-kernels`' `onednn_ext.h`.

- **Why zml's current "weight-only int8" is a dead end (0.5-1.2x) while this w8a16 is fast:** zml's
  `QuantizedLinear.weightOnly()` computes `x.dot(self.dequantWeight(x.dtype()))`, i.e. it
  MATERIALIZES a full bf16 weight (`weight.convert(f32).mul(scale)`) and then does a bf16 GEMM ->
  it reads int8 but writes+reads bf16, destroying the bandwidth win. The oneDNN w8a16 kernel FUSES
  the dequant into the matmul epilogue -> no materialization -> real bandwidth win. Matching this
  fusion in zml is the core task.

## 3. The zml integration problem (the hard, open part -- research first)

zml emits StableHLO; the oneAPI PJRT plugin (`libpjrt_oneapi`, a PREBUILT binary -- MODULE.bazel
`use_repo(oneapi, "libpjrt_oneapi")`) lowers it to oneDNN. Two facts constrain us:
- The prebuilt plugin IGNORES XLA_FLAGS / ONEDNN_VERBOSE dumps (empty output), so you CANNOT
  inspect the emitted HLO or oneDNN dispatch from the outside. Validate by BEHAVIOR (A/B timings,
  byte-identical outputs) -- or build a plugin you can instrument.
- zml HAS a CustomCall surface: `zml/kernel.zig` (stablehlo.custom_call, output-operand aliasing,
  `callTpuCustomCall` for TPU). Whether the oneAPI/XPU PJRT honors XLA FFI custom calls registered
  for the XPU backend is UNKNOWN and is the first thing to determine.

Candidate integration paths (evaluate, pick, or invent):
1. **Emit an HLO pattern the plugin's oneDNN pass already fuses into weight-decompression.** Test
   whether `dot_general(bf16_act, s8_weight)` with `preferred_element_type=bf16` + a
   convert/broadcast-mul the plugin recognizes lowers to the fused w8a16 (no materialization).
   Cheapest if it works; likely it does not (XLA usually materializes the convert). Probe via the
   M=1 A/B time (fused ~= bf16-weight bandwidth/2; materialized ~= worse than bf16).
2. **XLA FFI / custom call to oneDNN.** Register an XPU custom call that calls the SAME oneDNN
   `dnnl_matmul_w8a16_int8` / `dnnl_matmul_w8a8_int8` (reuse `kernels/*.h` verbatim). Requires the
   PJRT plugin to expose the FFI registry, and a way to get the SYCL queue/engine from PJRT. This is
   the sglang/vLLM approach adapted to XLA. Most robust if the plugin allows it.
3. **Build our own oneAPI PJRT plugin / XLA oneDNN pass** (from the intel-extension-for-openxla /
   xla source) with the weight-decompression fusion + instrumentation. Heaviest, but removes the
   prebuilt-binary blindfold and lets us profile/disassemble the actual GEMM.
4. **Bypass XLA for the hot matmuls**: a zml custom op that runs a hand-written XeTLA/DPAS int8
   kernel (Zig + SYCL/Level-Zero, or ESIMD/CM, or raw Gen assembly) for the specific qwen shapes.
   This is the "hand-write assembly" endgame; do it for the shapes that dominate after profiling.

## 4. Concrete work plan (profile-driven; do NOT guess -- measure each step)

- **P0 -- PROFILE the GDN internals (DONE at the layer level: GDN=79%, int8-attn=21%; now go one
  level deeper).** The 48 GDN layers dominate decode. Each GDN layer = bf16 projections (in_proj_qkv
  [10240,5120], in_proj_z/b/a, out_proj, conv1d -- all nn.Linear bf16) + an f32 recurrent delta-rule
  scan (`zml.nn.GatedDeltaNet.forward` = a `stablehlo.while` over s steps; runs Kv steps at verify).
  Split the GDN time between the bf16 projections (bandwidth-bound, SHOULD amortize at M>1) and the
  f32 sequential scan (compute-bound, does NOT amortize -- likely the real verify blocker) + conv1d.
  Also confirm at s=1 vs s=2 to see which parts scale with s. Instrument like the ZML_PROFILE_LAYERS
  timer (session.zig / inference.zig Runner) but inside GatedDeltaNet.forward, or compile
  projections-only vs scan-only GDN variants. THIS decides the primary kernel target.
- **P1 (PRIMARY, ~79%) -- GDN optimization.** Two sub-levers, prioritized by P0:
  (a) INT8 the GDN bf16 projections via the fused w8a16 weight-decompression matmul (section 3) --
      halves their weight bandwidth (48 layers of bf16 proj is the bulk of decode bytes). Needs an
      int8 GDN checkpoint (or on-the-fly quant) + the fused kernel; today the GDN is entirely bf16.
  (b) FUSED/CHUNKED delta-rule scan kernel (sglang/FLA "chunked_gated_delta_rule") to replace the
      naive per-step `stablehlo.while` f32 scan -- this is the compute-bound part that blocks MTP
      verify amortization (Kv sequential steps). A hand-written XeTLA/SYCL/Zig chunked scan is the
      likely endgame. This is on the critical path for BOTH decode and the MTP verify.
- **P2 (SECONDARY, ~21%) -- Fused w8a16 decode matmul for the full-attn int8 layers** (section 3).
  Make the M=1/M=2 int8-weight x f16-act matmul bandwidth-bound. Bench at the real qwen shapes
  (q_proj [12288,5120], o_proj [5120,6144], gate/up [17408,5120], down [5120,17408]) at M=1,2,4,8.
  Success = int8-weight decode ~2x faster than today AND >= bf16-weight bandwidth. Wire into
  `QuantizedLinear` as the decode path (replace the materializing weightOnly + the s8xs8 dotAcc at
  small M). The SAME fused w8a16 kernel serves P1(a).
- **P3 -- Confirm/keep the w8a8 prefill XMX path** (dotAcc at large M). Re-bench with the sweep
  harness (`//examples/w8a8_sweep`) to confirm 2.1-2.6x still holds; fix if regressed.
- **P4 -- MTP amortization + overhead trim** (after P1/P3 make the verify amortize): re-measure the
  drive loop (should jump to ~1.9x). Then trim the ~10ms/iter overhead: the accept catch-up should
  use a KV-only MtpPrefill-style update (no lm_head/sampler); cut host<->device syncs (batch the
  token readbacks); consider per-step GDN state emission to drop the snapshot; sweep K (2,3) once
  the verify is cheap (accept decays 0.84/0.57/0.46 per draft position, so larger K helps only if
  the verify truly amortizes). Re-validate BYTE-IDENTICAL each time.
- **P5 -- P2P GPU comms / TP=2 collectives.** o_proj/down_proj are row-parallel; today they reduce
  in bf16 (the wedge-safe path). Evaluate reducing int8/int32 partials (less bytes), P2P access
  (CCL_TOPO_P2P_ACCESS -- currently forced 0 to dodge the wedge; A/B the BW carefully), and whether
  in-graph Shardy all_reduce can be replaced/augmented. See `zml/ZML_TP_ALLREDUCE.md` and the
  memory `zml-tp-allreduce-pushar` (push-AR was ruled out; single-process in-graph is the win).
  DANGER: TP=2 has a stochastic BCS/oneCCL wedge (reboot-only on this box) -- run attended.

## 5. Hardware facts (Battlemage / B70) -- the roofline you are optimizing against

From memory `b70-int8-xmx-roofline` (verify against it): INT8 XMX = 2x bf16 peak, ~367 TOPS int8 /
~608 GB/s per card. `preferred_element_type=s32` is MANDATORY for int8 dot (dotAcc already does it).
Real qwen shapes are DPAS-aligned (M8/N16/K32). **M=1 decode is the GEMV trap: s8xs8 int8 GEMM at
M=1 dispatches to oneDNN's GEMV path and NEVER XeTLA/DPAS** -- this is the crux (llama.cpp #21517
class). int8 saturates ~245 TFLOP/s (~67% of peak) at large M; nk weight {n,k} layout is optimal
(kn is 1.7x slower -- do NOT transpose); i32-store beats bf16-store; the ~1.66x->2x gap is the
s8xs8 +128 tax + prologue-quant. M=1 int8 GEMV measured ~309 GB/s vs bf16 ~434 GB/s.

## 6. Build / run / GPU discipline (this box, LOCAL, shared with prod)

- REPO (commit here): `/mnt/vm_8tb/github/b70_ai_things`, branch `zml-w8a8-optimize`. The zml
  contribution is a PR-ready patch `zml/patches/zml_w8a8.patch` (~3.1k lines) + browsable copies
  under `zml/examples/llm/` + docs.
- BUILD CLONE (edit + build here; git-ignored upstream zml at HEAD 89b0908c + our working-tree
  changes): `/mnt/vm_8tb/b70/zml`. After edits, regenerate the patch + sync copies BACK to the repo,
  then commit ON THE REPO (cwd matters -- `cd` persists; do the `git commit` from the repo dir):
    cd /mnt/vm_8tb/b70/zml && git add -A && git diff --cached HEAD -- examples zml/tensor.zig \
      > /mnt/vm_8tb/github/b70_ai_things/zml/patches/zml_w8a8.patch && git reset -q
    cp examples/llm/models/qwen3_5/*.zig examples/llm/main.zig examples/llm/models/common_quant.zig \
      /mnt/vm_8tb/github/b70_ai_things/zml/examples/llm/... (as changed)
    cd /mnt/vm_8tb/github/b70_ai_things && git add -A && git commit ...   # trailer below
  Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- CPU build (fast, type-check only, no GPU, ~160s):
    cd /mnt/vm_8tb/b70/zml && ~/.local/bin/bazelisk build //examples/llm:llm --config=release
- GPU run (oneAPI, TP=2, ATTENDED): `./bin/gpu-run bash zml/run_w8a8_serve_tp2_gpu.sh` (13.7 baseline)
  / `zml/run_w8a8_mtp_drive_tp2_gpu.sh` (MTP). Env inside: ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  (BOTH cards; `level_zero:0` = ONE card; NOT `level_zero:gpu:0` -> parse error),
  CCL_TOPO_P2P_ACCESS=0, ZE_FLAT_DEVICE_HIERARCHY=FLAT. The 27B W8A8 is 32.6 GiB > one card's 30.3
  GiB usable -> TP=2 ONLY (single card OOMs).
- GEMM microbench harness: `//examples/w8a8_bench` (single shape) + `//examples/w8a8_sweep`
  (--shape=all --mset --variant --layout=both, median/min/max, %-of-peak). Extend these for w8a16.
- GOTCHAS: (a) `bazelisk` under `gpu-run` LEAKS a bazel daemon holding `gpu.lock.*` for ~3h ->
  ALWAYS `~/.local/bin/bazelisk shutdown` after a gpu-run and check `fuser /mnt/vm_8tb/b70/gpu.lock.0`
  is empty before restoring prod (the serve scripts already do this). (b) TP=2 BCS/oneCCL wedge is
  REBOOT-ONLY on this box; run TP=2 attended; `bin/xpu-health` before/after; `bin/xe-reset` if
  wedged. GuC firmware pinned 70.54.0. zml's single-process PJRT has NOT wedged across ~10 TP=2 runs
  but stay disciplined. (c) A transient VFS "reached unreachable code" in the concurrent safetensors
  loader (`file.File.getFileHandle`) can crash a load ~1/6 runs -- just re-run. (d) Notifications
  interrupt foreground `sleep`; wait on background tasks via run_in_background + an until-loop.

## 7. Daily driver (prod) -- restore when the GPU is free

The production daily driver (sglang W8A8 TP=2 on :18080 + WebUI :3000) is DOWN for this work.
Restore: `DD_API_KEY=$(cat /mnt/vm_8tb/b70/secrets/dd_api_key) ./vllm/daily_driver_serve.sh start`
then confirm `:18080/health` 200 + `bin/xpu-health` HEALTHY. Stop again for GPU work with
`./vllm/daily_driver_serve.sh stop` (frees the lease). Keep it down only while actively on the GPU.

## 8. Validation methodology (non-negotiable)

- Correctness first: any int8 GEMM change must keep the greedy serve output BYTE-IDENTICAL to the
  13.7 baseline ("What is the capital of France? Answer in one short sentence." topk=1 ->
  "The capital of France is Paris."). For MTP, byte-identical to MTP-off greedy is the oracle.
- Perf: microbench the kernel in isolation (w8a8_bench/sweep) at real qwen shapes AND M=1,2,4,8,
  then end-to-end decode t/s vs 13.7. Report %-of-608GB/s (decode) and %-of-367TOPS (prefill).
- Instrument with host wall-clock between device syncs (the prebuilt plugin gives no HLO/oneDNN
  dump). If you build your own plugin, add real oneDNN verbose + a SYCL/Level-Zero profiler.

## 9. References

- Kernels (the blueprint): `kernels/README.md`, `kernels/int8_gemm_w8a8.h`,
  `kernels/int8_gemm_w8a16.h`, `kernels/int8_gemm_kernel.patch`. sglang build recipe +
  shim: `research/w8a8/W8A8_BUILD.md`, `sglang/patches/w8a8_shim.py`; vLLM: `vllm/images/int8g/`.
- zml int8: `zml/tensor.zig` (dotAcc/dotGeneralAcc), `zml/examples/llm/models/common_quant.zig`
  (QuantizedLinear: forward / forwardQuant / weightOnly / rowParallelW8A8 / dequantWeight),
  `zml/examples/llm/models/qwen3_5/model.zig` (Mlp/SelfAttn/GatedDeltaNet + the MTP modules).
- zml docs: `zml/ZML_W8A8.md` (scheme + progress), `zml/ZML_MTP_PLAN.md` (MTP design + STATUS),
  `zml/W8A8_FEASIBILITY.md`, `zml/ZML_TP_ALLREDUCE.md`.
- JOURNAL.md 2026-06-30..07-01 entries (W8A8 + MTP arc, config->command->result->verdict).
- Memories: `b70-int8-xmx-roofline`, `zml-w8a8-cpu-validated`, `w8a8-fused-int8-kernels-mtp`
  (sglang got 23.8 t/s = 2.6x bf16 with fused kernels+MTP -- the existence proof),
  `zml-tp-allreduce-pushar`, `sglang-mtp-works-on-xpu`.
- Prior B70 int8/GDN analysis: `docs/kernel/*` (12_mtp_specdecode_plan.md,
  21_gdn_spec_capture_issue.md, and the int8 GEMV/GEMM notes).

## 10. First moves for the new session

1. Read `kernels/*` + `zml/ZML_MTP_PLAN.md` STATUS + this file. Bring the daily driver down.
   (The layer-level split is already done: GDN=79%, int8-attn=21%, via ZML_PROFILE_LAYERS=1 in
   inference.zig Runner + session.zig.)
2. P0-deep: split the GDN's 79% between bf16 projections (should amortize) and the f32 recurrent
   scan (does not) + conv1d, at s=1 vs s=2 (one attended TP=2 run). This picks P1(a) vs P1(b) order.
3. Determine the integration path (section 3): probe whether a bf16xs8 HLO pattern fuses (cheap
   A/B at M=1), else scope the XLA FFI custom-call-to-oneDNN route (reuse kernels/*.h).
4. Land the biggest GDN lever first (int8 projections via fused w8a16, and/or the chunked scan);
   bench in isolation then end-to-end; keep byte-identical. Then the 21% int8 full-attn matmul.
5. Re-measure MTP drive -- expect the ~1.9x to appear. Then trim overhead + P2P comms.

The MTP is already done and correct; this session's job is the GDN + int8 compute path that lets it
(and plain decode) actually be fast on the B70. Commit + push at each measured, byte-identical
milestone. NOTE the surprise: the GDN linear-attention -- not the int8 GEMM -- is the main decode
cost; do not over-invest in the int8 matmul before profiling the GDN internals.
