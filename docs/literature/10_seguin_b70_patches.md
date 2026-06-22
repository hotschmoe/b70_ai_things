# 10 -- steveseguin/b70-optimization-lab: patches + MTP + qwen3.6-int8, mined for OUR stack

Date: 2026-06-22. Source: https://github.com/steveseguin/b70-optimization-lab (cloned; main + 5 codex/* branches
reviewed). Companion to [docs/P2P_GPU.md](../P2P_GPU.md) sec B/H. Most of Steve's ~70 patches are MiniMax-M2.7-MoE or
llama.cpp/GGML specific (skip). This doc extracts ONLY what applies to our vLLM-XPU compressed-tensors 27B/35B int8 work.

================================================================================
1. PATCHES / KNOBS APPLICABLE TO US (prioritized try-list)
================================================================================
HIGH VALUE (try on the 27B W4A8 single-card, where we have decode headroom):
- **n-gram GPU speculative decoding** (`patches/vllm-pp2-ngram-ppguards-active`): README says n-gram HELPED Qwen FP8
  (49.58 vs 48.09 t/s) though it hurt MiniMax. Our 27B is the Qwen case -> a free decode lever. Serve flag:
  `--speculative-config '{"method":"ngram","num_speculative_tokens":N,"prompt_lookup_max":...}'`. Worth an A/B on our W4A8.
- **async engine + static `compile_sizes=[1]`**: their single biggest decode combo (48.09 t/s; no-async = 27.31). We
  already pass compile_sizes=[1]; CONFIRM async scheduling is on (v1 default) -- a 1.7x swing if it's off.
- **gpu_memory_utilization 0.90-0.97 for KV headroom**: matches what we did for the single-card W4A8 (0.97).

GRAPH/COMM (only relevant once we patch TP=2 -- see sec H of P2P_GPU; we already got graph-captured TP=2 via SYCLKERNELS=1):
- `patches/vllm-xpu-graph-noop-communicator-capture` + `vllm-xpu-force-graph-with-comm-experiment` +
  `strict-gate-record-custom-allreduce-env`: his clone-safe-allreduce route (VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP etc.).
  We achieved the same end (capturable TP=2 allreduce) with the oneCCL `CCL_ENABLE_SYCL_KERNELS=1` flag, NO source patch.
  His patches are the fallback if SYCLKERNELS=1 proves unstable on a future model/host.
- `vllm-xpu-cudagraph-partition-collectives-negative`: NEGATIVE -> their handoff says `unset
  VLLM_XPU_CUDAGRAPH_PARTITION_COLLECTIVES` + `unset VLLM_XPU_CUDAGRAPH_STATIC_INPUT_COPY` for the promoted path.

SKIP (not our arch): all `VLLM_MINIMAX_*` (QK-RMS/oproj/MoE-delay allreduce fusions -- MiniMax linear-attn specific,
and ALL measured NEGATIVE vs his 37.5 ref anyway); all `GGML_*` (llama.cpp); EP/expert-parallelism (comm-dominated,
below TP4); oneCCL worker/affinity/threshold knobs (all regressed -- "leave default oneCCL alone").

================================================================================
2. MTP on B70 -- Steve's work (we asked specifically). VERDICT: not viable yet; root cause now known.
================================================================================
- **Loader patch** `patches/vllm-qwen35-mtp-force-block-fp8-clean` (OUR arch, qwen3_5_mtp): adds
  `VLLM_QWEN35_MTP_FORCE_FP8_BLOCK` + `VLLM_XPU_BLOCK_FP8_REQUANT`. Handles the hybrid case where the TARGET body is
  compressed-tensors FP8 but `mtp.safetensors` is **block-FP8** -- it builds `Fp8Config(weight_block_size=[128,128],
  activation_scheme="dynamic", ignored_layers=[<prefix>.fc])` for the MTP module so it loads. (A LOADER fix, not perf.)
- **Serve flag** (from `qwen36-35b-a3b-fp8-requant-c48-mtp.env`): `--speculative-config '{"method":"mtp",
  "num_speculative_tokens":3}'` + `--generation-config vllm`. So MTP serving on the 35B IS wired (method=mtp, k=3).
- **But it does NOT pay off on B70:** README perf MTP = **2.36 tok/s eager / 1.84 compiled** (catastrophic). And the
  int8 branch names the ROOT CAUSE (quote): *"The suppressed bonus is fed back as the next draft, then the exact target
  rejects it. This is a VERIFIER/KV/INPUT-POSITION BOUNDARY problem, not a draft-quality problem."* Spec paths are
  "gated on oracle k=1 parity repair" before any ngram/MTP work.
- **Reconciles + deepens OUR MTP finding** (docs/literature/09, our -19%): we attributed it to graph capture; Steve
  shows the deeper bug is the spec-decode VERIFIER rejecting correct drafts due to a KV/position-boundary mismatch on
  XPU. So MTP-on-B70 needs a verifier/KV-position fix in the vLLM-XPU spec-decode path, not just a graph or a better head.
  => For us: MTP stays parked; if we revisit, the test is `--speculative-config '{"method":"qwen3_5_mtp",
  "num_speculative_tokens":3}'` on a W8A16/FP8 body + the block-FP8 mtp loader, and the thing to watch is acceptance
  (k=1 parity), not t/s. W8A8-vs-W4A16 body precision is second-order to this verifier bug (consistent with doc 09).

================================================================================
3. branch codex/qwen36-quark-int8-tracking -- he HAS the 35B int8 MoE SERVING (via AMD Quark, not GPTQ)
================================================================================
This is the most important find for our Q6/Q7 (the 35B int8 we deferred, serve-gated on docs/kernel/18):
- **Quark W8A8 INT8** of Qwen3.6-35B-A3B serves on **4x B70 (TP4)** at **~99.4-99.8 tok/s corrected decode**, ~10
  ms/token, 32K ctx, ~8.58 GiB/rank. So an int8-act MoE DOES run on B70 vLLM-XPU -- via Intel/AMD **Quark** quant +
  a "persistent W8A8 MoE layerlet" / fused-MoE kernel path. This is concrete proof our docs/kernel/18 Track-A is real.
- **Method delta:** he used **Quark** (the AMD/Intel quantizer) to emit the W8A8 MoE, NOT llmcompressor-GPTQ (our path,
  which was multi-day at ~25-30 min/layer). Quark may be the faster/working production path for the 35B int8 -> worth
  evaluating Quark for our Q6/Q7 instead of grinding GPTQ. (FIND/COMMUNITY before BUILD, per QUANTS_TODO.)
- **Corroborations of our results:** (a) W8A8 with OFFSETS (asymmetric) REGRESSED (96 vs 99 t/s) + failed provenance ->
  symmetric W8A8 is right (we use sym). (b) **TP2 slower than TP4** (91 vs 99) with provenance drift -> matches our
  "TP is a capacity play, comm tax grows" finding (P2P_GPU H.6). (c) top-128 hot-expert tables: no win.
- Other branch configs of note: `qwen36-35b-a3b-fp8-requant-c48{,-mtp}.env` (FP8 W8A16 requant path + the MTP variant),
  `qwen3-vl-30b-a3b-fp8.env` (a 30B-A3B FP8 slot ~ our Qwen3-30B-A3B small-MoE kernel target, QUANTS_TODO sec 7).

================================================================================
4. CORROBORATIONS of our independent findings (confidence boosters)
================================================================================
- **fp8 KV cache broken on XPU:** README "E5M2 fails in XPU FlashAttention; E4M3 reached 37.15 (negative)". We hit the
  EXACT fp8_e5m2 KV reject 3x -> use fp16 KV. Confirmed not-us.
- **MTP catastrophic** (sec 2): matches our -19%.
- **TP2 = capacity not speed** (sec 3b): matches P2P_GPU H.6.
- **Gen3/PCIe BW caps allreduce:** "PCIe4 13.79 GB/s vs PCIe5 27.88" -> our Gen3 cross-die is worse still (P2P_GPU B.4/H.6).
- **>512-token prefill chunks fail** (Intel IGC compiler failure at 1024) -> keep max_num_batched_tokens<=512 if we chunk.

================================================================================
5. CONCRETE NEXT EXPERIMENTS FOR US (cheap -> deep)
================================================================================
1. [cheap] n-gram speculative on the 27B W4A8 single-card: `--speculative-config '{"method":"ngram",...}'` A/B vs plain
   decode (his Qwen FP8 +1.5 t/s). Our biggest free decode lever.
2. [cheap] confirm async scheduling is ON in our serve (1.7x risk if off).
3. [medium] evaluate **Quark** as the 35B int8 production path (he shipped it; our GPTQ was multi-day) -- unblocks Q6/Q7
   far cheaper than grinding llmcompressor.
4. [deep/parked] MTP: only worth it after the spec-decode verifier/KV-position bug is fixed in vLLM-XPU (his "oracle k=1
   parity"). Track upstream; not a quant problem.
