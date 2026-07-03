# DFlash on XPU -- feasibility spike (2026-07-03)

**Verdict: GO -- DFlash serves COHERENTLY on the dual-B70 box, TP=2, first try, zero code
changes.** Slower than the NEXTN MTP baseline at the spike settings (over-drafting on the
cold bench workload); the follow-up is accept-length tuning + drafter-cost reduction, not
porting work. Part of the 2026-07-03 faster-DD session (docs/20260703_faster_dd_plan.md B2).

## What DFlash is

Block-diffusion drafting (Z Lab, ICML 2026, arXiv:2602.06036): a small draft model emits a
whole block of draft tokens in ONE parallel forward pass (non-causal within the block),
conditioned on target hidden states injected as KV. On CUDA: accept length ~6.5 vs EAGLE-3
~4.2, ~5x lossless. vLLM v0.24.0 ships the whole path in-tree (v1/spec_decode/dflash.py,
model_executor/models/qwen3_dflash.py, method auto-detect, VLM-target support).

## The drafter

`z-lab/Qwen3.6-27B-DFlash` (HF) -> `models/files/qwen3.6-27b/dflash-draft/` (3.3 GB bf16,
NOT in manifest.yaml yet). 5 layers (4x SWA-2048 + 1 full), hidden 5120, block_size 16,
taps target layers [1,16,31,46,61], shares target embedding/LM-head space (vocab 248320).

## Spike results (W8A8 target, TP=2, PIECEWISE capture, IN=2048/OUT=128, PREFIXCACHE=0)

| config | TG c1 | TTFT | notes |
|---|---|---|---|
| NEXTN MTP spec=3 (baseline) | 30.24 | 739 ms | the shelf config |
| DFlash spec=15 | 19.06 | 912 ms | coherent, capture 3s/1.42GiB, CAPSIZES 1,2,4,8,16 |
| DFlash spec=7 | CRASHED at init | | EngineCore died during health wait (shm_broadcast "cancelled"); post-teardown probes transiently HUNG (cleared on re-probe ~2 min later, no reset needed) |

## [2026-07-03 session 2] REAL-workload telemetry OVERTURNS the cold-bench verdict

vllm/dflash_accept_probe.py (8 cumulative coding turns, temp 0, 400 tok/turn, per-turn
/metrics deltas), same TP=2 PIECEWISE PREFIXCACHE=0 B70_MRV2=0 serves:

| serve | tok/s | accept_len | acc_rate | per-pos |
|---|---|---|---|---|
| MTP spec=3 | 27.80 | 2.837 | 0.612 | p0 89% p1 75% p2 20% (1-layer head wall) |
| DFlash bf16 spec=15 | **41.55 (+49%)** | **4.210** | 0.214 | accepts decay smoothly out to p14 |

The cold random-text bench (19.06, above) was UNREPRESENTATIVE -- hypothesis 1 confirmed.
On real coding DFlash bf16 spec=15 beats the MTP shelf config by +49% single-stream.

## Drafter W8A8 quant: two integration walls (both root-caused, config-fixed)

vLLM's in-tree qwen3_dflash has TWO paths that break naive per-linear quantization:
1. `precompute_and_store_context_kv` F.linear's the RAW fused k/v weights (bypasses the
   quantized wrapper) -> int8 k/v = dtype crash at init (BFloat16 x signed char).
2. q/k/v fuse into ONE qkv_proj module that must be uniformly quantized -> int8 q with
   bf16 k/v = loader KeyError qkv_proj.weight_scale.
FIX baked into vllm/quant_dflash_drafter.py: attention QKV stays bf16 (~315 MB / 5 layers);
int8 = o_proj + gate/up/down (the byte bulk). Output models/files/qwen3.6-27b/dflash-draft-w8a8-rtn.

spec=7 CAVEAT: the crash means the spec sweep is NOT free -- treat further DFlash starts as
crash-prone TP=2 inits (wedge discipline: one attempt, verify xpu-health between, never chain).
Unknown whether spec=7 itself or CAPSIZES=1,2,4,8 (vs 16-row) was the trigger; diagnose in a
dedicated session with B70_DEBUG=1 before sweeping.

Command (shelf serve.sh, no edits needed):

    SPEC='{"method":"dflash","model":"/models/qwen3.6-27b/dflash-draft","num_speculative_tokens":15}' \
    CAPSIZES="1,2,4,8,16" PREFIXCACHE=0 ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start

## What did NOT break (the feared walls, all clear)

- **vLLM issue #41190** (Qwen3.6-hybrid-GDN + TP=2 + spec crash at num_accepted_tokens_event):
  did NOT reproduce -- serve, probe, and bench all clean. (Our stack: XPU torch.Event + the
  csag/mamba-ptr shims; the crash may be CUDA-event-lifetime-specific.)
- **Non-causal draft attention on XPU**: the drafter's use_non_causal path ran on the XPU
  attention backend without modification, output coherent.
- **flash_attn dependency**: none -- the in-tree drafter uses vLLM's generic Attention layer,
  which routes to the platform backend.
- **W8A8 target + bf16 drafter**: no quant-path interference (drafter is a separate model dir;
  the MTP-bf16 shim block is inert for dflash).
- **VLM target**: fine (dflash supports mm targets; drafter is text-only by design).
- Graph capture: drafter + target both init under PIECEWISE; capture finished 3s / 1.42 GiB.

## Why it lost to MTP at the spike settings (hypotheses, in test order)

1. **Over-drafting**: spec=15 verifies 16 query tokens/step; on the random-ish bench text the
   accept length is low, so most verify compute is waste. (vLLM warned
   max_num_scheduled_tokens=1936 from the spec settings.) -> sweep spec {5,7,11,15} and use a
   REAL coding/agentic workload; the CUDA tau~6.5 numbers are on Math500/coding, not random text.
2. **Drafter weight reads**: 3.3 GB bf16 per draft pass ~ 5-6 ms/step of pure bandwidth on one
   card -- vs the NEXTN head's near-zero. An int8/W8A8 quant of the drafter is the obvious cut
   (halves it) once accuracy is verified. The drafter is also a candidate for the small-M int8
   DPAS kernel (plan C1) -- 5 layers at M=16 is exactly the shape it targets.
3. **Drafter capture**: verify whether the draft forward records into PIECEWISE or runs eager
   (5 layers x launch overhead). llm_base_proposer supports PIECEWISE dispatch keys; confirm
   engaged (adjust_cudagraph_sizes_for_spec_decode / initialize_cudagraph_keys logs).
4. Hidden-state gather from 5 target layers each step (extra reads + a cross-TP gather).

## Follow-ups (priority order)

1. Accept-length telemetry on a real coding workload (SpecDecoding metrics / prometheus
   vllm:spec_decode_* series) for spec {5,7,11,15} -- decide the crossover vs MTP.
2. If accept >= ~4 on real work: quantize the drafter W8A8 (compressed-tensors, GPTQ) and
   re-bench; then check drafter-capture engagement.
3. Longer term: the C1 small-M int8 kernel makes BOTH the MTP verify and the DFlash draft
   cheap; DFlash is the main consumer of a fast M=8..16 int8 path.
4. Consider `z-lab/Qwen3.6-35B-A3B-DFlash` for the MoE entry (its decode is eager-slow; a
   high-accept drafter may be worth more there).

## [2026-07-03 session 3] KV eater ROOT-CAUSED; group_size padding patch; 128k DD-viable

The last session's "GPU KV cache = 24,281 tokens (vs 320k)" was a MISMEASUREMENT. Real numbers
(w8a8-rtn spec=15, TP=2, UTIL=0.90, MAXLEN=253952): Available KV 9.88 GiB/card, per-token 57,096
B/tok/card, max_model_len caps at ~178,304 (PREFIXCACHE 0 and 1 both ~176-178k -- align mode is not
the collapse). DFlash already serves ~178k context; the 24k blocker does not exist.

Three compounding eaters (init logs):
1. Drafter = 5 DENSE full-attention layers (~10,240 B/tok/card). Its config declares 4x
   sliding_attention(2048)+1 full, but in-tree DFlashQwen3Attention passes no per_layer_sliding_window
   -> all 5 run full.
2. KV-group PADDING WASTE: the 5-layer drafter type makes group_size = min(16 full, 48 mamba, 5 draft)
   = 5, so the 16 target full-attn layers pad to 20 (25% waste on the dominant KV) + mamba 48->50
   (4.17%) = 6 wasted layers (~21% of per-token). MTP {16,48} -> group_size 16, zero pad.
3. mamba/attn page unify -> 896-token attention blocks (same as MTP).

FIX: vllm/patches/kv_cache_utils_gcd.py (mount via serve.sh B70_EXTRA_MOUNTS over
vllm/v1/core/kv_cache_utils.py). group_size = argmin total padding layers over [min,max], with
group_size >= min_layer_count so the drafter stays in ONE group (vLLM asserts "All drafting layers
should belong to the same kv cache group"; group_size=1 CRASHES the worker -- gracefully, no wedge).
DFlash -> gs=8 (pad 6->3, groups 15->9); MTP -> gs=16 UNCHANGED; eagle -> 13 UNCHANGED. max_model_len
178,304 -> 186,368.

Serve command (128k DD candidate, VALIDATED gate 12/12 + accept 48.8 t/s / 4.71 == prior, +75% vs MTP):

    P=/mnt/vm_8tb/github/b70_ai_things/vllm/patches/kv_cache_utils_gcd.py
    T=/opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
    SPEC='{"method":"dflash","model":"/models/qwen3.6-27b/dflash-draft-w8a8-rtn","num_speculative_tokens":15}' \
      CAPSIZES="1,2,4,8,16" PREFIXCACHE=0 MAXLEN=131072 B70_EXTRA_MOUNTS="$P:$T:ro" \
      ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start

To push past 178k (untested): all-sliding drafter (uniform sliding_window=2048 -> one group,
assertion-safe) shrinks the drafter's own ~10 KB/tok; risk = the 1 full->sliding layer may cost accept
and the precompute single-slot-mapping on a windowed cache is unverified. 128k is the accepted fallback.

## [2026-07-03 session 3 cont] all-sliding drafter = full 253952 ctx but -28% accept

To push past ~186k, window the drafter's own ~10 KB/tok KV. Its true 4-sliding+1-full arch can't be
honored (2 kv groups; DFlash requires 1). Uniform sliding_window=2048 on all 5 layers (one group,
assertion-safe) via vllm/patches/qwen3_dflash_swa.py, gated B70_DFLASH_SWA=1 (inert by default).

Result (SWA + gs patch, MAXLEN=253952): FULL context fits (GPU KV 273,661 tokens, 1.08x conc), gate
12/12 coherent -- but accept 4.71 -> 3.37 and t/s 48.8 -> 35.4 (+27% vs MTP, down from +75%). Accept
drops even at ctx<2048 (window not limiting) -> it's context-KV degradation on the windowed cache
(precompute single-slot-mapping), not truncation.

Decision table (w8a8-rtn drafter, 8-turn real coding):
  MTP spec=3                      accept 2.84  27.8 t/s  253,952 ctx
  DFlash full drafter + gs patch  accept 4.71  48.8 t/s  ~186k ctx   +75%  <-- 128k DD winner
  DFlash all-sliding + gs patch   accept 3.37  35.4 t/s  253,952+    +27%  (only if >186k needed)

The 128k daily driver = FULL drafter + gs=8 patch (do NOT set B70_DFLASH_SWA). All-sliding is a
documented >186k lever at a steep accept cost.

## [2026-07-03 session 4] TP-worker "shm_broadcast cancelled" crash ROOT-CAUSED + FIXED

The crash that reverted the DD to MTP is fixed at ~zero perf cost. ROOT CAUSE: DFlash is in vLLM's
`EagleModelTypes` (config/speculative.py) -> async scheduling AUTO-ENABLES for it. But only DFlash runs
`precompute_and_store_context_kv` (qwen3_dflash.py), which writes ALL context K/V by slot_mapping into the
drafter cache every draft pass via `do_kv_cache_update`->`basic_cache` (no slot-validity check). Under async
scheduling (step N+1 prepared while step N in flight), a request cancelled in that window leaves a STALE
slot_mapping -> the drafter writes into a freed/reused block -> hard TP-worker fault (rare; clean Exit 0 =
the shm_broadcast "cancelled" blocker) or soft GDN "!!!!" poison. MTP is immune (no context-KV precompute).

FIX: `--no-async-scheduling` on the DFlash path (now the `DFLASH=1` default in serve.sh) -> the drafter
always sees the committed batch, race removed by construction. Validated: accept probe 35.16 t/s / acc_len
3.24 (== async-on, ~zero cost, +26% vs MTP), 7-min concurrent cancellation soak + c4 bench clean (0 crash
sigs, coherent), gate_concurrent_coherence.py 18/18 PASS. Serve unchanged otherwise:

    DFLASH=1 DFSWA=1 MAXLEN=253952 PORT=18080 NAME=b70_daily_0 \
      ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start

CAVEATS: the original async-ON crash was NOT reproducible in-session (25 min of heavy load) so this is a
source-grounded deterministic fix, not a strict crash-A/B. DFlash accept ~0% on random/non-coding text (its
win is coding-specific); MTP stays flat-robust ~2.9 across workloads.
