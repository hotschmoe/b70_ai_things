# fp8 KV on the NVFP4 TP=2 daily driver -- A/B test plan (RESEARCH_TODO Track 11i)

Goal: decide whether fp8 KV (with calibrated scales) can replace bf16 KV (KV_FP8=0) on the
NVFP4 27B TP=2 daily driver -- restoring ~2x KV capacity at 128k -- WITHOUT bringing back the
2026-07-06 late-generation repetition collapse.

Coordinator owns the GPU lease and runs every command below. TP=2 uses both cards, so the live
DD must be stopped first (`docker rm -f b70_daily_0`). ASCII only. Format: config -> command ->
result -> verdict.

--------------------------------------------------------------------------------
## Background (what is already known -- do not re-litigate)

- The ModelOpt NVFP4 27B checkpoint declares `quantization_config.kv_cache_scheme={num_bits:8,
  type:float}` but ships NO k/v scales -> vLLM defaults every scale to 1.0 ("Using KV cache
  scaling factor 1.0 for fp8_e4m3").
- POST-GPU CORRECTION (docs/20260708_...investigation.md + JOURNAL "session 3 cont"): fp8
  scale=1.0 is NEAR-LOSSLESS on THIS checkpoint. Measured per-layer amax: K 11.7-21.8, V
  7.5-133.0 -- ALL far below e4m3 max 448, so scale=1.0 does NOT clip; e4m3 is a float so "bottom
  of range" is not low precision. On the CLEAN single-card path (TP=1, no-MTP, fused, GRAPH=1)
  fp8 scale=1.0 ran clean over 3500 forced tokens, gate 4/4, needle @118,856 tok retrieved.
- The XPU FlashAttention backend DOES consume layer._k_scale/_v_scale (proven: 3 injected scales
  -> 3 distinct temp-0 hashes). Calibration is mechanically live.
- Calibrated scales already exist: `vllm/nvfp4/kv_scales_nvfp4_27b.json` (16 full-attention
  layers), injected by sitecustomize block (10) via `NVFP4_KV_SCALES_FILE`, wired to the serve
  knob `KV_SCALES=`. On this checkpoint they are near-neutral vs scale=1.0 (no clipping to fix).
- `--calculate-kv-scales` is a DEAD END here: HybridAttentionMambaModelConfig hard-disables it
  for GDN/Mamba hybrids, and vLLM #37554 shows it corrupts scales on GDN. Offline calibration or
  bf16 KV only.

## THE ONE RISK TO ISOLATE

The 2026-07-06 repetition ("...community community...") that made us set KV_FP8=0 was bisected on
the **TP=2 + MTP** path. The clean single-card refutation was **TP=1, MTP OFF**. So we have NOT
yet separated the two variables on the DD's own config:
  (a) fp8 KV precision, vs
  (b) the TP=2 + MTP verify path (spec-decode + captured graph + push-AR).
It is entirely possible the repetition is a property of (b) and independent of KV precision, in
which case fp8 KV is safe on TP=2 too. The A/B below is built to answer exactly this by holding
TP=2 fixed and moving KV precision and MTP independently.

--------------------------------------------------------------------------------
## A/B matrix (5 arms, TP=2 throughout, one card-pair)

| arm | KV        | scales     | MTP | purpose                                                        |
|-----|-----------|------------|-----|---------------------------------------------------------------|
| A1  | bf16      | n/a        | 5   | CONTROL = the current shipped DD (known-good, KV_FP8=0)        |
| A2  | fp8       | 1.0 (stock)| 5   | does the OLD repetition reproduce on TP2+MTP now?             |
| A3  | fp8       | calibrated | 5   | does calibration change A2? (target config if coherent)       |
| A4  | fp8       | calibrated | OFF | isolate: KV precision vs MTP as the repetition cause          |
| A5  | fp8       | calibrated | 5   | CAPACITY read at MAXLEN=131072 (same as A3; read KV tokens)   |

Read A3 and A5 together (same config) -- A5 is just the capacity/needle read of A3. Run A1->A4 for
coherence, then A3/A5 for the payoff number. If A2 is clean, calibration (A3) is a formality; if A2
repeats and A3 is clean, calibration is the fix; if A2 and A3 both repeat but A4 is clean, the
fault is the MTP path, not KV precision (then keep bf16 KV OR ship fp8+MTP-off for long-ctx).

--------------------------------------------------------------------------------
## Exact serve commands

Common DD flags (identical across arms): MODE=fused GRAPH=1 CAPSIZES=1,2,4,8 MAXLEN=131072
MAXSEQS=8 UTIL=0.85 PUSH_AR=1 PUSH_AR_GRAPH=1 TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3.
Stop the live DD first: `docker rm -f b70_daily_0`. Reuse PORT=18080. Set API_KEY to the DD key.
KV knobs are the ONLY thing that changes:
  - bf16 KV  -> `KV_FP8=0`                       (strips kv_cache_scheme, bf16 cache)
  - fp8 stock-> (omit KV_FP8, it defaults to 1)  (checkpoint fp8, scale=1.0)
  - fp8 calib-> (omit KV_FP8) + `KV_SCALES=/mnt/vm_8tb/github/b70_ai_things/vllm/nvfp4/kv_scales_nvfp4_27b.json`
  - MTP off  -> omit MTPTOK ; MTP on -> `MTPTOK=5` (with CAPSIZES; the serve guard forces MAXSEQS>=8)

REPO=/mnt/vm_8tb/github/b70_ai_things ; cd $REPO

A1 (control, bf16 KV, MTP5):
```
NAME=b70_kvab PORT=18080 TP=2 MODE=fused GRAPH=1 MTPTOK=5 KV_FP8=0 \
  CAPSIZES=1,2,4,8 MAXLEN=131072 MAXSEQS=8 UTIL=0.85 PUSH_AR=1 PUSH_AR_GRAPH=1 \
  TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3 SERVED_FORCE=hotschmoe-dd \
  API_KEY=$DD_KEY ./bin/gpu-run bash vllm/nvfp4/serve_nvfp4_27b.sh
```

A2 (fp8 KV, stock scale=1.0, MTP5) -- omit KV_FP8 (defaults 1), no KV_SCALES:
```
NAME=b70_kvab PORT=18080 TP=2 MODE=fused GRAPH=1 MTPTOK=5 \
  CAPSIZES=1,2,4,8 MAXLEN=131072 MAXSEQS=8 UTIL=0.85 PUSH_AR=1 PUSH_AR_GRAPH=1 \
  TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3 SERVED_FORCE=hotschmoe-dd \
  API_KEY=$DD_KEY ./bin/gpu-run bash vllm/nvfp4/serve_nvfp4_27b.sh
```

A3 / A5 (fp8 KV, CALIBRATED scales, MTP5) -- the target config:
```
NAME=b70_kvab PORT=18080 TP=2 MODE=fused GRAPH=1 MTPTOK=5 \
  KV_SCALES=$REPO/vllm/nvfp4/kv_scales_nvfp4_27b.json \
  CAPSIZES=1,2,4,8 MAXLEN=131072 MAXSEQS=8 UTIL=0.85 PUSH_AR=1 PUSH_AR_GRAPH=1 \
  TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3 SERVED_FORCE=hotschmoe-dd \
  API_KEY=$DD_KEY ./bin/gpu-run bash vllm/nvfp4/serve_nvfp4_27b.sh
```

A4 (fp8 KV, calibrated, MTP OFF) -- drop MTPTOK; CAPSIZES may stay (unused without spec):
```
NAME=b70_kvab PORT=18080 TP=2 MODE=fused GRAPH=1 \
  KV_SCALES=$REPO/vllm/nvfp4/kv_scales_nvfp4_27b.json \
  CAPSIZES=1,2,4,8 MAXLEN=131072 MAXSEQS=8 UTIL=0.85 PUSH_AR=1 PUSH_AR_GRAPH=1 \
  TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3 SERVED_FORCE=hotschmoe-dd \
  API_KEY=$DD_KEY ./bin/gpu-run bash vllm/nvfp4/serve_nvfp4_27b.sh
```

Verify each arm loaded the intended KV mode BEFORE testing:
  - `docker logs b70_kvab 2>&1 | grep -i "KV cache\|scaling factor\|kv_cache_scheme\|inject"`
    A1: "kv_cache_scheme stripped / bf16". A2: "Using KV cache scaling factor 1.0 for fp8_e4m3"
    (and NO block-10 inject lines). A3/A5: block-10 "injected KV scales ... k=.. v=.." x16 layers.
  - `curl -s $AUTH http://192.168.10.5:18080/v1/models` -> served id hotschmoe-dd (Model Identity).

--------------------------------------------------------------------------------
## THE capacity metric to read (the payoff)

From each arm's startup log, read the KV pool token capacity at MAXLEN=131072:
```
docker logs b70_kvab 2>&1 | grep -iE "GPU KV cache size|Maximum concurrency|available_kv_cache|# GPU blocks"
```
vLLM prints "GPU KV cache size: N tokens" and "Maximum concurrency for <MAXLEN> tokens per request:
Kx". Record N for A1 (bf16) vs A3/A5 (fp8). TARGET: fp8 ~= 2x bf16 (single-card measured 1.98x).
Expected on TP=2 (2-card pool): bf16 ~385k tokens -> fp8 ~757k tokens at 128k MAXLEN. Success = fp8
roughly doubles the pool AND holds coherence. (bf16 already FITS 128k on TP=2; the fp8 win here is
headroom / concurrency / room toward 200k, not fitting 128k which bf16 already does.)

--------------------------------------------------------------------------------
## Coherence checks (run per arm; A2/A3/A4 are the ones under suspicion)

The repetition is a LATE-generation, load-dependent collapse -> short prompts will not catch it.
Use forced long decodes and concurrency, exactly the conditions that exposed it in July.

1. LONG single-stream forced decode (repetition watch). Forces 4000 decode tokens, ignore_eos:
   ```
   BASE=http://192.168.10.5:18080 KEY=$DD_KEY TOKENS_PER=4000 TARGET=4000 CONC=1 \
     LABEL=kvab_$ARM python3 vllm/nvfp4/soak_leak.py
   ```
   CAVEAT: soak_leak.py's built-in "GARBAGE?" flag only trips on single-char garbage
   (len(set(tail))<5); it will NOT catch the WORD-level "...community community..." loop that was
   the actual 2026-07-06 fault. So for the repetition check, fire one long completion and scan for a
   repeated n-gram directly:
   ```
   curl -s -H "Authorization: Bearer $DD_KEY" -H 'Content-Type: application/json' \
     http://192.168.10.5:18080/v1/completions \
     -d '{"model":"hotschmoe-dd","prompt":"Write a long detailed essay on the history of computing.",
          "max_tokens":4000,"temperature":0,"ignore_eos":true}' \
   | python3 -c 'import sys,json,collections; t=json.load(sys.stdin)["choices"][0]["text"]; \
     w=t.split(); g=collections.Counter(tuple(w[i:i+4]) for i in range(len(w)-3)); \
     top=g.most_common(1)[0] if g else (("",),0); \
     print("top 4-gram x%d:"%top[1], " ".join(top[0])[:80]); \
     print("REPETITION-FAIL" if top[1]>15 else "rep-ok")'
   ```
   PASS = top 4-gram repeats < ~15x through 4000 tokens (A1 bf16 is the clean baseline; compare
   A2/A3/A4 against it). This is the direct analogue of the bf16-clean-3500 / fp8-repeat-985
   bisection. Use soak_leak.py (above) in parallel for crash + single-char-garbage detection.

2. CONCURRENT forced-decode soak (the DD's real failure regime -- the 2026-07-06 repetition was
   under load):
   ```
   BASE=http://192.168.10.5:18080 KEY=$DD_KEY TOKENS_PER=3000 TARGET=36000 CONC=6 \
     LABEL=kvab_${ARM}_conc python3 vllm/nvfp4/soak_leak.py
   ```
   PASS = 0 aborts AND no "!!!!"/repetition on any of the 6 streams through 36k tokens (matches the
   DD's own crash-free soak criterion in memory daily-driver-is-sglang-w8a8-27b).

3. NEEDLE retrieval at depth (uses the extra fp8 capacity):
   ```
   PROBE_HOST=http://192.168.10.5:18080 NEEDLE_DEPTH=120000 python3 vllm/nvfp4/kv_gate.py
   ```
   PASS = capital-of-France / 17+26=43 / gold=Au AND needle "7391-ZULU" retrieved at ~120k tokens.
   (If KEY is enforced, add it inside kv_gate via PROBE auth or run against an open /health serve;
   kv_gate.py currently sends no auth header -- set the serve API_KEY empty for the test window, or
   extend kv_gate to add the bearer.)

4. AGENTIC parity (the DD's actual job): run a couple of real coding tasks through the pi.dev /
   omp.sh / hermes harness against hotschmoe-dd. PASS = finish_reason=stop with real code, no
   thinking-mode runaway (THINK_BUDGET=4096 default) and no repetition.

5. SCALE-APPLIED sanity (confirms A3 differs from A2): fire the SAME fixed prompt at temperature 0
   against A2 and A3 and hash the output. Distinct hashes confirm the calibrated scales are being
   consumed (mirrors the single-card 3-hash proof). If A2 and A3 hash identically, block-10 inject
   did not take -> re-check the "injected KV scales" log lines.

--------------------------------------------------------------------------------
## Decision rule

- A2 clean AND A3 clean  -> fp8 KV is safe on TP=2+MTP; ship A3 (calibrated, for robustness) as the
  DD. The old KV_FP8=0 was over-cautious; the 2026-07-06 fault was NOT KV precision. Update the DD
  launch (drop KV_FP8=0, add KV_SCALES=), re-sweep-gate, then flip. Record KV-token gain from the
  capacity read.
- A2 repeats, A3 clean     -> calibration is the fix on TP=2 (unlike single-card where it was
  neutral); ship A3.
- A2 AND A3 repeat, A4 clean -> the repetition is the TP=2+MTP path, independent of KV precision.
  fp8 KV is fine WITHOUT MTP -> offer a long-context fp8 profile (A4) for 128k+/200k work, keep
  bf16+MTP for the interactive DD. Escalate the MTP-path repetition separately.
- A3 AND A4 both repeat     -> fp8 KV genuinely degrades TP=2 (differs from single-card); keep
  KV_FP8=0. Document the TP=2-vs-TP=1 divergence.

Fallback at every step: the current shipped DD (A1, bf16 KV, KV_FP8=0) is known-good; revert to it
if any arm wedges. Do not leave a failing arm serving.
