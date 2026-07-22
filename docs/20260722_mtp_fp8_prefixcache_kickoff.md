# 2026-07-22 kickoff: root-cause + FIX prefix-cache hits = 0 under MTP x fp8-KV

Session-kickoff prompt. Paste "follow docs/20260722_mtp_fp8_prefixcache_kickoff.md" into a fresh
session. Assumes CLAUDE.md + memory are loaded; facts below override stale memory. This is THE
highest-value open perf item: fixing it gives the daily driver MTP decode speed (60.8 t/s code)
AND prefix-cache reuse (warm agentic turns) at 100k ctx simultaneously.

## The bug (measured 2026-07-22, JOURNAL entry (a))

On vLLM 0.25.1 XPU (image `vllm-xpu-env:int8g-v0251`, torch 2.12), Qwen3.6-27B NVFP4, single card,
GRAPH=1 PIECEWISE, prefix caching enabled (mamba align mode), identical ~2k-token prompt sent 3x
(temperature 0), reading `vllm:prefix_cache_{queries,hits}_total` from /metrics:

| config                          | hits/queries | repeat-prompt wall |
|---------------------------------|--------------|--------------------|
| no-MTP + fp8 KV (calibrated)    | 3200/7284    | 2.5s -> 1.1s       |
| MTP5  + fp8 KV (calibrated)     | **0/7284**   | flat               |
| MTP5  + bf16 KV                 | 1664/7284    | 3.2s -> 1.3s       |

Each feature alone caches fine; the COMBINATION zeroes every hit. The live DD (MTP5+fp8@100,352)
showed 0/35,784 over a whole bench session. Historical corroboration: 2026-07-16 TP=2 MTP5 + bf16
KV measured 57% hit. Also note MTP5+bf16 hits (1664) < no-MTP+fp8 (3200) on the same probe -- MTP
may already halve reuse even when it "works"; understand why while you are in there.

## Prime hypothesis (unverified -- verify FIRST)

The model is a hybrid: 48 GDN/mamba layers + 16 full-attn layers + 1 MTP drafter full-attn layer.
KV groups are formed per (layer type, page size, dtype) in vllm/v1/core/kv_cache_utils.py, and a
prefix-cache hit for a hybrid requires the hit to intersect across ALL kv-cache groups
(find_longest_cache_hit per single-type manager, then intersection). With fp8 KV the 16 main attn
layers cache in fp8, but the MTP drafter layer is UNQUANTIZED bf16 (its module is created by the
proposer outside the quantized path) -> its KV group has a different dtype/page-size -> group
layout mismatch (init logs show "Add 3 padding layers" + "attention block size 1664 to ensure
attention page size >= mamba page size") -> the drafter group never produces a hit -> the
intersection is always empty. With bf16 KV all attn layers share one group -> hits work.
If true, candidate fixes (in rough preference order):
  1. Make the drafter layer's KV fp8 too (align its kv_cache spec/dtype with the main attn group;
     scales for it can be 1.0 -- e4m3 amax headroom is proven huge on this ckpt) -> one group.
  2. Teach the intersection to tolerate/skip the drafter group IF vLLM can recompute drafter KV
     for cached prefixes (check what upstream does for eagle/MTP + APC on CUDA -- the drafter
     needs prefix KV for its attention; a hit that skips drafter KV must trigger a drafter
     prefill or the proposer breaks. Do NOT ship a fix that silently corrupts drafter KV).
  3. Port an upstream fix if one exists (search vLLM issues/PRs: spec decode + fp8 kv_cache +
     prefix caching; also check whether MTP+fp8+APC is broken upstream on CUDA too).

## How to reproduce (verbatim, card 1)

Research mode: `docker stop b70_daily_1` frees card 1 (card 0 KEEPS SERVING the DD -- never touch
it, never lease card 0). Restore with `docker start b70_daily_1` when done. Health-check between
crashy runs: `./bin/xpu-health --card 1`.

ZERO-HITS repro (the DD replica config, short ctx for fast loads):
```
K=$(cat /mnt/vm_8tb/b70/secrets/dd_api_key)
API_KEY="$K" NAME=pcache_b PORT=18077 CARD=1 TP=1 IMG=vllm-xpu-env:int8g-v0251 \
MODE=fused GRAPH=1 MTPTOK=5 MAXSEQS=8 CAPSIZES=1,2,4,8 UTIL=0.95 MAXBATCH=2048 MAXLEN=32768 \
KV_FP8=1 KV_SCALES=$PWD/vllm/nvfp4/kv_scales_nvfp4_27b.json \
PREFIXCACHE=1 SERVED_FORCE=pcache-b THINK_BUDGET=0 B70_EXTRA_ENV="B70_EMBED_INT8=1" \
./bin/gpu-run --card 1 bash vllm/nvfp4/serve_nvfp4_27b.sh start
```
Probe (expect hits=0 on this config; >0 with MTPTOK= or KV_FP8=0):
```
python3 -c "print('Explain this code in detail.\n' + 'def f(x):\n    return x*2\n' * 220)" > /tmp/probe.txt
for i in 1 2 3; do curl -s http://localhost:18077/v1/completions -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json; print(json.dumps({'model':'pcache-b','prompt':open('/tmp/probe.txt').read(),'max_tokens':16,'temperature':0}))")" >/dev/null; done
curl -s http://localhost:18077/metrics -H "Authorization: Bearer $K" | grep -E "prefix_cache_(queries|hits)_total"
```
A/B knobs: `MTPTOK=` (off) and `KV_FP8=0` (bf16, drop KV_SCALES). Init failures are cheap and
wedge-free at TP=1; each serve cycle ~2.5 min.

## Instrumentation starting points (in-image, mount-not-rebuild)

House pattern: ALL fixes go into `vllm/nvfp4/patches/sitecustomize.py` as a new numbered block
(next free = (14)), env-gated, default-off until gated. Read blocks (9)/(10)/(13) for the style.
Read code from the image: `docker run --rm --entrypoint cat vllm-xpu-env:int8g-v0251 /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py` etc.
- `vllm/v1/core/kv_cache_utils.py` -- block hashing, kv-cache group formation, the "Add 3 padding
  layers" path, `get_kv_cache_configs`.
- `vllm/v1/core/kv_cache_manager.py` + `single_type_kv_cache_manager.py` -- `get_computed_blocks`,
  per-group `find_longest_cache_hit`, the cross-group intersection. Hook these to LOG per-group
  hit lengths on the probe -- that one log line will likely prove/kill the hypothesis in minutes.
- `vllm/v1/worker/gpu_model_runner.py` + the MTP proposer (`llm_base_proposer.py`) -- how the
  drafter layer registers its KV spec; where kv_cache_dtype comes from per layer.
- The serve stack already patches KV behavior: sitecustomize block (10) injects calibrated fp8
  scales post-load (NVFP4_KV_SCALES_FILE); KV_FP8=0 works by STRIPPING kv_cache_scheme from a
  patched config.json (serve script). The drafter is forced-bf16 by design.
- Check upstream via a consultant agent: vLLM GitHub issues/PRs for "spec decode prefix caching
  fp8" / eagle + APC; and whether torch-xpu has anything relevant. Use agents heavily for code
  reading + upstream research; the COORDINATOR alone touches the GPU, serially.

## Correctness gates for any fix (a wrong fix silently corrupts KV -- gate hard)

1. Probe hits > 0 on MTP5+fp8 AND warm repeat-prompt wall drops (2.5s -> ~1.1s class).
2. BYTE-IDENTICAL temp-0 outputs: same long prompt on (a) fresh server (no cache) vs (b) warm
   server (cache hit) -- compare full completions, not eyeballs. Repeat with a >1664-token prefix
   (one full align block) and a mid-block-boundary prefix.
3. MTP accept rate unchanged (~73% overall on code probes; /metrics spec_decode_num_accepted/drafts).
4. Needle at depth through the cache: kv_gate.py NEEDLE_DEPTH=93000 (KEY= env supported) 4/4, run
   TWICE back-to-back (second run exercises cached prefix + fp8 + MTP together).
5. gate_concurrent_coherence 18/18; bench_code c1 >= ~60 t/s retained (fix must not tax decode).
6. Soak: 30k single-stream (soak_leak.py) + 40k concurrent (soak_concurrent.py) clean.
Then: flip the DD (rolling, replica 1 first -- config lives in rdy_to_serve/vllm/qwen36-27b-nvfp4/
serve.sh TP=1 branch; make the fix default-ON there once gated), re-run the IN=2048 bench
(vllm/nvfp4/bench_2048.py -- counts reasoning deltas), update README banner/footnote (the "hits
ZERO" caveat), JOURNAL (newest at bottom, config->command->result->verdict), memory, commit+push.

## Environment facts (current, override stale memory)

- Box HEADLESS, cards symmetric (~126-128 TFLOPS). Kernel 7.1. Reboot = only hard wedge recovery
  (headless did NOT free modprobe -r xe). TP=1 work carries no wedge risk class; oneCCL/P2P
  cautions apply only to TP>1 (do not run those here).
- DD (production, do not disturb): DP=2 NVFP4 replicas b70_daily_0 (card 0 :18091) + b70_daily_1
  (card 1 :18092) behind nginx :18080, served id hotschmoe-dd. Replica config = shelf wrapper
  TP=1 default: MODE=fused GRAPH=1 MTPTOK=5 MAXSEQS=8 CAPSIZES=1,2,4,8 UTIL=0.95 (OPERATOR MAX,
  never higher) MAXBATCH=2048 (>=1600 forced by mamba align) MAXLEN=100352 KV_FP8=1 + calibrated
  scales + PREFIXCACHE=1 + parsers + B70_EMBED_INT8=1 (block (13), frees 1.18 GiB -- REQUIRED for
  MTP@100k; also adjusts model_memory_usage or vLLM's KV budget ignores the free).
- systemd units installed + current (b70-daily-driver DP=2, b70-dd-watchdog watches both replicas).
- Key JOURNAL entries: 2026-07-21 (d)-(j), 2026-07-22 (a). README top banner has the current state.
- GPU lease: EVERY GPU touch via ./bin/gpu-run --card 1. Commit+push often. ASCII only.

## Success criteria

1. Root cause identified with a logged/instrumented proof (not just a theory), journaled.
2. A gated fix (sitecustomize block, env-gated, default-on in the shelf once it passes ALL six
   correctness gates) -- or, if the fix is genuinely upstream-hard, a journaled NO-GO with the
   exact upstream blocker + the best fallback recommendation (no-MTP@102400 for cache reuse vs
   current MTP@100k without; the operator decides the interim default).
3. DD flipped to MTP5 + fp8 + working prefix cache at 100,352 ctx, soak-gated, README/JOURNAL/
   memory updated, all pushed.
