# One full-model FP8 cache step, prepared only

CONFIG -> Current Qwen GPTQ on immutable14ee, card1/port18235. FP16 compute,
E4M3FN KV, Triton attention, FP32 SSM, prefixON with extra_buffer, eager/MTP0,
context8192/chunk512/c4 and fresh cache. Skip automatic image warmup. The
embedded fresh artifact is be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062
at /opt/b70/calibrated-kv/fresh-scales.json. The runtime owner confirmed no
loader edits after f82;14ee inherits that validated loader overlay unchanged.

COMMAND -> The exact gated launch is recorded in fp8_cache_plan.json. It uses
the existing per-card serving lifecycle, strict health/cleanup, and requires
the completed14ee pair/numerical gates plus prior FP16 text viability. No
GPU action was performed for this preparation.

The owned job first requires one target loader receipt naming the fresh
artifact, Triton/FP8, TP1 rank0, sixteen covered attention layers and no query
scale application. It then runs the frozen26-case temperature0/seed42 text
probe. Only after that passes does it run the four-history/four-turn concurrent
streaming tool probe with360 records and shared cache. Require32 checks,
32 attempts, zero bangs and a strictly positive cached-token counter delta
across that concurrent fixture. The metric gate counts only hotschmoe-dd
local-device/total cached tokens, excluding other models and storage counters.
This avoids treating earlier tiny-probe cache hits as evidence for the fixture.

RESULT -> CPU tests pass receipt coverage/identity, metric filtering, early
failure stopping the later probe and positive cache-delta enforcement.
Frozen sources and generated argv pass checks: prefix extra_buffer enabled,
graphs disabled, no speculative algorithm, exact FP8 dtype/scale path and
Triton backend. Startup bound1200s, each probe bound1500s with300s HTTP request
bounds, aggregate job bound3300s. Output remains fresh and no launch occurred.

VERDICT -> One meaningful full-model FP8/cache candidate ready for scheduling.
Passing the synthetic attention oracle does not qualify these model/cache
paths. This job is still unqualified. Graph and greedy MTP remain separate
future feature steps; no sampling overlay, native vision or full feature-parity
claim. Positive cache counters and coherence do not establish cache state
identity beyond this bounded fixture.

Raw plan: /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang14ee-fp8-cache-plan/plan.json
