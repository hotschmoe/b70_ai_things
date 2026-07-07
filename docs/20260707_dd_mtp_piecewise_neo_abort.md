# DD crash 2026-07-07: NEO linear_stream.h:84 abort (MTP-verify x piecewise cudagraph)

Status doc. config -> command -> result -> verdict. Newest evidence appended.

## Summary of the incident

- Container `b70_daily_0` (vllm/qwen36-27b-w8a8 TP=2, v0.24.0, PIECEWISE capture,
  MTP spec=3, prefix-cache on, vision on) died `07-07 00:39:26`.
- Both TP workers aborted simultaneously with:
  ```
  Abort was called at 84 line in file:
  ../../neo/shared/source/command_stream/linear_stream.h
  ```
  -> `Worker proc VllmWorker-0 died unexpectedly` -> EngineCore fatal ->
  clean vLLM shutdown (container Exited(0); the `RuntimeError: cancelled` /
  `EngineDeadError` traces below the abort are downstream shm-broadcast fallout,
  NOT the cause).
- dmesg CLEAN (no xe/GuC/DEVICE_LOST/reset). This is a pure host-side software
  abort in the Intel Compute Runtime (NEO) command-stream builder, NOT the TP=2
  hardware wedge and NOT a DEVICE_LOST.

## State at the moment of the abort (from dump_input)

- Single request, decode step. `num_computed_tokens=50512` (~50K ctx),
  `num_output_tokens=1202`.
- `num_scheduled_tokens=4` = 1 + 3 MTP spec tokens.
  `scheduled_spec_decode_tokens=[-1,-1,-1]` (MTP verify path active).
- `num_common_prefix_blocks=[0,0,0,61]`.
- `max_seq_len=200000`, `kv_cache_dtype=auto` (bf16 KV), `cudagraph_mode=PIECEWISE`,
  `cudagraph_capture_sizes=[1,2,4,6,8]`, `enforce_eager=False`, spec method=mtp
  num_spec_tokens=3, TP=2, prefix caching on.

## Q1: was the session over the 128K ctx limit?

NO. The crashing request was at ~50K computed tokens, well under 128K. Served
max_seq_len was 200000 (raised from the historical 131072). Context LENGTH of the
request was not the direct trigger. Open hypothesis: raising MAXLEN 131072->200000
changes KV-block/command-buffer sizing even for shorter requests -> must test.

## Confirmed cross-quant

- commit b638297 recorded the SAME crash class on the NVFP4 config ("crashed ~1h
  in ... MTP-in-piecewise-graph"). The revert to W8A8 did NOT escape it. So the
  abort is structural to **MTP-verify x piecewise XPU cudagraph**, not quant-specific.
- memory [[nvfp4-dd-repetition-crash-rootcause]] independently flagged a
  "MTP-verify-in-piecewise-XPU-cudagraph NEO linear_stream.h:84 abort (needs both
  MTP+capture)".

## PRIOR ART IN THIS REPO (this abort was already root-caused once)

The 2026-06-25 w8a8 MTP campaign (JOURNAL ~L4270-4438) already hit and localized
this exact abort. Established then:

- faulthandler (B70_DEBUG=1) caught the ground-truth traceback:
  ```
  torch/xpu/graphs.py:108 in replay
  vllm/compilation/cuda_graph.py:360 in __call__
  vllm/model_executor/models/qwen3_5_mtp.py:418 in forward   (MTP DRAFTER forward)
  vllm/v1/spec_decode/llm_base_proposer.py:642 in propose
  ```
- Root cause (then): NEO `LinearStream::getSpace` overflow during torch.xpu graph
  REPLAY of the MTP-drafter captured forward, ACCUMULATING over ~5000 spec steps
  (~20k cumulative gen tokens / ~20 min) until the command stream can't fit the
  next command -> abort. Silent worker death -> shm_broadcast cancelled -> EngineDead.
- Bisect (v0.23): MTP-on + capture -> CRASH; MTP-off + capture -> STABLE;
  MTP-on + enforce-eager -> STABLE (survived 40m / 37k tokens + full agentic eval).
  PUSH_AR EXONERATED (crashes without it too, just slower). => crash needs BOTH
  MTP AND graph capture; drop either and it is stable.
- Fix picked THEN: enforce-eager + MTP (stable, ~16 t/s) OR MTP-off + capture.

## THE CORRECTION (2026-07-07)

RESEARCH_TODO 11h and the 2026-07-06 NVFP4 bisection concluded the abort was
"specific to the NVFP4 fused-kernel graph" and that "W8A8 runs MTP-verify inside
the captured graph fine." TODAY REFUTES THAT: W8A8 + MTP3 + PIECEWISE hit the
IDENTICAL abort. So:

- The abort is NOT quant-specific. It is the generic MTP-drafter-graph-replay
  command-stream overflow, present on BOTH NVFP4 and W8A8.
- The v0.24.0 "PIECEWISE + MTP restarts=0" validation (2026-07-03) and the
  "W8A8 keeps MTP+graph" claim were UNDER-SOAKED. v0.24.0 raised the crash
  threshold (crash at ~3h of real use instead of ~20 min) but did not remove it.
- "Stable W8A8 keeps MTP at 36-43 t/s" (used to justify the 2026-07-06 revert)
  described the CRASHING config. The real stable W8A8 options are enforce-eager+MTP
  (~16 t/s) or capture+MTP-off. The DD decision rests on a false premise and must
  be revisited.

## Two candidate mechanisms on v0.24.0 (to disambiguate on GPU)

The old rate (~20k tokens -> crash) does NOT fit today (server up ~3h). Either the
leak rate dropped a lot, or a second mechanism is in play:

- M1 ACCUMULATION: per-replay L0 command/event handle not reclaimed -> LinearStream
  grows monotonically across spec steps -> overflow. (2026-06-25 mechanism.)
- M2 CONTEXT-GATED SINGLE STEP: a single captured decode step at long context
  (~50K KV) submits a command buffer proportional to KV blocks; past a size it
  overflows the linear stream on ITS OWN, no accumulation needed. Fits today's
  crash (single request, 50K ctx, num_common_prefix_blocks=[..,61]) AND the
  operator's Q1 intuition that 128K-maxlen serving did not crash.

SHARP FIRST EXPERIMENT: fresh serve (DD config, B70_DEBUG=2 leak checker), fire ONE
~50K-token forced-decode (ignore_eos) MTP request immediately.
 - crashes on first long request        -> M2 (context-gated single-step overflow)
 - survives, only dies after long soak   -> M1 (accumulation leak)
The leak-checker (ZEL_ENABLE_BASIC_LEAK_CHECKER) names the accumulating handle for M1.

## What linear_stream.h:84 actually is (NEO source, researched 2026-07-07)

`LinearStream::getSpace(size)` in intel/compute-runtime aborts via
`UNRECOVERABLE_IF(sizeUsed + size > maxAvailableSpace)` (and a sibling
`+ batchBufferEndSize` check in `ensureContinuousSpace`). It is a FIXED-CAP
command-buffer overflow: it fires when a single indivisible encoded command-stream
write cannot fit the current command buffer AND the path cannot chain to a fresh
one (immediate command list / CSR ring, or a single command list larger than one
whole command-buffer allocation). maxAvailableSpace is the per-buffer byte cap of
the command-buffer graphics allocation -- NOT heap/KV memory. So this is about how
many BYTES of GPU commands one submission encodes (kernel launches, walkers,
barriers, semaphore/pipe-control waits), not model memory.

Consequence: a captured MTP-verify decode graph whose encoded command list exceeds
the (~2MB-class) default command-buffer cap, on a path that can't chain, aborts.
Longer context => more KV-block attention commands => more bytes in the SAME
captured-size graph => tips over the cap. This unifies M1 and M2: the real variable
is encoded command-stream BYTES per submission.

Novelty: the agent found NO public issue tying linear_stream.h:84 to graph
capture / spec-decode. A characterized repro + fix would be a genuine upstream
contribution (compute-runtime / torch-xpu-ops / vLLM-XPU).

### TOP CANDIDATE FIX (keeps MTP + capture = fast): enlarge the command buffer
NEO debug key, no rebuild, env-only:
```
NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=4096   # then 8192 / 16384
```
If the captured MTP-verify command list merely exceeds the default cap, a bigger cap
holds it -> stable WITH capture AND MTP (the best case RESEARCH_TODO 11h wanted).
Only helps the growable-container path; if it's a non-chainable CSR-ring stream it
won't, and we fall back to bounding command bytes (smaller CAPSIZES / fewer spec
tokens / shorter captured decode) or the proven eager+MTP.

Ranked fix ladder:
1. OverrideCmdListCmdBufferSizeInKb=big (env, keeps MTP+capture) -- TRY FIRST.
2. Reduce encoded bytes per graph: fewer spec tokens, smaller CAPSIZES, cap
   captured decode ctx.
3. enforce-eager + MTP (PROVEN stable, ~16 t/s) -- reliable fallback.
4. capture + MTP-off (fast decode, loses MTP acceptance).

## MECHANISM LOCALIZED (binary disasm, 2026-07-07)

Disassembled `at::xpu::XPUGraphImpl::replay()` in libtorch_xpu.so (torch 2.12+xpu):
replay submits the executable SYCL `command_graph` via `submit_with_event` onto the
stream's in-order queue and NEVER synchronizes (no queue.wait / synchronize / event
destroy in the function body). Each replay therefore (a) appends a graph-exec command
into the queue's NEO immediate command list and (b) allocates a sycl::event / L0
event-pool handle that is dropped un-waited. NEO reclaims both only on a full queue
synchronize.

- Normal single-token decode reclaims implicitly: the sampled-token D2H copy forces a
  per-step queue sync.
- MTP breaks it: LLMBaseProposer.propose (llm_base_proposer.py:613-687) fires
  (num_spec_tokens-1) x (num piecewise sub-graphs) replays per outer decode step,
  keeps all draft tokens on-GPU (torch.stack at the end), and does NOT host-sync
  between draft steps. Across thousands of steps the un-reclaimed commands/events pile
  into the queue's NEO command list until LinearStream::getSpace can't fit the next
  command -> linear_stream.h:84 abort. Needs BOTH MTP (the sync-free replay loop) AND
  capture (the graph-exec commands). Matches every prior observation.
- Call path: qwen3_5_mtp.py forward (@support_torch_compile -> N piecewise
  CUDAGraphWrapper sub-graphs) -> cuda_graph.py:360 replay (capture-once, replay-by-
  descriptor; NO per-call recapture) -> torch/xpu/graphs.py:105 replay -> C++
  XPUGraphImpl::replay submit_with_event. CUDAGraph is torch.xpu.XPUGraph (rebound in
  xpu_model_runner.py:52-54).

### Candidate fixes (all rebuild-free; ranked)
1. FIX-SYNC (principled, keeps MTP+capture): one torch.xpu.synchronize() per propose()
   call -> restores the per-step reclaim cadence normal decode already has. Negligible
   cost (one sync per decode step). Patch point: wrap LLMBaseProposer.propose via
   sitecustomize. UPSTREAM-WORTHY (torch XPUGraphImpl::replay should use
   submit_without_event or drain per replay; the vLLM-side sync is the portable fix).
2. FIX-EVT (cheapest if it works): UR/L0 event-cleanup env so discarded events are
   reclaimed without a full sync (e.g. UR_L0_REUSE_DISCARDED_EVENTS=1, or the immediate-
   command-list event-cleanup threshold). Env-only.
3. FIX-BUF (symptom only): NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=large ->
   bigger command buffer just DELAYS an unbounded accumulation; not a true fix, but a
   useful mechanism corroborator (if it only postpones the crash, accumulation confirmed).
4. Fallbacks: enforce-eager+MTP (proven) or capture+MTP-off.

## Experiment log

### E1 (2026-07-07): baseline repro at MAXLEN=131072 -- did NOT crash
config -> DD config exactly (MTP3 + PIECEWISE + PUSH_AR + prefix cache + vision),
  MAXLEN=131072, B70_DEBUG=1 (faulthandler). Probe: 6x forced-decode (ignore_eos)
  of ~55K-token prompts, 3000 tokens each = 18k cumulative decode tokens.
command -> vllm/repro_campaign.sh (CFG_LABEL=e1_baseline_ml131k).
result -> 6/6 CLEAN, prompt_tok=55148, comp_tok=3000 each, decode ~10-14 t/s at
  55K ctx. NO abort. Container healthy after.
verdict -> v0.24.0 leak RATE is far lower than v0.23 (old crash ~20k tokens / ~20min;
  here 18k tokens clean). ALSO: a 55K-context request did NOT crash on run0 -> refutes
  a pure "single long-context submission too big" (M2) trigger. It IS accumulation, but
  slow -- the real DD crash took ~3h of production traffic. Two suspects for the gap:
  (a) MAXLEN=200000 (real) vs 131072 (E1); (b) single-stream 18k tokens vs hours of
  CONCURRENT traffic exercising capture sizes 6/8. Q1 (128K maxlen) partly answered:
  128K maxlen with a 55K request is NOT sufficient to crash on its own; the operator's
  "didn't crash at 128K" memory is consistent with lower cumulative/concurrent load then,
  not a hard maxlen threshold. NEXT: concurrent soak (E1c) to force it fast.

### E1c (2026-07-07): concurrent soak on the live MAXLEN=131072 container -- RUNNING
config -> reuse E1's healthy container. 6 concurrent forced-decode streams, ~8K ctx
  (fast decode = max replay rate), 4000 tok/req, ceiling 900k tok / 45 min.
command -> vllm/soak_concurrent.py e1c_concurrent_ml131k.
result -> *** CRASH (HTTP500) at tok=96000 t=1453s (~24min) reqs=24. faulthandler:
  "Abort was called at 84 line ... linear_stream.h" + "torch/xpu/graphs.py line 107 in
  replay" + EngineDeadError. REPRODUCED with the exact signature.
verdict -> RELIABLE REPRO. Threshold on v0.24.0 ~= 96k decode tokens under 6-way
  concurrency (~24min). Single-stream 18k (E1) was clean -> CONCURRENCY is the accelerant
  (saturates GPU = max replays/sec, and exercises capture sizes 6/8). MAXLEN=131072 ->
  Q1 CLOSED: maxlen is NOT the trigger; the operator's "no crash at 128K" was lower load.
  Baseline crash threshold for fix A/B = survive >> 96k tokens (target ~220k = 2.3x).

### FIX-SYNC (2026-07-07): per-step torch.xpu.synchronize -- FAILED (same threshold)
config -> DD config + B70_XPU_CG_SYNC_STEPS=1 (block 5: sync every decode step, no
  recapture). Confirmed ENABLED in both TP workers. Same 6-way concurrent soak.
result -> *** CRASH (HTTP500) at tok=96000 t=1433s -- NEARLY IDENTICAL to baseline
  (96000/1453s). Generation throughput ~80-90 t/s throughout (sync had NO perf cost
  AND no benefit). 4 abort lines (linear_stream.h:84) in container log.
verdict -> [KEY NEGATIVE] a full queue synchronize every decode step does NOT reclaim
  the accumulation. Refutes the "submit_with_event drops reclaimable events -> sync
  frees them" hypothesis (normal decode ALREADY syncs per step via sampled-token D2H,
  and it still crashes; adding another sync changed nothing). The growth is in a
  command list that is reset only by graph DESTRUCTION/RE-INSTANTIATION, not by a
  queue drain. => the fix must RECAPTURE (or reset) the captured graphs, or bound the
  replay count. Confirms the Tier F (block 3) design direction. FIX-EVT (event-cleanup
  env) is now also unlikely (sync would have reclaimed events). NEXT: block 3 recapture.

### FIX-RECYCLE (2026-07-07): recapture all graphs every N steps -- RUNNING
config -> DD config + B70_XPU_CG_RECYCLE_STEPS=2000 (block 3: torch.xpu.synchronize +
  clear_all_graphs every ~2000 decode steps -> bounded command list, keeps MTP+capture).
  Same 6-way concurrent soak to 220k tokens.
result -> *** CRASH at tok=24000 t=426s -- EARLIER than baseline and NOT a linear_stream
  abort (abort-lines=0 captured pre-teardown). A recapture fired around step ~2000 and
  crashed the engine itself.
verdict -> [FAILED, and instructive] block 3's _recycling_call calls
  CUDAGraphWrapper.clear_all_graphs() from INSIDE a wrapper __call__ -- it clears EVERY
  captured graph including `self`, the one about to be replayed THIS step, mid-batch under
  6 concurrent reqs -> unsafe recapture during an active step -> engine crash (different
  failure, not the NEO abort). clear_all_graphs/clear_graphs DO exist (cuda_graph.py:173/230),
  so it is a RACE, not a missing API. Safe recapture needs a quiescence point (no in-flight
  step) which a monkeypatch inside the hot wrapper cannot guarantee during live serving.

## CONCLUSION (2026-07-07): cheap-fix space exhausted; root cause definitive

Root cause DEFINITIVE and REPRODUCED: XPU graph REPLAY (at::xpu::XPUGraphImpl::replay,
submit_with_event, no reset) accumulates NEO command-list entries per replay; the MTP
propose loop fires ~spec x pieces replays/step, so the command list overflows
(linear_stream.h:84) after ~96k decode tokens under concurrent load. Needs BOTH MTP and
capture. NOT quant-specific (W8A8 == NVFP4), NOT maxlen-gated (crashes at 128K), NOT the
decode-while-prefill "!!!!" bug. sglang never hit it because it never captures MTP.

Cheap fixes tried and RULED OUT:
- per-step torch.xpu.synchronize (FIX-SYNC): no effect -- sync does not reclaim the
  command-list growth (normal decode already syncs per step and still crashes).
- recapture-every-N (block 3 Tier F): crashes when it fires (clears the in-flight graph
  mid-step); racy under concurrent load.
- event-cleanup env (FIX-EVT): not attempted -- FIX-SYNC already showed a full drain does
  not reclaim, so reclaimable-events is not the mechanism.
- buffer enlarge (OverrideCmdListCmdBufferSizeInKb): only multiplies the threshold; the
  growth is unbounded, so it delays, never fixes.

Truly stable configs (no capture-replay accumulation): enforce-eager + MTP (PROVEN
2026-06-25: 40min soak + full agentic eval, 0 crashes, ~16 t/s, keeps MTP) is the only
leak-PROOF config that keeps MTP. capture + MTP-off still leaks via the target graph, just
~spec-times slower (~290k tokens) -> not truly stable for a multi-day DD.

Remaining REAL-fix options (all substantial, tracked as future work):
1. torch-level: rebuild libtorch_xpu so XPUGraphImpl::replay uses submit_without_event or
   resets/updates the command list per replay (the upstream-worthy root fix). Novel -- no
   public issue exists.
2. safe periodic recapture: coordinate a quiescence barrier (drain all in-flight reqs,
   pause scheduler) before clear+recapture. Engine-level, non-trivial.
3. reduce replays/step: FULL_DECODE_ONLY capture (1 replay/step vs N pieces) or drafter-eager
   + target-captured -> pushes threshold ~spec x out; may still crash on long sessions.
4. fast + auto-heal: keep capture+MTP (fast ~40 t/s) + systemd Restart=on-failure so the
   ~few-hourly clean abort self-recovers in ~1 min (accept periodic blips).

DD restore decision (speed vs blips) -> deferred to operator.

## NVFP4 fix campaign (2026-07-07 session 2): faster repro + drafter-eager

Operator decision: chase the fix on NVFP4 (crashes far sooner = faster iteration; and it is the
biggest beneficiary -- higher 4-bit decode ceiling + top quality, currently fully blocked).

### NVFP4 repro finding: the fast crash needs TP=2
- NVFP4 TP=1 (single card) MODE=fused GRAPH=1 MTP5 KV_FP8=0: SURVIVED 9000 forced tokens
  single-stream at ~50-75 t/s -- NO crash. TP=1 has no TP collectives in the captured graph.
- NVFP4 TP=2 (both cards) same config: *** CRASH at ~8-12k tokens single-stream (~6 min), EXACT
  NEO abort. faulthandler traceback:
  ```
  torch/xpu/graphs.py:107 replay
  vllm/v1/spec_decode/llm_base_proposer.py:667 propose   <- drafter self.model() in the draft loop
  gpu_model_runner.py propose_draft_token_ids
  ```
- MECHANISM REFINEMENT: at TP=2 the capture-safe all-reduce (all_gather shim, block 3) is recorded
  INTO the captured graph on EVERY replay -> each replay encodes the collective's commands too ->
  ~10x more command bytes/replay than TP=1 -> crashes ~10x sooner. This is why NVFP4 TP=2 crashes far
  sooner than W8A8 TP=2 (nvfp4_gemm ALSO encodes more/replay), and why single-card NVFP4 is fine.
  The abort is IN the drafter propose loop (llm_base_proposer.py:667) -> drafter-eager should target it.
- Reliable NVFP4 repro for fix A/B: TP=2 single-stream forced decode, crash at ~8-12k tokens / ~6 min.

### DRAFTER-EAGER (B70_XPU_DRAFTER_EAGER=1) -- RUNNING
config -> NVFP4 TP=2 fused GRAPH=1 MTP5 KV_FP8=0 + drafter forced to CUDAGraphMode.NONE (block 4b):
  MTP drafter runs eager (no graph replay = no leak from its propose loop), target decode stays
  PIECEWISE-captured. Soak to 50k tokens (5x the ~10k crash point).
result -> FIRST TRY INVALID: block failed to load (AttributeError -- the class is
  SpecDecodeBaseProposer, not LLMBaseProposer as agent-2 labeled it) -> ran as baseline,
  crashed at ~8k. Fixed the class name, RETRIED:
result (retry) -> *** SURVIVED 44,000 tokens / 11 reqs / ~32 min, NO abort *** (baseline
  crashes at ~8-12k). Block confirmed ENABLED in both TP workers. Decode ~22-26 t/s
  single-stream on random forced text.
verdict -> [FIX VALIDATED] drafter-eager FIXES the NEO abort, rebuild-free, keeping MTP
  AND target-decode capture. Mechanism confirmed: the drafter's sync-free replay burst was
  the dominant leak source; running it eager removes it. The target decode graph still
  replays (captured) but its accumulation is slow enough to be practically stable (same as
  the graph+MTP-off "B4" config that ran as the NVFP4 DD). Toggle: B70_XPU_DRAFTER_EAGER=1
  (sitecustomize block 4b). Class = vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer,
  method initialize_cudagraph_keys forced to CUDAGraphMode.NONE.
  COST: ~22-26 t/s on random text (vs eager+MTP ~15-16 -> a WIN; vs the crashing captured
  config ~31-48 -> a haircut, because the drafter runs eager). On real CODE, MTP accept is
  ~99% so drafter-eager should beat graph+MTP-off (B4) -- TO MEASURE next.
  NOTE: this is quant-AGNOSTIC (SpecDecodeBaseProposer is shared vLLM code) -> ports to W8A8
  unchanged. The full-speed ceiling (keep the drafter captured too) needs the torch-level
  root fix (XPUGraphImpl::replay -> submit_without_event / reset per replay; rebuild) -- the
  upstream contribution, tracked as the follow-on.

### drafter-eager decode measurement (NVFP4 TP=2, 2026-07-07)
config -> NVFP4 TP=2 fused GRAPH=1 MTP5 KV_FP8=0 + B70_XPU_DRAFTER_EAGER=1.
result -> CODE prompt (streaming): decode ~17 t/s, MTP mean acceptance length 4.6-5.04
  (near-max for spec=5 -> MTP quality FULLY retained). Random forced (non-stream soak):
  ~22-26 t/s. (Streaming undercounts vs non-stream; treat ~22-26 as the raw number.)
verdict -> [IMPORTANT] drafter-eager decode is CAPPED by the eager-drafter cost: even at
  accept 5.04 on code, decode ~= random text. So the ~99% accept does NOT translate to
  speed -- running 5 eager drafter forwards/step eats it. drafter-eager (~22-26) therefore
  lands ~= the existing stable graph+MTP-off B4 (25-31), NOT the crashing captured+MTP
  (31.9 / 48-50 warm). It is a VALID, STABLE fix that PROVES the root cause and keeps MTP,
  but it is NOT a speed win over B4. Reclaiming the full 31-48 (drafter captured too) needs
  the torch-level root fix (XPUGraphImpl::replay submit_without_event / reset-per-replay).
  Possible drafter-eager tuning: lower MTPTOK (fewer eager drafter forwards/step) -- untested.

## TORCH-LEVEL ROOT FIX + ENV SHORTCUT (2026-07-07 session 3, binary+source research)

Located the exact leak site and fix (pytorch core, NOT torch-xpu-ops):
- `aten/src/ATen/xpu/XPUGraph.cpp:183` `XPUGraphImpl::replay()` does
  `queue.ext_oneapi_graph(*graph_exec_)` -- and ALL `ext_oneapi_graph` overloads route through
  `submit_with_event`, allocating a host-visible sycl::event per replay that pins its immediate-
  command-list segment until the UR L0 cleanup pass runs. No sync in replay.
- ROOT FIX (diff): replace with `queue.submit_without_event(empty_properties, [&](handler& cgh){
  cgh.ext_oneapi_graph(*graph_exec_); })`. submit_without_event IS in the shipped SYCL
  (libsycl.so.8). Compiled into libtorch_xpu.so -> INCREMENTAL relink
  (`cmake --build build --target torch_xpu`), minutes, but needs the full pytorch-xpu build env
  (USE_XPU=ON, oneAPI 2025.3). No IPEX in the image (stock upstream torch-xpu).

### THE ENV SHORTCUT (rebuild-free -- testing now)
The UR L0 adapter reclaims immediate-command-list event/command space only every N completed
events: `UR_L0_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD` DEFAULT 20. So the command list
grows for 20 replays before any reclaim -> under a fast MTP replay burst it overflows before the
pass runs. This EXPLAINS why FIX-SYNC failed: torch.xpu.synchronize waits for completion but does
NOT trigger the adapter's cleanup pass. Setting THRESHOLD=1 forces reclaim after every replay.
Candidates (rebuild-free, ranked): (1) UR_L0_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD=1
[+ UR_L0_REUSE_DISCARDED_EVENTS=1]; (2) UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 (regular batched lists,
fence reset); (3) UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1. If (1) holds the CAPTURED+MTP config
(31-48 t/s) -> full-speed fix with NO rebuild and NO drafter-eager haircut. TEST: nvfp4_urcleanup_tp2.
Rebuild-free interception if env fails: rotate the XPU stream every N replays (fresh immediate
command list); keep_graph=True re-instantiate reclaims the graph's list but not the queue's.

ENV TEST RESULTS (NVFP4 TP=2 captured+MTP, no drafter-eager):
- THRESHOLD=1 + REUSE_DISCARDED_EVENTS=1: CRASH at ~12k tok (~= baseline). Reclaims EVENTS but
  NOT the command-list command SEGMENTS that actually grow -> no fix. IMPORTANT byproduct: decode
  throughput logged the FULL captured+MTP speed and its DEGRADATION: 43.5 -> 34 -> 30 -> 27 -> 20
  t/s as the command list grows, then crash. Confirms full speed ~43 t/s AND that the growing
  LinearStream progressively slows submission (not just a hard cap).
- UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 (regular batched lists + fence reset): CRASH at ~12k tok,
  throughput slower + erratic (27->18->25->17->13). No fix.
=> ENV-SHORTCUT PATH EXHAUSTED. Neither event-cleanup nor batched-lists reclaims the per-replay
   graph-exec command APPENDS. Only the source-level submit_without_event fix (or not appending
   per replay) addresses it. Next: the libtorch_xpu.so relink OR ship drafter-eager + upstream the
   diff. NOTE the heavy cost: a source pytorch-xpu build (first full build hours; ABI must match the
   prebuilt torch 2.12 the image + vllm_xpu_kernels were built against -- real drop-in risk).

### sglang as a fix vehicle? NO (evaluated 2026-07-07)
sglang uses the SAME torch.xpu.XPUGraph/XPUGraphImpl::replay primitive -> identical leak
(backend-independent torch bug). It carries an EXTRA blocker vLLM already solved: capturable
cross-device all-reduce at TP=2 (sglang/graph_mtp closed it as a dead end 2026-06-25; vLLM's
piecewise records collectives fine). sglang's only viable MTP+capture shape is "target captured,
draft eager" = EXACTLY B70_XPU_DRAFTER_EAGER (no new speed), and its draft-captured path already
HUNG (2026-06-28). NVFP4 on sglang = multi-day custom-kernel port for zero gain (the nvfp4 oneDNN
op is vLLM-ABI). VERDICT: do the torch fix on vLLM (helps both backends); keep sglang as the
eager fallback. (Agent-reviewed against sglang/graph_mtp/README.md + docs/20260625_...campaign.md.)

## TORCH-LEVEL FIX BUILT + VALIDATED (2026-07-07 session 3)

Built a patched libtorch_xpu.so from pytorch source @ 7661cd9 (exact torch 2.12.0+xpu SHA),
ABI-matched to the prebuilt (cxx11abi=1, gcc-13, oneAPI 2025.3, USE_XCCL=ON). The fix in
XPUGraph.cpp replay(): `queue.ext_oneapi_graph(*graph_exec_)` -> `execute_graph(queue, *graph_exec_)`
(the PUBLIC event-less free function from enqueue_functions.hpp; submit_without_event is a private
internal template). Build recipe: docs/torch_xpu_build_recipe.md; patch: vllm/patches/xpugraph_*.patch.
Build gotchas fixed: setvars return-at-top-level, cmake 4.3.4 too new (->3.31), incomplete submodules
(skip CUDA flash-attention), USE_XCCL needed CCL 2021.17 (libccl.so.2.0).

VALIDATION (all PASS):
- ABI gate: NEW defines 75 XCCL syms (== prebuilt); 0 MISSING of the 23 symbols other torch libs
  import from torch_xpu; symbol count 128219 vs 128209. patchelf normalized RPATH + 3 MKL NEEDED.
- import torch: loads clean with the patched .so (no undefined symbols), cxx11abi True.
- GPU smoke (gpu-run --card 0): xpu available True; XPUGraph replayed 300,000 times cleanly in 13s
  (directly exercises the patched replay() -- the leaking function -- no abort).
- END-TO-END: NVFP4 TP=2 captured+MTP (the CRASHING config, NO drafter-eager) + patched .so overlay,
  soak past 55k tokens -> (result pending; baseline crashes at ~8-12k, degrades 43->20 t/s).

## STATE / DECISION (2026-07-07 end of session 2)

- Root cause: DEFINITIVE (disasm) + REPRODUCED (NVFP4 TP=2 ~8-12k tok; W8A8 96k concurrent).
- Fix ladder result:
  - enforce-eager+MTP: stable, ~15-16 t/s (proven fallback).
  - graph+MTP-off (B4): stable-ish, 25-31 t/s, NO MTP (current NVFP4 stable DD).
  - drafter-eager (NEW, validated): stable, ~22-26 t/s, KEEPS MTP -- but ~= B4 speed.
  - captured+MTP (was DD): 31.9/48-50 warm -- CRASHES (~8-12k NVFP4 / ~96k W8A8).
- To get captured+MTP speed WITHOUT the crash = the torch-level fix (rebuild libtorch_xpu:
  XPUGraphImpl::replay must not leak per replay -- submit_without_event or command-list reset).
  This is the novel upstream contribution (no public issue exists). Multi-hour (torch-xpu-ops
  build). => operator decision on whether to invest now.
