# W8A8 27B MTP + Graph: Stability + Performance Campaign (2026-06-25)

Status: ACTIVE (design complete; GPU execution gated on operator go-ahead).
Owner: research.
Model: `Qwen3.6-27B-W8A8-sqgptq-mtp-graft`, served id `qwen36-27b-w8a8-sqgptq-mtp`,
image `vllm-xpu-env:int8g`, TP=2, MTP spec=3.

This is the tracking doc for closing the gap between the SAFE eval path
(enforce-eager, ~9-16 t/s) and the FAST-but-crashing captured path
(PIECEWISE + MTP, ~34.8 t/s). Format throughout: config -> command -> result -> verdict.

Related: `FINDINGS.md` (186-201, 142-160), `MTP_TODO.md`, `27b_w8a8_research.md`,
`docs/handoff_decode_push_ar.md`, `JOURNAL.md` (2026-06-25 entries ~4206-4442),
`docs/20260624_devicelost_thoughts.md` (the DIFFERENT, hardware wedge).

---

## 0. TL;DR / decision

- The 27B-W8A8 MTP crash is **root-caused with high confidence and independently
  re-confirmed**: it is the **MTP/eagle drafter forward colliding with PIECEWISE
  cudagraph capture/replay on the TP=2 XPU path**. It appears **iff MTP-on AND
  graph-on**; removing either fixes it. Mechanism: the drafter's captured-graph
  Level-Zero (NEO) command buffer accumulates across ~5000 spec-decode replays
  until it overflows (`LinearStream::getSpace`), at ~16-23 min / ~20-28k generated
  tokens. Terminal form is non-deterministic: either a silent worker hang
  (`sample_tokens` RPC timeout) or a worker SIGABRT (`Fatal Python error: Aborted`).
  Both surface as `EngineDeadError`.
- **Critical correction vs the hardware wedge:** these crashes leave **both GPUs
  HEALTHY** (post-crash `xpu-health` returns HEALTHY; zero `DEVICE_LOST`,
  `OUT_OF_RESOURCES`, oneCCL, or NEO command-stream signatures). This is a
  software engine death, **not** the J.15/H.13 DEVICE_LOST box-wedge that needs a
  reboot. Do not conflate them. (Wedge risk in THIS campaign is therefore low; see
  section 8.)
- **W8A8 kernel and PUSH_AR are exonerated** (both present in survivors; PUSH_AR
  swapped graph.so->torch.so still crashed). Do NOT "fix W8A8 first."
- **The decisive lever (new):** vLLM 0.23 has a built-in
  `speculative_config.enforce_eager` field that disables cudagraph for the
  **drafter only** while leaving the **target model graphed**. One-line config.
  This is both the highest-ROI candidate fix AND a clean discriminator.

Order of operations: run Tier E (drafter-eager) first; it is cheap, low-risk, and
its outcome tells us whether the cure is "keep target captured + MTP at ~35 t/s"
or "the target hybrid spec-state path is also implicated."

---

## 1. Root cause (confirmed)

Evidence (forensics over 9 logs in `agentic-eval/results/logs/`, plus the prior-art
record):

Config -> outcome matrix (all TP=2, W8A8, max_num_seqs=1, push_ar engaged):

| Run | MTP | cudagraph | outcome | terminal signature |
|---|---|---|---|---|
| stress_isolate | spec=3 | PIECEWISE | CRASH req5 ~16m | `RuntimeError: cancelled` -> EngineDead |
| mtpon_faulthandler | spec=3 | PIECEWISE | CRASH req6 ~19m | `Fatal Python error: Aborted` (SIGABRT) |
| pushar_graph0 / 30m | spec=3 | PIECEWISE | CRASH req7 ~23m | `sample_tokens` RPC timeout (hang) |
| bisect_nomtp | OFF | PIECEWISE | SURVIVE 40m | -- |
| fix_eager / 40m | spec=3 | NONE (eager) | SURVIVE 40m / 37k tok | -- |
| apc_isolate (int4, TP=1) | none | PIECEWISE | SURVIVE | -- (control) |

Clean 2x2: crash requires **MTP-on AND cudagraph-on**. Disable either -> stable.

Faulting stack (faulthandler, from `w8a8_crashlog_30m.txt` + JOURNAL 4273-4291):
```
Abort  neo/.../command_stream/linear_stream.h:84   (NEO L0 command buffer full)
  torch/xpu/graphs.py:108  replay
  vllm/compilation/cuda_graph.py:360
  vllm/model_executor/models/qwen3_5_mtp.py:418  forward   <- the MTP drafter
  vllm/v1/spec_decode/llm_base_proposer.py:642   propose
```
Crash-step scheduler dump: `scheduled_spec_decode_tokens=[-1,-1,-1]`,
`num_computed_tokens=17399, num_output_tokens=3041` (~20.4k context). Acceptance was
HEALTHY (~70%) right until the stall -- this is a resource-accumulation hang, not a
correctness/garbage failure.

Threshold is consistent across all three deaths: ~16-23 min / ~20-28k cumulative
generated tokens / req 5-7. PUSH_AR's graph recording only changes the accumulation
RATE (full push_ar -> NEO abort ~19m; PUSH_AR_GRAPH=0 -> hang ~23m), not the cause.

Upstream corroboration: vLLM #25368 (Qwen3-Next MTP on TP: worker dies no-traceback
-> shm_broadcast "cancelled" -> EngineDeadError, stable with MTP off -- "essentially
our stack"); #41530, #41190, #40756 (MTP-TP long-run silent deaths). vLLM XPU
spec-decode is documented EXPERIMENTAL and lists only n-gram/EAGLE/EAGLE3; MTP on
XPU is upstream-untested, and every Intel XPU spec-decode example uses
`--enforce-eager`.

Distinct from "Bug B" (resolved 2026-06-24): that was numerical garbage (`!!!!`)
from ejecting TP collectives to eager, fixed by the capture-safe all_gather shim.
Bug B = correctness; this campaign = sustained-load resource overflow.

---

## 2. The performance prize

Decode t/s, single-stream, temp=0 coherence-gated probe (scripts/111;
`27b_w8a8_research.md` 246-259):

| Config | decode t/s | accept_rate | accept_len | stable? |
|---|---:|---:|---:|---|
| eager, no-MTP | ~4.1 | -- | -- | yes |
| captured, no-MTP | 18.10 | -- | -- | yes |
| eager, MTP spec=5 | 10.43 | 0.36 | 2.80 | yes |
| **enforce-eager MTP spec=3 (SHIPPED FIX)** | **~9-16** | ~0.9 draft | 3.7 | **yes** |
| captured, MTP spec=3 (WINNER, CRASHES) | **34.82** | 0.512 | 2.53 | NO (this crash) |
| captured, MTP spec=4 | 30.56 | 0.368 | 2.47 | NO |
| captured, MTP spec=5 | 26.10 | 0.258 | 2.29 | NO |

Decode is monotonically DECREASING with spec count (1-layer MTP head, useful draft
horizon ~3 tokens) -> spec=3 is the shipped default. The prize for a stable captured
path is roughly **2.5-2.7x decode throughput** over the current enforce-eager fix.

Prefill/TTFT is a separate axis: 84% collective-bound, addressed by PUSH_AR
(3.8x TTFT, +80-126% agg throughput; shipped shelf default, independent of this
campaign).

---

## 3. Q1 answer: does graph capture and/or MTP hurt concurrency at c<=8?

Operating regime stated: never over 8, rarely over 4, mostly 1-2.

1. **Graph capture is concurrency-NEUTRAL within c<=8.** The capture set is
   `cudagraph_capture_sizes=[1,2,4,6,8]`, which already covers every batch up to 8.
   Each size is a separate captured graph; the only costs are capture time (~2 s)
   and ~1.3 GiB graph memory. At c>8 the engine pads up to the max captured size or
   falls back to eager for that step -- functional, just unoptimized. So as long as
   your real batch sizes are <=8 (they are), capture HELPS and never hurts decode.
   (With MTP spec=K, the per-step uniform-decode shape is `1+K` query length per
   seq; the capture sizes above are in the batch/num-seqs dimension and already
   cover c<=8. If you ever raise max_num_seqs above 8, extend CAPSIZES to match.)

2. **MTP is a latency lever, not a throughput lever; it erodes -- and can reverse --
   at concurrency.** At c1 (bandwidth-bound) MTP is a clear win; as the batch grows
   the verify step goes compute-bound and the speculative win shrinks. Measured on
   the int4 sibling (ctx=2048): c1 tg 29.78 -> 46.69 (MTP helps), but **c4 REGRESSES**
   (agg out 51.69 -> 40.56, TTFT 3.40 -> 4.44 s). At TP=2 the crossover is a bit more
   favorable to MTP because it also amortizes the all-reduce tax, but the direction
   is the same. Net for your regime: MTP is a strong win at c1-2, roughly
   break-even at c4, and you should not rely on it past c4-6.

3. **The crash is concurrency-INDEPENDENT** -- it is driven by cumulative drafter
   graph REPLAYS (~5000), i.e. total generated tokens, not batch size. Each decode
   step replays the drafter once and advances the whole batch by ~accept_len
   tokens, so higher concurrency reaches a given token count in FEWER steps ->
   fewer replays -> if anything it DELAYS the replay-count crash. Concurrency is not
   the crash axis and not a mitigation to chase; the fix is in sections 4/7.

Practical guidance: keep MTP for the c1-2 latency win; ensure CAPSIZES covers the
batch sizes you actually serve; do not size the system around c>4 MTP throughput.
For GREEDY EVALS specifically, scores are concurrency-invariant -- the eval pins
c=1 (thinking-on) / c=4 (thinking-off) only to set the wall-clock operating point.

---

## 4. Evaluation of the external assistant's matrix

The external analysis (no access to our record) was directionally good. Reconciled
against our root cause and our already-run experiments:

RIGHT and adopted:
- "Do not fix W8A8 first; W8A8/PUSH_AR are probably innocent." Correct (proven).
- "Target model graph ON, MTP drafter graph OFF is the highest-ROI fix." Correct,
  and BETTER than they knew: vLLM 0.23 exposes this as a first-class config field
  (`speculative_config.enforce_eager=true`) -- no patch needed for the first cut.
  This is our Tier E and runs first.
- "Try `cudagraph_mode=NONE` (keep inductor compile, drop graph replay) before the
  nuclear `--enforce-eager`." Worth a cheap measurement (Tier B): since the crash is
  in REPLAY, NONE should be stable, and may beat full eager on speed for free.

ALREADY SETTLED (do not re-run as discovery):
- PIECEWISE + MTP is the crashing config (that IS the bug); we already have it.
- MTP-off captured is stable but loses the MTP win (already known).
- `--enforce-eager` is the current stable fix (already shipped per-config).

WRONG / not applicable to OUR root cause:
- "Restrict graph capture sizes to avoid command-stream overflow." Does NOT help.
  The accumulation is PER-REPLAY of whichever captured size is hit (mostly size 4
  at max_seqs=1, spec=3), not per-distinct-captured-size. Fewer capture sizes ->
  same per-replay growth -> same crash. (We will still narrow CAPSIZES in Tier E/B
  for capture speed, not as a fix.)
- "Sweep num_speculative_tokens K=1..5 as a fix." K only changes drafter replays
  per step, so lower K DELAYS the crash but also lowers perf (decode is monotonic in
  K). Not a fix; at most a fallback knob. Low priority.
- "Pin an older oneAPI/torch stack / Intel validated BOM." Heavyweight; our crash is
  a specific vLLM spec-decode-graph interaction with a clean config lever, so we try
  the config lever before re-imaging. Kept as a late fallback only.
- FULL / FULL_DECODE_ONLY: on stock `:int8g`/v0230 these are KERNEL-GATED
  (`spec_query_start_loc must have size [num_spec_decodes+1]` baked into
  `_xpu_C.gdn_attention`; FULL_DECODE_ONLY also hits the SYCL-Graph
  `work_group_scratch_memory` wall). They need an Intel vllm_xpu_kernels fix, so they
  are deferred behind a kernel patch, not in the first matrix.

---

## 5. Lever hierarchy (ranked by ROI)

1. **Tier E -- drafter eager, target captured** (`speculative_config.enforce_eager=true`,
   GRAPH=1 PIECEWISE). Removes the exact crash locus (drafter replay) while keeping
   the target body + verify captured. If stable -> ~near-35 t/s production fix.
   Also the discriminator (see contingency in section 0). One config flag.
2. **Tier B -- cudagraph_mode=NONE + MTP** (CGMODE=NONE, GRAPH=1). No replay
   anywhere, keep inductor compile. Should be stable; measure whether it beats
   enforce-eager on speed -> a free eval win even if E fails.
3. **Tier F -- drafter graph periodic reset/recapture** (monkeypatch). Only if E
   fails but we still want FULL drafter capture: bound/reset the drafter graph's
   command stream every N replays. Higher effort.
4. **num_spec_tokens fallback** (K=1/2 captured). Delays crash; minor; only if E/B
   both fail and we want some capture.
5. **FULL_DECODE_ONLY** -- behind an Intel GDN-spec kernel fix. Later frontier.

---

## 6. Test matrix

All runs: serve via the recipe (`rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh`
through `rdy_to_serve/_common/lib.sh`) under `bin/gpu-run` (both cards). Knob ->
env mapping is in section 7. Soak = drive long generations until PASS-token target
or crash; then a real aider+swe smoke.

| Tier | What it tests | Key env (on serve.sh) | Expectation |
|---|---|---|---|
| A (control) | shipped enforce-eager fix | `GRAPH=0` | STABLE, ~9-16 t/s (baseline) |
| Bug-repro | reproduce the crash, fixed seed | `GRAPH=1` (default PIECEWISE+MTP spec3) | CRASH ~20k tok (confirm harness sees it) |
| **E** | **drafter eager, target captured** | `GRAPH=1 SPEC='{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'` | hopefully STABLE at ~near-35 t/s |
| B | no replay, keep compile | `GRAPH=1 CGMODE=NONE` | STABLE; speed vs A is the question |
| E-K2 | drafter eager + spec=2 | as E with `num_speculative_tokens:2` | only if E marginal |
| no-MTP cap (control) | captured, MTP off | `GRAPH=1 B70_NOMTP=1` | STABLE 18.1 t/s (sanity) |
| F | drafter graph reset (patch) | `GRAPH=1` + `B70_FORCE_DRAFTER_GRAPH_RESET=<N>` shim | STABLE at full capture (if E fails) |

Run order: A (quick baseline) -> Bug-repro (confirm harness) -> **E** -> branch on E:
- E STABLE: lock E as the new fix; measure E vs B vs captured-crashing perf; ship.
- E STILL HANGS: it is the TARGET hybrid spec-state path (codex contingency:
  `postprocess_mamba_align_gpu`, GDN spec metadata under graph). Fall to B (no
  target replay) for stability, and open a target-state investigation.

---

## 7. Knob -> env mapping (consumed in `rdy_to_serve/_common/lib.sh`)

Set as env when invoking `serve.sh`; for the eval path set
`EVAL_SERVE_ENV="K=V ..."` in `agentic-eval/configs.sh` (per-config).

| Knob | Env | Default (recipe / eval) | Effect |
|---|---|---|---|
| enforce-eager vs graph | `GRAPH` | 1 / eval forces 0 | 0 -> `--enforce-eager`; 1 -> `--compilation-config` capture |
| cudagraph_mode | `CGMODE` | PIECEWISE | CC JSON `cudagraph_mode` (NONE/PIECEWISE; FULL kernel-gated) |
| capture sizes | `CAPSIZES` | `1,2,4,6,8` | CC `cudagraph_capture_sizes` |
| compile_sizes | `COMPILESZ` | empty | must stay empty for spec-decode |
| raw spec JSON | `SPEC` | empty | full `--speculative-config` JSON (overrides MTPTOK) -- this is how Tier E sets `enforce_eager:true` |
| MTP spec tokens | `MTPTOK` | 3 | builds spec JSON when SPEC empty |
| MTP off | `B70_NOMTP=1` | -- | clears spec entirely |
| max_num_seqs | `MAXSEQS` | 8 / eval 1 (think-on), 4 (off) | `--max-num-seqs` |
| arbitrary docker env | `B70_EXTRA_ENV="K=V ..."` | -- | inject any `-e` (e.g. the Tier F shim flag) |
| push all-reduce | `PUSH_AR`,`PUSH_AR_GRAPH` | 1,1 | keep ON (independent of this campaign) |

Note: there is no single env that injects a full `--compilation-config` JSON into
the recipe; vary capture via CGMODE/IGP/CAPSIZES/SPLITOPS. `--speculative-config`
DOES have the clean `SPEC` raw-JSON hook, which is exactly what Tier E needs.

Tier E exact serve command (assembled by lib.sh):
```
GRAPH=1 \
SPEC='{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  bin/gpu-run bash rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh
```
Verify in the startup log that the TARGET shows `cudagraph_mode=PIECEWISE` AND a
"Capturing CUDA graphs" pass, while the drafter (`eagle_head`) does NOT capture
(no drafter capture pass / drafter dispatch mode NONE).

Tier E fallback (only if the config field is shown to FAIL on a HEALTHY card -- NOT exercised
in Run 1; the field parsed fine and the cell died on a pre-wedged card): chain the
drafter-eager monkeypatch shim (codex sketch in the campaign codex consult; shim file created then),
activated by `B70_EXTRA_ENV="B70_FORCE_MTP_DRAFTER_EAGER=1"`. It patches
`SpecDecodeBaseProposer.initialize_cudagraph_keys` to force `CUDAGraphMode.NONE` for
the drafter only. Source basis: codex read of vLLM 0.23
`vllm/v1/spec_decode/llm_base_proposer.py`.

---

## 8. Pass / fail criteria and execution protocol

Pass criteria (stricter than the failure we are fixing):
- Soak >= **50k** generated tokens (>=2.5x the ~20k crash threshold); prefer
  **100k**. The old crash hit at ~20-28k, so 50k clears it with margin.
- No worker hang (no `sample_tokens` RPC timeout), no `Fatal Python error: Aborted`,
  no `EngineDeadError`.
- No NEO/Level-Zero command-stream error; no `DEVICE_LOST`/`OUT_OF_RESOURCES`.
- MTP acceptance unchanged from the pre-crash baseline (accept_len ~2.5-3.7).
- Post-soak `xpu-health` HEALTHY on both cards.
- THEN a real aider(smoke=5) + swe(smoke=0:3) run completes without engine death.

Execution protocol:
- Always under `bin/gpu-run` (locks both cards for TP=2).
- Pre-flight `bin/xpu-health` before each serve; graceful `docker stop -t30`
  teardown (lib.sh `b70_teardown` -- never `rm -f` a live TP worker, that is the
  wedge trigger).
- **Wedge risk in THIS campaign is LOW**: the MTP+graph crash leaves GPUs HEALTHY
  (proven). The reboot-class wedge comes from killing TP workers mid-collective /
  chained worker-init crashes (a DIFFERENT failure). The lib.sh guards
  (`b70_preflight`/`b70_teardown`/`b70_wait_healthy`) cover that.
- `B70_AUTO_RESET=1` makes a detected wedge auto-recover -- on THIS display-attached
  box that escalates to `sudo reboot` (modprobe -r xe cannot unload the
  console-attached driver). Operator decision: leave OFF (stop + report, default) or
  ON (auto-reboot if wedged). Do not chain TP=2 starts after any teardown that threw
  a DEVICE_LOST in shutdown without a health re-probe.

---

## 9. Results (to fill in as runs complete)

| Date | Tier | config | decode c1 | decode c4 | accept_len | outcome | verdict |
|---|---|---|---:|---:|---:|---|---|
| 06-25 | A | GRAPH=0 enforce-eager MTP3 | 12.78 | 9.47 | 1.18 | STABLE | baseline (shipped fix) |
| 06-25 | repro | GRAPH=1 PIECEWISE MTP3 | 34.89 | 19.09 | 1.13 | runs (perf only) | 2.73x A single-stream; crashes only on long soak |
| 06-25 | E | drafter-eager (spec enforce_eager=true) | -- | -- | -- | INCONCLUSIVE | died at model-load rotary .cos() on an ALREADY-degraded card 0 (err40->err20); the config parsed fine and was NOT exercised. RETRY on a fresh card. |
| 06-25 | B | GRAPH=1 CGMODE=NONE MTP3 | -- | -- | -- | BLOCKED | pre-flight found card 0 WEDGED; never started |

### Run 1 findings (2026-06-25, campaign_120_perf)

1. CONFIRMED PRIZE: captured PIECEWISE+MTP3 single-stream decode = 34.89 t/s vs enforce-eager
   12.78 t/s = **2.73x**. Cross-validates the prior 34.82 (scripts/111) under a different
   (vllm bench serve, random IN=512/OUT=256) methodology. c4 per-stream: 19.09 vs 9.47 = 2.02x.
2. accept_len in the bench (~1.1-1.2) is ARTIFICIALLY LOW because `--dataset-name random` feeds
   gibberish tokens -> MTP draft rarely matches. So most of repro's 2.73x is the TARGET-BODY graph
   capture, not MTP acceptance. Implication: keeping the target captured (Tier E/B) should retain
   most of the win even before MTP acceptance recovers on coherent text. (Re-measure the winner with
   the coherent scripts/111 probe for the true MTP-inclusive number.)
3. NEW BINDING CONSTRAINT (supersedes the "low wedge risk" note in section 8): the cumulative-TP2
   wedge tripped after only ~3 TP=2 serves this session (A ok, repro ok, E-init wedged card 0). Cell E's
   death was a pre-existing card-0 degradation surfacing at the FIRST model-load GPU op
   (`mrope.py:_compute_cos_sin_cache -> freqs.cos()`), err40 OUT_OF_RESOURCES -> err20 DEVICE_LOST --
   NOT the enforce_eager config. So: rapid back-to-back TP=2 serve/teardown is NOT viable here; the
   campaign needs an xe-reset (reboot) between serves OR the cumulative-TP2 wedge root-caused first.
   This wedge now gates ALL TP=2 experimentation and is arguably higher priority than the MTP crash.
   (Note: the MTP+graph crash itself still leaves GPUs HEALTHY; this wedge is the SEPARATE init-time
   failure mode. Two distinct failures, do not conflate.)
4. The lib.sh guard worked: pre-flight xpu-health caught the wedge before cell B and HALTED the sweep
   (no blind chaining onto a wedged box).

---

## 10. Open questions / upstream

- Does `speculative_config.enforce_eager=true` plumb through on vLLM-XPU 0.23 (the
  field exists; XPU path untested)? Tier E answers it; fallback shim ready.
- If the target hybrid spec-state path (not the drafter) is implicated
  (`postprocess_mamba_align_gpu`, GDN UNIFORM_BATCH metadata under graph), that is a
  deeper target-side fix and a candidate upstream report alongside #25368.
- A true full-capture fix (bound/reset the drafter graph command stream across
  replays, Tier F) is the upstream-worthy patch if we want FULL drafter capture.
- FULL_DECODE_ONLY needs an Intel GDN+spec kernel fix
  (`spec_query_start_loc` assert; SYCL-Graph scratch wall).

## 11. References

- Logs: `agentic-eval/results/logs/w8a8_crashlog_30m.txt` (definitive crash trace,
  == `w8a8_crashlog_pushar_graph0.txt`), `w8a8_crashlog_40m.txt` (enforce-eager
  survivor), wrappers `w8a8_{stress_isolate,mtpon_faulthandler,bisect_nomtp,
  bisect_pushar_graph0,fix_eager}.log`, `apc_isolate_apc0.log`.
- Prior art: `FINDINGS.md` 186-201 / 142-160; `JOURNAL.md` 2026-06-25 (4206-4442),
  Bug B 3627-3735, push-ar 4029-4113; `27b_w8a8_research.md` (esp. 246-264);
  `MTP_TODO.md` 6-25 / 73-105 / 129-183.
- Config: `agentic-eval/configs.sh` 26-32 / 80-97; `agentic-eval/serve/serve_config.sh`;
  `rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/{serve.sh,patches/sitecustomize.py}`;
  `rdy_to_serve/_common/lib.sh`.
- vLLM internals (codex): `vllm/v1/spec_decode/llm_base_proposer.py`
  (`SpecDecodeBaseProposer.initialize_cudagraph_keys`, separate
  `cudagraph_dispatcher`), `vllm/v1/worker/gpu_model_runner.py`
  (`_check_and_update_cudagraph_mode`, `postprocess_mamba_align_gpu`).
- Upstream: vLLM #25368, #41530, #41190, #40756, #40880; pytorch #187277.
