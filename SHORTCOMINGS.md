# SHORTCOMINGS.md -- open blockers + dead ends

Where the README keeps its headlines, this is where the honest negatives live: what is
blocked, what we tried that did not work, and the upstream bugs we are waiting on. Keeps the
README clean. Positive optimization lessons live in `research/LESSONS.md`; raw measurements in
`FINDINGS.md`; the chronological log in `JOURNAL.md`.

## Open blockers

- **vLLM "!!!!" garbage under concurrency -- vLLM is PAUSED.** Any vLLM serve that batches a
  prefill alongside an in-flight decode eventually emits endless `!!!!` and then poisons a
  shared KV/SSM block so even later single requests return NaN (a `docker restart` clears it).
  Root-caused to the XPU GDN/Mamba recurrent path: a NaN in the GDN forward propagates through
  attention's multiply-by-zero (`0*NaN=NaN`). The CUDA-side fixes are already in vLLM 0.23.0
  (#35219 zero-freed-SSM-blocks, #43961 MLA+linear, #44700 route-to-recurrent); the remaining
  XPU-side defect is **vllm-xpu-kernels #172** ("support fp32 ssm_state in chunk_fwd_o", NOT
  DONE). Tracking issue for our exact box+symptom: **vLLM #38994** (Qwen-3.5 garbled output,
  Intel backend, 2x Arc B70, open since 2026-04). No local fix on 0.23.0 -> we serve on sglang,
  which does not co-batch prefill+decode this way. Resume vLLM when #172 lands.
  (Evidence: JOURNAL 2026-06-27 research+A/B; mechanism verified upstream.)
  NUANCE (2026-06-29, the 35B W8A8 *MoE*): the int8 MoE entry is MUCH more robust than the dense GDN
  models -- a mixed-prompt soak was perfectly CLEAN at c4/720s (228 req) and showed only ONE rare,
  non-spreading, self-healing transient at c8 (1/416 = 0.24%; the server stayed coherent, no restart).
  vLLM 0.23's zero-freed-SSM-block fix holds on XPU for this MoE. So the MoE entry is closer to usable
  (with output validation) than the paused dense path; still gated behind the general pause + slow.
- **sglang W8A8 MoE (35b-a3b) -- UNBLOCKED 2026-06-29 (was wrong: it is a loader port, not a kernel
  build).** The old claim "sglang/XPU has no int8 fused-MoE kernel" was FALSE: sglang ships the
  pure-Triton `fused_moe` `use_int8_w8a8` path AND a `W8A8Int8MoEMethod` in-tree. It now SERVES
  coherently (`rdy_to_serve/sglang/qwen36-35b-a3b-w8a8`): the 256 experts run true int8, dense
  linears dequant->bf16. The only real fix was a 1-line shim (the stock per-token int8 act-quant uses
  `tl.extra.cuda.libdevice.round`, which does NOT link on triton-xpu) plus loader plumbing for Quark's
  1-D weight scales. REMAINING LIMITATION (not a blocker): decode is eager-slow (~8 t/s) -- graph
  capture / NEXTN MTP / fused int8 dense linears are the open decode levers (same follow-ups the 27B
  W8A8 path took). See `research/w8a8/SGLANG_MOE_PLAN.md` + JOURNAL 2026-06-29.
- **TP=2 BCS/oneCCL hardware wedge (reboot-only).** A TP>1 worker-init crash (or `P2P_ACCESS=1`)
  can corrupt the cross-GPU oneCCL/Level-Zero state so every later TP=2 serve `DEVICE_LOST`s at
  warmup -- and sometimes degrades even single-card ops. Recovery on THIS display-attached box
  is a full reboot (`modprobe -r xe` fails: `xe` drives the framebuffer). Guarded by
  `bin/xpu-health` (pre-flight probe) + `bin/xe-reset`; never chain crash-prone TP=2 starts.
  See `docs/P2P_GPU.md` and the GPU Discipline section of `AGENTS.md`.
- **W4A16 / W4A8 vision+MTP graft is UNVERIFIED on-GPU.** `models/graft_w4_complete.sh` rebuilt
  them mechanically (complete checkpoints), but they have not been coherence-gated yet. The
  W4A16 case in particular got a spliced multimodal config (its quant was authored text-only) --
  serve + a few prompts + one image before trusting. See `models/REFACTOR.md`.

- **[FOLLOW-UP, opened 2026-07-03] DFlash TP-worker crash blocks it as the daily driver.** DFlash
  speculative decoding (vLLM in-tree, `DFLASH=1` in `rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh`) is
  a validated *performance* win but NOT production-stable on this TP=2 hybrid, so the DD stays MTP.
  Two instability modes seen (JOURNAL 2026-07-03 session 3):
  1. **Hard crash under concurrent load (the blocker).** ~4 min into a live all-sliding-drafter DD, a
     random-text bench (c4) coincident with a WebUI request killed a TP worker ->
     `shm_broadcast.py acquire_read RuntimeError: cancelled` -> `EngineDeadError` -> graceful engine
     shutdown (Exit 0, no wedge, cards stayed healthy). SAME `shm_broadcast cancelled` signature as the
     2026-07-03 DFlash spec=7 spike crash at init. The dying worker's OWN traceback (upstream of the
     shm-cancelled read) is the thing to capture.
  2. **Soft "!!!!" GDN poison** on a request CANCELLED mid deep-prefill (40K-ctx); `docker restart` /
     `bin/dd-watchdog` heals it. Same GDN/mamba SSM-state family as the paused-vLLM "!!!!" blocker above.
  ROOT-CAUSE PLAN (dedicated session; DD is MTP meanwhile, DFlash is one flag away):
  (a) reproduce deterministically = concurrent `vllm bench serve --dataset-name random` c4 + a cancelled
      request, `B70_DEBUG=1` (faulthandler) or `=2` (L0 validation/leak + UR/CCL/vLLM debug) to dump the
      worker py+C traceback on the fatal signal; (b) suspect the drafter's `precompute_and_store_context_kv`
      single-slot-mapping interacting with chunked-prefill + the wide spec-verify batch (1+DFTOK) under
      concurrency, and the all-sliding drafter's windowed KV cache eviction mid-flight; (c) test DFTOK
      smaller (narrower verify), `--no-enable-chunked-prefill`, and DFSWA=0 vs 1 to localize; (d) the fix is
      likely a guard around the cancelled-request path in the DFlash proposer / a fp32/zeroing of the drafter
      SSM state. Refs: JOURNAL 2026-07-03 session 3, `vllm/DFLASH_XPU.md`, extracted source in the spike memo.
  Accept/decode research that STANDS (do not re-litigate): all-sliding drafter holds accept 3.6@100K + fits
  full 253952 + beats MTP decode at 40K (21.3 vs 17.7); stock full drafter COLLAPSES at depth (4.5->1.6).

## Dead ends (tried, measured, did not work)

- **XPUGraph multibucket capture for serving.** Custom XPU cuda-graph decode capture is great
  single-stream (int4 bs1/maxreq1 = 23.5 t/s) but COLLAPSES under concurrency: multibucket-nopad
  4.67 t/s, multibucket-pad 7.36, maxreq4 -> 3.55, c4 -> 0.75. Graph capture is a single-stream
  driver only; for concurrent serving, prefer MTP-eager. (scripts/141, 144; `research/LESSONS.md` row 6.)
- **graph + MTP stacked.** The captured speculative-decode forward HANGS under XPU graph capture
  -- the two cannot be combined today. (JOURNAL 2026-06-26/27; scripts 134-144.)
- **sglang's native `--enable-cuda-graph` flag on XPU.** No-op: graph_off 9.54 vs graph_on 9.46
  vs graph_on_si8 9.36 -- it stays eager. Our win came from the custom XPUGraph path, not the flag.
- **"Cheap" MTP flags (continuous steps, overlap schedule).** No single-stream uplift on int4
  (15.30 baseline -> 15.30-15.37); overlap-on actually hurt aggregate + TTFT. (scripts/134.)
- **TP=2 GRAPH=1 for W8A8 decode.** A ceiling, not a lever: decode is all-reduce-bound at TP=2
  (+5% only, prefill/TTFT regress). Eager + MTP is the TP=2 config. (JOURNAL 2026-06-27.)
- **P2P (`CCL_TOPO_P2P_ACCESS=1`) inside a vLLM TP>1 serve.** DANGER -- wedges the multi-GPU
  state box-wide (see Open blockers). Refused unless `I_KNOW_P2P_WEDGES=1`. (`docs/P2P_GPU.md`.)
- **vLLM PP=2 (pipeline parallel) for the 35B MoE.** PP=2 + vLLM's default async scheduling CRASHES on
  the first request (`KeyError req_id_to_index` in `scheduler.update_from_output`, the V1 PP
  `step_with_batch_queue` path). `--no-async-scheduling` does NOT fix it at `GRAPH=1` (same KeyError +
  capture). PP=2 ONLY serves at EAGER + `--no-async-scheduling` (c1 6.92 decode -- +44% over TP=2 eager
  4.80, confirming the all-reduce is a real eager tax) but eager is ~6x slower than TP=2 GRAPH=1 (41.8),
  which PP forfeits the capture lever to. Net: TP=2 GRAPH=1 is the only competitive multi-card vLLM
  config for this MoE. (JOURNAL 2026-06-29; `EXTRA_ARGS` knob in `_common/lib.sh` for the repro.)
- **vLLM W4A8 (27B, int8g) at `GRAPH=1`.** Engine init OOMs: the PIECEWISE capture buffers leave
  only 0.32 GiB for KV (needs 0.66 GiB for max-len 8192; est. usable max len 2496). So the W4A8
  vLLM entry has no working captured config -- EAGER (~6 t/s) is its only path. Raise
  `gpu-memory-utilization` or drop `max-model-len` to unblock. (2026-06-29 best-to-best bench.)

## Bench methodology note

`bin/serve-sweep --bench` must run each shelf entry at ITS OWN config -- it now honors each
entry's `GRAPH` default instead of forcing `GRAPH=0`. The old eager-forcing default undercounted
every graph-capture config ~4x (e.g. vLLM 27b-int4 8.4 -> 28.6 t/s, sglang W4A8 10.3 -> 27.3 t/s).
`--smoke` still forces eager (it only checks boot+coherence, where speed does not matter).
