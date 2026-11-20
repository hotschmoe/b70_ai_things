# Orchestrator notes for Codex (check at each stage boundary)

- 2026-11-20 02:15 UTC: campaign started. No changes yet.
- 2026-11-20 02:25 UTC: analyst report analyst/opus_sched_prefill_restore_walk.md is in (claude-opus-5-5). Read it.
  Top finding (orchestrator confirmed the source): the DD port maps old STRATA_NIGHT_ATTN_BATCH=1 to upstream
  STRATA_BATCH_ATTN_ROWS=1, but upstream disables the batched-rows path when any row's QSA kv_mode==1 (streamed KV),
  engine/sycl/src/core/verify.cpp ~1541-1547, and the DD always streams (--kv-resident 65536). Old night batched
  attention WITH streamed KV (src-ep verify.cpp:1372-1420). feature_map.md:73 is wrong on this point. Likely the
  bulk of the C2/C4 regression vs old night (window +3.2 ms at C2, +6.0 ms at C4 in the logs).
  Direction: in Stage B make sure the B2/B3 traces count QSA attention launches per window and per row so this is
  confirmed or killed. Make it the FIRST Stage C item: carry the old night streamed-KV batched attention (default-off,
  e.g. STRATA_BATCH_ATTN_ROWS_STREAMED=1 or similar), parity vs per-row path, then C2/C4 A/B. Also note the report's
  findings on cold-C4 prefill serialization (Python admission lock + single prompt session; CUDA-only BCO /
  STRATA_SLOT_READ / STRATA_PIPED_CONCURRENT) and the forced-restore TTFT breakdown (sync park ~0.44 s, Python ~0.53 s,
  tail read with 0% KV in VRAM ~0.2 s) - verify those in B4/B5.
- 2026-11-20 02:45 UTC: analyst/fable_decode_window_walk.md is in (claude-fable-5-2). Read it. It independently confirms
  the streamed-KV attention batching gap (96/228/492 extra launches per window at S=1/2/4). Additional items to
  verify in Stage B and fold into the Stage C ranking:
  a. Inert DD switches in this config: STRATA_GDN_COMMIT_LAYERS needs exactly 6 rows (verify.cpp:3727-3733) so
     commit runs ~110 launches per slot; PAIR_GRAPH needs depth 2; NIGHT_GDN_GRF has no 256-GRF batch-groups form;
     MTP_PARALLEL_ARGMAX only applies to the full-vocab head. Confirm with trace launch counts.
  b. Admission regression: park-from-slot 1478/1577 ms on 0216 vs 44/40 ms on old night, and draft_first 21-29 ms
     vs 0.7-3.7 ms. Cause unknown - check whether the old PARK_MOVE / direct slot copy carry is actually active
     (log markers) and include it in B5.
  c. Expert union: export the real distinct-experts-per-layer count at R=2..8 and how often groups of 3 or 5-7 rows
     re-read weights (kernels take 4/2/1 rows at a time).
  d. Cheap config-only arm early in Stage C: STRATA_BATCH_MTP_DEPTH=2 and =3 at C2/C4 (fill the window; the analyst
     estimates +8-25% per stream at C2), plus a lone slot in batch mode (88 vs 134.6 tok/s).
  e. The analysts DISAGREE on card balance: Opus ranks shared expert on card 1 as the best balance option; Fable
     argues card 1's chain already gates card 0 so it will not shorten the layer. Settle it from per-layer ep_wait
     direction (which card waits on which, per layer) in B1-B3 before building either.
- 2026-11-20 03:30 UTC (user direction): concurrency targets extend to C8 and ideally C16, with the hard gate that C1
  single-stream must NOT regress. After B5, add scenarios B7 = C8 and B8 = C16 (if it loads) using a copied config
  with --parallel 8 / 16 (keep 200K max-context and RAM-backed KV; lower --kv-resident or parking only if load
  requires it and record it). For C2/C4/C8/C16 record latency, not just throughput: TTFT p50/p95, inter-token
  latency p50/p95 per stream, window time distribution, queueing/admission time per request, and aggregate tok/s.
  Report what limits C8/C16 (slots, KV VRAM, expert union reads, per-slot serialized work, host). Every Stage C
  candidate is gated on C1 within noise of the Stage-A winner.
- Stage B capture lesson noted: keep STRATA_WATCHDOG_S raised only in diagnostic copies; split multi-scenario captures
  into one capture per scenario so a flush cannot kill the run.
- 2026-11-20 04:15 UTC: FYI a separate TensorFold/Qwen3.8-27B-pi-adapted probe is being prepared (CPU only now: Zig build,
  LoRA merge, RTN quantization under bin/cpu-run). Later it will need GPU time; the orchestrator will schedule that at one of
  your stage boundaries and tell you here first. Expect cpu-run lease waits. Your campaign priorities are unchanged.
- 2026-11-20 04:15 UTC GPU worker: no GPU workload is running. The Stage C candidate release build holds cpu-run; after it releases I plan the remaining card-0 parity retest (about 2 minutes), then matched C1/C2/C4 serves on cards 0+1 (about 5 minutes each for original, switch-off, switch-on), C8 if qualified, then depth arms. I will report boundary/lease availability here; do not start another GPU workload while a pair-leased serve is active.
- 2026-11-20 04:21 UTC GPU worker: candidate binary built (SHA256 1914c6f0...); the TensorFold CPU build acquired cpu-run before the queued checked-in parity rebuild. Both GPUs are free. I am waiting for that heavy CPU phase rather than starting a memory-intensive server concurrently; matched windows require the quiet CPU lease.
- 2026-11-20 04:25 UTC GPU worker: checked-in parity build is already queued for cpu-run; the preceding synthetic parity passed and the candidate binary is ready. To avoid leaving a model server loaded while waiting for a large CPU/RAM job, please sequence subsequent TensorFold heavy prep between my matched serve arms where practical. Each serve has about 1 minute startup plus 3-4 minutes under cpu-run --quiet; pair GPU lease remains coordinated.
- 2026-11-20 04:35 UTC GPU worker: checked-in parity build completed. Running the two small parity tests pinned to card 0 under gpu-run for about 2 minutes; card 1 remains free. TensorFold LoRA merge now holds the CPU lease, so no serving/timed window will overlap it.
- 2026-11-20 04:36 UTC GPU worker: checked-in parity retest passed and card 0 is free. The TensorFold LoRA merge is holding cpu-run; both GPUs remain free until it finishes and the paired Stage C serve begins.
- 2026-11-20 04:44 UTC GPU worker: the LoRA merge and copy completed; the four-variant quantization job immediately acquired cpu-run. Both cards remain free. Candidate parity passed; C1/C2/C4 comparisons still await an uncontended CPU window.
- 2026-11-20 05:05 UTC GPU worker: CPU lease is free; starting original-binary Stage C control on cards 0+1 under pair lease, expected about 5 minutes. Timed clients will take cpu-run --quiet.
- 2026-11-20 05:10 UTC ORCHESTRATOR: GPU YIELD REQUEST. TensorFold prep is finished (no more heavy CPU from it for now). After you
  complete the current attention-rows Stage C A/B (original / switch-off / switch-on, C1/C2/C4) and its post-health, please
  yield both GPUs and the CPU lease for the TensorFold GPU probe (~60-90 min): append a line "GPU YIELDED <time>" here,
  then poll this file every 2 minutes (sleep loop) until a line "GPU RESUME" appears, then continue with C8/C16 and the
  depth arms. While waiting you may do CPU-light analysis (ledger, parsers, source reading) but no gpu-run and no
  heavy cpu-run. If you are mid-arm, finish the arm first.
- 2026-11-20 05:12 UTC GPU worker: original control completed with clean teardown; starting candidate switch-OFF server on cards 0+1, expected 5-7 minutes, with quiet CPU client windows.
- 2026-11-20 05:19 UTC GPU worker: candidate switch-OFF arm passed coherence and exited cleanly. Starting candidate switch-ON on cards 0+1 under pair lease, expected 5-7 minutes; quiet CPU clients.
- 2026-11-20 05:26 UTC GPU worker: attention ON clients passed and server stopped cleanly. Running both-card post-health now, expected under 2 minutes; will release both GPUs and append GPU YIELDED immediately afterward. C4 outputs differ OFF/ON, requiring follow-up before promotion.
GPU YIELDED 2026-11-20 05:26:03 UTC
GPU RESUME 2026-11-20 06:10:20 UTC (TensorFold probe done)
