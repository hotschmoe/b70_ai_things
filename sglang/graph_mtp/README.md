# graph_mtp -- XPUGraph capture of the W8A8 MTP verify forward (campaign workspace)

Opened 2026-07-02 (perf campaign; JOURNAL entries same date). Goal: capture the TARGET_VERIFY
forward (and later the draft chain) of the sglang W8A8 fused+MTP TP=2 daily driver to break the
launch-bound ~220ms eager iteration (~25 t/s -> target 40+).

## State (end of 2026-07-02 session)

WORKS (first ever captured MTP verify on sglang-XPU):
  GRAPH_BACKEND=breakable + EAGER_COLL=all + decode-attn triton + spec-attention-mode decode
  + --cuda-graph-bs 2 3 4 -> serves, coherent, "cuda graph: True", accept ~5.4.
  BUT SLOW: c1 9.9-10.2 t/s vs 25.06 eager (JIT-independent; rerun with warm caches identical).
  Cause: ~129 collective eager-breaks/forward, each costing a python replay_fn + output copy +
  per-segment sycl-graph submission; segments too small to amortize. Also bs=1 pads to bs=2 (2x
  verify work) because of BUG 5 below. c4/soak crashes the server (untriaged).

BLOCKED fast shapes:
  - FULL capture + recorded oneCCL: captures, first gen coherent, then replay DEADLOCKS both ranks
    (host-staged oneCCL halves never re-execute). RUN 4 watchdog stack in JOURNAL.
  - FULL/breakable + PUSH_AR_GRAPH=1 (K.6 capturable push-AR): ar_allreduce_graph RECORDS but
    XPUGraph.capture_end HANGS finalizing the graph. MINIMAL REPRO: push_ar_record_test.py
    (2 ranks, records on both, capture_end never returns, timeout). The K.6 path was only ever
    proven inside vLLM torch.compile capture (different torch/SYCL runtime than this image).
    -> NEXT SESSION: debug the ext_codeplay native-command + IPC-event-pool interaction with
    torch-2.12 sycl command_graph finalize; scripts/104/106 cpp + P2P_GPU.md K are the sources.

## The five stacked bugs fixed this session (all in sglang/patches/xpu_cudagraph.py + woq_shim.py)

1. weak_ref_tensor NotImplementedError on XPU at the first eager break, MASKED by BCG __exit__
   double-ending the segment -> capture_end on dead sycl graph -> segfault. Fallback module
   (identity refs) pre-seeded in sys.modules. Real fix: adopt sgl-kernel-xpu #251 (from_blob).
2. XPUGraph.capture_begin lacks capture_error_mode kwarg -> _B70XPUGraph adapter (woq_shim).
3. BCG _slice_output/_copy_output_to_buffer don't handle LogitsProcessorOutput -> dataclass recursion.
4. BCG slices by shape_key.size (=bs) but spec outputs have bs*num_tokens_per_bs rows -> 11x
   truncation. Scaled via stashed num_tokens_per_bs.
5. bs=1 (M=11) hits "could not execute a primitive" (oneDNN) under recording at ~segment 101;
   bs 2/3/4 fine. Sidestep: --cuda-graph-bs 2 3 4. TODO bisect the primitive.

Also fixed on the way: eagle_worker_v2 held a stale by-name build_tree_kernel_efficient when
imported before mtp_tree_xpu (woq_shim order + defensive re-bind) -- likely a fragment of the
06-28 "install breaks spec" mystery.

## Isolation microbenches (all PASS -- torch.xpu graph machinery itself is fine on B70)

- ccl_between_segments_test.py: 2-rank oneCCL AR/AG between pooled segment captures + replay.
- triton_segments_test.py: 200 pooled segments each with a triton kernel.
- alloc_between_segments_test.py: allocating eager breaks between pooled segments.
- push_ar_record_test.py: the ONE FAILURE -- ar_allreduce_graph records, capture_end hangs.
- (inline) oneDNN int8_gemm_w8a16 in pooled segments: OK at M=11.

## Files

- ../graph_mtp_verify_ab.sh -- the experiment driver (knobs: GRAPH_BACKEND, EAGER_COLL=all|ag|0,
  EAGER_ATTN, PUSHAR, DRAFT_GRAPH, SPEC_STEPS, MAXBS, BS_LIST, BCG_TRACE, SEGVBT, EXTRA_ARGS).
- src/ -- image source snapshots for study. last_run_*.log -- saved serve logs.
- segv_bt.c/.so -- LD_PRELOAD SIGSEGV native-backtrace printer (how BUG 1 was found).
- NOTES_codex.md -- codex hazard notes on the runner/eagle paths.

## SOLVED LATE-SESSION: the capturable AR (spin-kernel sync)

Bisect (118b_push_ar_graph_bisect.cpp): ext_codeplay_enqueue_native_command -- even EMPTY -- breaks
XPUGraph capture_end finalize on DPC++ 2025.3/torch 2.12; the IPC posted-write kernel itself records
fine. MODE 4 = pure-SYCL spin sync (device seq counters + system-scope atomics on scratch-tail flag
pages): records + replays 5/5 correct in the 2-rank microbench. Wired into push_ar_xpu.py
(ar_allreduce_graph_spin preferred; all_gather emulated as zero-padded push-AR).
RUN 23 (FULL backend + spin collectives): capture COMPLETES (the 3x-hung config) but the first live
verify REPLAY hangs after prefill.

## Next-session priorities

1. Fix the run-23 replay hang: (a) per-AR-NODE flag slots instead of the single global seq pair
   (robust against any per-rank replay-count asymmetry, e.g. bucket-init replays), (b) add hang
   diagnostics to graph_mtp_verify_ab.sh failure path (py-spy both schedulers + GPU busy snapshot
   before teardown). If fixed -> FULL capture with ALL collectives in-graph = the real 2x+ shape.
2. bs=1 oneDNN primitive bisect (removes the 2x padding tax for single-stream).
3. c4/soak crash triage (breakable shape) -- lower priority if FULL shape lands.
4. Upstream: file the BCG bugs (upstream_issues.md) + the ext_codeplay-native-cmd finalize bug
   (clean repro: push_ar_record_test.py MODE 2); adopt sgl-kernel-xpu #251.
