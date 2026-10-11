Work in /mnt/vm_8tb/github/b70_ai_things as hotschmoe. Read AGENTS.md and
 docs/20261008_flashnext_udq4xl_campaign.md first. Preserve unrelated dirty
 changes. Continue the actual FlashNext GPU campaign; prioritize running the
 model, usable HTTP serving, and measured prefill/decode latency. Avoid adding
 another validation framework or repeating historical test sweeps.

Performance targets from the user: at least2000 tok/s prefill and50 tok/s
 decode on the selected artifact. Treat current rates as far below target.
 Compare actual source/recipes in Steve, Sergio and Strata; prefer experiments
 that address batching, latency and transfer overlap, rather than more wrappers.

Latest continuation state (2026-10-11):
- Balanced source43 24/24 producer and independent public checks PASSED.
  Short54/four-output warm diagnostic: PP936.8ms, decode245.9ms (16.3tok/s).
  Only18185/24576 experts resident; this is not sustained speed/fullmath.
- Source45 grouped-F16 built all8 and passed six isolated GPU operator cases.
  Real model OFF passed, but ON completed0 requests and exited139 after Xe
  null-address VM faults/CCS reset and libsycl segfault. Posthealth/full4 passed;
  kernel-fault gate FAILED. Do NOT rerun unchanged grouped45 ON.
  Failed root: F17/source45-grouped-f16-model-v1-run-v1.
- Source47 corrects the ordinary-heap metadata allocation concern with host
  USM and shared ownership/context-only cleanup. Cause remains unconfirmed.
  Its fresh all8 build PASSED (304s), all72/all8 rootverified; SDK is
  /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T223956Z-0b555kgw,
  receipt31a2227f. Larger-ne single/twoqueue GPU fixtures PASSED, each six
  cases with exact outputs/owned retirement/posthealth/no kernel faults.
  Corrected47 OFF completed twoGEN4 requests: warm524 prefill15482.3ms
  (33.8tok/s), decode366.0ms (10.9tok/s). ON failed after PP320of524
  with UR_RESULT_ERROR_DEVICE_LOST, Xe null/low-address faults at23:09:41
  and engine exit139 at23:10:50. Its container is removed; per-arm strict
  and compiled pair posthealth passed. Final actor closing gates still live
  when updated. Inspect actual50457 and report before proceeding.
  Do not retry unchanged47 ON or infer model reliability from fixtures.
  Proposal: strata/flash-next/native47-grouped-f16-usm-engine-build-proposal-v1.json.
  Fixture evidence is in F17/grouped-f16-operator47-singleq-v2 and
  grouped-f16-operator47-twoq-v2. Source46 HC rowbatch build is queued
  under94364 with indefinite lease wait; previous28224 timed out124
  BEFORE compilation. Do not duplicate94364. GroupedF16 muststay0
  on46/48. Source48 default-off singleton mirror staging is committed;
  kernels/mirror_t1_staging_harness_v1.cpp and .txt cover bounded routing
  and copy, not math/speed. Build proposal native48-mirror-t1-staging-engine-build-proposal-v1.json.
- Independent original first HC computation PASSED with fresh identity close.
  Original packet differs one scale byte; allcodes/sumbytes match. Conditional
  independent encoder of native mixed values matches nativepacket bitwise.
  This is a first arithmetic seam, not full48 qualification. CPUbulk row-FMA
  helper was built/tested and preserved at /mnt/vm_8tb/b70/build/hc-rows-dot32-cpu-v1.
- Existing NO_HOST/device-plan mirror reads feed mapped RAM directly to GEMV.
  The existing ptr2 staging path is bypassed by device-planned mirror groups;
  --pcie-mode kernel alone cannot fix this. Source-only audit is in agent/root
  history; inspect verify.cpp and resident_plan_kernel before patching routing.
  Do not equate logical blob bytes with physical PCIe traffic.

Hardware/model: two Intel B70s, 64 GiB total VRAM, approximately128 GiB host
 RAM; fixed kernel7.1.0-070100, Compute Runtime26.22. Exact Unsloth
 Qwen3.8-Flash-Next UD-Q4_K_XL across four GGUF shards. Runtime root:
 /mnt/vm_8tb/b70. Every real GPU touch MUST use bin/gpu-run. Root coordinates
 GPU leases; never overlap GPU experiments. Use OMP_NUM_THREADS=1,
 MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1.

Current concrete action: the genuine source43 mode1 API baseline was launched
 through bin/gpu-run on2026-10-11, using this prepared directory:
 /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-api-mode1-baseline-v3-prepared-v1
 The producer completed successfully in522 seconds: all eight HTTP requests
 coherent, owned teardown clean, pre/post health and full4 checks passed.
 Report SHA256: c894ba56fd719b84b447f51de666390e28b7ea9e528d662bb4f4a17c2affbae9.
 No live baseline server remains; do not mistake this for shelf qualification.
 Original public reader V3 exited1 with KeyError: WorkingDir. New V4 reader
 preserves original source and normalizes only missing image WorkingDir to
 empty, matching observed terminal Config. V4 public check PASSED, including
 fresh closing full4, reusing saved GPU evidence without another model run.
 Public receipt is source43-api-mode1-baseline-v4-root-public-binding-v1.json.
 Use validate_source43_api_baseline_v4.py; preserved V3 remains failed. The exact actor is
 strata/flash-next/run_source43_api_baseline_v3.py; port18083, primary served
 name hotschmoe-dd, GPU0, ctx2048/prefill64/PLE65536, MTP0, cacheOFF,
 batch0, parallel1. Canonical registry now has174 entries, including the exact
 source43 alias. The actor exercises six serial coherence requests and two
 overlapping HTTP requests, then owns teardown and post-health. HTTP overlap
 is FIFO serving, not parallel GPU compute. If terminal, run the corresponding
 validate_source43_api_baseline_v3.py with a fresh output path; inspect errors
 first rather than rerunning an expensive actor blindly.

A real warm source43 GPU profiling producer passed. Results root:
 /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-warm-pti-short-v5-ON-run-v1
 Its Chrome trace contains98621 GPU operation rows/115 kernel names. Public
 closing validation subsequently caught a SINGLE flipped bit in one cached
 shard3 page. Full O_DIRECT SHA matched the publisher; buffered SHA differed
 on exactly4096 bytes at offset10933211136 (one byte changed at+2796).
 Only that page was discarded using POSIX_FADV_DONTNEED; buffered reread now
 matches direct. Evidence: sibling warm5-public-page-recurrence-v1, especially
 full-direct-buffered-comparison.json and page-specific-recovery.json. No
 model file was written and no global cache flush was performed. Cause remains
 unknown; relevant kernel logs were absent, EDAC counters unavailable. Keep
 current identity and health checks; do not call this resolved hardware.

Next priorities:
1. Inspect the completed API baseline/public reader result. Warm repeated
   requests logged13.4-13.9 tok/s prefill and9.8-10.7 tok/s generation
   on23/51-token prompts and5/9 generated tokens. These tiny-output log
   observations are not a sustained decode benchmark. Keep hotschmoe-dd first.
2. Measure actual uninstrumented prefill and decode with fixed model/config,
   emitted-token counts, warmup, TTFT and token gaps. Avoid converting profiler
   durations or kernel sums into production tok/s. Compare mode/storage paths
   deliberately, not across unmatched configurations.
3. Short-read0 is ALREADY tested. Source43 shortread0V2 warm54 prefill
   fell4297.0->1516.2ms; decode unchanged374.3->374.6ms. Do not repeat
   this sweep. Long524 is already batched. Decode lanes32 gained only
   about1.9percent; no extra lane sweep. Prior chunk64->512 long524
   improved33.6->85.7inputtok/s with changed output, not promotion.
   Next substantial costs are grouped batching (currently faults), quant
   native kernels, HC row batching, and mapped-host expert staging.
4. Compare genuine two-GPU execution. Existing source43 paired CLI control
   passed bitwise equality against single GPU (four head/192 residual arrays).
   Strata currently uses serial layer split32/16 and a51216-byte/token pinned
   host bridge, not custom push-allreduce or direct P2P. No full-model push arm:
   loaded push graph still stalls at first submission.
5. Profile host gaps/submissions, expert streaming and transfer overlap as well
   as kernels. Observed large kernel-duration groups include native_gu_port
   and native_down_port; HC projection/norm also have many small launches.
   Duration sums are not additive critical-path or physical bandwidth evidence.
   A source-only default-off HC down+inject fusion prototype is in progress;
   inspect agents/files before duplicating work. No new SDK observer just to
   collect more fields. Test promising default-off candidates with focused
   numerical checks before full-model comparison.

Limits: full original-model numerical qualification remains open (full48 head
 NMSE about0.007888); successful lifecycle/bitwise observer controls do not
 prove original-model fidelity. Full-state prefix cache, persisted restore,
 true1/2/4/bounded6-stream usability and shelf qualification remain unfinished.
 Do not promote a shelf or claim speed/stability without matched identity,
 coherence, health and teardown. Research serving and honest timings can
 proceed with these limits explicit. Keep CPU synthetic spoofing clearly
 labeled; do not pass synthetic bandwidth as real model throughput.

Record new evidence CONFIG -> COMMAND -> RESULT -> VERDICT; byte-append ASCII
 to JOURNAL.md (it contains historical Unicode). Commit/push coherent changes
 without staging unrelated JOURNAL additions, serving edits or prebuilt libs.
 Do not restore archive binaries, change the host kernel or reboot casually.
 The user authorized focused GPU experiments and wants fewer frivolous tests,
 not another long harness-development phase. Deliver actual measurements.
