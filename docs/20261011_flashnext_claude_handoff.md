Work in /mnt/vm_8tb/github/b70_ai_things as hotschmoe. Read AGENTS.md and
 docs/20261008_flashnext_udq4xl_campaign.md first. Preserve unrelated dirty
 changes. Continue the actual FlashNext GPU campaign; prioritize running the
 model, usable HTTP serving, and measured prefill/decode latency. Avoid adding
 another validation framework or repeating historical test sweeps.

Hardware/model: two Intel B70s, 64 GiB total VRAM, approximately128 GiB host
 RAM; fixed kernel7.1.0-070100, Compute Runtime26.22. Exact Unsloth
 Qwen3.8-Flash-Next UD-Q4_K_XL across four GGUF shards. Runtime root:
 /mnt/vm_8tb/b70. Every real GPU touch MUST use bin/gpu-run. Root coordinates
 GPU leases; never overlap GPU experiments. Use OMP_NUM_THREADS=1,
 MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1.

Current concrete action: the genuine source43 mode1 API baseline was launched
 through bin/gpu-run on2026-10-11, using this prepared directory:
 /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-api-mode1-baseline-v3-prepared-v1
 Inspect its report.json, supervisor.log and owned processes before doing
 anything. Do not duplicate the launch or assume it passed. The exact actor is
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
1. Finish the active API baseline and report actual coherent responses, timings
   and any failures. Keep hotschmoe-dd first in /v1/models.
2. Measure actual uninstrumented prefill and decode with fixed model/config,
   emitted-token counts, warmup, TTFT and token gaps. Avoid converting profiler
   durations or kernel sums into production tok/s. Compare mode/storage paths
   deliberately, not across unmatched configurations.
3. Test --short-read0 against default64 for the54-token short prompt: default
   runs53 prefill rows as26 T2 windows plus one T1; short-read0 forces batched
   Prefill::run. The524-token long prompt ALREADY uses batched prefill, so this
   change cannot explain poor long-prefill speed. Existing focused control:
   strata/flash-next/run_source43_shortread0_v1.py and
   docs/20261011_flashnext_source43_shortread0_control_v1.md. Its frozen
   historical registry173 association needs a minimal deliberate update for
   current174, or use a direct controlled serving experiment. Do not build
   another wrapper chain to do a one-flag comparison.
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
