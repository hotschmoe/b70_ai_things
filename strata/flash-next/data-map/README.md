# Strata Flash-Next data map

Open `index.html` in any browser (file://, offline, no network). It shows
where data lives and moves for the Flash-Next IQ3_XXS Strata EP serving
path, which operations are expensive or wait, and how strong the evidence
is for each claim. `operation-data.json` is the source of truth;
`index.html` embeds a copy of it.

Snapshot: 2026-10-15, S30 (S22 operation trace plus readiness, real-head,
production PLE, EP publication fix/native ISA, paired-head serving,
draft-counter carry correction and full winning-candidate RAM recovery).
Runtime source:
`/mnt/vm_8tb/b70/build/strata-ab/src-ep` (upstream Strata 296b59d2 base
plus working-tree overlays for EP/PUSH/accounting/S19-S21 and the
default-OFF S26 head-pair prototype and S28 explicit EP barrier scope).
S30 also updates the Python optional draft-counter carry helpers.
Timings reference preserved binaries, not
the changing source directory. Evidence:
`docs/20261014_flashnext_strata_optimization.md` S10 and S20-S22, plus raw
summaries under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.

## The data path in one pass

1. Request: HTTP (serve/server.py) -> chat template -> tokenize -> native
   admission. Admission picks one of two 200000-token slots and checks
   resident slots and the 32 GiB / 4-entry RAM parking cache.
2. Cache: a parked conversation is restored into the slot
   (conversation_kv_restore; S20: about 420-430 ms per full restore at 200k,
   3.17 GB API payload, not physical PCIe bytes). A miss means prefill.
3. Prefill (S10 evidence, older binary): chunked sparse attention + GDN on
   GPU0, expert dequant + GEMM on both cards, PUSH FP32 reduction from GPU1
   posted into GPU0 memory.
4. Decode, one batch window: host packs 6 rows (2 slots x seed + 2 proposals
   at depth 2) -> 48 layers of [hyper-connections (hc 4, low rank 320) ->
   QSA attention (12 layers) or GDN (36 layers) -> dense/shared mmvq ->
   Q8_1 quant -> router top-10 of 512 -> EP push to GPU1 / local experts on
   GPU0 -> GPU1 experts -> rows returned -> combine] -> six-row Q5_K
   full-vocab head (N 2560, V 248320) -> verify/accept -> commit -> four
   serial one-row draft heads -> emit.
5. Park: when a slot is overwritten, the completed slot is saved to RAM.

That decode sequence describes the S22 trace. S29's opt-in variant pairs
the two same-depth draft heads (source expects two C2 calls per full-depth2
cycle); its kernel-by-kernel serving trace is still queued.

The EP link is Level Zero IPC-mapped device memory with kernel-side flag
counters (sycl/include/strata/core/ep_mem.hpp), not a collective.

## Reading the numbers

- Status: measured / inferred / source-only / unknown / parked / rejected.
  Every box and quote carries one.
- Trace numbers come from the S22 control unitrace: 8 interior replay
  cycles, BOOTTIME minus MONOTONIC_RAW shift. Per-cycle values are
  summed service times / 8. They overlap and are NOT a latency budget.
  ep_wait time is spin, not work.
- Tracing slows the cycle (interior median 59.17 ms vs clean native 49.12-
  49.27 ms; the cold average of 130 ms is misleading). Use the clean
  benchmark table for performance claims.
- Logical bytes are arithmetic from shapes and block sizes, labelled as
  estimates. No physical PCIe bytes or bandwidth saturation are claimed.
- Production CPU-path costs are not estimated; standalone synthetic preparation
  is labeled separately. See [readiness probes](../lowlevel_probe/README.md).

## Maintenance

All commands run from this directory with system python3, no dependencies.

    python3 -I build_map.py --import-ops   # re-read S22 steady summaries -> operations[], stage_rollups
    python3 -I build_map.py                # re-embed operation-data.json into index.html
    python3 -I build_map.py --check        # exit 1 if embed stale, JSON broken, or non-ASCII

- Edit stages, gaps, hypotheses and benchmarks by hand in
  operation-data.json, then re-embed.
- Kernel-to-stage assignment lives in RULES in build_map.py (first regex
  match wins). Unmatched kernels land in `unattributed`; assign them from
  source rather than guessing.
- For a new trace, point CONTROL/ASYNC at the new summarize_trace.py
  output (never the raw 300 MB trace) and update CYCLES and the cycle facts.
- Keep files ASCII-only.

Validation: embedded JSON/ASCII checks and a CPU-only DOM run exercised all
40 detail views, 5 phase filters, text filtering and 13 sortable columns.
Browser layout has not been visually inspected. PLE is enabled on NVMe through the resolved second GGUF symlink; its
row-cache/O_DIRECT pread worker path is mapped, while joined request costs
remain unmeasured. The primary IQ3_XXS GGUF is on SATA. Source paths marked
project-repository refer to this B70 repo; base/overlay paths refer to Strata.

S25 adds the actual-weight C1/2/3/4/6/8 head curve and bounded production
PLE miss-ticket waits. These are separate scopes from the S22 operation
rollups; source-only C2 batching remains a hypothesis until serving tests.

S26 records an unresolved standalone EP flag-mode freshness failure and
a CPU-built, GPU-untested head-pair prototype. The reference trace remains
the frozen S21 binary. The live src-ep source now has the optional S26
prototype; reference source snapshots/provenance are recorded in the JSON.
Both-card/compiled-collective post-health passes. Failed-arm timing data
does not establish a serving gain or serving-output corruption.

S27 observes the intended wait epoch in debug failures: first receiver
copy consumes current poison while immediate second/third copies pass.
Native/debug arms and post-health are recorded separately. This narrows
the question without proving its cause. S28 corrects the S27 claim that
native BMG ISA was unavailable: hidden .text.*.asm files were present.

S28's same-binary default mirror fails112/648 first-read poison words;
explicit system scope on the producer workgroup barrier passes65/65.
The actual patched EP translation unit with native wait/no extra reads
also passes65/65. Decoded preflag UGM fence changes none.group to
invalidate.gpu; no compiler-bug/legality conclusion follows. Frozen engine
152055a6 passes small C2 output prefixes, drafting, math/reasoning/tools,
normal teardown and final both-card/compiled-collective health. Headpair
remains OFF and unqualified. No new speed/fullcache/shelf claim. See
[S28 evidence](../../../docs/20261015_strata_ep_publication_scope.md).

S29 tests that same frozen152055a6 engine with headpairOFF/ON: median C2
coding97.881->102.279 aggregate tok/s (+4.493%,3pairs), all six exact
request/output hashes match, both functionals/normalteardown/health PASS.
Native emit weighted7.033->5.380ms/window, ON1675paired/19solo. New C2
per-kernel times and driver-call deltas are not traced; S22's fourC1 heads
remain the reference operation rollup. Full depth2/headpair cache and
long-context performance are still unqualified. A pre-existing batch->solo
counter carry bug loses prior draft totals, so acceptance comparisons are
invalid; client wall-time and output comparisons remain valid. See stageS29.

S30 fixes that carry loss for new requests; native summed344accepted/
417offered equals finalserver/API after a real batch-to-solo handback.
Six CPU tests and functionals pass. Same frozen152055a6/headpairON/depth2
now passes four199501-token histories, two active256-token replies, and
four forced RAM returns:199839held reused/31new/0retained-history replay.
Draft KV byte-readback passes, peaknativeRSS16.255GiB, zero swap/OOM.
Native663.3-685.1ms/clientTTFT2.992-6.918s are correctness-run diagnostics,
not a clean speed comparison. Normal teardown/health is in stage evidence.
The overlap gate is depth-aware: mean kept rows must exceed3 at depth2.

Existing native timers from the first full pair show selection224/225ms
at the first12288-token block ->6278/6284ms at block16; attention grows
2291/2303->2630/2636ms. These are two ordinal sequences without A/B
attribution or exact context offsets, not a joined GPU budget. Value-only
cost remains unisolated. All170 S22 operation rows remain unchanged.
