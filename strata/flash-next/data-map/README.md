# Strata Flash-Next data map

Open `index.html` in any browser (file://, offline, no network). It shows
where data lives and moves for the Flash-Next IQ3_XXS Strata EP serving
path, which operations are expensive or wait, and how strong the evidence
is for each claim. `operation-data.json` is the source of truth;
`index.html` embeds a copy of it.

Snapshot: 2026-10-15, S25 (S22 operation trace plus readiness, real-head
and production PLE diagnostics). Runtime source: frozen
`/mnt/vm_8tb/b70/build/strata-ab/src-ep` (upstream Strata 296b59d2 base
plus working-tree overlays for EP/PUSH/accounting/S19-S21). Evidence:
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
