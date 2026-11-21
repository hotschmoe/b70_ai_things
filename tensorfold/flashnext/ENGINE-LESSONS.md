# Engine lessons (running design notes for a future in-house Zig engine)

Owner directive 2026-11-21: the engine architecture is designed separately. This file records what we learn porting
Strata's hotschmoe-dd feature set onto TensorFold Flash-Next (Zig/Level Zero, two B70s). Append dated bullets under the
right heading, with an evidence path. Facts, not opinions; say what it cost.

## TensorFold architecture: what works

- Zig + Level Zero + OpenCL C kernels: a small, readable host path; the critical-path ledger (markers <1% overhead) made
  launch gaps and host stalls visible per op (regen.sh, B/ledger/). Recommend: build the ledger in from day one.
- Descriptor windows (XPU_FN_DESC): one precomputed submission per decode window cut host-side work; worth C4 +14% /
  C8 +21% over the classic path at int-rf2.
- Host unit tests that run without a device (CPU references for kernels, mocked-engine server tests) let five workers
  build features on the CPU while production held both GPUs (2026-11-21).

## TensorFold architecture: what hurts

- Feature flags are mutually exclusive in places where they should compose: descriptor windows vs paged int8 KV were
  exclusive, and desc mode preallocated KV for abi.max_rows=16 x --context, capping context at ~20-24K (2026-11-21).
  Lesson: KV memory must be a shared page pool sized by a budget, never rows x max context, and the fast path must be
  built over the paged representation from the start.
- The Flash-Next XPU path shipped greedy-only (server refused temperature > 0); sampling was an afterthought, so the
  first sampled path reads top-64 rows back to the host and runs at ~60 vs ~95 tok/s greedy. Lesson: design the
  sampler (device-side top-k + per-position keyed draw + spec-decode acceptance) into the decode window.
- Request-field allowlist in openai.zig silently dropped unknown sampling fields. Lesson: unknown fields log once;
  supported-field tables live next to the capability flags.
- The release binary embeds the worktree path, so the same commit built elsewhere has a different sha; gates must
  re-check the frozen binary.

## Strata features that were hard to port, and why

- Long context + parking (Strata: 200K, int8 KV, --kv-resident 65536 + RAM-backed slots): TensorFold had no paged
  fast path and the eviction work lived on an unmerged branch.
- Vision: Strata's CPU sidecar (strata-vision + mmproj) is reusable as a process; the work is the prompt splice
  (pad-token rows replaced by embeddings on both TP ranks, M-RoPE positions, encoder cache). ~930 ms encode per image.
- API parity: Strata applies model-card sampling defaults when the client sends none, clamps max_tokens to the room
  left instead of erroring, and supports tool_choice required/named; clients poll /slots.

## Kernel and hardware lessons (B70, see HARDWARE.md)

- Plain read ~604 GB/s; M=1 GEMV reaches 96-97%. Raw GGUF 98-byte q3 blocks get 17-21% of bandwidth vs ~90% for an
  aligned split layout: repack at load. Codebooks in SLM are 2.3x faster than __constant.
- Vector path goes ALU-bound at M >= 8 rows; DPAS holds 87-93%: concurrency needs DPAS decode kernels.
- ~3 us per short kernel, 2.1-2.3 us immediate append: launch count is the C1 currency.
- D2H ~4.5 GB/s (platform limit) vs P2P push ~13 GB/s: avoid host readback on the decode critical path.
- Device memory compression makes memset buffers read at 3.2 TB/s: never benchmark on them.

## Scheduler and serving lessons

- Host-bound first (event churn, host MTP), then device-bound: fix host overhead before kernels.
- Recorded command windows hung until signal events + drain were explicit; a recorded path needs its own tests.
- Prefill over paged desc KV measured ~865 tok/s at 9K tokens vs 1560 at 2K on the dense path (2026-11-21, VIS gate):
  long-prompt prefill must be benchmarked at the real prompt sizes (58-90K), not 2K.
- Benchmarks must include the production features (sampling, long context, vision, tools), not only greedy 16K.

## Backend portability

- (to fill: what in TensorFold is Level Zero specific vs portable; OpenCL C kernels vs SYCL/ESIMD; what Strata's SYCL
  stack gives that L0 does not.)

## Operations

- Five concurrent zig builds (~2.5 GB each) + agents caused a global OOM (2026-11-21 17:59) that killed the user systemd
  manager; enabled units auto-started on its restart. At most two builds host-wide, under cpu-run limits.
