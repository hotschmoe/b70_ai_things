# Flash-Next: next Strata development steps, 2026-10-09

CONFIG -> CPU-only source audit. Keep the selected Unsloth UD-Q4_K_XL bytes
and existing llama.cpp correctness campaign. No GPU execution, builds,
checkout changes, or weight downloads. Storage facts supplied for planning:
8 TB SATA SSD has about 5 TiB free; /mnt/cache NVMe has about 671 GiB free.

COMMAND -> Read Strata AGENTS.md and existing local source; run read-only
`git ls-remote origin HEAD refs/heads/main` and the same for `b70`; inspect
native hyper-connection dispatch, GGUF expert mirrors, and split-stage setup.

RESULT -> Both remote tips still match the locally available references:

| Repository | HEAD/main | Commit date |
| --- | --- | --- |
| Niko1221/Strata | fb58e0dbc8399662c0e47c76578c6e878b14f6cf | 2026-10-08 |
| maxfridbe/Strata_B70 | 99f3dbd0b21d1401b3769e0c0d963913607f380b | 2026-10-03 |

There is no newer upstream main revision beyond our initial fb58e0 pin to
pull for this investigation. The B70 remote is a divergent historical line,
not a newer replacement. Use upstream's existing SYCL tree as the development
base; transfer individual relevant findings deliberately. The pinned upstream
commit itself made stage-buffer pinning opt-in after corruption in an IQ3_S
release gate. Do not enable STRATA_STAGE_PIN globally as an assumed fix.
[Upstream pin](https://github.com/Niko1221/Strata/commit/fb58e0dbc8399662c0e47c76578c6e878b14f6cf),
[B70 pin](https://github.com/maxfridbe/Strata_B70/commit/99f3dbd0b21d1401b3769e0c0d963913607f380b).

Two concrete gaps make focused development preferable to starting a new
multi-GPU engine:

1. Native Q8 hyper-connections are not complete on SYCL. NativeDense can
   attach original Q8_0 bytes through `hc_q8`, and verify.cpp supplies them,
   but `sycl/src/kernels/cuda/fused_gr.dp.cpp:1882` explicitly states that
   Q8_0 reads are not built. CUDA's fused_gr.cu has a Q8 dispatch to port.
   Merely setting STRATA_HC_Q8 does not establish source-preserving SYCL math.
   The default projection interface reads BF16. This affects the chosen
   GGUF's Q8 hyper-connection matrices independently of expert caching.
2. Pinned host expert mirrors cover only the first stage. In
   `sycl/src/program/generate.cpp:4691`, `mirror_end` is the first split point;
   later caches are created afterward. The code explains that allocation
   ownership is tied to the first GPU's context. Moreover,
   `sycl/src/kernels/cuda/verify_kernels.dp.cpp:2354` stores only one global
   residency-pointer/mirror-table pair, then derives offsets from it.
   A second-stage mirror therefore needs per-stage ownership and dispatch,
   not simply a larger global RAM cap.

Source evidence at the pinned revision:
[native loader](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/sycl/src/core/native_dense.cpp),
[SYCL hyper-connection kernels](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/sycl/src/kernels/cuda/fused_gr.dp.cpp),
[CUDA reference](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/kernels/cuda/fused_gr.cu),
[stage driver](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/sycl/src/program/generate.cpp),
[mirror dispatch](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/sycl/src/kernels/cuda/verify_kernels.dp.cpp).

Preferred incremental gates:

1. Keep llama.cpp as the current measurement/control lane. Its repeat/history
   drift still needs explanation; do not promote it as a validated fidelity
   oracle merely because the constrained answer checks pass. Freeze settings
   and compare token IDs, first-token logits, and sequence-state reset.
2. In parallel, implement an opt-in SYCL native Q8_0 hyper-connection path.
   Preserve GGUF blocks and F32 activation arithmetic; cover attention/FFN
   down/up, final mixer, and exact injection weights. Leave
   STRATA_HC_Q8_INJECT off: converting original F32 injections to Q8 would add
   another quantization. If the existing BF16 injection representation is
   used, verify exact representability from the actual source tensor first.
   Cover both prompt and single-token/verify paths; enabling only one still
   leaves a mixed arithmetic contract. Gate each matrix against a CPU
   dequantize-to-F32 reference, then composed hyper-connection outputs,
   repeated calls, and request-history tests. Reuse gr_parity and the CUDA
   implementation as references, not as proof of SYCL parity.
3. Only after one-card byte/math gates, qualify existing serial layer-split
   handoffs on the two B70s. Begin without MTP, overlap, peer expert execution,
   or adaptive cache changes. Exercise a split on one device before the real
   two-device handoff; check intermediate residuals and final logits.
4. Add stage-1 mirrors after both caches are populated. Each stage owns its
   pinned allocations, device-address table, residency association, and
   teardown lifetime. Pass the table explicitly or bind it to a stage/device
   context; do not share the present single global pointer pair across GPUs.
   Allocate only experts absent from that stage's VRAM cache. Use one global
   host-memory budget across both stages, preserving OS, PLE, prompt and
   workspace headroom. The current default MemAvailable-minus-4-GiB policy
   must not be applied independently twice. First test immutable placement;
   then cache eviction/refill, forced misses, fallback, graph capture/replay,
   and teardown. Verify GPU-consumed bytes against GGUF and distinguish
   resident hits, RAM-mirror reads, and storage reads in counters.
5. Profile before relocating weights. A full RAM mirror removes recurring
   expert SSD reads but still uses PCIe. If storage remains material, compare
   identical PLE/direct-I/O and expert-miss workloads on SATA versus NVMe,
   preserving hashes and memory budgets. Move only the measured bottleneck
   working set; free NVMe space alone is not evidence of a speed benefit.

VERDICT -> Stay on llama.cpp for the active correctness/control investigation
while developing Strata's Q8 fidelity path first, then per-stage host mirrors.
This is bounded kernel and memory-lifecycle work on an existing split engine,
not multi-GPU support from scratch. Mirror ownership and graph lifetime are
more invasive than the Q8 dispatch port. Neither effort has a measured speed
or completion-time promise. GPU gates must use bin/gpu-run, matched identity,
health, coherence, teardown, and post-health; source preparation can proceed
in parallel without sharing live GPU execution.

## Concurrency follow-up: one user, usually one or two streams

CONFIG -> Target normal concurrency is 1-2, occasional 4, and at most 4-6.
This makes latency and preserving the one-stream expert cache more important
than maximizing large-batch throughput.

COMMAND -> Inspect serve/server.py, serve/test_parallel.py, docs/BATCHING.md,
and the actual pinned SYCL generate.cpp slot allocation/admission/pipeline.

RESULT -> Strata is serial by default, but it already has real opt-in
multi-sequence decoding. `server.py:3466` holds the FIFO only without batch
slots. With slots, `generate_batched:1176` serializes prompt admission on a
control lock, then releases it at line 1326 while each active request consumes
its own token queue. Requests share one model/engine and expert cache, not
separate complete model instances. Each slot has its own recurrent, KV,
indexer, and PLE state on every GPU stage (`generate.cpp:3811`). It can move a
solo request into a batch slot when another arrives. The Python parallel tests
use a fake engine; they are not GPU or model-fidelity evidence.
[Server](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/serve/server.py),
[batching documentation](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/BATCHING.md).

The SYCL engine also implements cross-stage conversation-group pipelining:
`generate.cpp:8828` has per-group token/position state, one in-flight group per
stage, and a pump that starts groups on free stages. This permits GPU0 and
GPU1 to work on different conversations. It is incremental admission and
multi-sequence decode, but its scheduling contract is not identical to
llama-server's continuous batching. Two concrete differences require tests:

- SYCL defaults `batch_groups=1` and parses the argument with atoi (lines 594,
  1788). Generic docs describing automatic groups are ahead of this path.
  Use explicit `--batch 2 --batch-groups 1` as the first concurrency control,
  then compare `--batch-groups 2`; for four slots compare groups 1 and 2.
- The SYCL prompt routine at line 10046 returns directly into the ordinary
  prompt pipeline when `piped` is true. That bypasses its explicit
  between-chunk batch-decode routine. Therefore long-prompt fairness with
  pipelined groups must be measured, not inferred from generic documentation.

Allocation can reduce the slot count or disable batching if fewer than two
slots fit. Assert the actual INFO batch_slots and server concurrency.serving,
not just command-line intent. Slot state comes out of VRAM otherwise available
to experts; exact cost depends on context and KV configuration.

VERDICT -> Supporting 2/4 streams does not require writing a new Strata request
scheduler. It does require qualifying existing SYCL slot state, admission,
cancellation, request reset, model fidelity, and pipeline synchronization,
plus the previously identified Q8/mirror work. Target two slots first and keep
four as an explicit capacity profile; leave six unprovisioned unless a real
need appears. Compare with llama-server at matching total/per-request context
and actual active slots, using 1/2/4 concurrent clients. Include a late short
request during a long prefill, cancellation/reuse, and return from two streams
to one. Measure per-stream latency, TTFT, inter-token stalls, memory and expert
hit rates, not only aggregate tokens/second. Until those gates pass, Strata is
a development candidate for tiered memory serving, not the declared winner.
