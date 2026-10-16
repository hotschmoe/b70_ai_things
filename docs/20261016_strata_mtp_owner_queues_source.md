# Paired MTP owner queues: source proposal, 2026-10-16

S127 prepares a default-OFF scheduling change. It is unapplied, unbuilt and
untested at this checkpoint. The evidence supports a conditional source design;
it supplies no latency result, hardware-overlap measurement or completed KV
byte-equivalence proof.

CONFIG -> `STRATA_BATCH_MTP_OWNER_QUEUES=1` would record each slot's head-free
MTP forward on its existing owner queue. Both forwards must complete before
the unchanged paired head/tail runs on A's queue. The tail completes before
either owner starts the next depth. Each graph replays on its recording queue;
whole-pair graphs are bypassed in this mode. The flag is immutable for the
pair's graph-cache lifetime. OFF retains the existing queue/launch/prefetch
ordering. The proposal changes scheduling, not kernel arguments or arithmetic.

COMMAND -> Read the actual MTP record_front/record_rest/forward and their
callee scratch contracts, make external V1/V2 candidate copies, and run only
read-only apply/reverse checks and source-projection checks. No compiler,
GPU/model execution or live source change was performed. Exact patches and
source paths/hashes are in the [S127 evidence index](../strata/flash-next/lowlevel_probe/mtp_owner_queues_source_s127/source-copy-index.json).

RESULT -> V2 preserves V1's scheduling mechanism and closes three source-review
conditions:

- ON pre-staging readiness uses `wait_and_throw` on each owner's main and side
  queues. The original `idle()` waits only `cs_` using `wait()`. Its DPCT macro
  catches ordinary standard exceptions and returns an error code; it does not
  ordinarily terminate the process or independently drain `side_`. A successful
  recorded side-to-main join covers side work, but a failure before that join
  needs an explicit side drain.
- The existing SYCL exception handler remains. An added standard-exception
  handler drains owner queues/branches before allocating an error string in ON
  mode, and rethrows when OFF to preserve its prior behavior. Missing output
  after a completed tail names the owner and depth. Failure never authorizes a
  second draft after possible KV writes.
- A bounded callee audit found no unkeyed mutable global device-scratch blocker
  on the current head-free paths. This does not establish runtime pointer
  disjointness or arbitrary host-thread safety.

| State examined | Source ownership consequence |
| --- | --- |
| HC `down_partials` and `slice_partials` | Mutex-protected maps keyed by queue pointer; first use can allocate40,960+829,440=870,400B (850KiB) at source-default constants. |
| Grouped/shared experts, router, quantizers, norm/RoPE | Working arrays are supplied by the caller or workgroup-local; grouped selection/last-path state is host atomic telemetry, not shared GPU output scratch. |
| QSA attention/KV/window IDs | The capturable APIs receive owner scratch, step pointers, page tables and KV pools. The legacy process-global host-scalar step buffer is not reached by these MTP calls. |
| Lookup tables/weight registries | Read-only during forward use after initialization; per-device table initialization and load/unload registry changes must not be mistaken for mutable per-token scratch. |

The HC allocations may already exist from B's earlier solo/prefill work; source
inspection cannot prove that. They are retained for process lifetime with raw
queue-pointer keys, so queue/pointer lifetime and address reuse matter. Thus
"no new buffers" is too strong: there is no new explicit owner arena, but using
B's queue may materialize existing lazy allocations. These are logical byte
counts, not measured VRAM deltas or proven capture-allocation success.

Actual `claude-opus-5-5` (56.541901904s) and `claude-fable-5-1` (92.955938251s)
returned `SOURCE_DESIGN_GO` with these corrections. Both exited0 in their own
resumed sessions; the parent read the complete responses. The [filtered review
records](../strata/flash-next/lowlevel_probe/mtp_owner_queues_source_s127/reviews/root-disposition-derived.json)
preserve actual IDs, packet/response hashes and dispositions without cumulative
usage. V2 implements root-authorized corrections; it was not sent for a second
model round. Model statements suggesting deterministic offers, complete KV
proof, a noise band, or a causal hardware-overlap conclusion are not adopted.

Opus recommended a conditional standard-exception catch that drains ON and
rethrows OFF. Fable preferred a local prefetch catch to preserve OFF behavior.
Root selected the conditional catch: OFF still rethrows, while ON also covers
other standard exceptions during active graphs. Both required side-queue
readiness and a callee scratch audit. Their complete actual responses are
retained beside the filtered review records; their stronger runtime claims
remain proposals, not evidence.

VERDICT -> Retain as a source-only candidate for a separately authorized main
build and finite matched comparison. No private framework or redundant model
round was added. Any future result must include all host joins and CPU PLE
hints, strict immediate work/output checks and normal teardown/health. Current
MTP forwards issue no source-visible GPU PLE host-service request; that bounded
argument expires if a later forward adds one and does not generalize to the
main verifier. S125 remains a separate Q6 pre-expansion source lead.
