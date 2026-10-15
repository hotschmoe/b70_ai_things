# Strata current 32K request profile (S90, 2026-10-15)

CONFIG -> One diagnostic request on the frozen S89 executable
`b2a610ea23856ad290cbca2a0dcdec4de676f3cdb519ddc0a01c5321942636be`.
QSA tiled ON, Q6 centering OFF, HC common, S49 prefetch OFF, global RPW1,
A2/PF0, batch MTP depth2/headpair ON, native EP149/peer363 per layer,
all24576 experts resident. INT8 KV, two200000-token slots, 32GiB/four-entry
RAM parking, prefill workspace12288. The pinned e41 image, native process,
profiler, effective flags, model continuity and source pins are recorded.
No source or build change was made for this stage.

COMMAND -> Root reused S89 post-health only after checking continuity, then
started unitrace paused with the native process. After loading, it resumed
before the first generation: the exact retained S38 request, changing only
max_tokens to32. There was no generation warmup. Collection paused/stopped
in cleanup; one subsequent unprofiled arithmetic request flushed tracing.
Normal TERM -> native QUIT, full CID+NAME absence and wrapper exit0 preceded
both-card and compiled two-rank collective10 post-health, all exit0. One
stopped, stable trace was parsed after teardown; a later analysis read only
its compact CSVs. No duplicate serving run or raw-trace reparse.

The recorded phase commands used `bin/cpu-run` for startup and
`bin/cpu-run --quiet` for collect/stop, with the detached server owning the
pair GPU lease. The analysis explicitly selected the v2 run directory,
not its historical v1 default. Exact scripts and closure records are in
[the S90 record directory](../strata/flash-next/lowlevel_probe/current32k_profile_s90/README.md).
They retain historical absolute paths and must not be executed in place.

RESULT -> The one request used32728 fresh prompt tokens, cache0, and32
completion tokens; the retained greedy prefix and usage/native-work checks
passed. The native summary reported27599ms prompt processing,2576ms decode,
19accepted/33offered drafts and2checkpoints. Client TTFT was28.134s; the full
instrumented request bracket was30.225416s. These are diagnostic observations,
not clean throughput or an OFF/ON comparison.

The valid441678985-byte trace contains2732936 events;781466 overlap the
request window, with zero boundary crossers. The separate8129-row operation
CSV preserves device, engine/thread, exact names, geometry and intervals.
Its SHA256 is2178af0595785aefc8a5c1ea70d36a7ad9feb23eae5460bd4046fac6b4a72d9f.
Raw trace SHA256 is6799ce895caff68c80197c68549be70ab1b7020ea0875bae3156bf4fd9120730.
Neither the raw trace nor full CSVs are copied into the repository; an
external inventory and bounded50-row current top-operation CSV are retained.
The170 historical map operations, rollups and operation metadata stay exact.

| Current primary observation | Calls | Recorded time | Scope |
| --- | ---: | ---: | --- |
| Full32-query INT8 attention chunk | 12264 | 5689.599621ms | Contained service, two engine rows combined by name+device |
| Matching full32 merge | 12264 | 697.009584ms | Same aggregation policy |
| Attention family | 25048 | 6473.357250ms | Clipped service sum; broader than the full32 pair |
| S89 tiled scorer | 768 | 792.502816ms | Actual named GPU calls; contained service |
| Original scorer, all contexts | 960 | 345.393500ms | Clipped service sum |
| Top-k, all contexts | 1728 | 187.579278ms | Contained service |
| GDN recurrence | 108 | 1751.303ms | Selected exact operation rows |
| Primary `copy_f32_wide` | 144 | 738.071083ms | Contained service; activation-copy source scope |

Service sums and unions overlap and are not additive latency budgets. The
full32 attention rows require aggregation across both engine/thread rows:
the largest row alone has12232calls and omits32calls. The merge likewise
requires12225+39. Current main-prompt geometry32721 =12288+12288+8145 gives
12 QSA layers * (384+384+254) =12264 full32 calls; each layer also has a
17-query tail. This reconciles invocation counts without deriving exact
native prefill timestamps.

For256-query selection, the three main chunks imply576/576/384 scorer calls.
The4097-block guard selects0/396/372 =768 tiled calls, matching actual GPU
kernel names, rather than only the host activation marker. Original main
scoring contributes576/180/12 =768;192 additional original calls remain
attributed only to small-final/capture/decode handling. They are not all
labeled steady decode. Top-k1728 =1536main+192residual is consistent.

Current source identifies the144 `copy_f32_wide` calls as48MoE layers across
three chunks copying FP16 activation bits to the peer mapping. Its logical
8,041,512,960bytes are shape arithmetic, not measured PCIe traffic. The
fixed4096-workgroup grid does not encode the transfer byte count.

The primary recorded-operation union is17.239233s; its uncovered intervals
sum12.986183s within this request bracket. The peer union is4.883815s, on a
separate device clock. These are recorded intervals, not physical occupancy
or idle; the cards are not joined or added. Long gaps near the end precede
`ep_bump` and cannot be called main-prefill gaps. Ordered GDN envelopes are
partial source motifs, not complete chunk boundaries.

Host `zeCommandListAppendMemoryCopy` has681 calls and14.404145s interval
union. Only143.952ms overlaps primary recorded gaps;14.260193s coincides
with recorded primary activity. The extracted host CSV lacks queue/device
arguments and correlation IDs. This overlap neither assigns host calls to
a device nor proves transfer service, P2P bytes or removable waiting.

Some earlier65-68ms same-primary gather-to-dequant motifs justify a source
readiness audit: current prefill enqueues the primary gather, then the peer
expert path, before the first primary expert dequant. This is a source
ordering fact and a hypothesis, not a measured cause. Rejected S7 async
submission and S38 alias/gather changes are not queued again.

Actual source reviews responded as `claude-opus-5-5` (44.645s) and
`claude-fable-5-1` (48.436s), both GO with root-resolved conditions. Root
revalidated recent health, the explicit analysis path and the profiler event
category. Opus's S30 RAM-recovery analogy was rejected as parser-memory
proof; current available memory and stopped trace size were checked instead.
The later combined S90-post/S91-source reviews responded as the same model
IDs in fresh recorded sessions (39.007s/60.518s), both accepting S90's bounded
reconciliation. Full answers, completions, sessions and root dispositions
are copied; original event streams and packets remain external.

VERDICT -> S90 is a completed current-workload diagnostic, without a clean
speed delta, critical-path saving, hardware-idle, bandwidth, stability,
default/daily/shelf promotion or target-achievement claim. The first unwarmed
request includes admission/capture/prefill/decode; S89 had prior C2/4K work
and256completion tokens. Differences cannot be assigned solely to profiling.
The host clock correction changed945.036us between before/after brackets;
there is no cross-card clock join or exact native prefill epoch endpoint.

The selected next lead is a private score-only12-head reduction within the
existing SG32/CHUNK64/NC1 attention body, retaining query/value/softmax/merge
arithmetic and strict scratch/output gates. This differs from rejected
S11 query batching, S12 lane ownership and S13 value pipeline changes.
S91 is separate: no runtime result or production change is claimed here.
The2000prefill/150per-stream goals remain open; S30 RAM recovery evidence
is retained without another cache or daily-restore run.

Evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
Authoritative completion is `current32k-profile-s90-root-runtime-decision.json`
and combined-review `root-s90-postdecision.json`. Historical preparation and
initial collection records preserve their earlier pending states; they are
superseded by the completed analysis and root decisions, not rewritten.
