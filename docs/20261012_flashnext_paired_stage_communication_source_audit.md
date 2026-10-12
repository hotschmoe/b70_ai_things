# Paired target communication: actual source52 and58

CONFIG: Source52 SDK /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww,
receipt8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94.
Source58 SDK /mnt/vm_8tb/b70/build/strata-native58-fresh52-v2-20261012T204001Z-id823bdv,
receipt9d39acecc1b34c641311e831107eddb280ef5830ef1e85f3c785d3ede8777490.
Target24/24, HC4,N2560. Current original-draft baseline sets device-plan1,
NO_HOST1, stage-mirrors1, segmented pinned mirrors1024MiB, mirrorT1-staging0.
Stock custom push-AR and collective transport are not enabled by this path.
Source58 cooperative prefill changes host scheduling, not the checked handoff
allocation/import/export bodies. No model/provider/GPU/compile actions here.

COMMAND: Bounded reads of the named actual source snapshots and small receipt/
blueprint metadata. Source hashes and identical52/58 prefill transfer-body
checks saved in /tmp/b70-source52-58-paired-communication-source-audit-v1.json.
No source changes, physical counters, GPU property probes or payload reads.
The inspected actual58 genuine-construction recipe likewise contains
stage-mirrors1/device-plan1/NO_HOST1; its transport recipe is not an actual
physical traffic measurement. Verify handoff import/export bodies are also
byte-identical between52 and58, although the complete verify.cpp files differ.

RESULT: Explicit stage communication is host-staged, not a direct GPU0->GPU1
USM-device copy or all-reduce.

- Verify constructor: source52 sycl/src/program/generate.cpp7046,7095-7123
  allocates sycl::malloc_host on the current in-order queue and gives each stage
  the same handoff pointer. sycl/include/strata/core/verify.hpp146-153 defines
  packet HC*N +N +HC floats per row: [R40960B,bo10240B,inj16B],51216B/row.
  The allocated capacity uses kVerifyMaxT, not just the current request T.
- Verify graph: sycl/src/core/verify.cpp1668-1675 exports these three arrays
  through copy_from_mapped kernels. Despite the name, its implementation is
  an ordinary float4 pointer load/store kernel, elementwise.dp.cpp417-431,
  542-566; the destination is the host-USM handoff. Stage1 graph imports the
  same three arrays into its device R/bo/inj at verify.cpp867-873. T1 transfers
  51216B per leg; T2 transfers102432B per leg. These are logical operand extents,
  not measured physical link traffic/cache transactions.
- Verify completion: verify.cpp2327 waits the source compute queue;2418 waits
  the expert copy queue only if used;2434-2436 then invokes stage1.run recursively
  on the same host thread. Stage1 is not submitted before this source-stage
  completion in this serial run path. Each run scopes OnDevice(device_),2098.
  The split pipeline/run_slots/pl_launch APIs are separate and not evidence of
  overlap in the current serial original-MTP path.
- Prefill: prefill.cpp103 defines D10240. It owns two host-USM buffers at986-992,
  each chunk*D*4. At4402-4411 source52 (4498-4507 source58) it queue-copies device
  R to host and waits the source compute queue. It waits the prior direct
  successor future, passes this host pointer to stage1 and launches std::async
  at4428-4437 (4524-4533 source58). Stage1 queue-copies host R to its device R at
  2279-2280 source52 (2338-2339 source58; same exact body). T512 is20971520B/20MiB EACH leg;
   512 complete rows would traverse41943040B/40MiB logical bytes across both legs.
  Double buffering allows source computation for another chunk while the
  successor processes the previous one. The future/get protects reuse;
  final next_run_.get closes outstanding successor work. Cooperative58 may
  service parked work while awaiting that successor; it preserves the copies.
- final_R_all(): verify.hpp271 recursively returns the LAST verifier's R_
  pointer. It performs no transfer back toGPU0 or host. The original drafter
  ownsGPU1: loaded_main_adapter_pair_v3.hpp.bind_target checks last stream
  context/device and actual USM device ownership before consuming this pointer.
  Its check_hidden confines rows to that final allocation or the authenticated
  LAST-prefill allocation. Thus original own-draft catchup has no reverse
  hidden-state bridge. Host argmax/logprob readbacks are separate operations.
- Commit: verify.cpp2676-2677 keeps the source-stage queue wait when next_ is
  present, then forwards commit at2731. wait_commit additionally closes each
  pending event and forwards at2764. These are state/decision fences, not
  evidence that the interconnect was idle.

Expert spill is a different data path. generate.cpp4971-5006 enumerates missing
experts for each stage and refuses uncovered experts when NO_HOST is enabled.
GgufExpertSource::mirror_stage, sycl/src/core/gguf_expert_source.cpp217-257,
allocates stage-owned sycl::malloc_host segments, reads exact original expert
bytes into them, and uploads a device pointer table after completion. These
are pinned host weight copies, distinct from the original read-only GGUF mmap
and from stage-to-stage activation buffers. Resident GPU cache pointers and
host-mirror pointers feed the device-built plan at verify.cpp1422-1431.
Grouped GPU kernels can consume mirrored weights directly; optional ptr2
fetch/rebase at1576-1582 stages them into device scratch when the explicit
staging route is enabled. Current original-MTP baseline explicitly disables
mirrorT1-staging. A mirror pointer says where operands reside; startup mirror
size is not per-token transferred traffic.

NO_HOST at verify.cpp2137/2210 skips the per-layer CPU expert-doorbell servicing
loop. It does NOT mean no CPU launch, no host weight operands, no RAM activation
handoff or no CPU PLE publication. PLE gather/host flag publication remains at
2138-2157 and2215-2225. Device-plan rows use zero CPU expert contributions at
1588-1591. Inferring CPU expert math from mirrored host operands is incorrect.

Existing timing is host elapsed. target_stage_run GPU0 encloses recursive
GPU1 run, so stage0-inclusive minus nested stage1 is host-exclusive elapsed,
not physical GPU0 idle. ReplayWait includes queued kernels, mapped-memory
reads/writes, dependencies and completion; draft_record_wait includes its
own kernels and preparation. Teacher target_prefill_part_run encloses own
catchup callbacks and asynchronous successor work. Nested spans must not be
added. Source-context selection/USM ownership is pinned here; actual cross-stage
native context handles were not measured by this audit.

VERDICT: Host-staged activation handoff is source-supported. Its contribution
versus host-mirrored expert access and computation is not measured separately.
The best next physical seam is a bounded loaded two-stage trace of the exact
handoff export/import kernels (verify) and known20MiB D2H/H2D commands(prefill),
joined by actual source graph/queue/window(T,pos) and allocation descriptors.
Use per-device raw-clock intervals/unions and existing calibrated mappings only
where proven. Generic copy_from_mapped also serves other operands, so kernel
name alone cannot attribute a handoff. Separate those intervals from grouped
expert kernels whose pointer plans reference host mirrors. Do not infer bus
bandwidth, saturation or idle from host waits or logical byte totals.

A direct-device transport is a separate proposed control: first query actual
peer accessibility under the pair lease, then demonstrate a loaded two-device
owned-buffer oracle with exact context, data/coherence and retirement before
changing a model handoff. No custom push-AR model arm is justified without its
loaded oracle. No new transport/kernel patch is proposed by this source audit.
