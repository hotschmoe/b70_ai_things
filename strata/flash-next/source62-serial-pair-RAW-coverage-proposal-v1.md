CONFIG

Source-only proposal after actual optimized62 RAW capture. The admitted trace spans9.353s; observed any-device union3.645s and CPU API union2.144s overlap1.637s, leaving5.201s uncovered by these two recorded categories. This is not device idle. Original inspector/source/raw files stay immutable. No new model run, build, payload read or physical-bandwidth inference.

Current gaps and exact sources:

- inspect_source62_cached_RAW_clock_v1.py37 selects submit_host_ns inside client SEND-before/DONE-receive. Work submitted earlier but executing inside the window is excluded. Report selected-submit scope separately from a future complete clock-roster overlap view; do not relabel the old result.
- Inspector75 explicitly rejects claims of stage mapping, internal preparation, complete operations and continuous RAW/monotonic drift. Device_origin UUID is genuine, but no native logical stage is bound to it.
- generate.cpp2198 rejects H36 for every layer split. Current pair trace has H36OFF, so existing Verify/StageInputs/PlePrefetch/PleGather/ReplaySubmit/ReplayWait/Commit scopes do not contribute native intervals.
- verify.cpp451-465 constructs compute/copy/shared queues on actual device_; init515 establishes actual lb/le. run2112-2141 records logical context; run2327-2329 encloses the original wait. Earlier-stage run2436 delegates synchronously to next_->run, so its Verify interval contains stage1 and cannot be summed with stage1.
- verify.cpp2441 Sample scope exists only for stochastic/history sampling. Greedy selection is in the graph; mapped h_out_ reads2452 are presently unlabeled. Add a distinct host mapped-output-read scope, never call it measured GPU sampler duration.
- Prefill2195-2197 has host60 marks;2267 records preparation ready,2325 chunk prepare,2435/2440 PLE future get,2629 expert plan,3503/3505 existing router wait. Existing host60.hpp ctor filters pos24/count>=512 and generate10489 requires layer cooperation. These cannot observe current batch0/RID2/pos1536/remaining512 without an explicit successor.
- Prefill2250 bind_stage_helper() can swap m.pp to another GPU. UUID alone cannot attribute every GPU1 kernel to stage1. Logical target range, native queue role and actual backing device must be separate.

COMMAND / concrete next source port

A new bounded flag STRATA_SERIAL_PAIR_RAW_METADATA=1 is defaultOFF. Preserve old H36 single-device mode byte semantics. Permit H36 only additionally for explicitly parsed split_at={24}, split_devs={1}, serve/batch0/pipeline0/noMTP/no suffix or lookup/adapt0/peer-disabled. Reject alternate branches. Retain4-request/MAX64/prompt2048/event32768 caps, owner-thread/overflow/source/session-close checks. For this new mode only use CLOCK_MONOTONIC_RAW and label the actual clock in metadata; default oldCLOCK_MONOTONIC stays unchanged.

The new standalone source prototype original-mtp53/serial_pair_native_queue_mapping_v1.hpp has32 metadata rows, source binding/PID/TID/queue variant/device/context and RAW time. Its source never calls a timestamp query, explicit wait, barrier, GPU allocation or device sampling operation. get_native(queue) may materialize a backend queue; call ordinary owner mappings before READY/timed requests and disclose that initialization cost. This prototype is not wired or compiled yet.

Reviewable call sites:

1. verify.cpp after actual lb/le/source handoff validation515-523: record verify.cs/copy/shared using cs_/copy_/sh_cs_, actual device/range. No per-layer hook.
2. Prefill init1020-1024 after m.cs/m.device: record owned prefill.cs with actual stage range, after bounds are established. Stage ranges may be assigned by a later setter: place the hook at the first point BOTH are final, not assume init defaults.
3. Prefill2250 after successful bind_stage_helper swap: at most one row per selected run for active helper m.pp->s with logical target m.device/LB/LE and actual queue-derived device. Bind RID/slotgen/pos/T. This covers ownership metadata, not all internal helper subqueues; unmatched handles remain unknown.
4. Existing host60 Trace/marks remain the preparation instrument. A new branch admits only RID2/pos1536/T512 with existing owner publication and8-run/64-row caps, RAW timestamps. Stage workers keep their own row arrays; do not use single-writer H36 on async Prefill workers. Default historical filter/clock stays unchanged.
5. verify.cpp2452 mapped output reads get an appended distinct H36 kind. Existing stochastic Sample scope stays unchanged. GEN requestscope9803/Reset9804-10120/PromptSpan10735/TEmit11649/DONE11911 remain the existing native ownership/token joins.

Join rules: same original native PID/namespace/source binding, unique live device handle to clock device_origin UUID, then regular_CQ native handle to PTI queue_handle OR immediate_CL native handle to command_list_handle. PTI normalization records both handles (prepare_pti_unitrace_clock_export_v1.py44); ze_collector ZeCommand764 queue_ is a CQ and can be null, never silently reinterpret CL as CQ. Graph-internal/unmatched queues remain unassigned. The current clock instances lack context_handle; a native context row is ownership evidence only, not a completed PTI context join. Adding tool context would require a separately versioned metadata-only source export from existing collector command-list/queue descriptors, not fabricated instance context.

RESULT

Existing-source audit and bounded source prototype only. No executable native/tool patch or runtime integration is claimed. Actual compile/controller and typed reader are future narrow ports after review. Existing intervals already establish observed unions; coverage remains incomplete even with additional host spans unless the actual roster proves completeness.

VERDICT

This is the smallest identified reliable next localization: paired native host spans + native handle-role mapping + selected Prefill preparation marks. It can assign covered time and identify still-uncovered intervals; it cannot establish physical idle, bus saturation or savings of5.201s. Do not sum nested stage/Verify/Prompt scopes or overlap CPU waits with device work additively. Stop here for the requested pause; no new experiments launched.
