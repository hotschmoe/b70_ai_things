# Independent host trace review

CONFIG -> Frozen startup hook in raw collective-host-trace, phase0328900c
runner and distributed route. Review only; no hook applied or GPU execution.

COMMAND -> Inspect exact runner, custom-op route and communicator sources;
rerun test_cpu.py. Reviewed hook SHA256: a65751a864ac454d3cbb48219d99c2c5d603f7b1a7bbc11fc54533a62fd101e1.

RESULT -> Source-pinned all-reduce/all-gather fake implementations do not call
the communicator. Requiring use_custom_op_call keeps this wrapper below the
opaque collective boundary instead of tracing JSON/metadata code into Dynamo.
The runner wrapper retains the original inference-mode-decorated callable.
Metadata access uses shape/dtype/device/stride; no values are copied or read.
Original clone, async c10d call and existing work.wait remain unchanged.

Review fixes completed by the hook owner: separately hash inherited base
communicator source, filter actual communicator group to scoped TP group and
record its identity, and protect dummy_enter with the TLS-restoring finally.
The revised CPU fixtures pass. No instrumented GPU compile/capture smoke has
run as part of this independent review.

VERDICT -> Suitable for bounded startup profile observations, with host timing
perturbed and compilation behavior still requiring runtime checking. Python
returns are not device-completion events. Nested method calls are not unique
native collective counts. Graph replay may bypass Python entirely. This is
not a live accepted-count/cache producer-consumer trace. Truncation makes logs
incomplete. Phase-only runner hash must reject the MRV1 candidate until a
separate explicit source review/repin; never bypass that gate.
