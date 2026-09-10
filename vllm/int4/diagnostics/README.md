# R276 request-to-GDN tracing

CONFIG -> Exact R276 source, optional host-only diagnostics for the confirmed
one-token prefill routing defect. No live image or serving process was changed.
COMMAND -> Prepare patches/r276-gdn-request-trace.patch; install this directory's
gdn_request_trace.py as vllm/b70_gdn_request_trace.py in a separate diagnostic
image. Set B70_GDN_TRACE_DIR to a mounted private output directory. Leave it
unset to disable tracing. Use the same diagnostic patch on stock and phase-fixed
images if comparing outputs; it does not apply the phase fix itself.
RESULT -> Nine CPU-only fixture tests pass. Exact installed-source hashes,
patch application, syntax and the three host hook locations pass source checks.
The added helper contains no device transfer, .item(), event query, or
synchronization calls. No image was built and no GPU test was run for this trace.
VERDICT -> Ready for a separately identified diagnostic run, not production
promotion. Host logging can perturb timing and therefore is not a performance
measurement or proof that an asynchronous race cannot occur.

The three hooks are:

1. End of GPUModelRunner._prepare_inputs: snapshot request IDs, scheduled counts,
   host computed/prompt boundaries, original admission computed count, and the
   runner's CPU-written accepted-count working buffer after existing handling.
2. After target CommonAttentionMetadata construction: bind those rows to the
   exact CPU query-offset object after checking per-row lengths. Shallow cache
   group copies retain this object. Capture builds clear the binding.
3. GDNAttentionMetadataBuilder.build, after its non-spec/spec branch: emit actual
   classification only for effective query length 1 with is_prefilling true.

Each worker writes its own gdn-trace-PID.jsonl, mode 0600. No prompt text, token
IDs, tool arguments, cache salts, authentication headers or credentials are
recorded. Request IDs remain visible for correlation with server/RPC records.
The trace records counts and phase decisions, not conv/SSM state values or
physical cache-block pointers. The caller supplies a private directory.

A directly mapped event with route=recurrent-decode, prompt_tokens much larger
than 1, effective_query_len=1, and is_prefilling=true establishes that a long
request reached the confirmed bad routing decision. Correlating its req_id with
a bang response would connect the Pi symptom to that path. It would not alone
prove that no other defect contributed. route=initializing-prefill on the fixed
image verifies the corresponding routing correction.

initial_computed_host preserves the scheduler's new-request computed count even
when the suspicious singleton occurs after a later chunk. A positive value
shows work was already considered computed at admission; it is not by itself
proof of local prefix-cache provenance. remaining_prompt_host and scheduled
counts locate the logical boundary. common_prefix_len is the builder's cascade
prefix argument and must not be interpreted as total per-request cache hits.

Async caveat: computed_host and accepted_working_host are explicitly host
snapshots. The helper never reads input_batch.num_accepted_tokens_cpu, whose
D2H copy may still be in flight. In the asynchronous default-one path, the
accepted value is labeled non-authoritative. The trace adds no event wait to
make it authoritative. Metadata from a new draft/split CPU query-offset object
is marked mapped_target=false; it is never silently assigned target request
IDs. A diagnostic error disables logging without failing inference.

Validation:

    python3 -m unittest discover -s vllm/int4/diagnostics -p test_gdn_request_trace.py -v

Raw evidence:
/mnt/vm_8tb/b70/results/bang_isolation_20260910/gdn-request-trace/

## Experimental strict card health

xpu_health_strict.sh deliberately remains outside shared bin/ shelf tooling.
It rejects NaN/Inf in both matrix outputs, requires an exact positive marker
and exit0, and verifies owned-container removal after timeout. Invoke only
under the appropriate bin/gpu-run lease and with an explicit immutable --img.
The retained shared probe can accept HEALTH_OK False; its earlier HEALTHY
summaries establish only that its old gate passed, not a finiteness proof.

CONFIG -> Same health workloads, stricter finite/exit/cleanup gates.
COMMAND -> python3 vllm/int4/diagnostics/check_xpu_health_strict.py;
new-image-preflight under both leases using exact vLLM0.29 and SGLang pointer
candidate1660 images, separate fresh compiled collective caches, P2P0.
RESULT -> Twelve CPU/mock cases pass. Both new image stacks pass strict
per-card checks and numeric compiled two-rank collective health. Native
libraries and host stack are unchanged by this diagnostic script.
VERDICT -> Used via explicit experiment health-probe override. This is not
formal shared-shelf qualification; bin/ remains unchanged. Shell finite checks
are not a complete numerical kernel correctness or model coherence suite.
