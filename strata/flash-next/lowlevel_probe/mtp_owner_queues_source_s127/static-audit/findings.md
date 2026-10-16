# S127: head-free MTP function-static/global device-state audit

CONFIG -> SOURCE ONLY, 2026-10-16. Current source2d26e686 and frozen S127
proposal; no implementation, build, GPU, payload/binary/trace read or model call.
Audit record_front824-925, record_rest941-1155 (head=false), forward815-821,
their actual callees, and idle. Pins/excerpts are retained beside this note.

COMMAND -> Follow call arguments through current kernels and local DPCT SDK;
inspect static maps/global selectors/allocation sites. This is a bounded source
reachability/ownership audit, not pointer-disjointness, runtime scheduling,
compiled-store or concurrency proof.

RESULT -> No unkeyed mutable GLOBAL DEVICE SCRATCH blocker found on these
head-free paths. Correct the earlier claims as follows.

| Reachable symbol / source | Mutable state, key and capture/lifetime facts |
| --- | --- |
| fused_gr::down_partials, fused_gr.dp.cpp1759-1768 | Mutex-protected static map keyed by sycl::queue*. Lazy40960B allocation (4*8*320 floats), pointer captured by fused_gr_read_multi1931. |
| fused_gr::slice_partials655-664 | Separate mutex/map by queue*. Default sliced path1956-1964 lazily allocates829440B (80*8*324 floats), used for down slices/reduction. |
| fused_gr_read2074-2085 | Alternate single-read adapter also has queue-keyed xn map,40960B. Actual MTP calls multi directly; its final fallback gr_read is gr.dp.cpp469, not this adapter. |
| new_grouped, s2_expert_grouped762-765/1579-1590 | NO static device allocation. g_last_path is a host atomic diagnostic; g_select_old is a host atomic override. Real gate/up, Q8 and scales are carved from caller hit_scratch_; group/count/destination/output buffers supplied by owner. Concurrent diagnostics may overwrite each other, not GPU work buffers. |
| shared_expert475-539 | Carves gate/up/intermediate/scales from sh_scratch_. NativeSharedWeights::q8_1=xq_ is owner-local. Routed work uses hit_xq_/hit_xs_, so it does not overwrite shared branch xq_. native_bf16 host selector must remain fixed. |
| native_mmvq small path / iq4nl_values540,2511/2538 | A16B global-memory lookup table is lazily initialized per DEVICE, not queue (SDK get_ptr_impl930-943). init may allocate/upload on first use; later calls reuse it. It is not forward scratch. wq206-211 allows Q8_0/Q4_0 projections, whose SmallTraits ignore this lookup pointer; kernel use of IQ4 table must not be generalized. Q8/Q6 packed registries are original-weight-keyed immutable copies during forwards, modified only by registration/unregistration APIs. No concurrent pack/unpack is allowed. |
| NativeEmbed::gather_dev285 / iq_embed_rows4633 | Uses shared table read-only, owner tok_/emb_, supplied queue; no function-static allocation in this entry. IQ codebook constants are read-only. |
| native_qsa norm173/219, native_rope228+, native_bf16::mmvf861 | Workgroup-local reductions/accessors and supplied input/output/gamma. Host enable/environment selectors, no persistent mutable device scratch found in these entries. |
| native_router_top10 / router_top10; quantize_q8_0_scaled343; native_quantize_q8_1 | Router logits/ids/weights and quant outputs/scales are supplied owner pointers. No lazy device allocation in these entries. Local reductions are per invocation; host feature flags are not device buffers. |
| qsa_decode_attn_batch643+, KV *_step/*_steps, window_ids1634 | Partials/m/l carve supplied attn_scratch_; KV/page/status/host mirrors use owner st_; IDs/steps are explicit owner pointers. No per-call global device scratch on these signatures. |

HC map facts matter: down_partials is requested on every default multi read;
sliced mode also requests slice_partials. Thus a previously unused B.cs_ can
add870400B (850KiB), even though the queue itself already exists. Earlier solo/
prefill work may have initialized it; source cannot establish that it did.
These maps retain allocations for process lifetime, have no erase/free here,
and retain raw queue-address keys after queue destruction. No owner-queue
address reuse may occur while captured users are alive. Queue-keyed storage
is safe from A/B alias only with DISTINCT live queues and ordered use within
each in-order queue; same queue does not provide concurrent scratch ownership.
Lazy malloc has no explicit null check in these helpers. Cold capture/load
allocation failure remains an error/health concern, not permission to retry.
"No new buffers" must become "no new explicit owner arena; existing queue-keyed
HC allocations may be materialized for B". These bytes are logical allocations,
not an observed free-memory delta or capture-allocation success.

Two exclusions are explicit. qsa.dp.cpp790 has an UNKEYED process-global d_step
for OLD host-scalar wrappers; MTP instead calls capturable *_step/*_steps with
step_ (qsa865/kv_q8193/kv_q4272), so step_upload/step_scratch are not reachable
from the audited forwards. verify_kernels::ProfileHostClock2606 is context-keyed
host USM plus writer, used only by gpu_stamp; MTP supplies no stamp buffer.
Neither exclusion extends to arbitrary verifier/legacy API callers.

idle semantics (mtp.hpp155-165): cs_->wait() is wrapped in DPCT_CHECK_ERROR.
It waits for cs_ completion, DOES NOT explicitly wait side_, and DOES NOT call
wait_and_throw to request asynchronous error delivery. The macro catches
std::exception, prints it and returns an error code; idle then returns false.
The outer sycl::exception catch calls exit1 if an exception escapes the wrapper;
calling all ordinary idle failures fatal is too broad. Normal successful
record_rest enqueues side_ fork/join events1040-1068/1101-1105 before combine,
so cs_ completion transitively covers joined side work. An early failure before
the join invalidates that inference. New error/drain handling must explicitly
cover cs_ AND side_, report errors, and preserve captured-pointer lifetime.

VERDICT -> Conditional source ownership argument survives; NO runtime/build
GO implied. Keep queues/selectors/registry stable, budget possible850KiB B HC
materialization, capture only with stable pointers, and independently drain
both owner queues plus their side branches on failed submission/capture.
Function-static host selectors and per-device HC attr/chunk/split caches are
not device output buffers; current serial host submission avoids a claim of
thread-safe arbitrary concurrent host reconfiguration. No all-readonly claim,
allocation-free claim, numerical equivalence or realized overlap follows.
