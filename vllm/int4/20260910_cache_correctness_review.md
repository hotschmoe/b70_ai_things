# R276 cache correctness review, 2026-09-10

CPU-only investigation of image
`sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad`.
Raw evidence: `/mnt/vm_8tb/b70/results/bang_isolation_20260910/cache-correctness/`.
No GPU operations, serving changes, or native builds were performed by this
review. These results establish source behavior, not the cause of the Pi events.

## Accepted-token ordering: actionable MRV1 candidate

CONFIG -> Exact installed `gpu_input_batch.py` and `gpu_model_runner.py`, NumPy
CPU arrays, simulated previous-step D2H landing before or after row movement.

COMMAND -> `python3 vllm/int4/test_r276_accepted_counts.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/source-review/installed/vllm`

RESULT -> All 16 expected outcomes passed. The test executes the complete
installed `swap_states()`, `condense()`, and `_get_active_token_count()` methods,
and the actual accepted-count gather AST. Unrelated block-table operations and
empty auxiliary dictionaries are stubbed; no GPU backend is imported.

With old request counts A=1, B=4, swapping requests to B,A after an early D2H
correctly moves the host buffer to [4,1]. The subsequent installed gather uses
the old row map again and produces [1,4], assigning each request the wrong
accepted count. Late D2H landing makes that gather correct. Waiting before row
moves AND consuming the current-order counts is correct under either landing.

An important negative control: pure `condense()` alone was correct in this
test, because its old high rows remained intact. `condense()` followed by
`swap_states()` failed. Saying that every condensation alone necessarily causes
double permutation would overstate the finding.

VERDICT -> The R276 MRV1 source contains the defect targeted by
[vLLM PR 53919](https://github.com/vllm-project/vllm/pull/53919). The upstream
report includes one-token degeneration under heavier CPU starvation and labels
the issue MRV1-only. Neither TP=2 nor Intel hardware is required by this
mechanism. A continuously serial request occupying unchanged row zero does not
exercise cross-request permutation. Serial turnover, other traffic, and async
overlap require separate examination; Pi session counts alone do not measure
those transitions.

`patches/r276-accepted-count-order.patch` is a separate minimal source candidate.
It adds the earlier wait and removes the redundant gather together. It is not
GPU-qualified. Applying only the wait is not a valid repair. Candidate and
baseline source hashes are in `patch-manifest.json`. GDN phase initialization is
a separate mechanism and is not included in this patch.

## Backward state copies: reject a blanket guard pending stronger evidence

CONFIG -> Exact installed `postprocess_mamba_fused_kernel` scalar control flow,
interpreted on CPU with scalar load/store and copy recording stubs. No Triton
kernel was launched. Compare a separate candidate that suppresses src > dest.

COMMAND -> `python3 vllm/int4/test_r276_backward_copy_contract.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/source-review/installed/vllm/v1/worker/mamba_utils.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/cache-correctness/r276-mamba-backward-copy-experimental/v1/worker/mamba_utils.py`

RESULT -> For block size 16, computed=13, scheduled=4, draft=3, accepted=3, and
running-state column 1, the installed equations produce running position 14,
accepted boundary 16, and a copy from column 1 to column 0 with token bias 2.
The blanket guard suppresses that copy. This is consistent with publishing an
accepted earlier boundary from the later running scratch state; source-column
order alone does not prove the snapshot is wrong.

VERDICT -> Do not assume Sergio's three-site backward-copy guard is a valid
general repair for this R276 configuration. A guard that avoids contamination
may also suppress a required cache publication. The CPU result does not prove
either GPU outcome, but it is sufficient to withhold promotion. The candidate
patch is retained only in raw experimental evidence, not the repository patch
set. Official [issue 53505](https://github.com/vllm-project/vllm/issues/53505)
reports connector-dependent boundary reconciliation corruption; it does not
establish a universal rule prohibiting every backward column copy. Our original
serve had no CPU KV connector/offload configured.

## EAGLE drop: an API discrepancy is not yet a complete state proof

CONFIG -> Exact installed Mamba manager and coordinator, earlier installed
manager CPU regression, and Sergio's coarse/fine drop source.

COMMAND -> Review `vllm/int4/test_mamba_eagle_drop.py`, the installed
`single_type_kv_cache_manager.py`, and `kv_cache_coordinator.py` under the earlier
investigation directory `20260910T013217Z`.

RESULT -> The Mamba finder accepts `drop_eagle_block` but ignores it in both
coarse and fine branches. The coordinator deliberately gives Mamba no EAGLE
margin, documenting that the drafter has no Mamba layers. A manager-only drop
therefore changes the reconciled prefix boundary, rather than simply making
Mamba equivalent to full attention. Sergio also offers one hash-unit versus one
whole-page drops in the fine branch, which are materially different policies.

VERDICT -> The existing coarse patch and CPU test demonstrate lookup behavior,
but do not establish that the reused snapshot is contaminated in our workload.
Do not combine an unqualified fine-branch policy with the accepted-count or GDN
candidate. A serial warm-prefix replay can exercise this mechanism, unlike the
specific cross-request permutation case above. Capture boundary, accepted count,
state source/destination, and snapshot identity before selecting a drop policy.

## Suggested isolation order

Run the separate GDN phase candidate first against its tiny fresh-prefill
trigger and the Pi reconstructions. For the MRV1 accepted-count candidate, use
at least two requests with unequal accepted counts and forced decode/prefill
reordering, then the actual Pi history replay. Record row IDs and counts around
the copy event and reorder to establish that a failing request traversed the
mechanism. A modern MRV2 arm can be informative because its device-resident
accepted-count path differs, but passing it would not isolate this fix from
other backend changes. Retain matched cache, graph, MTP, KV dtype, model identity,
health, and teardown evidence for each GPU arm.

## Draft-only INT4 head: no proven TP shape defect from the available source

The retained Steve r62 source patch implements row-local symmetric quantization
of the already loaded vocabulary shard: `num_tokens, hidden = weight.shape`,
groups run along hidden size, packed weights are transposed, and scales have
shape `(hidden / group_size, local_vocab_rows)`. A shallow module copy preserves
the existing shard metadata while copying the buffer/parameter registries before
registering private packed buffers. There is no obvious requirement for a global
vocabulary offset inside this row-local packing step. This does not validate the
native GEMM layout or distributed logits path, and the retained patch is not an
independent byte-for-byte export of the running image.

Two attempted CPU-only image exports stalled before a container shim appeared;
their owned docker clients were terminated. No devices were exposed. Do not
equate Sergio's TP>1 prohibition for his separate implementation with proof that
this implementation has the same defect. A matched draft-INT4-off control remains
useful after the more specific GDN screen.
