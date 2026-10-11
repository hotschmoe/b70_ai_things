# Source41 materialized indexer candidate V2

CONFIG -> Source40 remains unchanged. The isolated source40 helper completed
18 frames and independent public admission (parent 5cbea445..., binding953b5fa1...).
Its actual global-half-store/separate-load path matched IEEE binary16 expansion
and complete state across all three APIs. This is fixture-conditional component
evidence, not normal model, reference fidelity, cache or compiler-cause authority.

COMMAND -> Apply the new 0041 candidate only to a fresh source40-derived SDK.
The proposal inherits the exact d87740 source40 build recipe and adds one patch.
There are 69 explicitly hashed patched sources, 31 earlier added headers, 41
ordered patches and all eight ABI outputs plus a fresh upload oracle required.
No compile, model, GPU, registry change or old SDK edit by the source agent.

RESULT -> STRATA_QSA_INDEXER_MATERIALIZED_HALF is resolved once: absent/0 is
OFF, 1 is ON, every other value refuses. ON additionally requires the native
indexer path, so the legacy standalone fallback cannot silently bypass it.
OFF returns the original raw pointer before pointer validation, queue access,
kernel launch or scratch access; conditional arenas preserve the old carve order.

ON stores raw keys into caller-owned global half storage, then a separate kernel
loads them into caller-owned expanded F32. The unchanged production append
functions receive expanded values; original projections/raw, gamma, epsilon,
RoPE, positions, pooling, norm, score/selection and remaining math are unchanged.
Every row uses the original relative device position before raw is read. Rejected
commit positions (<0 or >=max_cells) write zero scratch and never read raw.
Separate Store and Load require the actual owning in-order queue. No allocations,
host synchronizations or graph objects are introduced within these functions.

Actual SYCL production caller inventory:

| Owner | Original call coordinates | Scratch and order |
| --- | --- | --- |
| QsaBuffers standalone | layer.cpp:1286 | one 128-element half+F32 pair; qsa_buffers_bytes/init |
| Verifier forward | verify.cpp:1236/1243 | nQ*MT*ID arena; original qi/tb offset and kStepCount stride |
| Verifier accepted commit | verify.cpp:1981/1985 | same arena; original commit+2, stride1, MT capacity |
| Verifier grouped accepted commit | verify.cpp:2961 | original qsa_index*MT+u and commitb owner-row offset |
| Prefill batch/single fallback | prefill.cpp:3034/3040 | T*128 QSA overlay arena; qsa_set_bytes and carve in same order |

Verifier existing arena count/carve/initial zeroing and graph retirement/free
own the new spans. Prefill existing count/carve/bytes_needed/relayout and queue
retirement own its spans, with no cache-state or checkpoint payload additions.
Standalone arena reserves six bytes per key plus conservative alignment allowance.
All sources are pinned to their actual SDK40 bytes, not inferred CUDA coordinates.

Persisted SessionConfig gains a conditional ON key before its existing config
fingerprint. OFF omits the key, retaining the prior fingerprint. ON/OFF restoration
must be negatively tested against actual saved headers before any cache promotion.
A process cannot change resolved mode; in-process live cache and stage ownership
remain subject to fresh source41 qualification. No historical source40 cache or
model proof is transferred.

Fresh qualification order:

1. Fresh source/flags/archive/header/all-eight-ABI compile and upload390 oracle,
   normal owned teardown, per-card/compiled health, kernel and newfour/pages.
2. Fresh linked-source41 isolated own-raw/gamma control: OFF raw, ON actual half
   bytes/expanded, all three APIs, direct plus two replays, accepted counts0/1/2,
   grouped owner rows, invalid positions and wrong raw-store control. All buffers
   and graphs retired. The current source40 fixture is independent input provenance.
3. Fresh source41 OFF model versus current source40 matched OFF targets: full48
   and head plus29 QSA fields; numerical equivalence is an actual gate.
4. Fresh source41 ON model prefix4, whole2/1/1 windows, own-reference inputs only.
   Actual native captures are comparison targets only. Compare storage first,
   then independently owned layer3 trajectory/full48/head descriptively. No fitted
   tolerance, original-reference patch or promised head improvement.
5. Fresh natural functional/repeat and concurrent/source-state isolation. Full
   cached-prefix/cancellation/eviction/persisted mode-refusal tests are separate
   requirements; source40 V11 cache campaign continues unchanged meanwhile.

VERDICT -> Reviewable source candidate and fresh build proposal only. Nothing
is runtime-qualified or promoted. Compiler lowering cause, latency and full
model arithmetic remain false until their own actual evidence exists.

CPU:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_indexer_materialized41_candidate_cpu_v2 -q

V1 is held and unchanged. V2 mirrors Verifier owner fields into both canonical
and SYCL public headers. All wrappers pass scratch bases and integer offset/capacity;
no null pointer arithmetic occurs while OFF. ON validates the complete bounded
owner extents and offset/count before forming any scratch slice pointers. Two
additional controls inspect every actual wrapper and both header layouts.
