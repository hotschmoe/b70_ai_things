# Native HC prompt and verifier source routes

CONFIG -> Pinned Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf;
patch0001+0002+0003 overlay; original selected UD-Q4_K_XL tensors unchanged.
CPU-only source preparation, no external checkout mutations or GPU access.

COMMAND -> Prepare patch0004-sycl-native-hc-prefill-verifier-routes.patch;
apply series to a private temporary overlay; compile actual shared dispatch
headers and invoke CPU mocks under ASan/UBSan. Reproduce:
python3 strata/flash-next/test_native_hc_routes_cpu.py.

RESULT -> Sequential application passes. CPU shared binding tests pass2
valid layer/head forms and12 rejected source type/shape forms. Write shape
checks pass8 row counts and26 capacity/count rejections. Mock launch receipts
validate5 contiguous chunk layouts (1,2,8,9,17 rows) and4 prelaunch null/empty
rejections. Device launchers are mocks; no arithmetic or model-path execution
was validated by this CPU test.
Patch SHA256: 18ddbdeb0db1951cc52ee836938c00f96d70db13d2ce38725136208e5089cb72.

VERDICT -> Concrete source now connects ordinary decode, prompt HC and
verifier HC through one F32 composition and one exact-source binding contract.
All explicit residual writes share a new F32 write primitive with the same
sigmoid/FMA contract as composed pending writes. MTP remains rejected.
SYCL TU compilation, native uploads, model route coverage, logits, state
isolation, handoff and serving are still unqualified. Source wiring is not
runtime fidelity evidence; retain the integration/model-use gate.

## Prompt route

Native mode makes gr_unfused() true, so both the region-size calculation and
carving reserve T*D original F32 xn workspace. This also disables all existing
write/norm, up/mix and control-vector/norm fusion in native mode. Shared F32
lo/gate/rs and output mixed/injection buffers remain stage/session-owned.
The prompt owner's carve establishes contiguous token-major arrays; unsupported
missing workspace or oversized row count is rejected. No pointer array is
interpreted as contiguous data.

Every attention/FFN read dispatches contiguous slices of up to8 rows to the
same hc_native_read_f32 used by ordinary decode. Native HC never consumes
xn16 or lo16. After HC completes in queue order, existing to_f16/to_bf16
functions derive mixed_h and mixed_bf (including an optional low BF16 part)
for downstream model operators. These conversions preserve the downstream
operator's existing precision boundary rather than feeding reduced precision
back into HC. The compatibility HC branch remains entirely behind the native
mode else clause.

Every prompt half writes its residual with the shared native write primitive,
chunked up to8 rows. Normed bookkeeping resets; each later native read computes
fresh per-stream RMS from the fully committed residual. Control vectors apply
after the explicit native FFN write; PLE and stage transfer receive committed
R. The extra F32 xn workspace costs40960 bytes per prompt token and must be
included in placement and chunk-size budgets. MTP draft_kv rejects native mode
independently, in addition to the existing MtpDrafter::load rejection.

## Verifier route

Verifier's existing Rt(t) expression proves its residual rows are contiguous:
R_+t*HC*N. Native mode checks the needed F32 workspace and valid row count.
A one-row call for each verifier row handles repeated/grouped windows without
assuming slot-state pointer arrays are matrices. xn/lo/rs have distinct row
slices. The raw up gate uses the owning session's BlockBuffers F32 gate scratch;
all calls are serialized on the same in-order queue and each read finishes
its mixed output before that scratch is reused. Two sessions must own separate
BlockBuffers as required by the existing session lifecycle contract.

The attention read never sees a native deferred previous FFN write: every
post(l,group) commits its FFN output explicitly. The FFN read explicitly writes
the preceding attention output before reading. Its projection outputs remain
in the verifier's independent inj2 row slices. The PLE boundary skips the old
previous-FFN materialization in native mode, preventing a second write.

Native mode disables fused-head residual handling. The final FFN write is
committed before the final native mixer, which invokes ordinary lm_head_mix
(or lm_head if its native head is absent) with each row's R/mixed pointers.
Earlier stages hand off committed R; bo/inj may still travel in the existing
handoff layout, but the next native stage does not apply those values again.
Control vectors apply only after explicit native writes. Native mode produces
no qfuse images; the normal downstream quantization runs. Incompatible qfuse,
gr_v3 and legacy HC flags remain rejected by the common policy.

## Remaining source and runtime gates

Parent must compile the full overlay in enabled/disabled builds. Changed
SYCL sources are layer.cpp, prefill.cpp, verify.cpp and hc_native_composition.cpp;
0003 uploader and MTP/policy changes still need their compile evidence.
GPU tests must qualify the new standalone write against the pending-write
contract, including exact repeat/chunked output bytes, guard regions and
unchanged source inputs. Shared dispatch CPU tests do not exercise queues.

Then require runtime receipts covering every native read/write and source
binding: each prompt half, ordinary halves, verifier halves, final mixer,
large-prompt chunk boundaries, PLE boundary, control vectors, stage handoff,
prefill-to-decode transitions and one-row verifier versus ordinary decode.
Check all intermediate tensors and full logits against declared tolerances,
then independent and concurrent histories, teardown and post-health. Failures
must abort the request; native mode never silently uses BF16 HC or retries on
partly updated state. MTP stays unavailable until its independent routes have
an additional qualified source increment.

## Exact-artifact dispatch follow-up

CONFIG -> Corrected0003 source-rank contract and F04 fresh header/payload
receipts. Norm source rank1[10240] must not be confused with HC's logical4x2560
activation layout or its normalized row view[10240,1].

COMMAND -> Regenerate shared dispatch checks with explicit source rank and
actual shape metadata; derive CPU fixtures from every fresh inventory tensor
and cross-check them against the parent's original-payload SHA256 receipt.
Run test_native_hc_routes_cpu.py under CPU ASan/UBSan.

RESULT -> Current0004 SHA256:
abc9772abf3c363a57a1a1a040f9ffff45d5dc964cd7fd09782ad3ec8690dfd5.
Actual387 descriptor checks and97 layer/head binding forms pass, alongside
12 malformed binding,8 write shape,26 shape rejection,5 chunk layout and4
prelaunch rejection cases. The source audit checks dtype, original rank/shape,
bytes, source offsets and bounds before dispatch. Old synthetic norm metadata
is superseded; no source math or primitive arithmetic changed.

VERDICT -> CPU contracts now use the exact selected artifact's actual source
metadata. Header tests with mocked device launchers remain insufficient to
prove SYCL compilation, uploads, device arithmetic or whole-model fidelity.

Stage-bound follow-up: the same F04-derived CPU test now exercises7 actual
HC effective-bound contract cases. A24/24 split expects192 first-stage source
tensors and195 final-stage tensors including head. These are source descriptor
counts only; upload ownership must still be observed at runtime.0004 hash is
unchanged; its prerequisite0003 is now207642b5.
