# Native expert E2 operator fixture and isolated link

CONFIG -> Actual SDK52/source012115Z-_kwur0ww, receipt8f6c09aa. Candidate is
cache_source35's expert-pair-reuse-source-proposal-v1.json6ee8e6b0 and fresh
constructed iq_kernels.dp.cppedea6431. No model, engine, prefill or GGML provider.
The new cached flag requires two processes, original OFF0 then ON1. No setter
or flag mutation can pretend to change a captured kernel.

COMMAND -> Prepare the candidate with the existing
prepare_expert_pair_reuse_v1.py --output NEW_SOURCE_DIRECTORY. Compile only its
new IQ object, using actual52 build.ninja750 flags/includes, then link this
fixture with that object FIRST, a COPY of original52 libstrata_kernels.a with
exactly iq_kernels.dp.cpp.o removed, original52 libstrata_core.a, and ze_loader.
Original archives and every frozen engine remain read-only. Root must record
current pre/post archive/object/source SHA and the exact ar-member roster:
old IQ count exactly1 before deletion and exactly0 after; no original IQ symbol
may be loaded. No whole-eight-target engine rebuild precedes operator evidence.
The literal compile/link argv and input identities are in the small source plan.
The author has read only source/header/Ninja/receipt metadata, not archives.

GPU root uses the established one-card lease, strict per-card health and fresh
kernel window, full /dev/dri exposure and the matching ZE_AFFINITY_MASK.
B70_IQ_EXPECTED_PCI must equal the actual assigned BDF; the helper requires
LevelZero, host-USM and the supported PCI query before expert work. Invoke:

    STRATA_EXPERT_PAIR_REUSE=0 fixture 0 --output /results/off
    STRATA_EXPERT_PAIR_REUSE=1 fixture 1 --output /results/on --compare /results/off

OFF and ON must be separately executed fresh processes with the same exact ELF,
fixture source, synthetic data, original environment and card. Root records
actual command/env/return-code and original terminal/EOF/removal. Do not inherit
old model or kernel performance authority from this compile.

RESULT -> Source-only standalone fixture. Four type pairs, two weight storage
modes, five directed plans =40 cases per process. No build/GPU run by author.
The original API native_expert_grouped handles the complete GU/SwiGLU/Q8/down
chain. Phase1 and phase2 are supported benchmark-only public selectors, not
invented exported GU/down symbols. row_entries and GU/down kernels are internal.

VERDICT -> Existing operator/lifecycle pieces are sufficient. Original
native_grouped_parity supplies the synthetic quant layout precedent, but its
broad type/shape sweep and destructor are not reused. This new bounded owner
queues-drain before heap graph deletion and USM free; failed drain exits86 while
retaining storage, failed free exits87. Host upload vectors outlive that owner;
readback errors retain before temporary host buffers are released. No model
load, tensor-row access, kernel arithmetic alteration, or original-reference
substitution exists. Actual bitwise and latency qualification remain pending.

## Exact dimensions and storage

n_embd2560, n_ff640, tokens4, cap_entries4, cap_groups2, grid_groups1.
Default original lane selection remains8; no lane/row sweep or AMD mode flag.
Original q8_1 input is four rows *80blocks *36bytes =11520bytes. The scratch
contract is three aligned F32 arrays of10240bytes plus aligned q8_1 bytes3072:
33792bytes total. Complete output is40960bytes, including untouched destinations.
These are buffer extents, not measured memory traffic or bandwidth.

| GU/down IDs | Original formats | GU row bytes | Down row bytes | Expert blob bytes |
| --- | --- | ---: | ---: | ---: |
| 12/7 | Q4_K/Q5_1 | 1440 | 480 | 3072000 |
| 12/8 | Q4_K/Q8_0 | 1440 | 680 | 3584000 |
| 13/7 | Q5_K/Q5_1 | 1760 | 480 | 3481600 |
| 13/8 | Q5_K/Q8_0 | 1760 | 680 | 3993600 |

Each case has two independently generated expert blobs, either device USM or
sycl::malloc_host USM with the same original bytes. Types12/13 use256-value
blocks of144/176bytes; types7/8 use32-value blocks of24/34bytes. Random packed
codes/subscales are valid; finite small original FP16 scale/min fields follow
the existing parity generator. F32 inputs are independently synthesized and
packed by the unchanged actual quantize_q8_1_rows API.

Plans: same expert E2 distinct token IDs; same expert E2 duplicate activation;
different experts [1,1] fallback; same expert [3] fallback; offset paired group
[1,2]. Token indices and destination rows are permuted/noncontiguous. Each case
saves full scratch, output and packed input for initial and changed inputs.
ON compares every byte with original OFF before writing its own artifact.
Synthetic original blob/input bytes are also saved and matched per type pair.
Within both processes, captured full-chain graph replay must match a direct
call, including after changed input;40 changed graph cases remain mandatory.

For positive distinct-token E2 cases only, each storage/type pair warms three
calls and records seven host-submit-plus-queue-wait samples per phase: phase1
GU+SwiGLU+q8_1, phase2 down against that original packed intermediate. Readback
and disk comparison occur outside timings. These are phase observations, not
pure GU timings, device execution, mapped-RAM bandwidth, or whole-model speed.
Actual bitwise failures are preserved; no tolerance fitting is permitted.
