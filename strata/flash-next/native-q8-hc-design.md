# Native Q8 hyper-connection: first source increment

CONFIG -> Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf; existing selected
UD-Q4_K_XL; CPU-only preparation. No external checkout edits, compiler runs,
GPU access, new weights, or model-format conversion.

COMMAND -> Audit SYCL NativeDense, layer, prefill, verifier/final mixer and
CUDA native Q8 reference. Prepare patch0001 and run git apply --check against
the pinned clean source. Record source hashes and primitive specifications in
native-q8-hc-primitive-plan.json.

RESULT -> A deliberately partial, default-off source patch adds two projection
primitives to the SYCL kernel library only when
`STRATA_SYCL_Q8_HC_PRIMITIVES=ON`:

- Original GGUF Q8_0 weights multiplied by original F32 activations: decode
  each FP16 scale and signed code into F32, then F32 FMA. No BF16 intermediate,
  activation quantization, reordered weights, or source mutation.
- Original F32 injection weights multiplied by F32 activations, with the same
  row layout and reduction. No STRATA_HC_Q8_INJECT conversion.

One subgroup computes each output row with a fixed reduction tree. The kernel
uses no scratch allocations or atomics and requires an in-order queue.
Descriptor bounds reject unsupported dimensions and undersized arrays before
submission. Device/context accessibility and nonaliasing remain caller
contracts. This is a correctness-first primitive, not an optimized fused HC
kernel, and its summation order is not CPU-serial or CUDA-bit-identical.

No model call site is connected. The patch therefore DOES NOT fix complete
model hyper-connections or qualify prompt/decode/final-mixer fidelity. Do not
enable STRATA_HC_Q8 for a model run on the strength of this patch. Existing
Strata's option can load hc_q8 pointers while its SYCL fused kernel ignores
them; that existing behavior is not a supported activation mechanism here.

The fixture plan covers production down [K=10240,M=320], up [K=320,M=10240],
and F32 injection [K=10240,M=4], with token counts 1,2,4,8: 12 shape cases plus
edge cases. The largest Q8 weight image is 3481600 bytes and its F32 reference
expansion is 13107200 bytes. CPU FP64 sums of exact dequantized source values
are the proposed numerical oracle. Preregistered NMSE <=1e-6 and normalized
maximum error <=1e-4 are proposals, not observed results. Require exact repeat
and concatenated-one-token versus multi-token GPU output bytes, guard regions,
unchanged source bytes, cancellation values, extreme signed codes, and F32
injection values that BF16 would change. No fixture was run on a GPU.

Integration is a separate, reviewable increment:

1. Own source projections per stage, with explicit dtype/shape/stride and
   original-data receipts. NativeDense already uploads Q8 HC copies. Add
   original F32 injection ownership instead of assuming the packed BF16 copy
   is exact. Keep the BF16 compatibility image only for the disabled path.
   Prefer a SYCL-specific native HC descriptor/owner over reinterpreting
   WeightRef::hc_q8 as an unrelated dtype.
2. Compose normalization, down, SiLU, up, mixing and injection using one
   documented F32 contract. Preserve the original per-stream RMS definition,
   gate scaling, mean, residual injection and pending-write ordering. Compare
   every intermediate against the model math/CPU reference; projection parity
   alone cannot validate this composition. Add the optional prior residual
   write and final-mixer no-injection cases explicitly.
3. Wire ALL execution routes behind one default-off runtime option, failing
   closed if any required native tensor or supported route is absent:
   - `sycl/src/prefill/prefill.cpp`: prompt HC uses its own matrix paths;
   - `sycl/src/core/layer.cpp`: ordinary single-token HC and output mixer;
   - `sycl/src/core/verify.cpp`: attention and FFN halves, repeated/grouped
     tokens, final mixer;
   - `sycl/src/kernels/cuda/fused_gr.dp.cpp`: either a native composed dispatch
     or explicit separate primitive composition, never silent BF16 fallback.
   MTP remains disabled; its separate HC paths are unqualified until another
   increment covers them. Existing STRATA_QFUSE and GR_V3 behavior must not
   silently bypass the new native contract.
4. Exercise prefill-to-decode transitions and the same token through ordinary
   decode and one-row verifier before full model screening. Test two active
   sequences with separate buffers, repeated request histories, and return
   to a single stream. Qualify one card first, then the existing two-stage
   handoff. Additional state must belong to a stage/session, not globals.

VERDICT -> Concrete primitive source is ready for review, with exact source
and patch hashes. Applying it to an independent source copy, building, and
running GPU primitives remain parent-controlled work under bin/gpu-run. Full
native HC coverage is explicitly incomplete and is the next integration gate;
second-stage RAM mirrors are outside this increment.

## Parent source-compile follow-up

CONFIG -> Independent patch overlay; same pinned oneAPI2026.1 runtime image
39992d70, no GPU devices exposed. Full pair lease acquired for compiler work.

COMMAND -> icpx -fsycl -std=c++17 -O2 -c hc_native_projection.cpp, overlay
headers followed by pinned SYCL/source includes.

RESULT -> Exit0; SYCL object generated. Raw command/log/hash receipt:
/mnt/vm_8tb/b70/build/strata-q8-hc-primitive-20261009-fwx35fqd/compile-receipt.json.
External source and baseline binaries unchanged.

VERDICT -> Source compilation passes. Device execution, numerical fixture
checks, composed HC math and full model integration remain unqualified.

CPU validation follow-up: test_native_q8_hc_cpu.py passed the actual descriptor
helper (12 shapes, 10 rejection cases) under ASan/UBSan, and validated fixture
decoding for all 63488 finite FP16 scales, signed-code extremes, and an F32
injection value changed by BF16 truncation. These validate host contracts and
fixture references only; they do not validate SYCL arithmetic. Null stream is
valid (default in-order queue); null weight/input/output pointers are invalid.
The final patch factors the descriptor into the header and has SHA256
7edd863134d9a660d21fc28e4b061dd2568e6013fc58a67a529a0dc686263efb.

Final snapshot compile follow-up: patch7edd8631 factors descriptor checks into
the header for CPU testing. Earlier426bc4ac receipt covers its preceding
snapshot only. Current snapshot icpx SYCL object compile passes, no devices;
receipt: /mnt/vm_8tb/b70/build/strata-q8-hc-final-20261009-tc6wgt0y/compile-receipt.json.
GPU numerical/model qualification remains outstanding.

## Leased GPU projection follow-up, 2026-10-09

CONFIG -> Same pinned patch7edd8631, source fb58e0d and runtime39992d70.
Synthetic original-layout Q8/F32 fixtures; no full-model calls or expert banks.
COMMAND -> run_native_q8_hc_gpu.py with the final overlay and F04 native-q8-hc-v2
output. The controller rebuilds tracked source, verifies overlay/patch identity,
owns both leases, and pins each separate process with ZE_AFFINITY_MASK.
RESULT ->72/72 cases per card,12 production shapes with6 input profiles each;
22 invalid descriptor/queue cases per card;72 numeric negative controls and
one guard-corruption negative control per card. Exact repeat after independent
buffers, concatenated single versus multi-token bytes,128-byte guards and
unchanged weight/input bytes all pass. Worst NMSE3.0529638501619164e-10;
normalized maximum error1.747273261445363e-5, under frozen1e-6/1e-4 gates.
Normal process exit/removal, strict per-card and compiled two-rank P2P0 pre/post
health pass; kernel journal has no selected fault signature. Initial v1 failed
GPU discovery because the nonroot container lacked DRM groups; no HC kernels
executed and post-health passed. v2 adds existing device group IDs. Preserve v1.
VERDICT -> Projection primitives are qualified for these synthetic per-card
fixtures. This does not qualify actual-model tensors, composed HC, routing,
prefill/decode/verifier transitions, concurrent state, two-stage handoff or speed.
Raw receipts: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/.
