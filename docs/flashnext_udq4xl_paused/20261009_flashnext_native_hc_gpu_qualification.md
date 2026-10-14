# FlashNext native HC device qualification, 2026-10-09

CONFIG -> Pinned Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf,
projection patch7edd8631 and composition patchbde63767. Runtime/build image
sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7;
oneAPI2026.1 compiler; fixed kernel7.1.0-070100-generic. No model inference,
expert banks, conversion, storage relocation, clock changes or serving promotion.
Original selected Unsloth UD-Q4_K_XL files remain unchanged. Fixtures synthesize
the same Q8_0/F32 types and production HC geometry; they do not sample model
tensors or prove complete model fidelity.

COMMAND -> The parent controller owns both leases through bin/gpu-run,
rebuilds source with no devices exposed, verifies the independent overlay
against the source patches, then launches one process per pinned card.
Existing render/video group IDs are supplied to each nonroot container.
SYCL_CACHE_PERSISTENT=0; in-order Level Zero queues. Strict per-card finite
matmul health and ten compiled P2P0 two-rank collectives bracket each run.

```sh
python3 strata/flash-next/run_native_q8_hc_gpu.py \
  --overlay /mnt/vm_8tb/b70/build/strata-q8-hc-final-20261009-tc6wgt0y \
  --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/native-q8-hc-v2
python3 strata/flash-next/run_native_q8_hc_gpu.py --composition \
  --projection-receipt /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/native-q8-hc-v2/receipt.json \
  --overlay /mnt/vm_8tb/b70/build/strata-native-hc-composition-20261009-pi3daji7 \
  --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/native-hc-composition-v1
```

Replay must select new output paths. The original directories are immutable
experiment records. The controller stores its source snapshot, fixture sources,
compile/launch commands, binary/source/patch hashes, JSONL case records, kernel
journal, process states, removal and health logs. Composition requires a passing
same-source projection receipt. No previously compiled object is substituted.

RESULT -> Both complete numerical runs pass on each B70:

| Gate, per card | Projection | Composition |
| --- | --- | --- |
| Cases | 72:12 shapes x6 profiles | 48:4 row counts x3 pending modes x2 injection modes x2 profiles |
| FP64 reference rows | Complete projection outputs | 344 intermediate stage records |
| Worst NMSE | 3.0529638501619164e-10 | 2.8520817958328582e-11 |
| Worst normalized maximum error | 1.747273261445363e-5 | 3.533520650739953e-5 |
| Exact repeat after independent buffers | PASS | PASS |
| Exact batch versus concatenated single rows | PASS | PASS |
| Guards and source/input immutability | PASS | PASS, with declared in-place residual writes |
| Numeric negative controls | 72 | 344 |
| Invalid projection descriptor/queue cases | 22 | Separate CPU composition descriptor gate |
| Normal exit/removal, pre/post health | PASS | PASS |

Preregistered gates were NMSE<=1e-6 and normalized maximum error<=1e-4,
with normalization max_abs_error/max(1e-6,max_abs_reference). No gate was widened.
CPU references decode exact FP16 scales and signed codes, preserve F32 source
values and accumulate in FP64. All output rows must be finite. Guard corruption
and deliberate output perturbations are rejected; absent/duplicate case or stage
records cannot pass the controller coverage check.

Projection profiles include signed-code extremes, cancellation, impulse inputs,
positive/negative/zero scales and representative finite subnormals. Twenty F32
injection cases per card explicitly differ from BF16-rounded controls.
Composition covers pending writes absent, out of place and in place; injection
present and final-mixer absent; ordinary and epsilon-sensitive residual inputs.
Checks expose pending residual, per-stream RMS, normalized activation,
down/pre-SiLU, post-SiLU, raw up gate, injection and mixed output. Pre-SiLU down
is recomputed from the composed normalized activation, rather than captured
inside the composition launch sequence.

Initial projection v1 failed GPU discovery because Docker did not inherit the
host user's DRM groups. No HC kernel ran; cleanup and post-health passed. Failed
receipt is retained. v2 adds the actual device group IDs and makes no arithmetic
change. Both passing runs have no selected fault signature in the kernel journal.

VERDICT -> Source-preserving HC projection and composition math pass these
bounded synthetic per-card gates. This is the first device-arithmetic evidence
for the default-off native path. It does not qualify original tensor upload,
full model routes, state reset, teacher-forced logits/quality, shared write
integration, two-device handoff, tiered experts, prefix reuse, concurrent
coherence, latency or shelf promotion.

The primary serving-development lane is now the pinned Strata native HC path:
it already supplies layer splitting, shared expert caches, slots and per-stage
in-memory prefix snapshots, so its known fidelity/ownership gaps are narrower
than the audited alternative ports. This is a development allocation, not a
backend performance verdict. Keep pinned llama.cpp for first-divergence
diagnostics and independent GGUF math comparisons; its history drift prevents
using its full-model output as an unquestioned authority. Independent CPU
exact-source references remain necessary. See the
[backend source audit](20261009_flashnext_backend_source_audit.md) for alternatives
and the [integration plan](../strata/flash-next/native-hc-integration-plan.md)
for remaining route/ownership gates.

Next required work: compile and numerically qualify the new shared residual
write; validate all native upload/source receipts; compare prompt/decode/verifier
and final-mixer routes on one card before the two-stage path. Then implement
stage-specific host mirrors and qualify complete prefix state reuse, one/two/four
streams and the bounded six-stream stress case. Clean matched latency work and
the verified shelf remain downstream gates.

## Shared write, source audit and full build follow-up

CONFIG -> Source native HC routes plus immutable generation HC/PLE/mirror
patches; no full Strata model inference. COMMAND -> native-hc-write-v1 GPU
fixtures; current GGUF header/source tensor audit; full source builds v1/v2/v3.
RESULT ->21/21 shared-write cases and15 rejection cases per card pass. Row
counts1/2/4/8/9/16/17 exercise shared chunk boundaries, exact single/chunk,
standalone/pending and repeat bytes. Worst NMSE6.25260364053364e-16 and
normalized maximum error6.158421892255321e-8. Guards, input immutability,
normal teardown and full pre/post-health pass. Source audit reads all387 actual
HC tensors and receipts SHA256/type/rank/shape/bytes. Original norms are rank1
[10240], not the initially assumed rank2[2560,4]; that source loader/binding bug
is corrected. All96 original injections are BF16-exact; some97 norm tensors
are not. Split ownership now follows explicit/effective stage bounds and
requires explicit trimmed native splits. Full CMake v1 fails PLE span type,
v2 fails shadowed SYCL verifier declarations, v3 succeeds with separate fixes
0008/0009. Prior failed receipts remain immutable.
VERDICT -> Shared write synthetic math and full-engine compilation pass.
Source upload/model quality, mirrors, prefix reuse, concurrency and speed remain
unqualified. Additional exact PLE key/value/conv GPU fixture is prepared; no
PLE device pass is claimed here. v3 receipt:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T050248Z-5ucibsj0/receipt.json.

## Original-source PLE device follow-up

CONFIG -> Original-layout synthetic Q8 key[2560,10240], value[2560,2560],
F32conv[4,10240], norms/embedding/residual; compiled v3 source and frozen gates.
COMMAND -> run_native_q8_hc_gpu.py --ple, native-ple-v1. RESULT ->7/7 cases,
70/70 FP64 intermediate checks and57 history rows per card pass;70 numeric
negative controls/card, positive reduced-precision controls, exact repeats and
chunk/single projection bytes, guards and immutable source/input bytes pass.
Worst NMSE1.2264844235673818e-13; normalized maximum error4.5531998639683306e-7.
Both normal exits/removals and strict per-card/compiled P2P0 pre/post-health pass;
no selected fault signature. VERDICT -> Synthetic source-preserving PLE math
passes; actual model ownership/history callsites and prefix state remain
unqualified. Fixture owns history-shift protocol; it does not observe the
engine's real cache/reset/slot behavior.
