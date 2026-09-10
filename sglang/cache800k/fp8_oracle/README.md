# SGLang Triton calibrated-scale numeric oracle

CONFIG -> SGLang source pin 2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed;
synthetic sequence length 67, Q heads 24, KV heads 4, dimension 256, FP16 inputs,
E4M3FN cache, page size 1, nonidentity slot mapping in 160 physical slots.
No model, weights, graphs, TP, MTP, server, or calibration-identity claim.

The distinct scale values are numeric fixtures taken from the prior layer3
artifact. The existing loader's install_validated_plan installs persistent FP32
buffers and exact float mirrors in a synthetic layer. This intentionally tests
its buffer contract without invoking or bypassing the real model provenance
gate. It does not make the old artifact valid for the current weights.

COMMAND -> CPU run in existing immutable R276 image, no devices/network,
2 CPUs and 4 GiB, with source and cache800k mounted read-only:

```text
python3 /candidate/fp8_oracle/oracle.py --device cpu \
  --source /source --loader-root /candidate/calibrated_kv
```

RESULT -> CPU exact-store-byte reference, untouched slots, clone protection and
independent K/V read-scale sensitivity pass. Raw evidence:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-fp8-oracle/cpu.json`.
The same fixture has 255 K bytes and 203 V bytes that differ when replacing the
actual FP16 division intermediate with a direct FP32-division-to-FP8 conversion.

VERDICT -> CPU/source preparation passes. No XPU execution or Triton numeric
qualification has occurred. A leased single-card run in the fresh image is the
next gate after parent-managed per-card and collective health.

## Actual paths covered

The XPU MHA FP8 store is PyTorch, not a Triton store kernel. It divides FP16 K/V
in place by scalar tensors, converts to FP8, reinterprets storage as uint8, then
scatters rows. The oracle calls actual set_kv_buffer and _store_kv_layer methods
on a minimal NHD storage fixture. Their normal _set_kv_buffer_impl dispatch is
used on XPU. CPU mode extracts these exact source methods, forces the same
naive scatter branch, and stubs only diagnostic/location/capture metadata.

The byte reference includes FP16 intermediate rounding. Untouched slots retain
distinct K/V sentinels. The direct pool API mutates input tensors; the oracle
checks this, protects caller inputs with clones, and checks the actual backend
forward_extend source contains paired K/V clones before the scale-aware write.

On XPU, actual decode_attention_fwd handles q1 with two KV splits. Actual
extend_attention_fwd handles q1 and q4. Both compare against a CPU FP32 softmax
reference. Decode consumes all 67 cached tokens. Extend consumes a dequantized
cached prefix and an original FP16 fresh suffix, matching the backend's two-stage
contract; it must not descale the fresh suffix again. Lengths cross the 64-token
attention tile boundary. Page-size-1 slots are permuted, so this does not test a
larger paged layout or its page-boundary arithmetic.

Each read runs with correct mirrors, only K descale doubled, and only V descale
doubled while cache bytes stay unchanged. All three outputs must match their own
reference and the altered scales must measurably change output. This detects
ignored, swapped, or coupled K/V read scales. It is a synthetic sensitivity test,
not an accuracy recommendation to change scales.

XPU tolerance is rtol 0.03, atol 0.015 against FP32 attention over the exact stored
FP8 values (or fresh FP16 suffix). Raw maximum error and RMSE are recorded.
Exact stored bytes and untouched slots have zero tolerance. Review measured
errors before interpreting a pass; no speed/stability claim follows from this.

## Parent launch preparation

Use the fresh triton-dense image once its required stack health has passed. Mount
this directory, calibrated_kv, and the exact source tree read-only. Under the
project GPU lease and the workload's matching single-card device pin, run:

```text
python3 /candidate/fp8_oracle/oracle.py --device xpu \
  --source /source --loader-root /candidate/calibrated_kv
```

This README deliberately contains no unleased Docker GPU command. The parent
owns card allocation, health, image identity, and teardown. The oracle checks
installed pool/backend/attention source bytes against the supplied tree before
using them, records hashes, and imports no archived ABI-specific code. A source
mismatch is a provenance gate, not a reason to replace the check with a warning.

Independent CPU/source review confirmed the actual decode/extend signatures,
24/4 grouped-head shape, unique permuted slots, scaled-store reference, and
cached-prefix/fresh-suffix treatment. No source blocker was found. This does
not cover execution of the full backend wrapper: clone protection is checked
at its AST call site, while the numeric read invokes its actual kernel entry
points directly. XPU execution remains unproven until the leased run.
