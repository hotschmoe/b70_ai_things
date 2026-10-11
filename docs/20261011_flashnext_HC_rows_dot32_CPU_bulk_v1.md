# Explicit CPU bulk HC FMA/XOR32 backend

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: First independent HC math completed previously. Original residual was
bitwise; normalized/mixed errors were about1.3e-14/1.6e-14 NMSE, with one packet
scale byte different and code/sum bytes equal. Native mixed through the same
encoder reproduced the native packet. Do not fit that scale or borrow it.
Scalar candidate9ad227f9 and executed first-HC sources remain unchanged.

The existing source35 bulk executable already has the desired row schedule,
but its file/subprocess transport and old runtime admissions are unnecessary
for this narrowly bounded CPU DSO. The new helper reuses EXACT dot32 body from
hc_f32_arithmetic35_host_v1.cpp. Only transport and environment checks change.

COMMAND: Authorized CPU-only build in39992/C388, networknone, oneCPU512MiB,
no device/group/model mounts, source readonly; no SYCL compiler flags:

```sh
g++ -O3 -std=c++17 -shared -fPIC -mfma -fno-fast-math -ffp-contract=off -frounding-math /source/hc_rows_dot32_ctypes_v1.cpp -o /out/hc-rows.so -lm
```

Actual owned build root:
/tmp/b70-hc-rows-dot32-owned-CPU-build-v2-447241
ELF SHA83c289ccd11ef6fed8cb9c0563328795f1fbd665110edace7efcfb4fc710fb47.
Created/terminal inspections retain exact original ID, command, labels, mounts,
resource/device predicates; original CLI exit0 and normal exact-ID removal0.
Receipt.json retains a FAILED case-sensitive absence-string predicate. Actual
Docker returned lowercase `error: no such object`; retirement-supplement.json
rehashes that original failed receipt and preserves successful normal terminal,
removal and independently captured authoritative absence. No old receipt was
rewritten to pass. Tests stdout/stderr/command/SHA are retained separately.
An earlier uninspected --rm CPU build is also retained, not used as lifecycle
qualification. Host compiler absence was an initial pre-test refusal.

Explicit integration, no scalar/global monkeypatch:

```python
from original_native_hc_bulk_f32_v1 import RowsDot32
backend = RowsDot32('/tmp/b70-hc-rows-dot32-owned-CPU-build-v2-447241/hc-rows.so',
                    '83c289ccd11ef6fed8cb9c0563328795f1fbd665110edace7efcfb4fc710fb47')
owned = backend.hc_read_original(projector, own_residual, 'blk.0.hc_attn_',
                                 original_epsilon)
```

OwnedOriginalNativeHcBulkF32 additionally offers the explicit rows_dot_elf and
rows_dot_sha256 keyword constructor for future whole48 use. Provider/state/
non-HC math stay inherited. No whole48 run is authorized by this source note.

RESULT: C ABI checks FE_TONEAREST, MXCSR FTZ/DAZ clear and actual input/output
gradual fmaf probes once per call, before and after. It refuses bad settings;
it does not reset caller rounding. Weights and operands remain finite bounded
F32 arrays. Row-local fmaf accumulators and simultaneous XOR32 additions use
the original body. Shared/paired transport permits normal square sums too.
Python adapter requires explicit current regular ELF hash; ABI1 is checked,
file bytes rechecked per call. This is file-byte association, not process-memory
proof. Norm and projection enter the backend explicitly; host sqrtf/expf,
separate mix, original Q8_1 encoder and other math stay unchanged.

Five tiny synthetic tests PASS0.007s against current scalar: shared/paired
rows; FMA cancellation and gradual input/output; norm bytes; nonRNE refused
without changing caller mode; wrongELF/nonfinite refused. No provider/GGUF,
GPU or whole-model calculation. No full-model elapsed speedup claim.

VERDICT: Ready opt-in CPU acceleration for the next original HC comparison.
Device rsqrt/exp and compiler mix contraction remain unqualified; packet scale
is not fitted; fullmodel/numerical qualification and tolerance gate remainfalse.
