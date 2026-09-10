# XPU FP8 store scalar semantics

CONFIG -> Exact SGLang image bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf,
actual MHA pool store, FP16 K/V, FP32 zero-dimensional device scale buffers,
67 tokens, four KV heads, dimension256, permuted physical slots. No model.

COMMAND -> Original strict oracle failed on card0. Separate leased diagnostic
retained normalized pool inputs and compared division, identical-input casts,
and scatter independently, including power-of-two scales. Raw evidence:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-fp8-byte-diagnostic/card0/`.

RESULT -> Device division differs from CPU FP32-scalar division at7837 K and
6441 V half values. This changes156 K FP8 bytes and zero V bytes. All normalized
values exactly match rounding the scale to FP16 before division, then rounding
the quotient to FP16. CPU/device FP8 conversion on identical half inputs and
actual scatter versus device recast match exactly. Independent K=.125,V=.25
power-of-two fixture matches throughout. The original failed evidence remains.

The corrected oracle explicitly computes the measured XPU reference using
FP64 division by the half-rounded scale, then FP16 and FP8 rounding. It checks
the normalized input and all stored bytes exactly, retains original full-FP32
scalar comparison metrics, and requires the independent power-of-two store
gate. Attention tolerances and query1/query4 scale-sensitivity checks are
unchanged. This is a reference correction supported by stage-isolated evidence,
not a tolerance increase or backend modification.

Read descales remain the original FP32 float mirrors. For this fixture the
effective write divisors are K=.0298309326171875,V=.0210418701171875; reads use
K=.0298287533223629,V=.02104317769408226. Ignoring FP8 and result rounding, their
read/write ratios differ from1 by approximately -7.31e-5 and +6.21e-5. Attention
references therefore dequantize the actual expected bytes using original
FP32 read scales. They must not silently replace read scales with half values.

Installed Torch XPU `ATen/native/xpu/sycl/Loops.h:705-728` distinguishes a CPU
scalar (captured as an opmath value) from a device scalar (BinaryFunctor argument
types). Its `BinaryFunctor` accepts typed arguments before invoking the opmath
function. This is consistent with the measured device-scalar conversion.
Installed headers alone do not supply the full division dispatch and type
promotion implementation, so the numeric experiment is the decisive evidence.
Copied headers live in the raw diagnostic's torch-source directory.

VERDICT -> Explained, deterministic scalar precision behavior; no evidence of
random FP8 casting or pool corruption in this fixture. CPU tests pin the measured
boundary examples and difference counts, plus power-of-two and invalid effective
scale cases. Corrected XPU attention execution still requires its own leased run.
