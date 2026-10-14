# Independent original HC F32 source schedule

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: The full48 primary remains original-weight and own-state. It already
encodes Q8_1 activations where its native consumers require them; no missing
packer hypothesis is introduced. The first HC unsupported step is narrower:
full48_owned_composition_storage_v1.py:123-131 uses NumPy mean/sqrt and matrix
multiply, while the actual source44 native HC uses fixed F32 FMA/XOR32 order.
No original reference file or actual raw result is rewritten.

Actual SDK44 source root:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T165803Z-5xbo5_f0/source.
HC composition: sycl/src/kernels/hc_native_composition.cpp:30-50.
HC projection: sycl/src/kernels/hc_native_projection.cpp:8-47.
These are source anchors only, not a transferred runtime proof.

COMMAND: New software entry points in original_native_hc_f32_v1.py:

- hc_read_original(projector, own_residual, original_stem, epsilon, inject=True)
- hc_write_original(own_residual, own_block, own_inject)
- OwnedOriginalNativeHcF32: existing whole48 own-state composition subclass,
  replacing only HC read/write. Its tokens/state/routing schedule is inherited.
- owned_first_packet(own_mixed): unchanged independent Q8_1 encoder.

Root's next useful comparison is ONLY the original embedding-driven first HC
normalized/low-silu/gate/inject/mixed plus packet against admitted NUM38 targets.
Construct the projector/provider from independently admitted original weights;
pass no captured activations, states, routes or selected experts into primary
math. Captured NUM38 data are comparison targets. Existing root identity/closing
obligations apply. No provider or GGUF bytes were read by this author.

The primitive default uses CPU libm fmaf through the existing source-schedule
host primitive. It is intentionally a correctness candidate, not a fast full48
benchmark: firstHC is the next bounded step. A future use of the already
qualified bulk CPU dot helper may accelerate the same schedule, but is not
implemented or claimed by this module. No SDK45 observer/build is needed.

RESULT: HC norm accumulates squares per32 lanes with FMA, then simultaneous
XOR16/8/4/2/1 additions; F32 mean+epsilon precedes reciprocal sqrt. Normalization
keeps (R*norm)*rs as two F32 multiplications. HC down/up decode original Q8_0
weights and apply FMA to F32 operands, then XOR32. They do NOT use the other
consumer's Q8_0 x Q8_1 d0*d1 integer-dot algebra. Injection reads original F32
weights through the same FMA/XOR schedule. HC mix uses separate rounded product
and sequential stream addition; write uses FMA. The old HC refinement's fused
mix is not selected silently.

CPU command (only tiny synthetic arithmetic controls):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_original_native_hc_f32_cpu_v1 -q
```

Five controls PASS0.057s: a known lane-order cancellation differs from double
sum, fused product cancellation retains its tiny result, original-row projection
cannot enter a packet projection, norm shape/finite refusal and full-size
synthetic HC write. No native comparison or expensive original computation.

VERDICT: This removes a specific unsupported reference arithmetic shortcut;
it does not establish bitwise device equivalence or solve head NMSE0.007888.
Host sqrtf reciprocal and expf remain candidates for ordinary sycl::rsqrt/exp,
compiler mix contraction remains unobserved, and non-HC reduction/dispatch
frontiers remain. Full48 own-state and original inputs are preserved; fullmath
qualification and tolerance fitting remain false.
