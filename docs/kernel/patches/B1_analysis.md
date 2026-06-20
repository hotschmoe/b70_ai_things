# B1: Drop the always-on symmetric SRC zero-point in int4_gemm_w4a8.h

Target kernel: `csrc/xpu/onednn/int4_gemm_w4a8.h` (oneDNN s8-activation x u4-weight ->
f16 grouped matmul, decode-critical at m=1).
Reference (clean) kernel: `csrc/xpu/onednn/int8_gemm_w8a8.h` (symmetric, scales only,
no zero-points anywhere).
Patch: `int4_gemm_w4a8_drop_src_zp.diff` (apply with `git apply` or `patch -p1`).

--------------------------------------------------------------------------------
## 0. The symmetric reality of our checkpoints (why this is safe)

Our production decode path feeds `int4_gemm_w4a8` from the fused op
`dynamic_per_token_int8_quant` (`csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp`).
That op is SYMMETRIC-ONLY by construction:

    // dynamic_per_token_int8_quant.cpp
    TORCH_CHECK(use_sym_quant, "...: only symmetric quant is implemented");
    // returns q [..., K] int8 (s8), scale [..., 1], zero_point [..., 1] int32 == 0

So at runtime:
- `mat1` is `at::ScalarType::Char` (s8)  -> `jd = joint_dtypes_t::s8_int4`.
- `m1_zp` is an all-zero int32 tensor of shape `[M, 1]`  -> `m1_zp.numel() == m`.

An all-zero src zero-point contributes nothing to the GEMM math (s8 src, zp=0).
oneDNN still emits an extra s32 zero-point-correction term across K when a src zp
attribute is configured. At decode (m=1) that is pure, repeated overhead on every
token. Dropping it is mathematically identical for the symmetric case.

This mirrors `int8_gemm_w8a8.h`, which sets only `DNNL_ARG_SRC` / `DNNL_ARG_WEIGHTS`
scales and never sets any zero-points (its header comment: "Symmetric-only (no
src/weight zero points)").

--------------------------------------------------------------------------------
## 1. Exact lines changed

The patch removes the SRC zero-point in all three places it appears, and adjusts
the cache key. Concretely, against the current file:

(a) f_attr lambda, per-token branch (was ~lines 71-75): delete
    `pattr.set_zero_points(DNNL_ARG_SRC, (1<<0)+(1<<1), {1,k}, s32);`

(b) f_attr lambda, per-tensor/else branch (was ~lines 82-86): delete
    `pattr.set_zero_points(DNNL_ARG_SRC, 0, {}, s32);`

(c) the runtime arg binding (was ~lines 143-150): delete the
    `matmul_ext.set_attribute(arg_off++, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
    m1_zp.data_ptr(), ...)` block.

(d) the cache key (was ~lines 119-120): change
    `zp_group_size = (m1_zp.numel() << 32) | m2_zp.numel();`
    to
    `zp_group_size = m2_zp.numel();`   (src-zp half pinned to 0 -- see section 2).

What is intentionally LEFT UNCHANGED:
- All `DNNL_ARG_SRC` / `DNNL_ARG_WEIGHTS` *scales* (both f_attr and runtime binding).
- The entire WEIGHT zero-point path (`m2_zp`), including the u4 zp=8 centering --
  see section 4. This patch is SRC-zp only.
- The op signature `int4_gemm_w4a8(... A_zp ...)` in `torch_bindings.cpp`. `m1_zp`
  is still accepted for ABI compatibility; it is simply no longer consumed.

The `arg_off` counter still self-increments correctly: removing one
`set_attribute(arg_off++, ...)` call drops exactly one slot. The trailing
`matmul_ext.execute(strm, engine, ..., arg_off)` uses the live counter, so the
attribute-argument count stays consistent. No hard-coded slot offsets exist in
this function, so nothing else needs renumbering.

--------------------------------------------------------------------------------
## 2. THE CACHE-KEY TRAP (verdict: must pin src-zp half to 0; we do)

The primitive is LRU-cached in `onednn_ext.h`
(`matmul_primitive_cache_t<Tt,Ts,F>::get`, lines ~764-821). The key is built at
lines 781-789:

    auto pri_key = concat(
        src_strides, wei_strides, m, n, k, int(b_type),
        scale_group_size, zp_group_size);

Crucially, `f_attr` (the lambda that decides whether `set_zero_points(DNNL_ARG_SRC,
...)` is called) is NOT hashed into the key. `f_attr` only runs on a cache MISS
(line 803, `f_attr(pattr);`). On a HIT the cached primitive is returned verbatim,
attributes and all. Therefore correctness depends entirely on the explicit key
fields distinguishing a "src-zp" primitive from a "no-src-zp" primitive.

Before the patch, `zp_group_size = (m1_zp.numel() << 32) | m2_zp.numel()`. Every
call sets the src zp, so the upper half always equaled `m` and there was no
ambiguity -- but also no symmetric variant existed.

After the patch we always build WITHOUT src zp. The danger is a FUTURE asymmetric
per-token call: it would have `src_zp.numel() == m`, and `m` is already a key field
-- so `m` ALONE does NOT disambiguate "src-zp present" from "src-zp absent". If we
left the old encoding, an asymmetric call at the same (m,n,k,strides,b_type,
sc_group_size, m2_zp.numel()) could either (i) collide and reuse a symmetric
primitive lacking the src-zp correction (silent numerical error), or (ii) be
distinguished only by accident.

Fix (in the patch): pin the src-zp half of `zp_group_size` to 0 ->
`zp_group_size = m2_zp.numel()`. This makes the key field that encodes src-zp
state read "no src zp" for every primitive this function now builds. The encoding
becomes a faithful, intentional flag: any future asymmetric path MUST set the
upper 32 bits back to `m1_zp.numel()` (the patch comment says exactly this), which
will then key-separate symmetric from asymmetric primitives. Pinning to 0 is both
necessary (otherwise a future asym path aliases) and sufficient (the symmetric
path is the only producer today, and it always emits 0).

Verdict: SAFE with the `zp_group_size = m2_zp.numel()` change included. Do NOT ship
the f_attr/arg-binding removal without the cache-key change.

Note: `sc_group_size` is untouched -- scales are unchanged, so the scale half of
the key is still correct.

--------------------------------------------------------------------------------
## 3. REF-FALLBACK RISK (verdict: omitting src-zp does NOT force ref; it helps)

oneDNN's GPU matmul jit/gemm implementation supports s8 (or u8) src with grouped
low-precision weights using src scales + weight scales + weight zero-points. Source
zero-points are an OPTIONAL attribute; omitting them removes a code path, it does
not remove a feature the optimized kernel requires. The clean `int8_gemm_w8a8.h`
already runs s8s8 -> f16 on this same Battlemage oneDNN with scales-only and no
zp, and it is our fast path -- direct evidence that "no src zp" stays on the
optimized impl on this hardware.

Adding an (all-zero) src zero-point can only ADD work: oneDNN materializes the s32
zp-correction term (sum over K of weights, scaled by the src zp) even when the zp
is zero, because the attribute presence -- not its runtime value -- selects the
code path at primitive-create time. So omitting it is strictly <= the cost of
keeping it; it never bounces a supported config to `ref`. Independently, IPEX
guidance (see section 5) states symmetric+int8 is the recommended high-performance
combination precisely because zero_point handling costs performance.

Verdict: no ref-fallback risk; expected to be neutral-to-faster, never slower.

--------------------------------------------------------------------------------
## 4. WEIGHT u4 zp=8 ASSESSMENT (secondary; assessed, NOT included)

What it is: in this kernel the weight is stored as u4 (`memory::data_type::u4`),
and the symmetric weight path passes a scalar zp of 8
(`zero_points = torch.tensor([8], int8)` in the test). That 8 is NOT a real
asymmetric quantization offset -- it is the u4->s4 STORAGE CENTERING: the true
signed weight is `s4 = u4 - 8`. So oneDNN subtracts 8 from every u4 weight to
recover the signed value before accumulation. It is doing real, useful work; it is
not redundant the way the all-zero src zp is.

Can it be dropped? Only by switching the weight memory dtype from u4+zp8 to a
native signed 4-bit type (`memory::data_type::s4`) and storing already-centered s4
weights. Then the zp=8 attribute could be removed entirely. Feasibility hinges on:
(1) oneDNN GPU matmul actually supporting `s4` weights with grouped scales on
Battlemage (the codebase only ever uses `u4` for int4 weights -- see every
`onednn_types_mapper<*_int4>` in `onednn_ext.h`, all map weight -> `memory::
data_type::u4`); and (2) re-packing the quantized checkpoints from u4 to centered
s4 (a format/repack change touching the quant scripts and the stored weights, not
just this header).

Value: the weight-zp correction is a per-GROUP term folded into the weight
dequant; at decode it is far cheaper than reloading weights, so the expected
speedup is small. Risk is high (new dtype path that may NOT be as well optimized
-> possible ref fallback; plus a checkpoint repack and a coordinated u4->s4 change
across `int4_gemm_w4a16` too, which shares the u4 assumption).

Verdict: do NOT bundle with this patch. The src-zp removal is zero-risk and self-
contained; the weight-zp removal is a separate, riskier format change with modest
upside. Recommend it as a follow-up experiment gated on confirming oneDNN s4
weight support on Battlemage (a 1-shape microbench: build an s4-weight primitive
and check it does not land on `ref` via ONEDNN_VERBOSE).

--------------------------------------------------------------------------------
## 5. IPEX cross-reference (validates the conditional-src-zp approach)

IPEX `csrc/gpu/oneDNN/QMatmul.h` (branch `xpu-main`) is the canonical Intel GPU
quantized-matmul reference. It sets the src zero-point CONDITIONALLY, not always:

    bool m1_need_zp = (m1.q_zero_point() != 0);
    ...
    if (m1_need_zp) {
        pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
    }
    ...
    if (m1_need_zp) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, m1_zp_m});
    }

i.e. the src zp attribute AND its runtime arg are both omitted when the activation
zero-point is zero (the symmetric case) -- exactly what this patch does, just
hoisted to "always omit" because our op's only producer is symmetric. IPEX also
comments `// wgh should never have zero point` and does not set a weight zp at all
(it uses s8 weights directly), which is the s4-native analogue of the section-4
follow-up.

IPEX/Intel quantization docs corroborate the perf rationale: "If the best
performance is desired, symmetric+int8 combination is recommended, as other
configurations may have lower performance due to the existence of zero_point."

Stronger still, IPEX's OWN W4A8 oneDNN path (`csrc/gpu/oneDNN/DnnlMatmulQuant.h`,
xpu-main) takes an `act_quant_mode` (PER_TENSOR / PER_M x SYM/asym) and sets the
activation zero-points CONDITIONALLY per mode -- it does NOT set a src zp for the
symmetric mode. So the exact kernel family we are patching is, in Intel's shipping
code, src-zp-conditional. (This was already documented in-repo:
`docs/kernel/05_int8_int4_optimization_survey.md` sections 2.1-2.2, both marked
[VERIFIED] against the raw xpu-main sources, which explicitly call this "lever B1".)

The same in-repo survey (section 1.2) notes oneDNN v3.8 "Improved int8 matmul
performance with zero-points support for source and weight tensors" -- i.e. zp
handling is a distinct (and historically slower) code path, reinforcing that
omitting it for the symmetric case is the fast direction.

Sources:
- IPEX QMatmul.h (xpu-main):
  https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/oneDNN/QMatmul.h
- IPEX DnnlMatmulQuant.h (xpu-main, the W4A8 path):
  https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/oneDNN/DnnlMatmulQuant.h
- IPEX GPU int8 quantization overview:
  https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/features/int8_overview_xpu.html
- In-repo: docs/kernel/05_int8_int4_optimization_survey.md (sections 1.2, 2.1-2.2)

--------------------------------------------------------------------------------
## 6. Codex sanity-check

Ran `codex exec` (gpt-5.5, high reasoning, read-only) on the cache-key, ref-fallback,
and weight-zp questions. It explored the actual repo sources and verified against
oneDNN docs + the v3.8 release notes before answering. Verbatim conclusions:

- Q1 CACHE-KEY: "Safe only if the convention is: no-src-zp primitives encode
  src-zp count as 0, and any future src-zp-present primitive encodes
  m1_zp.numel() in the upper half. m alone does not disambiguate, because both
  symmetric and asymmetric per-token calls can have the same m; the key needs an
  explicit 'src zp present/absent' signal. Pinning the src-zp half to 0 is
  necessary and sufficient for the current symmetric path, provided a future
  asymmetric path restores the upper-half encoding." -> matches section 2 exactly;
  the patch's `zp_group_size = m2_zp.numel()` is the required 0-pin, and the patch
  comment instructs the future asym path to restore `m1_zp.numel()` in the upper
  half.

- Q2 REF FALLBACK: "Omitting DNNL_ARG_SRC zero-points should not force ref;
  source zero-points are optional matmul attributes ... absence of src-zp is not
  the risky case. Configuring an all-zero src-zp still selects the zp-enabled
  primitive path, so it can carry real correction-path overhead even though the
  runtime values are zero." -> confirms section 3: dropping src-zp is the fast
  direction, no ref risk; the all-zero zp we remove was real overhead. Codex also
  recommends `ONEDNN_VERBOSE=2` to confirm the impl (already in the validation
  plan).

- Q3 WEIGHT-ZP: "The weight zp=8 is not semantic asymmetric quantization ... it is
  the u4 storage centering needed to interpret u4 as signed s4 = u4 - 8. oneDNN
  docs list both u4 and s4 ... so in principle native s4 weights could drop the
  zp=8, but that requires repacking/storing weights as signed int4 and proving the
  Battlemage impl stays optimized. Dropping weight-zp is much riskier than
  dropping src-zp because it changes the weight format and kernel dtype path,
  while all-zero src-zp is mathematically redundant." -> matches section 4: weight
  zp=8 stays; src-zp removal is the safe, self-contained win.

Codex sources: oneDNN matmul dev guide
(https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html) and oneDNN v3.8
release notes (https://github.com/uxlfoundation/oneDNN/releases/tag/v3.8).

--------------------------------------------------------------------------------
## 7. VALIDATION PLAN

### 7.1 IMPORTANT pre-existing test gap to fix first
`tests/test_int4_gemm_onednn.py::test_int4_gemm_w4a8` does NOT exercise the
symmetric s8 path that the patch changes. It calls
`dynamic_per_token_quant_ref(input, False, bits=8)` -- note `use_sym_quant=False`
-- so for BOTH `QuantMode.SYM` and `QuantMode.ASYM` it produces UINT8 activations
(`jd = u8_int4`) with a NONZERO per-token src zp. The `qmode` switch in that test
only toggles the WEIGHT zp, never the activation. So the existing test:
- still passes after the patch ONLY if the u8 nonzero-src-zp case it covers is now
  computed WITHOUT a src-zp correction -> it would FAIL (it relies on the src zp).

Therefore the patch is NOT drop-in for that test as written: it removes the src-zp
the existing u8-asym test depends on. Two clean options:

  Option A (recommended, matches production): change the w4a8 test to call
  `dynamic_per_token_quant_ref(input, True, bits=8)` so it generates s8 symmetric
  activations with a zero src zp (the real decode path). Then both qmodes exercise
  the symmetric src and the patch is correct. This is the honest test of what we
  ship.

  Option B: add a NEW parametrization (e.g. an `act_sym` axis) that runs the
  symmetric-s8 case alongside the legacy u8-asym case, and gate the op so the
  asym-activation case is rejected (our op is symmetric-only in production anyway).

The deliverable here is the kernel patch; the test change is a required companion
commit. Do not trust a "green" run until you confirm the test feeds s8 + zero zp.

### 7.2 Rebuild
On the GPU host, behind the GPU lease is NOT required for a pure CPU compile, but
the build script runs in a container; use the existing minimal builder:

    scripts/44_build_int8_kernel.sh

It builds ONLY `_xpu_C` (toggles FA2/MoE/GDN/etc OFF) -> minutes, and ends with a
`hasattr(torch.ops._xpu_C, "int8_gemm_w8a8")` registration check. Confirm the
build RC=0 and that `int4_gemm_w4a8` is still registered:

    python -c "import torch, vllm._xpu_ops; print(hasattr(torch.ops._xpu_C, 'int4_gemm_w4a8'))"

### 7.3 Correctness test (run on the B70 via scripts/gpu-run)
After fixing the test per 7.1:

    scripts/gpu-run python -m pytest tests/test_int4_gemm_onednn.py \
        -k "w4a8" -v

Expect the symmetric (s8, zero src-zp) cases to pass within the existing
`atol=3e-1, rtol=3e-1` tolerance. A regression here means either the cache-key
change is wrong or oneDNN bounced to ref (check ONEDNN_VERBOSE=1).

Optional belt-and-suspenders: set `ONEDNN_VERBOSE=2` for one shape and confirm the
chosen primitive is a jit/gemm impl, NOT `ref`, and that the verbose attr line no
longer lists `zero_points:src`.

### 7.4 Microbench delta to expect (decode, m=1)
Run the m=1 single-token GEMM microbench (the decode shape, e.g.
`(m,n,k)=(1,4096,11008)` from `MNK_FACTORS`) before vs after, via
`scripts/gpu-run`. Removing one s32 zp-correction pass over K at m=1 is a small but
real win: expect on the order of a few percent (low single digits) latency
reduction on the affected GEMMs, larger on K-heavy projections. It should be
strictly non-negative; any slowdown indicates an unexpected ref fallback and must
be investigated before merging. Record config -> command -> result -> verdict in
JOURNAL.md.

### 7.5 Rollback
The change is a pure attribute/key removal in one header. Revert =
`git checkout -- csrc/xpu/onednn/int4_gemm_w4a8.h` + rebuild. No data/format change,
no checkpoint repack (unlike the section-4 weight-zp follow-up).
