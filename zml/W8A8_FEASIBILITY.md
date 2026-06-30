# True W8A8 INT8 Linear In ZML -- Feasibility

Date: 2026-06-30
Clone reviewed: `/mnt/vm_8tb/b70/zml` HEAD `89b0908c`
Scope: read-only source inspection + web check. No GPU, no bazel build.
Question: can ZML express a TRUE W8A8 int8 linear (int8 weights AND int8 activations
-> int32 accumulate -> dequant) for qwen3_5/qwen3.6-27b dense, validatable first on the
XLA CPU backend.

All line numbers are from the clone above.

---

## TL;DR / Verdict

True W8A8 (int8 activations, int32 accumulate) IS expressible in ZML's IR. The
accumulation dtype is fully controllable at the low-level MLIR builder
`dialects.stablehlo.dot_general`, which takes an explicit `result_type`
(`mlir/dialects/stablehlo/stablehlo.zig:167-207`, result_type at line 171). StableHLO
`dot_general` permits a result element type different from the operands, and that result
type IS the accumulation type -- so `s8 x s8 -> s32` is a legal, standard op.

The catch is purely in ZML's high-level wrapper: `Tensor.dot` / `Tensor.dotGeneral`
hardcode `result dtype = operand dtype` (`zml/tensor.zig:1282`). A naive `i8.dot(i8)`
therefore produces an i8 result that overflows instantly. There are two clean ways out:

- (a) ZERO library change, CPU-validatable today: promote operands to i32 first --
  `x_i8.convert(.i32).dot(w_i8.convert(.i32), .d)`. The dot then accumulates in i32
  (the public wrapper sets result = operand = i32). Numerically this IS the int8xint8
  ->int32 result; it validates the dequant math on CPU with no patch.
- (b) ~15-line library addition `Tensor.dotGeneralAcc(rhs, ..., out_dtype)` that mirrors
  `dotGeneral` but sets the result type to `.i32`, feeding GENUINE s8 operands into
  `stablehlo.dot_general`. This is what you want for the GPU payoff, so the oneAPI/oneDNN
  backend can recognize an int8 GEMM and dispatch the B70 INT8-XMX path. Variant (a)'s
  inserted `convert(s8->s32)` ops may hide the int8 operands from the backend matcher.

We are NOT limited to weight-only int8. The only thing unverifiable from source is whether
the prebuilt oneAPI PJRT plugin actually lowers an s8 `dot_general` to INT8-XMX/DP4A on
B70 -- that must be measured. Everything else is present.

---

## 1. Does ZML expose an integer DOT (i8 x i8 -> i32)?

### The high-level wrapper forces result dtype = operand dtype (the blocker)

`Tensor.dot` (`zml/tensor.zig:1200-1216`) is a tag-based convenience that resolves
contracting/batching axes and delegates to `dotGeneral`.

`Tensor.dotGeneral` (`zml/tensor.zig:1272-1343`):
- line 1278: `stdx.debug.assert(lhs.dtype() == rhs.dtype(), ...)` -- operands must share a
  dtype (fine for s8 x s8).
- line 1282: `var res_shape: Shape = .{ ._dtype = lhs.dtype() };` -- **the result dtype is
  hardcoded to the operand dtype.** For s8 operands this yields an s8 result (overflow);
  there is NO `preferred_element_type` / accumulation-type parameter on the public API.
- line 1328-1342: emits `dialects.stablehlo.dot_general(...)` with the result type built
  from `res_shape` (`mlirx.Type.rankedTensor(mlirCtx(), res_shape)`, line 1332) and
  `.dot_precision = .fast` (line 1338).

It does NOT force a float path -- it will happily build an integer `dot_general`. It just
picks the wrong (operand) result dtype, so it cannot give you i32 accumulation as-is.

### The low-level builder DOES take an explicit result type

`mlir/dialects/stablehlo/stablehlo.zig:167-207`:
```zig
pub fn dot_general(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    result_type: *const mlir.Type,   // <-- line 171: caller chooses the result/accum type
    opts: struct { ...contracting/batching dims..., dot_precision: DotPrecision },
    location: *const mlir.Location,
) *mlir.Operation
```
So `s8 x s8 -> s32` is one call away: pass `result_type = rankedTensor(dims, i32)`. The
StableHLO op is valid with a result element type distinct from the operands.

### convert / cast

`Tensor.convert(self, to: DataType)` (`zml/tensor.zig:1112-1120`) maps directly to
`stablehlo.convert` and is the cast op (int->int promotion, int->float dequant,
float->int quantize). Tested for f4/f8 round-trips (test at 1122-1182). There is no
separate `cast` name; `convert` is it.

### DotPrecision / DotAlgorithm carries an accumulation type -- but it's float-only

`mlir/dialects/stablehlo/stablehlo.zig:110-161`: `DotPrecision` has a `.algorithm`
variant carrying a `DotAlgorithm` whose `accumulation` is built with `mlir.Type.float(...)`
(line 157). So the "dot algorithm" accumulation knob is for float component algorithms
(bf16x3 etc.), NOT for integer accumulation. Integer accumulation is expressed simply by
choosing an integer result_type, not via the algorithm attribute.

### nn.Linear

`zml/nn.zig:16-34`:
```zig
pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    tag: Shape.Tag,
    pub fn forward(self: Linear, x: Tensor) Tensor {
        var y = x.dot(self.weight, self.tag);          // float dot today
        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }
};
```
A `QuantizedLinear` would replace this `forward` with the quantize -> i32-dot -> dequant
sequence below.

### Existing quantized matmul helper?

No general one. The only quantization machinery is in the MoE Triton path and it is FP8,
not int8, and int8 is explicitly refused:
- `zml/moe/triton.zig:443-461` `quantizePerTokenGroupFp8` -- per-token-group FP8 activation
  quant feeding a Triton kernel (precedent for activation quant, but fp8 + Triton-kernel,
  not a stablehlo int8 dot).
- `zml/moe/triton.zig:644`, `zml/moe/mosaic_tpu.zig:71`, `zml/moe/metal.zig:61`:
  `use_int8_w8a8` / `w*_scale` paths return `error.UnsupportedQuantization`.

So there is NO ready-made int8 scaled-mm helper; we build it from primitives.

---

## 2. Ops available to express dequant

Per-channel weight scale + per-token activation scale needs: reduce-max, abs, divide,
round, clamp, int<->float convert, broadcast, multiply. All exist as public `Tensor`
methods (integer-capable: `binaryOp` only asserts equal operand dtype, `zml/tensor.zig:4308`;
the underlying stablehlo ops work on integer element types):

| Need | Method | Cite |
|---|---|---|
| abs (act amax) | `Tensor.abs` | `tensor.zig:331` (keeps dtype) |
| reduce max over axis | `Tensor.max(axis)` | `tensor.zig:3001` |
| reduce sum | `Tensor.sum(axis)` | `tensor.zig:1531` |
| elementwise multiply | `Tensor.mul` | `tensor.zig:1041` |
| elementwise divide | `Tensor.div` | `tensor.zig:1046` |
| scalar multiply | `Tensor.scale(v)` | `tensor.zig:1086` |
| scalar divide | `Tensor.divByConst(v)` | `tensor.zig:1076` |
| round-to-nearest-even | `Tensor.round` | `tensor.zig:1185` |
| clamp(min,max) | `Tensor.clamp(minT,maxT)` | `tensor.zig:1191` |
| floor / ceil | `Tensor.floor` / `Tensor.ceil` | `tensor.zig:1102` / `1107` |
| dtype convert (i32->f32, f32->i8) | `Tensor.convert(to)` | `tensor.zig:1112` |
| broadcast (explicit axes) | `Tensor.broadcast(shape, axes)` | `tensor.zig:2262` |
| broadcast (tag-aligned) | `Tensor.broad(shape)` | `tensor.zig:2304` |
| scalar const | `Tensor.scalar(v, dt)` | `tensor.zig:2179` |
| zero const tensor | `Tensor.constant(Value)` | `tensor.zig:2206` |

Gaps that force a custom op: NONE for the dequant elementwise math. The ONE gap is the
i32 accumulation in the dot itself (section 1) -- the public `dot`/`dotGeneral` will not
hand you an i32 result. Practical note: `binaryOp` requires equal operand dtypes
(`tensor.zig:4308`), so after the int32 dot you must `convert(.i32 -> .f32)` BEFORE
multiplying by the f32 act/weight scales; you cannot multiply an i32 tensor by an f32
scale directly.

---

## 3. Does XLA lower s8 dot_general to int32 on CPU? On oneAPI?

### CPU PJRT (numerical validation) -- YES, supported

StableHLO/XLA `dot_general` accepts integer operands with an s32 result; the result type
is the accumulation type (OpenXLA operation semantics). Integer matmul with int32
accumulation is a first-class, well-trodden pattern: x86 AVX-512 VNNI (`VPDPBUSD`:
u8 x s8 -> s32 dot-accumulate) and AMX-INT8 implement exactly this, and XLA's CPU backend
lowers integer `dot_general` through LLVM. For our purpose the key point is correctness:
`s8 x s8 -> s32` (or the i32-promote variant) produces numerically exact integer
accumulation on CPU, so the dequant math can be validated bit-for-bit there with NO GPU.
(See TF issue tensorflow/tensorflow#59530 "Status of int8 dot/conv with XLA" -- int8 dot
is supported in XLA; backend-specific lowering to a hardware int8 kernel is the variable
part, not the legality of the op.)

### oneAPI PJRT (the B70 INT8-XMX payoff) -- PLAUSIBLE, must be MEASURED

ZML loads a prebuilt, hermetic `libpjrt_oneapi.so` (artifact tag
`manual-2026-06-23T00-20-00Z`, see `REVIEW_intel_arch.md` section 1). That plugin is the
Intel OpenXLA line (`intel/intel-extension-for-openxla`): it compiles StableHLO, adds
Intel GPU passes, and dispatches through oneDNN. oneDNN HAS int8 matmul with XMX/DP4A fast
paths, so it is plausible the plugin maps an s8 `dot_general` (s32 result) onto an oneDNN
int8 GEMM and hits INT8-XMX. But:
- No public Intel doc confirms the s8-`dot_general`->XMX lowering pass specifically (June
  2026 search surfaced only generic "int8/int4 via oneDNN Graph API" notes for SDPA/Gated-MLP,
  not a documented dot_general int8 lowering).
- The exact build ZML pins is an opaque prebuilt artifact; its int8 dispatch behavior is
  not inspectable from this clone.
Conclusion: treat INT8-XMX dispatch as an empirical question. Validate numerics on CPU
first; then on B70 compare an s8-operand serve against a bf16 serve and profile to confirm
an int8 kernel actually ran (e.g. via the oneAPI/oneDNN verbose / Level-Zero kernel names).
This is also exactly where variant (b) of section 1 matters: feed real s8 operands, not
converted-to-i32 ones, so the backend can see the int8 GEMM.

---

## 4. Weight loading (compressed-tensors W8A8 -> ZML)

Our checkpoint `models/files/qwen3.6-27b/w8a8-sqgptq` is compressed-tensors: int8 weight
tensors + per-channel `weight_scale` (+ GDN/vision/MTP tensors). Two options were posed:
(a) load int8 weights + scales directly, (b) load bf16 and quantize in-graph.

### ZML reads int8 safetensors natively

`zml/safetensors.zig:753-773` `stringToDtype` maps the on-disk dtype strings:
```
"F32"->.f32  "F16"->.f16  "BF16"->.bf16  "F8_E4M3"->.f8e4m3fn
"I32"->.i32  "I8"->.i8  "U8"->.u8   (unknown -> error.UnsupportedDataType)
```
So an `I8` weight tensor and an `F32`/`BF16` `weight_scale` tensor are both loadable
directly. The tensor's dtype flows from the safetensors header into the `Shape`.

### How Linear weights are registered (and how a QuantizedLinear would carry both)

The model declares weight Tensors as graph placeholders bound to checkpoint keys via the
`TensorStore.View`:
- `TensorStore.View.createTensor(subkey, tags, partitioning)` (`zml/io.zig:169`, backed by
  `maybeCreateTensor` `io.zig:136-167`) reads the registry entry, applies tags +
  partitioning, builds `Tensor.fromShape(ptr.shape)` (the dtype is whatever the header
  says, e.g. `.i8`), and binds the tensor id to the key (`bindIdToKey`, `io.zig:42-52`).
- Today's Linear is built by, e.g., `SelfAttn.initProj` (`examples/llm/models/qwen3_5/model.zig:483-489`):
  ```zig
  fn initProj(store, partitions, bias_partitions) zml.nn.Linear {
      return .init(
          store.createTensor("weight", .{ .dout, .d }, partitions),
          store.maybeCreateTensor("bias", .{.dout}, bias_partitions),
          .d,
      );
  }
  ```
- Buffers are filled by `zml.io.load(Model, &inner, ...)` (called from `loadBuffers`,
  `model.zig:99`; def `io.zig:1074`), which reflects over the model struct's `Tensor`
  fields and streams each bound key into a device buffer (Direct per-shard writer for
  oneAPI, `io.zig:304-305`).

Because `load` simply walks every `Tensor` field of the model struct, a `QuantizedLinear`
that carries TWO bound tensors loads automatically -- no loader rewrite:
```zig
const QuantizedLinear = struct {
    weight: zml.Tensor,        // .i8,  {dout, d}
    weight_scale: zml.Tensor,  // .f32, {dout}
    bias: ?zml.Tensor = null,
    tag: zml.Shape.Tag,

    fn init(store: zml.io.TensorStore.View, partitions: anytype) QuantizedLinear {
        return .{
            .weight       = store.createTensor("weight",       .{ .dout, .d }, partitions),
            .weight_scale = store.createTensor("weight_scale", .{ .dout },     .{ .dout = .model }),
            .bias         = store.maybeCreateTensor("bias",     .{ .dout },     .{ .dout = .model }),
            .tag = zml.Shape.toTag(.d),
        };
    }
    pub fn forward(self: QuantizedLinear, x: zml.Tensor) zml.Tensor { ... }  // section 5
};
```

### Verdict: option (a) is clearly more tractable

The weights are ALREADY int8 in our checkpoint and ZML reads `I8` + `weight_scale`
directly; (a) needs only a new struct + `forward`, with `io.load` reflection doing the
rest. Option (b) (load bf16, requant in-graph) wastes the int8 artifact, doubles checkpoint
bytes, and adds a calibration step -- only useful as a fallback when no W8A8 artifact
exists. Caveat to verify on the real file: compressed-tensors stores int8 weights UNPACKED
(plain `I8` [out,in]) for W8A8 (sub-byte W4 uses a packed int32 layout that ZML's
`stringToDtype` would NOT understand) -- confirm the artifact's headers are `I8` and note
the `weight_scale` shape (`[out,1]` vs `[out]`) and any `weight_zero_point` (symmetric
GPTQ/sqgptq should be zero -> ignorable).

---

## 5. Smallest CPU-validatable experiment

Goal: a self-contained target that builds a W8A8 int8 linear on small random tensors, runs
on the CPU platform, and reports max-abs error vs a dequantized-weight bf16/f32 reference.
Template: `examples/sharding/main.zig` (platform `.auto` -> CPU, `.fromShape` placeholders,
`platform.compile(model, .forward, .{input})`, `Buffer.fromBytes`/`fromSlice`, `exe.call`,
`out.toSliceAlloc`).

### The W8A8 forward (real ZML API names)

```zig
const std = @import("std");
const zml = @import("zml");

const QuantLinear = struct {
    weight: zml.Tensor,        // {dout, d} .i8
    weight_scale: zml.Tensor,  // {dout}    .f32  (per output channel)

    pub fn forward(self: QuantLinear, x: zml.Tensor) zml.Tensor {
        // x: {token, d} f32 (bf16 in the real model)
        // 1) per-token activation scale = max(|x|) / 127
        const amax      = x.abs().max(.d);             // {token} f32
        const act_scale = amax.scale(1.0 / 127.0);     // {token} f32

        // 2) quantize activations to int8 (symmetric, per token)
        const inv = act_scale.broadcast(x.shape(), &.{x.axis(.token)}); // {token,d}
        var xq = x.div(inv).round();
        xq = xq.clamp(zml.Tensor.scalar(-127, .f32), zml.Tensor.scalar(127, .f32));
        const x_i8 = xq.convert(.i8);                  // {token, d} i8

        // 3) i8 x i8 -> i32 accumulate.
        //    CPU-validation variant -- NO library change (promote to i32, then dot):
        const acc = x_i8.convert(.i32).dot(self.weight.convert(.i32), .d); // {token,dout} i32
        //    True-s8 variant (after adding Tensor.dotGeneralAcc, see Verdict):
        //    const acc = x_i8.dotGeneralAcc(self.weight, &.{.{x.axis(.d), 1}}, &.{}, .i32);

        // 4) dequant: y = acc * act_scale[token] * weight_scale[dout]
        var y = acc.convert(.f32);                     // {token,dout} f32
        y = y.mul(act_scale.broadcast(y.shape(), &.{y.axis(.token)}));
        y = y.mul(self.weight_scale.broadcast(y.shape(), &.{y.axis(.dout)}));
        return y;
    }
};

// Reference: dequantize the SAME int8 weights, do a full-precision dot. The W8A8 output
// should match this up to activation-quant error.
const RefLinear = struct {
    weight: zml.Tensor,        // {dout, d} .i8
    weight_scale: zml.Tensor,  // {dout}    .f32
    pub fn forward(self: RefLinear, x: zml.Tensor) zml.Tensor {
        const w = self.weight.convert(.f32)
            .mul(self.weight_scale.broadcast(self.weight.shape(), &.{self.weight.axis(.dout)}));
        return x.dot(w, .d);   // {token,dout} f32
    }
};
```

### Harness (mirrors examples/sharding/main.zig)

- `var platform = try zml.Platform.auto(allocator, io, .{});` (CPU by default).
- Shapes: `input = Shape.init(.{ .token = 8, .d = 64 }, .f32)`,
  `weight = Shape.init(.{ .dout = 32, .d = 64 }, .i8)`,
  `weight_scale = Shape.init(.{ .dout = 32 }, .f32)`. All `.withPartitioning(.replicated)`
  (single CPU device).
- Placeholders via `zml.Tensor.fromShape(...)`; `model = QuantLinear{...}`.
- `var exe = try platform.compile(allocator, io, model, .forward, .{input}, .{});`
- Fill host arrays in Zig (random i8 weights in [-127,127], random f32 scales ~0.01-0.05,
  random f32 activations), make buffers:
  - i8 weight: `zml.Buffer.fromBytes(io, platform, weight_shape, .replicated, std.mem.sliceAsBytes(&w_i8))`
  - f32 scale/input: `zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(&arr))`
- `exe_args.set(.{ weight_buf, weight_scale_buf, input_buf });` `exe.call(...)`;
  `out.toSliceAlloc` -> `items(f32)`.
- Compile + call `RefLinear.forward` the same way; compute `max(|y_w8a8 - y_ref|)` and the
  relative error. Expect small (activation-quant-bounded, ~1e-2 relative); print PASS/FAIL.

### bazel target + run command (CPU, no oneAPI)

New dir `examples/w8a8/` with `main.zig` (above) and `BUILD.bazel` mirroring
`examples/sharding/BUILD.bazel`:
```python
load("@rules_zig//zig:defs.bzl", "zig_binary")
zig_binary(
    name = "w8a8",
    main = "main.zig",
    deps = ["//zml"],
    visibility = ["//visibility:public"],
)
```
Run on CPU (default `--@zml//platforms:cpu=True`, do NOT pass `--@zml//platforms:oneapi=true`):
```bash
cd /mnt/vm_8tb/b70/zml
bazelisk run //examples/w8a8
```
(Bazelisk fetches the repo-pinned Bazel 9.1.1; this build is CPU-only and touches no GPU.)

If you choose the true-s8 variant, first add `Tensor.dotGeneralAcc` to `zml/tensor.zig`
(it needs `mlirCtx()`/`currentBlock()`/`dialects`, all private to that file, so it must
live there -- it cannot be written from the example).

---

## 6. Verdict

True W8A8 (int8 activations, int32 accumulate) is EXPRESSIBLE in ZML today -- we are NOT
limited to weight-only dequant->bf16. The int32 accumulation is a property of the StableHLO
`dot_general` result type, which the low-level builder exposes
(`mlir/dialects/stablehlo/stablehlo.zig:171`). The only obstacle is that the public
`Tensor.dot`/`dotGeneral` hardcode result dtype = operand dtype
(`zml/tensor.zig:1282`), which is trivially worked around.

What's genuinely available now:
- int8 safetensors loading (`safetensors.zig:753`) + automatic two-tensor weight
  registration via `createTensor` + `io.load` reflection (Q4) -- so a `QuantizedLinear`
  carrying `weight:i8` + `weight_scale:f32` drops into the existing qwen3_5 loader with no
  loader rewrite.
- all dequant elementwise/reduce ops as public `Tensor` methods (Q2).
- i32 accumulation on CPU for numerical validation TODAY with ZERO library change, via
  `x_i8.convert(.i32).dot(w_i8.convert(.i32), .d)`.

What needs a small change / measurement:
- a ~15-line `Tensor.dotGeneralAcc(out_dtype)` in `tensor.zig` to feed genuine s8 operands
  into `dot_general` with an s32 result -- needed so the oneAPI/oneDNN backend can see and
  dispatch an int8 GEMM (the convert-to-i32 trick may hide int8 operands from the matcher).
- whether the pinned oneAPI PJRT plugin lowers s8 `dot_general` to B70 INT8-XMX/DP4A is
  UNVERIFIABLE from source and must be profiled on hardware (Q3).

Realistic effort:
- CPU W8A8 microbench example proving max-abs-error vs dequant-weight reference: ~1 day,
  no library change (`//examples/w8a8`).
- `Tensor.dotGeneralAcc` + a test: ~0.5 day.
- `QuantizedLinear` + wiring it into `qwen3_5` q/k/v/o, gate/up/down, lm_head, and the GDN
  in/out projections, loading our W8A8 checkpoint: a few days (plus the existing model-port
  gaps from `REVIEW_intel_arch.md`: `model_type` alias, vision tower, MTP head, which are
  orthogonal and much larger).
- The real unknown / research payoff: confirming an int8 kernel actually runs on B70 and
  beating/comparing against our oneDNN int8 sglang kernels -- empirical, GPU-time work.

Recommended first milestone: ship `//examples/w8a8` (CPU, zero library change, i32-promote
dot) that builds a per-token-act / per-channel-weight int8 linear and reports max-abs error
vs a dequantized-weight f32 reference. That locks the dequant math correctness independent
of any GPU, after which adding `dotGeneralAcc` and a B70 profiling run is the next step to
chase the INT8-XMX win.
