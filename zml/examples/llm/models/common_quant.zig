//! W8A8 int8 quantized linear, a drop-in for `zml.nn.Linear`.
//!
//! Implements the qwen3.6-27b w8a8-sqgptq scheme (see zml/ZML_W8A8.md section 1):
//!   - weight: int8, symmetric, per-output-channel (static, from the checkpoint)
//!   - input activation: int8, symmetric, per-token (per contracting-axis row), dynamic
//!   - i8 x i8 -> i32 accumulate, then dequant by act_scale[row] * weight_scale[channel]
//!
//! Same field/forward shape as `zml.nn.Linear` so it wires into the model unchanged:
//! `weight` is `{dout, d}` (now `.i8`), `tag` is the contracting axis (`.d`). It adds an
//! OPTIONAL `weight_scale` `{dout}` (or `{dout,1}`) tensor; `io.load` reflection loads both
//! automatically. This makes QuantizedLinear a UNIVERSAL drop-in: bind it everywhere a model
//! uses `nn.Linear`, and it auto-selects per checkpoint --
//!   - W8A8 checkpoint: `weight` is `i8` AND `weight_scale` is present  -> int8 W8A8 path.
//!   - bf16 checkpoint: `weight` is bf16 AND `weight_scale` is absent   -> plain bf16 dot
//!     (`maybeCreateTensor("weight_scale")` returns null, so the field stays `null`).
//! The presence of `weight_scale` on disk is the checkpoint discriminator -- no separate model
//! type, no comptime flag, no loader special-case (the static field names stay stable for the
//! `io.load` reflection that binds Tensor fields by name).
//!
//! NOTE (CPU vs GPU): the dot uses `Tensor.dotAcc(.i32, ...)` -- genuine s8 operands with an
//! i32 result -- NOT `x.convert(.i32).dot(w.convert(.i32))`. Both are numerically identical
//! (validated bit-for-bit on CPU, //examples/llm:quant_tests), but feeding TRUE s8 operands is
//! what lets the oneAPI/oneDNN backend recognize an int8 GEMM and dispatch the B70 INT8-XMX
//! (DPAS/DP4A) path; the convert-to-i32 form widens before the dot and hides the int8 operands
//! from that matcher. `dotAcc` lives in zml/tensor.zig (the only library change this needs).

const std = @import("std");

const zml = @import("zml");

pub const QuantizedLinear = struct {
    weight: zml.Tensor, // {dout, d} .i8 (W8A8) or {dout, d} .bf16 (plain) -- per-channel weight
    weight_scale: ?zml.Tensor = null, // {dout} or {dout, 1} .f32/.bf16 -- present iff W8A8
    bias: ?zml.Tensor = null, // {dout}   -- added in the accumulation dtype (f32/bf16)
    tag: zml.Shape.Tag, // contracting axis tag (e.g. .d)
    // Whether to dynamically int8-quantize the ACTIVATIONS (true = full W8A8) or keep them in
    // their float dtype against a dequantized int8 weight (false = weight-only int8). Set false
    // for ROW-PARALLEL layers (o_proj, down_proj) whose contracting axis is TP-SHARDED: the
    // per-token act-quant reduce-max over a sharded axis adds a cross-shard collective (an
    // all_reduce(MAX) for the scale and/or an i32 all_reduce(SUM) of the dot) that the oneAPI
    // oneCCL/Level-Zero plugin mishandles -> "!!!!" garbage under TP=2. The 5 COLUMN-PARALLEL
    // layers (q/k/v/gate/up) contract over the REPLICATED hidden axis, so int8 acts are safe
    // there. Weight stays int8 either way (memory/fit preserved). (bf16 metadata field, not
    // bufferized -- like `tag`.)
    act_quant: bool = true,

    pub fn init(weight: zml.Tensor, weight_scale: ?zml.Tensor, bias: ?zml.Tensor, tag: anytype) QuantizedLinear {
        return .{
            .weight = weight,
            .weight_scale = weight_scale,
            .bias = bias,
            .tag = zml.Shape.toTag(tag),
        };
    }

    /// Same projection but with the dynamic int8 ACTIVATION quant disabled (weight-only int8:
    /// bf16 activation x dequantized int8 weight). Use for TP-row-parallel layers (sharded
    /// contracting axis) where the sharded act-quant reduce breaks the collective. Weight stays
    /// int8 in VRAM; only the matmul sees a dequantized weight.
    pub fn weightOnly(self: QuantizedLinear) QuantizedLinear {
        var s = self;
        s.act_quant = false;
        return s;
    }

    pub fn forward(self: QuantizedLinear, x: zml.Tensor) zml.Tensor {
        // bf16/plain fallback: no weight_scale on disk -> ordinary full-precision linear. Lets
        // the SAME field type serve a bf16 checkpoint (q/k/v/o + gate/up/down behave exactly
        // like nn.Linear) so the model needs no per-checkpoint type swap.
        const wscale_opt = self.weight_scale orelse {
            const y_plain = x.dot(self.weight, self.tag);
            return if (self.bias) |bias| y_plain.add(bias.broad(y_plain.shape())) else y_plain;
        };

        // --- weight-only int8 path (act_quant=false): bf16 activation x dequant(int8 weight).
        //     No activation reduce-max, so no cross-shard act-quant collective -- the row-parallel
        //     all_reduce is a plain bf16 SUM (exactly what the bf16 path does, which works on TP=2).
        if (!self.act_quant) {
            const wf = self.dequantWeight(x.dtype());
            const y_woq = x.dot(wf, self.tag);
            return if (self.bias) |bias| y_woq.add(bias.broad(y_woq.shape())) else y_woq;
        }

        // --- full W8A8 int8 path (int8 weight + dynamic per-token int8 activation) ---
        // 1) per-token (per contracting-axis row) symmetric dynamic int8 activation quant.
        //    zml's reduce keeps the reduced axis as size 1, so act_scale broadcasts against x.
        const amax = x.abs().max(self.tag); // x.shape, contracting axis -> 1
        const act_scale = amax.scale(1.0 / 127.0);
        // clamp bounds must share x's dtype (x is bf16 in the real model, f32 in microbenches).
        const x_i8 = x.div(act_scale)
            .round()
            .clamp(zml.Tensor.scalar(-127, x.dtype()), zml.Tensor.scalar(127, x.dtype()))
            .convert(.i8);

        // 2) i8 x i8 -> i32 accumulate via true s8 operands (GPU INT8-XMX-ready; bit-identical
        //    to the convert-to-i32 path on CPU). See zml/tensor.zig Tensor.dotAcc.
        const acc = x_i8.dotAcc(.i32, self.weight, self.tag);

        // 3) dequant: y = acc(f32) * weight_scale[channel] * act_scale[row].
        const out_dtype = if (self.bias) |b| b.dtype() else wscale_opt.dtype();
        var y = acc.convert(.f32);
        // The compressed-tensors checkpoint stores weight_scale as [out, 1]; squeeze the
        // trailing singleton to a per-channel {dout} vector (a no-op if already rank-1).
        var wscale = wscale_opt.convert(.f32);
        if (wscale.rank() > 1) wscale = wscale.squeeze(wscale.rank() - 1);
        y = y.mul(wscale.broad(y.shape()));
        // drop the size-1 contracting axis; convert to f32 to match the f32 accumulator dtype
        // (act_scale carries x's dtype, which is bf16 in the real model).
        const act_scale_row = act_scale.squeeze(self.tag).convert(.f32);
        y = y.mul(act_scale_row.broad(y.shape()));
        y = y.convert(out_dtype);

        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }

    /// The dequantized weight (weight_int8 * weight_scale) in `dtype` -- the full-precision
    /// weight the int8 weight represents. The per-channel scale aligns by tag to the output
    /// (non-contracting) axis, so this works for both `{dout, d}` and the transposed
    /// `{d, dout}` (down_proj) layouts. Handy for parity references and bf16 fallbacks.
    /// With no weight_scale (a plain bf16 weight), returns the weight converted to `dtype`.
    pub fn dequantWeight(self: QuantizedLinear, dtype: zml.DataType) zml.Tensor {
        var ws = (self.weight_scale orelse return self.weight.convert(dtype)).convert(.f32);
        if (ws.rank() > 1) ws = ws.squeeze(ws.rank() - 1);
        return self.weight.convert(.f32).mul(ws.broad(self.weight.shape())).convert(dtype);
    }
};
