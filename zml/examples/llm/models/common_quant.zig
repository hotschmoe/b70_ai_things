//! W8A8 int8 quantized linear, a drop-in for `zml.nn.Linear`.
//!
//! Implements the qwen3.6-27b w8a8-sqgptq scheme (see zml/ZML_W8A8.md section 1):
//!   - weight: int8, symmetric, per-output-channel (static, from the checkpoint)
//!   - input activation: int8, symmetric, per-token (per contracting-axis row), dynamic
//!   - i8 x i8 -> i32 accumulate, then dequant by act_scale[row] * weight_scale[channel]
//!
//! Same field/forward shape as `zml.nn.Linear` so it wires into the model unchanged:
//! `weight` is `{dout, d}` (now `.i8`), `tag` is the contracting axis (`.d`). It adds a
//! `weight_scale` `{dout}` tensor; `io.load` reflection loads both automatically.
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
    weight: zml.Tensor, // {dout, d} .i8    -- per-channel symmetric int8 weight
    weight_scale: zml.Tensor, // {dout} or {dout, 1} .f32/.bf16 -- per output channel
    bias: ?zml.Tensor = null, // {dout}   -- added in the accumulation dtype (f32/bf16)
    tag: zml.Shape.Tag, // contracting axis tag (e.g. .d)

    pub fn init(weight: zml.Tensor, weight_scale: zml.Tensor, bias: ?zml.Tensor, tag: anytype) QuantizedLinear {
        return .{
            .weight = weight,
            .weight_scale = weight_scale,
            .bias = bias,
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn forward(self: QuantizedLinear, x: zml.Tensor) zml.Tensor {
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
        const out_dtype = if (self.bias) |b| b.dtype() else self.weight_scale.dtype();
        var y = acc.convert(.f32);
        // The compressed-tensors checkpoint stores weight_scale as [out, 1]; squeeze the
        // trailing singleton to a per-channel {dout} vector (a no-op if already rank-1).
        var wscale = self.weight_scale.convert(.f32);
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
    pub fn dequantWeight(self: QuantizedLinear, dtype: zml.DataType) zml.Tensor {
        var ws = self.weight_scale.convert(.f32);
        if (ws.rank() > 1) ws = ws.squeeze(ws.rank() - 1);
        return self.weight.convert(.f32).mul(ws.broad(self.weight.shape())).convert(dtype);
    }
};
