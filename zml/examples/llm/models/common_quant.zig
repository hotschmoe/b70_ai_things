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
//! NOTE (CPU vs GPU): the dot promotes both operands to i32 before contracting, because
//! zml's public `Tensor.dot` hardcodes the result dtype to the operand dtype. That gives
//! genuine int32 accumulation and is correct everywhere, but the inserted `convert(s8->s32)`
//! ops may hide the int8 operands from the oneAPI/oneDNN matcher and forfeit INT8-XMX. For
//! the GPU payoff (M4) switch the marked line to a real s8-operand dot via the
//! `Tensor.dotGeneralAcc(.i32)` helper.

const std = @import("std");

const zml = @import("zml");

pub const QuantizedLinear = struct {
    weight: zml.Tensor, // {dout, d} .i8    -- per-channel symmetric int8 weight
    weight_scale: zml.Tensor, // {dout} .f32 (or .bf16) -- per output channel
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
        const x_i8 = x.div(act_scale)
            .round()
            .clamp(zml.Tensor.scalar(-127, .f32), zml.Tensor.scalar(127, .f32))
            .convert(.i8);

        // 2) i8 x i8 -> i32 accumulate. CPU/correct-everywhere variant (promote to i32).
        //    GPU INT8-XMX (M4): replace with `x_i8.dotGeneralAcc(self.weight, ..., .i32)`.
        const acc = x_i8.convert(.i32).dot(self.weight.convert(.i32), self.tag);

        // 3) dequant: y = acc(f32) * weight_scale[channel] * act_scale[row].
        const out_dtype = if (self.bias) |b| b.dtype() else self.weight_scale.dtype();
        var y = acc.convert(.f32);
        y = y.mul(self.weight_scale.convert(.f32).broad(y.shape()));
        const act_scale_row = act_scale.squeeze(self.tag); // drop the size-1 contracting axis
        y = y.mul(act_scale_row.broad(y.shape()));
        y = y.convert(out_dtype);

        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }
};
