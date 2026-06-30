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

/// A per-token int8-quantized activation + its scale, produced ONCE and shared by every
/// column-parallel QuantizedLinear that contracts the same axis (q/k/v share the input_layernorm
/// output; gate/up share the post_attention_layernorm output). Sharing avoids re-emitting the
/// abs/max/div/round/clamp/convert prologue per projection (the act-quant-dedup decode lever).
pub const QuantAct = struct {
    x_i8: zml.Tensor, // {..., d} .i8        -- the int8 activation
    act_scale: zml.Tensor, // {..., 1} float  -- per-token scale, contracting axis kept as size 1
    tag: zml.Shape.Tag, // the contracting axis the scale was reduced over (must match the linear's)
};

/// Dynamic per-token symmetric int8 activation quant over `tag` (the contracting axis). `tag` must
/// be REPLICATED under TP (q/k/v/gate/up contract the replicated hidden axis) so the reduce-max is
/// shard-local. Identical math to forward's inline column-parallel prologue.
pub fn quantizeActivations(x: zml.Tensor, tag: anytype) QuantAct {
    const t = zml.Shape.toTag(tag);
    const amax = x.abs().max(t); // x.shape, contracting axis -> 1
    const act_scale = amax.scale(1.0 / 127.0);
    const x_i8 = x.div(act_scale)
        .round()
        .clamp(zml.Tensor.scalar(-127, x.dtype()), zml.Tensor.scalar(127, x.dtype()))
        .convert(.i8);
    return .{ .x_i8 = x_i8, .act_scale = act_scale, .tag = t };
}

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
    // ROW-PARALLEL shard-local full W8A8: when true (with act_quant true) the contracting axis is
    // TP-sharded (.model), so forward runs the int8 compute inside a `zml.ops.manualComputation`
    // block where that axis is LOCAL -- the per-token reduce-max is shard-local (no all_reduce(MAX)),
    // and the partials are summed with ONE explicit bf16 allReduce(SUM) (Megatron/vLLM row-parallel
    // int8). This recovers int8 acts on o_proj/down_proj (vs the weight-only TP fallback) while only
    // using the bf16 SUM collective the TP=2 path is proven coherent with. (bf16 metadata, not bufferized.)
    manual_act_quant: bool = false,

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

    /// Full W8A8 (int8 activations) for a ROW-PARALLEL layer whose contracting axis is TP-sharded
    /// (o_proj, down_proj): forward runs the int8 compute in a manual-sharding block so the
    /// per-token reduce-max is shard-local and the partials are summed with one bf16 allReduce(SUM)
    /// -- recovering int8 acts on those layers without the broken sharded-int8 collective. On a
    /// SINGLE device the allReduce no-ops, so it reduces to the plain column-parallel W8A8 (CPU
    /// parity holds). PROVEN COHERENT on TP=2.
    ///
    /// CALL-SITE REQUIREMENT: the activation entering forward MUST carry the .model sharding on the
    /// contracting axis (manualComputation localizes inputs by their declared sharding; merge/rename
    /// reset it to .unknown). Annotate at the model call site, e.g.
    ///   down_proj.forward(hidden.withPartitioning(.{ .dout = .model }))
    ///   o_proj.forward(x.rename(.{ .d_out_proj = .d }).withPartitioning(.{ .d = .model }))
    ///
    /// PERF FINDING (2026-06-30, qwen3.6-27b TP=2 decode): MEASURED SLOWER than `.weightOnly()` at
    /// M=1 -- 12.2 vs 13.0 tok/s -- because the act-quant prologue + manual-block/collective overhead
    /// exceeds the int8-XMX benefit when the GEMM is bandwidth-bound. So `.weightOnly()` is the
    /// DECODE-optimal default; prefer `.rowParallelW8A8()` only for prefill-heavy/batched (large-M) use.
    pub fn rowParallelW8A8(self: QuantizedLinear) QuantizedLinear {
        var s = self;
        s.act_quant = true;
        s.manual_act_quant = true;
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

        // --- ROW-PARALLEL shard-local full W8A8 (act_quant=true, manual_act_quant=true) ---
        // self.tag is TP-sharded (.model). Run the int8 compute inside a manual-sharding block so
        // the sharded axis is LOCAL: each shard quantizes its LOCAL slice with its OWN per-token
        // scale, does the local i8xi8->i32 dot, dequantizes the i32 partial to bf16 locally, then
        // ONE explicit bf16 allReduce(SUM) sums the partials (the collective the bf16 TP=2 path
        // proves coherent). No all_reduce(MAX), no i32 collective. On 1 device allReduce no-ops ->
        // identical to the column-parallel path below (CPU parity).
        if (self.manual_act_quant) {
            const wsh = self.weight.shape();
            const c_ax: u3 = wsh.hasTag(self.tag).?;
            const out_ax: u3 = if (c_ax == 0) 1 else 0;
            const global_out = x.shape().remove(self.tag).appendDim(wsh.dim(out_ax), wsh.tag(out_ax));
            const y_full = zml.ops.manualComputation(
                .{ x, self.weight, wscale_opt },
                global_out,
                .{ .tag = self.tag },
                (struct {
                    fn body(c: anytype, _: std.mem.Allocator, ins: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                        const xl = ins[0];
                        const wl = ins[1];
                        const wsl = ins[2];
                        const amax_l = xl.abs().max(c.tag); // shard-LOCAL per-token max
                        const act_scale = amax_l.scale(1.0 / 127.0);
                        const x_i8 = xl.div(act_scale)
                            .round()
                            .clamp(zml.Tensor.scalar(-127, xl.dtype()), zml.Tensor.scalar(127, xl.dtype()))
                            .convert(.i8);
                        const acc = x_i8.dotAcc(.i32, wl, c.tag); // local i8 x i8 -> i32 (INT8-XMX)
                        var yl = acc.convert(.f32);
                        var wscale_l = wsl.convert(.f32);
                        if (wscale_l.rank() > 1) wscale_l = wscale_l.squeeze(wscale_l.rank() - 1);
                        yl = yl.mul(wscale_l.broad(yl.shape()));
                        const act_scale_row = act_scale.squeeze(c.tag).convert(.f32);
                        yl = yl.mul(act_scale_row.broad(yl.shape()));
                        const y_local = yl.convert(xl.dtype()); // dequant to bf16 BEFORE the sum
                        return zml.ops.allReduce(y_local, zml.Tensor.add);
                    }
                }).body,
            );
            return if (self.bias) |bias| y_full.add(bias.broad(y_full.shape())) else y_full;
        }

        // --- COLUMN-PARALLEL full W8A8 (act_quant=true, manual_act_quant=false) ---
        // Contracting axis is REPLICATED (q/k/v/gate/up contract .d=.replicated), so the per-token
        // reduce-max is shard-local and the dot needs no cross-shard sum. Factored into
        // quantizeActivations + forwardQuant so the model can quantize ONCE and SHARE the int8
        // activation across linears that contract the same axis (q/k/v share input_layernorm; gate/up
        // share post_attention_layernorm) -- the act-quant-dedup lever. Bit-identical to the inline form.
        return self.forwardQuant(quantizeActivations(x, self.tag));
    }

    /// True iff this layer dynamically int8-quantizes its activations via the SHARED-act path
    /// (column-parallel full W8A8). Excludes weight-only (act_quant=false) and the manual
    /// row-parallel path (manual_act_quant=true, which must keep its own in-block quant), so the
    /// shared-act dedup in the model only fires where forwardQuant is the right consumer.
    pub fn usesActQuant(self: QuantizedLinear) bool {
        return self.act_quant and !self.manual_act_quant and self.weight_scale != null;
    }

    /// The int8 dot + dequant, consuming a pre-built (possibly SHARED) QuantAct instead of
    /// re-running the act-quant prologue. `qa` must have been quantized over THIS layer's
    /// contracting axis. Bit-identical to forward's inline column-parallel path.
    pub fn forwardQuant(self: QuantizedLinear, qa: QuantAct) zml.Tensor {
        std.debug.assert(self.weight_scale != null); // caller guarantees a W8A8 checkpoint
        std.debug.assert(qa.tag == self.tag); // shared scale was reduced over our contracting axis
        const wscale_opt = self.weight_scale.?;
        const acc = qa.x_i8.dotAcc(.i32, self.weight, self.tag); // s8 x s8 -> i32 (INT8-XMX)
        const out_dtype = if (self.bias) |b| b.dtype() else wscale_opt.dtype();
        var y = acc.convert(.f32);
        var wscale = wscale_opt.convert(.f32);
        if (wscale.rank() > 1) wscale = wscale.squeeze(wscale.rank() - 1);
        y = y.mul(wscale.broad(y.shape()));
        const act_scale_row = qa.act_scale.squeeze(self.tag).convert(.f32);
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
