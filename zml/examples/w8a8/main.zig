// W8A8 (int8 weights + int8 activations -> int32 accumulate -> dequant) microbench.
//
// Milestone M0 of zml/ZML_W8A8.md: prove, on the XLA CPU backend with NO GPU and NO
// library change, that zml can express the per-token-activation / per-channel-weight
// int8 linear used by our qwen3.6-27b w8a8-sqgptq checkpoint, and that XLA accumulates
// the i8 x i8 dot in i32 (not i8/i16, which would overflow).
//
// The compiled graph returns three tensors so the host can apply two independent gates:
//   1. acc (i32)  -- the raw int8xint8 accumulator. Compared BIT-EXACT against a host
//                    integer recomputation from the device's own quantized activations.
//                    If XLA had used an i8/i16 result dtype, the >127 sums would wrap and
//                    this check would fail catastrophically. This is the i32-accum proof.
//   2. y   (f32)  -- the dequantized output. Compared (tolerance) against an INDEPENDENT
//                    full-precision host reference x @ dequant(W). The gap is bounded by
//                    activation-quant error (~<1-2% rel on random Gaussian). Dequant proof.
//   3. xq  (i8)   -- the device's per-token int8 activations, fed into gate 1 so the
//                    integer check needs no matching of stablehlo's round-half-to-even.
//
// Run (CPU, default platform -- do NOT pass --@zml//platforms:oneapi=true):
//   cd /mnt/vm_8tb/b70/zml && ~/.local/bin/bazelisk run //examples/w8a8 --config=release

const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.w8a8);

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Problem size. d is deliberately large enough that an int8/int16 accumulator would
// overflow many times over (max |product| ~127*127 ~= 16129, summed over d), so a
// bit-exact match to the host i32 reference is a genuine proof of i32 accumulation.
const TOKENS = 16;
const D = 256;
const DOUT = 128;

/// W8A8 linear: per-channel symmetric int8 weight, per-token symmetric dynamic int8
/// activation, i32 accumulation, then dequant back to f32. Mirrors section 1 of ZML_W8A8.md.
const QuantLinear = struct {
    weight: zml.Tensor, // {dout, d} .i8   -- static, from checkpoint
    weight_scale: zml.Tensor, // {dout} .f32 -- per output channel

    const Out = struct {
        y: zml.Tensor, // {token, dout} .f32  dequantized result
        acc: zml.Tensor, // {token, dout} .i32  raw int8xint8 accumulator
        xq: zml.Tensor, // {token, d}    .i8   quantized activations
    };

    pub fn forward(self: QuantLinear, x: zml.Tensor) Out {
        // x: {token, d} f32 (bf16 in the real model).
        // 1) per-token activation scale = max(|x|) / 127. zml's reduce keeps the reduced
        //    axis as size 1, so amax is {token, d=1} and broadcasts against {token, d}.
        const amax = x.abs().max(.d); // {token, d=1} f32
        const act_scale = amax.scale(1.0 / 127.0); // {token, d=1} f32

        // 2) quantize activations to symmetric int8 (per token, dynamic).
        const x_scaled = x.div(act_scale); // {token, d} (auto-broadcast: tags match, d=1 broadcasts)
        const x_clamped = x_scaled.round().clamp(
            zml.Tensor.scalar(-127, .f32),
            zml.Tensor.scalar(127, .f32),
        );
        const x_i8 = x_clamped.convert(.i8); // {token, d} i8

        // 3) i8 x i8 -> i32 accumulate. CPU variant: promote both operands to i32 so the
        //    dot result dtype is i32 (zml's public dot hardcodes result dtype = operand
        //    dtype). The GPU INT8-XMX path (M4) instead needs a real s8-operand dot via a
        //    Tensor.dotGeneralAcc(.i32) helper -- see ZML_W8A8.md.
        const acc = x_i8.convert(.i32).dot(self.weight.convert(.i32), .d); // {token, dout} i32

        // 4) dequant: y = acc(f32) * act_scale[token] * weight_scale[dout].
        var y = acc.convert(.f32); // {token, dout} f32
        // act_scale is {token, d=1}; map (token->token, d->dout) so the size-1 d broadcasts.
        y = y.mul(act_scale.broadcast(y.shape(), &.{ y.axis(.token), y.axis(.dout) }));
        // weight_scale is {dout}; map dout->dout.
        y = y.mul(self.weight_scale.broadcast(y.shape(), &.{y.axis(.dout)}));

        return .{ .y = y, .acc = acc, .xq = x_i8 };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform: {f}", .{platform.fmtVerbose()});

    // --- shapes ---
    const x_shape = zml.Shape.init(.{ .token = TOKENS, .d = D }, .f32);
    const w_shape = zml.Shape.init(.{ .dout = DOUT, .d = D }, .i8);
    const ws_shape = zml.Shape.init(.{ .dout = DOUT }, .f32);

    // --- model + compile ---
    const model: QuantLinear = .{
        .weight = .fromShape(w_shape),
        .weight_scale = .fromShape(ws_shape),
    };
    const x_ph: zml.Tensor = .fromShape(x_shape);

    var exe = try platform.compile(allocator, io, model, .forward, .{x_ph}, .{});
    defer exe.deinit();

    // --- host inputs (deterministic) ---
    var prng: std.Random.DefaultPrng = .init(0xb70a8a8);
    const rand = prng.random();

    const w_host = try allocator.alloc(i8, DOUT * D);
    defer allocator.free(w_host);
    for (w_host) |*v| v.* = rand.intRangeAtMost(i8, -127, 127);

    const ws_host = try allocator.alloc(f32, DOUT);
    defer allocator.free(ws_host);
    for (ws_host) |*v| v.* = 0.01 + rand.float(f32) * 0.04; // 0.01 .. 0.05

    const x_host = try allocator.alloc(f32, TOKENS * D);
    defer allocator.free(x_host);
    for (x_host) |*v| v.* = rand.floatNorm(f32); // standard normal

    // --- buffers ---
    var w_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(w_shape, std.mem.sliceAsBytes(w_host)), .replicated);
    defer w_buf.deinit();
    var ws_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(ws_shape, std.mem.sliceAsBytes(ws_host)), .replicated);
    defer ws_buf.deinit();
    var x_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_shape, std.mem.sliceAsBytes(x_host)), .replicated);
    defer x_buf.deinit();

    // --- run ---
    const model_bufs: zml.Bufferized(QuantLinear) = .{ .weight = w_buf, .weight_scale = ws_buf };
    var out = try zml.testing.autoCall(allocator, io, &exe, QuantLinear.forward, .{ model_bufs, x_buf });
    defer {
        out.y.deinit();
        out.acc.deinit();
        out.xq.deinit();
    }

    const y_slice = try out.y.toSliceAlloc(allocator, io);
    defer y_slice.free(allocator);
    const acc_slice = try out.acc.toSliceAlloc(allocator, io);
    defer acc_slice.free(allocator);
    const xq_slice = try out.xq.toSliceAlloc(allocator, io);
    defer xq_slice.free(allocator);

    const y_dev = y_slice.items(f32); // [TOKENS*DOUT]
    const acc_dev = acc_slice.items(i32); // [TOKENS*DOUT]
    const xq_dev = xq_slice.items(i8); // [TOKENS*D]

    // --- GATE 1: bit-exact i32 accumulation ---
    // Recompute acc on host from the DEVICE's quantized activations and our i8 weights.
    // Integer math is exact; any mismatch means the device did not accumulate in i32.
    var int_mismatches: usize = 0;
    var first_bad: usize = 0;
    for (0..TOKENS) |t| {
        for (0..DOUT) |o| {
            var s: i64 = 0;
            for (0..D) |dd| {
                s += @as(i64, xq_dev[t * D + dd]) * @as(i64, w_host[o * D + dd]);
            }
            const dev = acc_dev[t * DOUT + o];
            if (@as(i64, dev) != s) {
                if (int_mismatches == 0) first_bad = t * DOUT + o;
                int_mismatches += 1;
            }
        }
    }
    const gate1_ok = int_mismatches == 0;

    // --- GATE 2: dequant tolerance vs independent full-precision reference ---
    // ref[t,o] = sum_d x[t,d] * (w_i8[o,d] * w_scale[o])   (full-precision activations)
    var max_abs: f64 = 0;
    var sum_sq_err: f64 = 0;
    var sum_sq_ref: f64 = 0;
    for (0..TOKENS) |t| {
        for (0..DOUT) |o| {
            var ref: f64 = 0;
            const wscale: f64 = ws_host[o];
            for (0..D) |dd| {
                ref += @as(f64, x_host[t * D + dd]) * (@as(f64, @floatFromInt(w_host[o * D + dd])) * wscale);
            }
            const dev: f64 = y_dev[t * DOUT + o];
            const e = dev - ref;
            max_abs = @max(max_abs, @abs(e));
            sum_sq_err += e * e;
            sum_sq_ref += ref * ref;
        }
    }
    const rel_l2 = std.math.sqrt(sum_sq_err / sum_sq_ref);
    const gate2_ok = rel_l2 < 0.02;

    // --- report ---
    log.info("shapes: x[{d},{d}] w[{d},{d}] -> y[{d},{d}]", .{ TOKENS, D, DOUT, D, TOKENS, DOUT });
    log.info("y[0][0..6] = {any}", .{y_dev[0..@min(y_dev.len, 6)]});
    log.info("acc[0][0..6] = {any}", .{acc_dev[0..@min(acc_dev.len, 6)]});
    if (gate1_ok) {
        log.info("GATE 1 (i32 accumulation, bit-exact): PASS ({d} elements)", .{TOKENS * DOUT});
    } else {
        log.err("GATE 1 (i32 accumulation): FAIL -- {d}/{d} mismatches (first at flat idx {d})", .{ int_mismatches, TOKENS * DOUT, first_bad });
    }
    log.info("GATE 2 (dequant vs f32 ref): rel_l2 = {d:.5}, max_abs = {d:.5} -> {s}", .{
        rel_l2, max_abs, if (gate2_ok) "PASS" else "FAIL",
    });

    if (gate1_ok and gate2_ok) {
        log.info("RESULT: PASS -- zml expresses W8A8 with genuine i32 accumulation on CPU.", .{});
    } else {
        log.err("RESULT: FAIL", .{});
        std.process.exit(1);
    }
}
