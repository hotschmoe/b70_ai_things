// M1 parity test for QuantizedLinear (zml/ZML_W8A8.md).
//
// Proves the reusable QuantizedLinear (examples/llm/models/common_quant.zig) is a drop-in
// for zml.nn.Linear: same forward signature, and its output matches a real nn.Linear fed
// the DEQUANTIZED weights up to activation-quant error. Because both use w_i8 * w_scale for
// the weight, the only difference is the per-token int8 activation quant, so a small rel_l2
// confirms the QuantizedLinear forward (act quant + i32 dot + dequant + bias) is correct.
//
// CPU only, no GPU. Run:
//   cd /mnt/vm_8tb/b70/zml
//   ~/.local/bin/bazelisk run //examples/llm:quant_tests --config=release

const std = @import("std");

const zml = @import("zml");

const common_quant = @import("common_quant.zig");
const QuantizedLinear = common_quant.QuantizedLinear;

const log = std.log.scoped(.quant_tests);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const TOKENS = 16;
const D = 256;
const DOUT = 128;

fn fillRand(comptime T: type, rand: std.Random, slice: []T, comptime kind: enum { weight, scale, normal }) void {
    for (slice) |*v| {
        v.* = switch (kind) {
            .weight => rand.intRangeAtMost(i8, -127, 127),
            .scale => 0.01 + rand.float(f32) * 0.04,
            .normal => rand.floatNorm(f32),
        };
    }
}

/// Run QuantizedLinear and a reference nn.Linear(dequant weights) on the same input and
/// return the relative L2 error between them.
fn parity(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, with_bias: bool) !f64 {
    const x_shape = zml.Shape.init(.{ .token = TOKENS, .d = D }, .f32);
    const w_shape = zml.Shape.init(.{ .dout = DOUT, .d = D }, .i8);
    const wf_shape = zml.Shape.init(.{ .dout = DOUT, .d = D }, .f32);
    const ws_shape = zml.Shape.init(.{ .dout = DOUT }, .f32);
    const b_shape = zml.Shape.init(.{ .dout = DOUT }, .f32);

    // --- host data ---
    var prng: std.Random.DefaultPrng = .init(0x1d11a);
    const rand = prng.random();

    const w_host = try allocator.alloc(i8, DOUT * D);
    defer allocator.free(w_host);
    fillRand(i8, rand, w_host, .weight);

    const ws_host = try allocator.alloc(f32, DOUT);
    defer allocator.free(ws_host);
    fillRand(f32, rand, ws_host, .scale);

    const b_host = try allocator.alloc(f32, DOUT);
    defer allocator.free(b_host);
    fillRand(f32, rand, b_host, .normal);

    const x_host = try allocator.alloc(f32, TOKENS * D);
    defer allocator.free(x_host);
    fillRand(f32, rand, x_host, .normal);

    // reference weight = dequant(int8 weight): wf[o,d] = w_i8[o,d] * w_scale[o]
    const wf_host = try allocator.alloc(f32, DOUT * D);
    defer allocator.free(wf_host);
    for (0..DOUT) |o| {
        for (0..D) |dd| wf_host[o * D + dd] = @as(f32, @floatFromInt(w_host[o * D + dd])) * ws_host[o];
    }

    // --- buffers ---
    var w_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(w_shape, std.mem.sliceAsBytes(w_host)), .replicated);
    defer w_buf.deinit();
    var ws_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(ws_shape, std.mem.sliceAsBytes(ws_host)), .replicated);
    defer ws_buf.deinit();
    var wf_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(wf_shape, std.mem.sliceAsBytes(wf_host)), .replicated);
    defer wf_buf.deinit();
    var b_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(b_shape, std.mem.sliceAsBytes(b_host)), .replicated);
    defer b_buf.deinit();
    var x_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_shape, std.mem.sliceAsBytes(x_host)), .replicated);
    defer x_buf.deinit();

    const bias_ph: ?zml.Tensor = if (with_bias) .fromShape(b_shape) else null;

    // --- QuantizedLinear ---
    const q_model: QuantizedLinear = .init(.fromShape(w_shape), .fromShape(ws_shape), bias_ph, .d);
    var q_exe = try platform.compile(allocator, io, q_model, .forward, .{zml.Tensor.fromShape(x_shape)}, .{});
    defer q_exe.deinit();
    const q_bufs: zml.Bufferized(QuantizedLinear) = .{
        .weight = w_buf,
        .weight_scale = ws_buf,
        .bias = if (with_bias) b_buf else null,
    };
    var yq = try zml.testing.autoCall(allocator, io, &q_exe, QuantizedLinear.forward, .{ q_bufs, x_buf });
    defer yq.deinit();

    // --- reference nn.Linear with dequantized weights ---
    const ref_model: zml.nn.Linear = .init(.fromShape(wf_shape), bias_ph, .d);
    var ref_exe = try platform.compile(allocator, io, ref_model, .forward, .{zml.Tensor.fromShape(x_shape)}, .{});
    defer ref_exe.deinit();
    const ref_bufs: zml.Bufferized(zml.nn.Linear) = .{
        .weight = wf_buf,
        .bias = if (with_bias) b_buf else null,
    };
    var yref = try zml.testing.autoCall(allocator, io, &ref_exe, zml.nn.Linear.forward, .{ ref_bufs, x_buf });
    defer yref.deinit();

    // --- compare ---
    const yq_slice = try yq.toSliceAlloc(allocator, io);
    defer yq_slice.free(allocator);
    const yref_slice = try yref.toSliceAlloc(allocator, io);
    defer yref_slice.free(allocator);
    const a = yq_slice.items(f32);
    const b = yref_slice.items(f32);

    var sum_sq_err: f64 = 0;
    var sum_sq_ref: f64 = 0;
    var max_abs: f64 = 0;
    for (a, b) |av, bv| {
        const e = @as(f64, av) - @as(f64, bv);
        sum_sq_err += e * e;
        sum_sq_ref += @as(f64, bv) * @as(f64, bv);
        max_abs = @max(max_abs, @abs(e));
    }
    const rel_l2 = std.math.sqrt(sum_sq_err / sum_sq_ref);
    log.info("with_bias={}: rel_l2 = {d:.5}, max_abs = {d:.5}", .{ with_bias, rel_l2, max_abs });
    return rel_l2;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const TOL = 0.02;
    var ok = true;
    inline for (.{ false, true }) |with_bias| {
        const rel = try parity(allocator, io, platform, with_bias);
        if (!(rel < TOL)) ok = false;
    }

    if (ok) {
        log.info("RESULT: PASS -- QuantizedLinear matches nn.Linear(dequant) within activation-quant tol ({d}).", .{TOL});
    } else {
        log.err("RESULT: FAIL -- parity exceeded tol {d}", .{TOL});
        std.process.exit(1);
    }
}
