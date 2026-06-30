// W8A8 INT8-XMX perf gate (zml/ZML_W8A8.md M4): time a bf16 GEMM vs an int8 (s8 x s8 -> s32)
// GEMM of the same shape on the B70, to answer whether the oneAPI PJRT plugin lowers an s8
// `dot_general` to the INT8-XMX/DP4A fast path.
//
//   - if int8 hits INT8-XMX: int8 GEMM should be roughly ~2x the bf16 GEMM throughput (B70
//     INT8 XMX is ~2x bf16), and int8 is the clear W8A8 win on zml.
//   - if int8 just widens to bf16/f32 before the dot: int8 will be ~the same or SLOWER than
//     bf16 -> W8A8 buys nothing on zml and we fall back to weight-only int8 (or stop).
//
// Pair with ONEDNN_VERBOSE=1 to see the dispatched primitive (s8:s8:s32 + an xmx/jit:gemm impl
// vs a reference/upconvert).
//
// GPU run (ONE card; daily driver down, under the gpu-run lease):
//   cd /mnt/vm_8tb/b70/zml
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 CCL_TOPO_P2P_ACCESS=0 ZE_FLAT_DEVICE_HIERARCHY=FLAT \
//   ONEDNN_VERBOSE=1 ~/.local/bin/bazelisk run //examples/w8a8_bench --config=release \
//     --@zml//platforms:cpu=false --@zml//platforms:oneapi=true -- --iters=100

const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.w8a8_bench);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    m: usize = 512, // tokens (rows)
    k: usize = 5120, // contracting (q_proj in-dim)
    n: usize = 12288, // output (q_proj out-dim)
    iters: usize = 100,

    pub const help =
        \\w8a8_bench [--m=N] [--k=N] [--n=N] [--iters=N]  -- time bf16 vs int8 GEMM
    ;
};

const Bf16Gemm = struct {
    w: zml.Tensor, // {n, k} bf16
    pub fn forward(self: Bf16Gemm, x: zml.Tensor) zml.Tensor {
        return x.dot(self.w, .k); // {m, n} bf16
    }
};

const Int8Gemm = struct {
    w: zml.Tensor, // {n, k} i8
    pub fn forward(self: Int8Gemm, x: zml.Tensor) zml.Tensor {
        return x.dotAcc(.i32, self.w, .k); // {m, n} i32 (true s8 operands -> INT8-XMX target)
    }
};

fn bf16Bits(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}
fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Time `iters` calls of an already-compiled exe with two input buffers; returns ns/call.
fn timeExe(io: std.Io, allocator: std.mem.Allocator, exe: *zml.Exe, w_buf: zml.Buffer, x_buf: zml.Buffer, iters: usize) !struct { ns_per_call: f64, out: zml.Buffer } {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ w_buf, x_buf });

    exe.callOpts(io, args, &results, .{ .wait = true }); // warmup (compile/jit caches)

    const n = @max(iters, 1);
    const t0 = std.Io.Timestamp.now(io, .awake);
    for (0..n) |_| exe.callOpts(io, args, &results, .{ .wait = true });
    const dur = t0.untilNow(io, .awake);
    const ns: f64 = @floatFromInt(dur.toNanoseconds());

    const out = results.get(zml.Buffer);
    return .{ .ns_per_call = ns / @as(f64, @floatFromInt(n)), .out = out };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform: {f}", .{platform.fmtVerbose()});

    const M = args.m;
    const K = args.k;
    const N = args.n;
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N));
    log.info("GEMM M={d} K={d} N={d}, iters={d} ({d:.1} GFLOP/call)", .{ M, K, N, args.iters, flops / 1e9 });

    var prng: std.Random.DefaultPrng = .init(0xb70be12);
    const rand = prng.random();

    const x_bf = zml.Shape.init(.{ .m = @as(i64, @intCast(M)), .k = @as(i64, @intCast(K)) }, .bf16);
    const w_bf = zml.Shape.init(.{ .n = @as(i64, @intCast(N)), .k = @as(i64, @intCast(K)) }, .bf16);
    const x_i = zml.Shape.init(.{ .m = @as(i64, @intCast(M)), .k = @as(i64, @intCast(K)) }, .i8);
    const w_i = zml.Shape.init(.{ .n = @as(i64, @intCast(N)), .k = @as(i64, @intCast(K)) }, .i8);

    // --- bf16 buffers ---
    const xb = try allocator.alloc(u16, M * K);
    defer allocator.free(xb);
    for (xb) |*v| v.* = bf16Bits(rand.floatNorm(f32) * 0.05);
    const wb = try allocator.alloc(u16, N * K);
    defer allocator.free(wb);
    for (wb) |*v| v.* = bf16Bits(rand.floatNorm(f32) * 0.05);
    var xb_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_bf, std.mem.sliceAsBytes(xb)), .replicated);
    defer xb_buf.deinit();
    var wb_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(w_bf, std.mem.sliceAsBytes(wb)), .replicated);
    defer wb_buf.deinit();

    // --- int8 buffers ---
    const xi = try allocator.alloc(i8, M * K);
    defer allocator.free(xi);
    for (xi) |*v| v.* = rand.intRangeAtMost(i8, -127, 127);
    const wi = try allocator.alloc(i8, N * K);
    defer allocator.free(wi);
    for (wi) |*v| v.* = rand.intRangeAtMost(i8, -127, 127);
    var xi_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_i, std.mem.sliceAsBytes(xi)), .replicated);
    defer xi_buf.deinit();
    var wi_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(w_i, std.mem.sliceAsBytes(wi)), .replicated);
    defer wi_buf.deinit();

    // --- bf16 GEMM ---
    const bf_model: Bf16Gemm = .{ .w = .fromShape(w_bf) };
    var bf_exe = try platform.compile(allocator, io, bf_model, .forward, .{zml.Tensor.fromShape(x_bf)}, .{});
    defer bf_exe.deinit();
    var bf = try timeExe(io, allocator, &bf_exe, wb_buf, xb_buf, args.iters);
    defer bf.out.deinit();

    // --- int8 GEMM ---
    const i8_model: Int8Gemm = .{ .w = .fromShape(w_i) };
    var i8_exe = try platform.compile(allocator, io, i8_model, .forward, .{zml.Tensor.fromShape(x_i)}, .{});
    defer i8_exe.deinit();
    var iq = try timeExe(io, allocator, &i8_exe, wi_buf, xi_buf, args.iters);
    defer iq.out.deinit();

    // --- sanity (not optimized away) ---
    const bf_out = try bf.out.toSliceAlloc(allocator, io);
    defer bf_out.free(allocator);
    const i8_out = try iq.out.toSliceAlloc(allocator, io);
    defer i8_out.free(allocator);
    const bf_first = bf16ToF32(bf_out.items(u16)[0]);
    const i8_first = i8_out.items(i32)[0];

    const bf_gflops = flops / bf.ns_per_call;
    const i8_gflops = flops / iq.ns_per_call;

    log.info("bf16: {d:.3} ms/call, {d:.1} GFLOP/s  (out[0]={d:.4})", .{ bf.ns_per_call / 1e6, bf_gflops, bf_first });
    log.info("int8: {d:.3} ms/call, {d:.1} GFLOP/s  (out[0]={d})", .{ iq.ns_per_call / 1e6, i8_gflops, i8_first });
    log.info("int8/bf16 speedup = {d:.2}x  (>~1.5x => INT8-XMX engaged; <~1x => widened, no win)", .{ bf.ns_per_call / iq.ns_per_call });
}
