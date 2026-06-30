// W8A8 GEMM/GEMV sweep (zml/ZML_W8A8.md follow-ups): push int8 toward the ~2x peak and
// profile the decode-shape (M=1) GEMV.
//
// Extends the M4 perf gate (//examples/w8a8_bench) from a single shape/dtype to a sweep over:
//   - the REAL qwen3.6-27b W8A8 projection shapes (q/k/v/o, gate/up/down at K=5120),
//   - an M (token-count) sweep from 1 (decode GEMV) to 4096 (prefill GEMM),
//   - four variants per (shape, M):
//       bf16  : pure bf16 GEMM                         (baseline)
//       i8    : raw s8 x s8 -> s32 dotAcc              (the proven INT8-XMX fast path, M4)
//       w8a8  : full QuantizedLinear (dynamic per-token act-quant + s8 dot + dequant), bf16 in/out
//               (the REAL serve path; isolates the quant epilogue overhead vs raw i8)
//       woq   : weight-only int8 (i8 weight dequant->bf16 in-graph, bf16 activation, bf16 dot)
//               (the GEMV alternative; tests whether XLA reads i8 from VRAM and expands in-reg
//                -- a bandwidth win at M=1 -- or materializes a bf16 weight first -- no win)
//
// Reports per config: ms/call, effective TFLOP/s (compute roofline), effective GB/s (memory
// roofline; weight bytes dominate at M=1), and the int8/bf16 speedup. GB/s near the B70 peak
// at M=1 => bandwidth-bound (int8 halves weight bytes => ~2x is the ceiling). TFLOP/s near the
// INT8-XMX peak at large M => compute-bound.
//
// CPU build just compiles (the timings are meaningless without the GPU). GPU run (ONE card,
// daily driver down, under the gpu-run lease):
//   cd /mnt/vm_8tb/b70/zml
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 CCL_TOPO_P2P_ACCESS=0 ZE_FLAT_DEVICE_HIERARCHY=FLAT \
//   ~/.local/bin/bazelisk run //examples/w8a8_sweep --config=release \
//     --@zml//platforms:cpu=false --@zml//platforms:oneapi=true -- --shape=all --m=0 --iters=100
//
// Flags:
//   --shape= all | q | k | v | o | gate | up | down | sq4096 | sq8192   (default all)
//   --m=     0 (sweep 1..4096) | <single M>                              (default 0)
//   --variant= all | bf16 | i8 | w8a8 | woq                             (default all)
//   --iters= timed calls per config (default 50)
//   --layout= nk (weight {n,k}, contract k -- model layout) | kn (weight {k,n}, contract k)
//             -- probe whether the XMX kernel prefers a transposed weight (default nk)

const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.w8a8_sweep);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    shape: []const u8 = "all",
    m: usize = 0,
    variant: []const u8 = "all",
    iters: usize = 50,
    layout: []const u8 = "nk",

    pub const help =
        \\w8a8_sweep [--shape=all|q|k|v|o|gate|up|down|sq4096|sq8192] [--m=0|N]
        \\           [--variant=all|bf16|i8|w8a8|woq] [--iters=N] [--layout=nk|kn]
        \\  Sweep bf16 vs int8 GEMM/GEMV over the real qwen3.6-27b W8A8 projection shapes.
    ;
};

const ShapeDef = struct { name: []const u8, k: i64, n: i64 };

// Real qwen3.6-27b W8A8 projections (N = out channels, K = contracting in-dim). hidden=5120,
// intermediate=17408, gated attn (q out = 24*2*256 = 12288), GQA kv (4*256 = 1024), o in =
// 24*256 = 6144. Plus two square shapes to see clean compute-bound peak behavior.
const SHAPES = [_]ShapeDef{
    .{ .name = "q_proj", .k = 5120, .n = 12288 },
    .{ .name = "k_proj", .k = 5120, .n = 1024 },
    .{ .name = "v_proj", .k = 5120, .n = 1024 },
    .{ .name = "o_proj", .k = 6144, .n = 5120 },
    .{ .name = "gate_proj", .k = 5120, .n = 17408 },
    .{ .name = "up_proj", .k = 5120, .n = 17408 },
    .{ .name = "down_proj", .k = 17408, .n = 5120 },
    .{ .name = "sq4096", .k = 4096, .n = 4096 },
    .{ .name = "sq8192", .k = 8192, .n = 8192 },
};

const M_SWEEP = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

const Variant = enum { bf16, i8, w8a8, woq };

const Layout = enum { nk, kn };

fn bf16Bits(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}
fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

// ---- models (one forward per variant; weight layout chosen by `lt`) -------------------------

fn weightShape(lt: Layout, n: i64, k: i64, dtype: zml.DataType) zml.Shape {
    return switch (lt) {
        .nk => zml.Shape.init(.{ .n = n, .k = k }, dtype),
        .kn => zml.Shape.init(.{ .k = k, .n = n }, dtype),
    };
}

const Bf16Gemm = struct {
    w: zml.Tensor, // {n,k} or {k,n} bf16
    pub fn forward(self: Bf16Gemm, x: zml.Tensor) zml.Tensor {
        return x.dot(self.w, .k); // {m,n} bf16
    }
};

const Int8Gemm = struct {
    w: zml.Tensor, // {n,k} or {k,n} i8
    pub fn forward(self: Int8Gemm, x: zml.Tensor) zml.Tensor {
        return x.dotAcc(.i32, self.w, .k); // {m,n} i32 -- true s8 operands -> INT8-XMX
    }
};

// Full W8A8 linear: dynamic per-token int8 activation quant + s8 dot + per-channel/per-token
// dequant. Mirrors examples/llm/models/common_quant.zig QuantizedLinear.forward exactly, so the
// extra cost over Int8Gemm is purely the act-quant prologue + dequant epilogue.
const W8A8Linear = struct {
    w: zml.Tensor, // {n,k} i8
    wscale: zml.Tensor, // {n} f32 (per output channel)
    pub fn forward(self: W8A8Linear, x: zml.Tensor) zml.Tensor {
        const amax = x.abs().max(.k); // {m, k=1}
        const act_scale = amax.scale(1.0 / 127.0);
        const x_i8 = x.div(act_scale)
            .round()
            .clamp(zml.Tensor.scalar(-127, x.dtype()), zml.Tensor.scalar(127, x.dtype()))
            .convert(.i8);
        const acc = x_i8.dotAcc(.i32, self.w, .k); // {m,n} i32
        var y = acc.convert(.f32);
        y = y.mul(self.wscale.broad(y.shape()));
        const act_scale_row = act_scale.squeeze(.k).convert(.f32);
        y = y.mul(act_scale_row.broad(y.shape()));
        return y.convert(.bf16);
    }
};

// Weight-only int8: dequant the i8 weight to bf16 in-graph, bf16 activation, bf16 dot. The
// question is whether XLA reads the i8 weight from VRAM and widens in registers (bandwidth win
// at M=1) or materializes a bf16 weight first (no win). bf16 in/out.
const WoqLinear = struct {
    w: zml.Tensor, // {n,k} i8
    wscale: zml.Tensor, // {n} f32
    pub fn forward(self: WoqLinear, x: zml.Tensor) zml.Tensor {
        const wf = self.w.convert(.bf16).mul(self.wscale.convert(.bf16).broad(self.w.shape()));
        return x.dot(wf, .k); // {m,n} bf16
    }
};

// ---- timing ---------------------------------------------------------------------------------

const Timing = struct { ns_per_call: f64 };

fn timeExe(io: std.Io, allocator: std.mem.Allocator, exe: *zml.Exe, bufs: anytype, iters: usize) !Timing {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(bufs);

    // warmup (jit/compile caches), then time.
    exe.callOpts(io, args, &results, .{ .wait = true });
    const n = @max(iters, 1);
    const t0 = std.Io.Timestamp.now(io, .awake);
    for (0..n) |_| exe.callOpts(io, args, &results, .{ .wait = true });
    const dur = t0.untilNow(io, .awake);
    const ns: f64 = @floatFromInt(dur.toNanoseconds());

    var out = results.get(zml.Buffer);
    out.deinit();
    return .{ .ns_per_call = ns / @as(f64, @floatFromInt(n)) };
}

// Bytes moved (read weights + read activations + write output), per variant. At M=1 the weight
// term dominates and is the GEMV roofline; int8 weight is half the bf16 weight.
fn bytesMoved(v: Variant, m: i64, k: i64, n: i64) f64 {
    const mf: f64 = @floatFromInt(m);
    const kf: f64 = @floatFromInt(k);
    const nf: f64 = @floatFromInt(n);
    return switch (v) {
        .bf16 => 2.0 * nf * kf + 2.0 * mf * kf + 2.0 * mf * nf,
        .i8 => 1.0 * nf * kf + 1.0 * mf * kf + 4.0 * mf * nf, // i32 out
        .w8a8 => 1.0 * nf * kf + 2.0 * mf * kf + 2.0 * mf * nf, // bf16 in/out, i8 weight
        .woq => 1.0 * nf * kf + 2.0 * mf * kf + 2.0 * mf * nf, // bf16 in/out, i8 weight read
    };
}

const RunResult = struct { ns: f64, gflops: f64, gbs: f64 };

fn runOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime v: Variant,
    lt: Layout,
    m: i64,
    k: i64,
    n: i64,
    iters: usize,
    rand: std.Random,
) !RunResult {
    const x_dtype: zml.DataType = if (v == .i8) .i8 else .bf16;
    const x_shape = zml.Shape.init(.{ .m = m, .k = k }, x_dtype);
    const w_dtype: zml.DataType = if (v == .bf16) .bf16 else .i8;
    const w_shape = weightShape(lt, n, k, w_dtype);
    const ws_shape = zml.Shape.init(.{ .n = n }, .f32);

    // ---- host data + device buffers ----
    var x_buf = blk: {
        if (x_dtype == .i8) {
            const h = try allocator.alloc(i8, @intCast(m * k));
            defer allocator.free(h);
            for (h) |*e| e.* = rand.intRangeAtMost(i8, -127, 127);
            break :blk try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_shape, std.mem.sliceAsBytes(h)), .replicated);
        } else {
            const h = try allocator.alloc(u16, @intCast(m * k));
            defer allocator.free(h);
            for (h) |*e| e.* = bf16Bits(rand.floatNorm(f32) * 0.05);
            break :blk try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_shape, std.mem.sliceAsBytes(h)), .replicated);
        }
    };
    defer x_buf.deinit();

    var w_buf = blk: {
        if (w_dtype == .i8) {
            const h = try allocator.alloc(i8, @intCast(n * k));
            defer allocator.free(h);
            for (h) |*e| e.* = rand.intRangeAtMost(i8, -127, 127);
            break :blk try zml.Buffer.fromSlice(io, platform, zml.Slice.init(w_shape, std.mem.sliceAsBytes(h)), .replicated);
        } else {
            const h = try allocator.alloc(u16, @intCast(n * k));
            defer allocator.free(h);
            for (h) |*e| e.* = bf16Bits(rand.floatNorm(f32) * 0.05);
            break :blk try zml.Buffer.fromSlice(io, platform, zml.Slice.init(w_shape, std.mem.sliceAsBytes(h)), .replicated);
        }
    };
    defer w_buf.deinit();

    var ws_buf = blk: {
        const h = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(h);
        for (h) |*e| e.* = 0.01 + rand.float(f32) * 0.04;
        break :blk try zml.Buffer.fromSlice(io, platform, zml.Slice.init(ws_shape, std.mem.sliceAsBytes(h)), .replicated);
    };
    defer ws_buf.deinit();

    // ---- compile + time the chosen variant ----
    const Model = switch (v) {
        .bf16 => Bf16Gemm,
        .i8 => Int8Gemm,
        .w8a8 => W8A8Linear,
        .woq => WoqLinear,
    };
    const model: Model = switch (v) {
        .bf16 => .{ .w = .fromShape(w_shape) },
        .i8 => .{ .w = .fromShape(w_shape) },
        .w8a8 => .{ .w = .fromShape(w_shape), .wscale = .fromShape(ws_shape) },
        .woq => .{ .w = .fromShape(w_shape), .wscale = .fromShape(ws_shape) },
    };
    var exe = try platform.compile(allocator, io, model, .forward, .{zml.Tensor.fromShape(x_shape)}, .{});
    defer exe.deinit();

    const t = switch (v) {
        .bf16, .i8 => try timeExe(io, allocator, &exe, .{ w_buf, x_buf }, iters),
        .w8a8, .woq => try timeExe(io, allocator, &exe, .{ w_buf, ws_buf, x_buf }, iters),
    };

    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(n));
    return .{
        .ns = t.ns_per_call,
        .gflops = flops / t.ns_per_call,
        .gbs = bytesMoved(v, m, k, n) / t.ns_per_call,
    };
}

// Run + log one int8 variant for a (shape, M). Factored out of the inline-for in main so the
// runtime error handling (catch/return) is not "comptime control flow inside a runtime block".
fn logVariant(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime v: Variant,
    want_variant: []const u8,
    lt: Layout,
    mu: usize,
    m: i64,
    sd: ShapeDef,
    iters: usize,
    rand: std.Random,
    bf16_ns: f64,
) void {
    if (!variantSelected(want_variant, v)) return;
    const r = runOne(allocator, io, platform, v, lt, m, sd.k, sd.n, iters, rand) catch |e| {
        log.err("{s} M={d} {s}: {s}", .{ sd.name, mu, @tagName(v), @errorName(e) });
        return;
    };
    const ratio = if (bf16_ns > 0) bf16_ns / r.ns else 0;
    log.info("{s:>10} {d:>6} {d:>6} {d:>6} | {s:>6} {d:>9.4} {d:>8.1} {d:>8.1} | {d:>6.2}x", .{
        sd.name, mu, sd.k, sd.n, @tagName(v), r.ns / 1e6, r.gflops / 1e3, r.gbs, ratio,
    });
}

fn variantSelected(want: []const u8, v: Variant) bool {
    if (std.mem.eql(u8, want, "all")) return true;
    return std.mem.eql(u8, want, @tagName(v));
}

fn shapeSelected(want: []const u8, name: []const u8) bool {
    if (std.mem.eql(u8, want, "all")) return true;
    return std.mem.eql(u8, want, name);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform: {f}", .{platform.fmtVerbose()});

    const lt: Layout = if (std.mem.eql(u8, args.layout, "kn")) .kn else .nk;
    log.info("sweep: shape={s} m={d} variant={s} iters={d} layout={s}", .{ args.shape, args.m, args.variant, args.iters, @tagName(lt) });
    log.info("{s:>10} {s:>6} {s:>6} {s:>6} | {s:>6} {s:>9} {s:>8} {s:>8} | {s}", .{
        "shape", "M", "K", "N", "var", "ms/call", "TFLOP/s", "GB/s", "vs-bf16",
    });

    var prng: std.Random.DefaultPrng = .init(0xb70b3c);
    const rand = prng.random();

    for (SHAPES) |sd| {
        if (!shapeSelected(args.shape, sd.name)) continue;

        const m_list: []const usize = if (args.m == 0) &M_SWEEP else &[_]usize{args.m};
        for (m_list) |mu| {
            const m: i64 = @intCast(mu);

            // bf16 baseline (always run if any variant needs the ratio, and if selected).
            var bf16_ns: f64 = 0;
            const want_ratio = !std.mem.eql(u8, args.variant, "bf16");
            if (variantSelected(args.variant, .bf16) or want_ratio) {
                const r = runOne(allocator, io, platform, .bf16, lt, m, sd.k, sd.n, args.iters, rand) catch |e| {
                    log.err("{s} M={d} bf16: {s}", .{ sd.name, mu, @errorName(e) });
                    continue;
                };
                bf16_ns = r.ns;
                if (variantSelected(args.variant, .bf16))
                    log.info("{s:>10} {d:>6} {d:>6} {d:>6} | {s:>6} {d:>9.4} {d:>8.1} {d:>8.1} | {s:>7}", .{
                        sd.name, mu, sd.k, sd.n, "bf16", r.ns / 1e6, r.gflops / 1e3, r.gbs, "1.00x",
                    });
            }

            inline for (.{ Variant.i8, Variant.w8a8, Variant.woq }) |v| {
                logVariant(allocator, io, platform, v, args.variant, lt, mu, m, sd, args.iters, rand, bf16_ns);
            }
        }
    }
    log.info("done.", .{});
}
