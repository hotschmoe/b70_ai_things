// W8A8 GEMM/GEMV sweep (zml/ZML_W8A8.md follow-ups): push int8 toward the ~2x peak, profile the
// decode-shape (M=1) GEMV, and CHARACTERIZE the kernel robustly (median timing, layout probe,
// per-call spread) so a one-off device stall can no longer masquerade as a kernel-selection cliff.
//
// Extends the M4 perf gate (//examples/w8a8_bench) from a single shape/dtype to a sweep over:
//   - the REAL qwen3.6-27b W8A8 projection shapes (q/k/v/o, gate/up/down) + two square shapes,
//   - an M (token-count) sweep from 1 (decode GEMV) up to 8192 (large prefill / saturation),
//   - five variants per (shape, M):
//       bf16  : pure bf16 GEMM                         (baseline)
//       i8    : raw s8 x s8 -> s32 dotAcc, i32 store   (the proven INT8-XMX fast path, M4)
//       i8b   : raw s8 x s8 -> s32 dotAcc, bf16 store  (i8 with the i32 epilogue down-converted
//               to bf16 -- isolates the 4B-vs-2B output-store cost and whether oneDNN fuses the
//               down-convert into the GEMM epilogue; i8b ~= i8 means store is free/fused)
//       w8a8  : full QuantizedLinear (dynamic per-token act-quant + s8 dot + dequant), bf16 in/out
//               (the REAL serve path; isolates the quant epilogue overhead vs raw i8)
//       woq   : weight-only int8 (i8 weight dequant->bf16 in-graph, bf16 activation, bf16 dot)
//               (the GEMV alternative; tests whether XLA reads i8 from VRAM and expands in-reg
//                -- a bandwidth win at M=1 -- or materializes a bf16 weight first -- no win)
//   - either weight LAYOUT (or BOTH back-to-back): nk = weight {n,k} (model layout, contract k)
//       vs kn = weight {k,n} (transposed, contract k). The dot contracts on the `.k` TAG either
//       way, so both are numerically identical; only the physical weight memory order differs.
//       This probes whether the oneDNN int8/bf16 kernel prefers a k-major or n-major weight.
//
// Timing is MEDIAN-of-iters with a separate warmup discard, and reports per-call min/max so a
// one-time stall (the M=2048 bf16 2502ms/call anomaly) is visible as max>>med instead of being
// averaged into the headline. A real shape cliff would instead show med itself elevated.
//
// Reports per config: median/min/max ms/call, effective TFLOP/s (+ % of the 367 INT8-TOPS peak),
// effective GB/s (+ % of the 608 GB/s peak), and the variant/bf16 speedup. GB/s near peak at M=1
// => bandwidth-bound (int8 weight is half the bf16 weight => ~2x is the ceiling). TFLOP/s near the
// INT8-XMX peak at large M => compute-bound. The %pk columns are vs the SAME references for every
// variant (367 TOPS, 608 GB/s), so bf16 tops out near 50% of the compute axis and int8's 2x
// headroom is read off directly.
//
// CPU build just compiles (the timings are meaningless without the GPU). GPU run (ONE card, daily
// driver down, under the gpu-run lease):
//   cd /mnt/vm_8tb/b70/zml
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 CCL_TOPO_P2P_ACCESS=0 ZE_FLAT_DEVICE_HIERARCHY=FLAT \
//   ~/.local/bin/bazelisk run //examples/w8a8_sweep --config=release \
//     --@zml//platforms:cpu=false --@zml//platforms:oneapi=true -- --shape=all --mset=coarse --iters=100
//
// Flags:
//   --shape=   all | q | k | v | o | gate | up | down | sq4096 | sq8192        (default all)
//   --m=       0 (use --mset) | <single M>                                     (default 0)
//   --mset=    coarse {1,512,2048,4096,8192} | full {1,2,..,8192} | decode {1,2,4,8}  (default coarse)
//   --variant= all | bf16 | i8 | i8b | w8a8 | woq                              (default all)
//   --iters=   timed (median'd) calls per config                              (default 50)
//   --warmup=  untimed warmup calls discarded before timing                   (default 5)
//   --layout=  nk (weight {n,k}) | kn (weight {k,n}) | both (run nk then kn)   (default nk)

const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.w8a8_sweep);

pub const std_options: std.Options = .{
    .log_level = .info,
};

// B70 (Battlemage/Xe2) roofline references: ~367 INT8 TOPS (= TFLOP/s of int8 MACs) and
// ~608 GB/s HBM. bf16 dense peak is ~half the INT8 peak, so against the 367 axis a perfect bf16
// GEMM reads ~50% and the int8 2x headroom over it is read off the %pk column directly.
const PEAK_TOPS: f64 = 367.0;
const PEAK_GBS: f64 = 608.0;

const Args = struct {
    shape: []const u8 = "all",
    m: usize = 0,
    mset: []const u8 = "coarse",
    variant: []const u8 = "all",
    iters: usize = 50,
    warmup: usize = 5,
    layout: []const u8 = "nk",

    pub const help =
        \\w8a8_sweep [--shape=all|q|k|v|o|gate|up|down|sq4096|sq8192] [--m=0|N]
        \\           [--mset=coarse|full|decode] [--variant=all|bf16|i8|i8b|w8a8|woq]
        \\           [--iters=N] [--warmup=N] [--layout=nk|kn|both]
        \\  Sweep bf16 vs int8 GEMM/GEMV over the real qwen3.6-27b W8A8 projection shapes.
        \\  Median-of-iters timing with per-call min/max; weight-layout (nk/kn) probe.
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

// M sets. coarse = characterization (decode + a few prefill/saturation points incl. the M=2048
// anomaly point and an M=8192 saturation point). full = the original fine sweep + 8192. decode =
// the bandwidth-bound GEMV regime only.
const M_COARSE = [_]usize{ 1, 512, 2048, 4096, 8192 };
const M_FULL = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
const M_DECODE = [_]usize{ 1, 2, 4, 8 };

const Variant = enum { bf16, i8, i8b, w8a8, woq };

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
        return x.dotAcc(.i32, self.w, .k); // {m,n} i32 -- true s8 operands -> INT8-XMX, 4B store
    }
};

// i8 with the i32 accumulator down-converted to bf16 in-graph (no dequant scale -- timing only).
// Isolates the output-store cost (4B i32 vs 2B bf16) and whether oneDNN fuses the down-convert
// into the int8-GEMM epilogue. If i8b ~= i8 the store is free/fused; if i8b is faster at large
// M*N the i32 store was a real bandwidth tax on the i8 path.
const Int8GemmB = struct {
    w: zml.Tensor, // {n,k} or {k,n} i8
    pub fn forward(self: Int8GemmB, x: zml.Tensor) zml.Tensor {
        return x.dotAcc(.i32, self.w, .k).convert(.bf16); // {m,n} bf16
    }
};

// Full W8A8 linear: dynamic per-token int8 activation quant + s8 dot + per-channel/per-token
// dequant. Mirrors examples/llm/models/common_quant.zig QuantizedLinear.forward exactly, so the
// extra cost over Int8Gemm is purely the act-quant prologue + dequant epilogue.
const W8A8Linear = struct {
    w: zml.Tensor, // {n,k} or {k,n} i8
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
    w: zml.Tensor, // {n,k} or {k,n} i8
    wscale: zml.Tensor, // {n} f32
    pub fn forward(self: WoqLinear, x: zml.Tensor) zml.Tensor {
        const wf = self.w.convert(.bf16).mul(self.wscale.convert(.bf16).broad(self.w.shape()));
        return x.dot(wf, .k); // {m,n} bf16
    }
};

// ---- timing ---------------------------------------------------------------------------------

// Median-of-iters with separate warmup discard and per-call min/max. Each call is timed
// individually (callOpts wait=true blocks until the device finishes), so a single stalled call
// shows up as a large max while the median stays clean -- the key to telling a one-off stall from
// a real shape cliff. The mean is also returned to expose the averaging artifact (mean >> median
// when a stall is present, as in the old mean-of-N harness).
const Timing = struct { med_ns: f64, min_ns: f64, max_ns: f64, mean_ns: f64 };

fn timeExe(
    io: std.Io,
    allocator: std.mem.Allocator,
    exe: *zml.Exe,
    bufs: anytype,
    warmup: usize,
    iters: usize,
) !Timing {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(bufs);

    // warmup (jit/compile/autotune caches), discarded.
    var wi: usize = 0;
    while (wi < @max(warmup, 1)) : (wi += 1)
        exe.callOpts(io, args, &results, .{ .wait = true });

    const n = @max(iters, 1);
    const samples = try allocator.alloc(f64, n);
    defer allocator.free(samples);
    for (samples) |*s| {
        const c0 = std.Io.Timestamp.now(io, .awake);
        exe.callOpts(io, args, &results, .{ .wait = true });
        const d = c0.untilNow(io, .awake);
        s.* = @floatFromInt(d.toNanoseconds());
    }

    var out = results.get(zml.Buffer);
    out.deinit();

    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (samples) |s| sum += s;
    return .{
        .med_ns = samples[n / 2],
        .min_ns = samples[0],
        .max_ns = samples[n - 1],
        .mean_ns = sum / @as(f64, @floatFromInt(n)),
    };
}

// Bytes moved (read weights + read activations + write output), per variant. At M=1 the weight
// term dominates and is the GEMV roofline; int8 weight is half the bf16 weight.
fn bytesMoved(v: Variant, m: i64, k: i64, n: i64) f64 {
    const mf: f64 = @floatFromInt(m);
    const kf: f64 = @floatFromInt(k);
    const nf: f64 = @floatFromInt(n);
    return switch (v) {
        .bf16 => 2.0 * nf * kf + 2.0 * mf * kf + 2.0 * mf * nf,
        .i8 => 1.0 * nf * kf + 1.0 * mf * kf + 4.0 * mf * nf, // i8 weight, i8 act, i32 out
        .i8b => 1.0 * nf * kf + 1.0 * mf * kf + 2.0 * mf * nf, // i8 weight, i8 act, bf16 out
        .w8a8 => 1.0 * nf * kf + 2.0 * mf * kf + 2.0 * mf * nf, // bf16 in/out, i8 weight
        .woq => 1.0 * nf * kf + 2.0 * mf * kf + 2.0 * mf * nf, // bf16 in/out, i8 weight read
    };
}

const RunResult = struct {
    med_ns: f64,
    min_ns: f64,
    max_ns: f64,
    mean_ns: f64,
    gflops: f64, // from median
    gbs: f64, // from median
};

fn runOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime v: Variant,
    lt: Layout,
    m: i64,
    k: i64,
    n: i64,
    warmup: usize,
    iters: usize,
    rand: std.Random,
) !RunResult {
    const x_dtype: zml.DataType = if (v == .i8 or v == .i8b) .i8 else .bf16;
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
        .i8b => Int8GemmB,
        .w8a8 => W8A8Linear,
        .woq => WoqLinear,
    };
    const model: Model = switch (v) {
        .bf16 => .{ .w = .fromShape(w_shape) },
        .i8 => .{ .w = .fromShape(w_shape) },
        .i8b => .{ .w = .fromShape(w_shape) },
        .w8a8 => .{ .w = .fromShape(w_shape), .wscale = .fromShape(ws_shape) },
        .woq => .{ .w = .fromShape(w_shape), .wscale = .fromShape(ws_shape) },
    };
    var exe = try platform.compile(allocator, io, model, .forward, .{zml.Tensor.fromShape(x_shape)}, .{});
    defer exe.deinit();

    const t = switch (v) {
        .bf16, .i8, .i8b => try timeExe(io, allocator, &exe, .{ w_buf, x_buf }, warmup, iters),
        .w8a8, .woq => try timeExe(io, allocator, &exe, .{ w_buf, ws_buf, x_buf }, warmup, iters),
    };

    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(n));
    return .{
        .med_ns = t.med_ns,
        .min_ns = t.min_ns,
        .max_ns = t.max_ns,
        .mean_ns = t.mean_ns,
        .gflops = flops / t.med_ns,
        .gbs = bytesMoved(v, m, k, n) / t.med_ns,
    };
}

// ---- reporting ------------------------------------------------------------------------------

fn logHeader() void {
    // med_ms is the headline (TFLOP/s, GB/s, %pk, vs-bf16 all derive from it). min/max bracket the
    // per-call spread; mean_ms reproduces the OLD mean-of-N number so the M=2048-style contamination
    // (mean >> med == a one-off stall) is read directly.
    log.info("{s:>10} {s:>6} {s:>6} {s:>6} {s:>3} | {s:>5} {s:>8} {s:>8} {s:>8} {s:>8} | {s:>7} {s:>5} | {s:>6} {s:>5} | {s}", .{
        "shape", "M", "K", "N", "L", "var", "med_ms", "min_ms", "max_ms", "mean_ms", "TFLOP/s", "%pk", "GB/s", "%pk", "vs-bf16",
    });
}

fn logRow(sd: ShapeDef, mu: usize, lt: Layout, vname: []const u8, r: RunResult, ratio: f64) void {
    const tflops = r.gflops / 1e3;
    log.info("{s:>10} {d:>6} {d:>6} {d:>6} {s:>3} | {s:>5} {d:>8.4} {d:>8.4} {d:>8.4} {d:>8.4} | {d:>7.1} {d:>5.1} | {d:>6.1} {d:>5.1} | {d:>6.2}x", .{
        sd.name, mu, sd.k, sd.n, @tagName(lt), vname,
        r.med_ns / 1e6, r.min_ns / 1e6, r.max_ns / 1e6, r.mean_ns / 1e6,
        tflops, 100.0 * tflops / PEAK_TOPS,
        r.gbs, 100.0 * r.gbs / PEAK_GBS,
        ratio,
    });
}

// Run + log one non-bf16 variant for a (shape, M, layout). Factored out of the inline-for in main
// so the runtime error handling (catch/return) is not "comptime control flow inside a runtime
// block".
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
    warmup: usize,
    iters: usize,
    rand: std.Random,
    bf16_ns: f64,
) void {
    if (!variantSelected(want_variant, v)) return;
    const r = runOne(allocator, io, platform, v, lt, m, sd.k, sd.n, warmup, iters, rand) catch |e| {
        log.err("{s} M={d} {s} {s}: {s}", .{ sd.name, mu, @tagName(lt), @tagName(v), @errorName(e) });
        return;
    };
    const ratio = if (bf16_ns > 0) bf16_ns / r.med_ns else 0;
    logRow(sd, mu, lt, @tagName(v), r, ratio);
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

    // layouts to run: nk, kn, or both (back-to-back so the kn/nk delta is read off adjacent rows).
    const want_layouts: []const Layout = if (std.mem.eql(u8, args.layout, "both"))
        &[_]Layout{ .nk, .kn }
    else if (std.mem.eql(u8, args.layout, "kn"))
        &[_]Layout{.kn}
    else
        &[_]Layout{.nk};

    // M list: an explicit --m=N wins; otherwise pick the named set.
    const single_m = [1]usize{args.m};
    const m_list: []const usize = if (args.m != 0)
        single_m[0..]
    else if (std.mem.eql(u8, args.mset, "full"))
        &M_FULL
    else if (std.mem.eql(u8, args.mset, "decode"))
        &M_DECODE
    else
        &M_COARSE;

    log.info("sweep: shape={s} m={d} mset={s} variant={s} iters={d} warmup={d} layout={s}", .{
        args.shape, args.m, args.mset, args.variant, args.iters, args.warmup, args.layout,
    });
    log.info("peaks: {d:.0} INT8-TOPS, {d:.0} GB/s (bf16 dense peak ~= half the TOPS axis)", .{ PEAK_TOPS, PEAK_GBS });
    logHeader();

    var prng: std.Random.DefaultPrng = .init(0xb70b3c);
    const rand = prng.random();

    for (SHAPES) |sd| {
        if (!shapeSelected(args.shape, sd.name)) continue;

        for (m_list) |mu| {
            const m: i64 = @intCast(mu);

            for (want_layouts) |lt| {
                // bf16 baseline (run if bf16 is selected OR a ratio is needed for another variant).
                var bf16_ns: f64 = 0;
                const want_ratio = !std.mem.eql(u8, args.variant, "bf16");
                if (variantSelected(args.variant, .bf16) or want_ratio) {
                    const r = runOne(allocator, io, platform, .bf16, lt, m, sd.k, sd.n, args.warmup, args.iters, rand) catch |e| {
                        log.err("{s} M={d} {s} bf16: {s}", .{ sd.name, mu, @tagName(lt), @errorName(e) });
                        continue;
                    };
                    bf16_ns = r.med_ns;
                    if (variantSelected(args.variant, .bf16))
                        logRow(sd, mu, lt, "bf16", r, 1.0);
                }

                inline for (.{ Variant.i8, Variant.i8b, Variant.w8a8, Variant.woq }) |v| {
                    logVariant(allocator, io, platform, v, args.variant, lt, mu, m, sd, args.warmup, args.iters, rand, bf16_ns);
                }
            }
        }
    }
    log.info("done.", .{});
}
