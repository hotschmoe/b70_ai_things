// M2 loader probe for the real compressed-tensors W8A8 checkpoint (zml/ZML_W8A8.md).
//
// Loads ONE real full-attention q_proj (layer 3) out of models/files/qwen3.6-27b/w8a8-sqgptq
// -- I8 weight [12288,5120] + BF16 weight_scale [12288,1] -- into QuantizedLinear via zml's
// own safetensors + TensorStore reflection, and validates it on the XLA CPU backend.
//
// The load is LAZY: only the two bound tensors stream off disk, NOT the 35 GB checkpoint.
//
// Two checks:
//   - the real artifact loads through QuantizedLinear (correct I8 + BF16[out,1] handling),
//   - QuantizedLinear(real weights) matches an in-graph dequant reference (the loaded int8
//     weight dequantized to f32, full-precision matmul) up to activation-quant error.
//
// CPU only, no GPU. Run:
//   cd /mnt/vm_8tb/b70/zml
//   ~/.local/bin/bazelisk run //examples/llm:quant_load_probe --config=release -- \
//     --model=/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/w8a8-sqgptq

const std = @import("std");

const zml = @import("zml");

const common_quant = @import("common_quant.zig");
const QuantizedLinear = common_quant.QuantizedLinear;

const log = std.log.scoped(.quant_load_probe);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8 = "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/w8a8-sqgptq",
    layer: usize = 3, // first full-attention layer
    tokens: usize = 4,

    pub const help =
        \\quant_load_probe --model=<dir> [--layer=N] [--tokens=N]
        \\
        \\ Load one real W8A8 q_proj into QuantizedLinear and validate on CPU.
    ;
};

// Probe model: a single QuantizedLinear bound to the real q_proj, plus an in-graph
// dequant reference computed from the SAME loaded int8 weight.
const Probe = struct {
    q: QuantizedLinear,

    const Out = struct {
        yq: zml.Tensor, // QuantizedLinear forward
        yref: zml.Tensor, // dequant(loaded int8 weight) full-precision matmul
    };

    pub fn forward(self: Probe, x: zml.Tensor) Out {
        // QuantizedLinear returns the activation dtype (bf16 here, matching the real model);
        // convert to f32 for an apples-to-apples compare with the f32 reference below.
        const yq = self.q.forward(x).convert(.f32);

        var ws = self.q.weight_scale.convert(.f32);
        if (ws.rank() > 1) ws = ws.squeeze(ws.rank() - 1);
        const wf = self.q.weight.convert(.f32).mul(ws.broad(self.q.weight.shape()));
        const yref = x.dot(wf, self.q.tag);

        return .{ .yq = yq, .yref = yref };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    // --- open the checkpoint registry (header-only; no tensor bytes read yet) ---
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // --- bind the real layer-N q_proj keys ---
    var key_buf: [256]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.self_attn.q_proj", .{args.layer});
    const qv = store.view().withPrefix(prefix);

    const w_shape = qv.getShape("weight") orelse {
        log.err("no q_proj.weight at layer {d} -- is it a full-attention layer? (3,7,11,...)", .{args.layer});
        std.process.exit(1);
    };
    const ws_shape = qv.getShape("weight_scale").?;
    log.info("loading {s}.weight {f} + .weight_scale {f}", .{ prefix, w_shape, ws_shape });

    const q: QuantizedLinear = .init(
        qv.createTensor("weight", .{ .dout, .d }, .replicated),
        qv.createTensor("weight_scale", .{ .dout, .one }, .replicated),
        qv.maybeCreateTensor("bias", .{.dout}, .replicated), // absent in this checkpoint -> null
        .d,
    );
    const probe: Probe = .{ .q = q };

    const din: i64 = w_shape.dim(1); // contracting dim (5120)
    const dout: i64 = w_shape.dim(0); // output channels (12288)
    const x_shape = zml.Shape.init(.{ .token = @as(i64, @intCast(args.tokens)), .d = din }, .f32);

    // --- compile, then stream ONLY the two bound tensors off disk ---
    var exe = try platform.compile(allocator, io, probe, .forward, .{zml.Tensor.fromShape(x_shape)}, .{});
    defer exe.deinit();

    var probe_bufs = try zml.io.load(Probe, &probe, allocator, io, platform, &store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &probe_bufs);
    log.info("loaded q_proj buffers (lazy: only this projection, not the 35GB checkpoint)", .{});

    // --- input ---
    const x_host = try allocator.alloc(f32, args.tokens * @as(usize, @intCast(din)));
    defer allocator.free(x_host);
    var prng: std.Random.DefaultPrng = .init(0x90a8);
    const rand = prng.random();
    for (x_host) |*v| v.* = rand.floatNorm(f32);
    var x_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_shape, std.mem.sliceAsBytes(x_host)), .replicated);
    defer x_buf.deinit();

    // --- run ---
    var out = try zml.testing.autoCall(allocator, io, &exe, Probe.forward, .{ probe_bufs, x_buf });
    defer {
        out.yq.deinit();
        out.yref.deinit();
    }

    const yq_slice = try out.yq.toSliceAlloc(allocator, io);
    defer yq_slice.free(allocator);
    const yref_slice = try out.yref.toSliceAlloc(allocator, io);
    defer yref_slice.free(allocator);
    const yq = yq_slice.items(f32);
    const yref = yref_slice.items(f32);
    log.info("yq shape {f} (len {d}); yref shape {f} (len {d})", .{ yq_slice.shape, yq.len, yref_slice.shape, yref.len });

    var sum_sq_err: f64 = 0;
    var sum_sq_ref: f64 = 0;
    var max_abs: f64 = 0;
    var n_finite: usize = 0;
    const n = @min(yq.len, yref.len);
    for (yq[0..n], yref[0..n]) |a, b| {
        if (std.math.isFinite(a)) n_finite += 1;
        const e = @as(f64, a) - @as(f64, b);
        sum_sq_err += e * e;
        sum_sq_ref += @as(f64, b) * @as(f64, b);
        max_abs = @max(max_abs, @abs(e));
    }
    const rel_l2 = std.math.sqrt(sum_sq_err / sum_sq_ref);
    const all_finite = n_finite == n;
    const ok = all_finite and rel_l2 < 0.05 and yq.len == yref.len;

    log.info("q_proj [{d}x{d}] -> y[{d}x{d}]; yq[0..6] = {any}", .{ dout, din, args.tokens, dout, yq[0..@min(yq.len, 6)] });
    log.info("all_finite={} ({d}/{d}); rel_l2 vs in-graph dequant ref = {d:.5}, max_abs = {d:.5}", .{ all_finite, n_finite, n, rel_l2, max_abs });
    if (ok) {
        log.info("RESULT: PASS -- real W8A8 q_proj loads into QuantizedLinear and flows correctly on CPU.", .{});
    } else {
        log.err("RESULT: FAIL", .{});
        std.process.exit(1);
    }
}
