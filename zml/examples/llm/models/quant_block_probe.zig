// M3 block-parity probe (zml/ZML_W8A8.md): wire QuantizedLinear into a real qwen3.6 block.
//
// The MLP block (down(silu(gate(x)) * up(x))) is the natural M3 slice: it is W8A8 on EVERY
// layer (only linear_attn/vision/mtp/lm_head are in the ignore list), and it is pure
// projections + SiLU -- no rotary, attention, KV cache, or GDN. This composes THREE quantized
// projections + a nonlinearity, so it exercises error accumulation through a real block.
//
// Loads layer-0 mlp (gate/up/down) from models/files/qwen3.6-27b/w8a8-sqgptq into a quantized
// Mlp and compares, on the XLA CPU backend, against a reference Mlp built from the SAME loaded
// int8 weights dequantized in-graph to bf16 (what the bf16 model effectively computes). The
// gap is the block's int8 activation-quant error. bf16 activations, matching the real model.
//
// CPU only, no GPU. Run:
//   cd /mnt/vm_8tb/b70/zml
//   ~/.local/bin/bazelisk run //examples/llm:quant_block_probe --config=release -- \
//     --model=/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/w8a8-sqgptq

const std = @import("std");

const zml = @import("zml");

const common_quant = @import("common_quant.zig");
const QuantizedLinear = common_quant.QuantizedLinear;

const log = std.log.scoped(.quant_block_probe);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8 = "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/w8a8-sqgptq",
    layer: usize = 0, // mlp is W8A8 on every layer; layer 0 (a GDN layer) is fine
    tokens: usize = 4,

    pub const help =
        \\quant_block_probe --model=<dir> [--layer=N] [--tokens=N]
        \\
        \\ Load a real W8A8 mlp block into a quantized Mlp and validate vs a bf16 dequant ref on CPU.
    ;
};

// Quantized qwen3.6 MLP block, mirroring examples/llm/models/qwen3_5/model.zig Mlp.forward but
// with QuantizedLinear projections. forward returns BOTH the quantized output and an in-graph
// dequant-weight bf16 reference computed from the same loaded int8 weights.
const QuantMlp = struct {
    up_proj: QuantizedLinear, // weight {dout, d} i8, contract .d
    gate_proj: QuantizedLinear, // weight {dout, d} i8, contract .d
    down_proj: QuantizedLinear, // weight {d, dout} i8, contract .dout (transposed layout)

    const Out = struct { yq: zml.Tensor, yref: zml.Tensor };

    fn refLinear(q: QuantizedLinear) zml.nn.Linear {
        return .{ .weight = q.dequantWeight(.bf16), .bias = q.bias, .tag = q.tag };
    }

    pub fn forward(self: QuantMlp, x_in: zml.Tensor) Out {
        const x = x_in.convert(.bf16); // real model activations are bf16

        // quantized block
        const hidden_q = self.gate_proj.forward(x).silu().mul(self.up_proj.forward(x));
        const yq = self.down_proj.forward(hidden_q);

        // reference block: same math, dequant(int8)->bf16 weights via nn.Linear
        const hidden_r = refLinear(self.gate_proj).forward(x).silu().mul(refLinear(self.up_proj).forward(x));
        const yref = refLinear(self.down_proj).forward(hidden_r);

        return .{ .yq = yq.convert(.f32), .yref = yref.convert(.f32) };
    }
};

fn loadQL(view: zml.io.TensorStore.View, name: []const u8, comptime w_tags: anytype, comptime ws_tags: anytype, comptime out_tag: anytype, comptime tag: anytype) QuantizedLinear {
    const v = view.withPrefix(name);
    return .init(
        v.createTensor("weight", w_tags, .replicated),
        v.createTensor("weight_scale", ws_tags, .replicated),
        v.maybeCreateTensor("bias", out_tag, .replicated), // absent in this checkpoint -> null
        tag,
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var key_buf: [256]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp", .{args.layer});
    const mlp_view = store.view().withPrefix(prefix);
    if (mlp_view.getShape("up_proj.weight") == null) {
        log.err("no mlp.up_proj.weight at layer {d}", .{args.layer});
        std.process.exit(1);
    }
    const w_up = mlp_view.getShape("up_proj.weight").?;
    const w_down = mlp_view.getShape("down_proj.weight").?;
    log.info("layer {d} mlp: up/gate {f}, down {f}", .{ args.layer, w_up, w_down });

    // up/gate: weight {dout, d}, contract .d, scale {dout, 1}. down: weight {d, dout}, contract .dout.
    const mlp: QuantMlp = .{
        .up_proj = loadQL(mlp_view, "up_proj", .{ .dout, .d }, .{ .dout, .one }, .{.dout}, .d),
        .gate_proj = loadQL(mlp_view, "gate_proj", .{ .dout, .d }, .{ .dout, .one }, .{.dout}, .d),
        .down_proj = loadQL(mlp_view, "down_proj", .{ .d, .dout }, .{ .d, .one }, .{.d}, .dout),
    };

    const din: i64 = w_up.dim(1); // model hidden size (5120)
    const x_shape = zml.Shape.init(.{ .token = @as(i64, @intCast(args.tokens)), .d = din }, .f32);

    var exe = try platform.compile(allocator, io, mlp, .forward, .{zml.Tensor.fromShape(x_shape)}, .{});
    defer exe.deinit();

    var mlp_bufs = try zml.io.load(QuantMlp, &mlp, allocator, io, platform, &store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &mlp_bufs);
    log.info("loaded mlp block (lazy: gate/up/down only)", .{});

    const x_host = try allocator.alloc(f32, args.tokens * @as(usize, @intCast(din)));
    defer allocator.free(x_host);
    var prng: std.Random.DefaultPrng = .init(0xb10c);
    const rand = prng.random();
    for (x_host) |*v| v.* = rand.floatNorm(f32);
    var x_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(x_shape, std.mem.sliceAsBytes(x_host)), .replicated);
    defer x_buf.deinit();

    var out = try zml.testing.autoCall(allocator, io, &exe, QuantMlp.forward, .{ mlp_bufs, x_buf });
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
    const all_finite = n_finite == n and yq.len == yref.len;
    // block-level int8 quant error composes through 3 projections + silu; allow a wider tol.
    const ok = all_finite and rel_l2 < 0.05;

    log.info("yq[0..6] = {any}", .{yq[0..@min(yq.len, 6)]});
    log.info("block rel_l2 (W8A8 vs bf16 dequant ref) = {d:.5}, max_abs = {d:.5}, all_finite={}", .{ rel_l2, max_abs, all_finite });
    if (ok) {
        log.info("RESULT: PASS -- real W8A8 mlp block matches the bf16 dequant block on CPU.", .{});
    } else {
        log.err("RESULT: FAIL", .{});
        std.process.exit(1);
    }
}
