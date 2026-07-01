const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5);
const Phase = common.Phase;

// libc getenv (binary links -lc); used only to gate the ZML_PROFILE_LAYERS per-layer-type timer.
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, shardings: common.Shardings) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        return .{
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, shardings.model),
            .rng = .init(),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    // If non-null, run() stores the post-layer-loop hidden buffer here (the last-layer output
    // BEFORE text_model.norm -- the `prev_hidden` the MTP head consumes) and transfers ownership
    // to the caller (who must deinit) INSTEAD of deinit-ing it. null -> hidden is freed as before.
    capture_hidden: ?*?zml.Buffer = null,
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe,
    decode: KernelExe,
    // NEXTN/MTP spec-decode exes (prefill s=seqlen + draft s=1), present iff the model has an MTP
    // head. null for bf16/non-MTP checkpoints -> the plain decode path runs unchanged.
    mtp: ?MtpExes,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        qwen_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        return .{
            .loaded_model = loaded_model,
            .prefill = try .init(
                allocator,
                io,
                platform,
                qwen_model,
                parameters,
                @intCast(parameters.prefill_tokens.dim(.s)),
                .prefill,
                false,
                progress,
            ),
            .decode = try .init(
                allocator,
                io,
                platform,
                qwen_model,
                parameters,
                @intCast(parameters.decode_tokens.dim(.s)),
                .decode,
                false,
                progress,
            ),
            .mtp = try MtpExes.init(allocator, io, platform, qwen_model, parameters, parameters.seqlen, progress),
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
        if (self.mtp) |*mtp| mtp.deinit();
    }
};

// The NEXTN/MTP speculative-decode executables, compiled once alongside prefill/decode. `prefill`
// fills the MTP's 1-layer self-attn KV over the prompt (s=seqlen, no lm_head); `draft` runs the
// head for one token (s=1) and returns the drafted token + updated cache + rng. See zml/ZML_MTP_PLAN.md.
// Speculative window: K=1 drafted token per step -> verify Kv=2 tokens [t, d_1] in one forward.
pub const MTP_K: usize = 1;
pub const MTP_KV: usize = MTP_K + 1;

pub const MtpExes = struct {
    prefill: zml.Exe, // model.MtpPrefill.forward @ s=seqlen -> updated SelfAttnCache
    draft: zml.Exe, // model.MtpDraft.forward @ s=1 -> (draft token, updated SelfAttnCache, rng)
    verify: KernelExe, // main model @ s=Kv (prefill-style sampler + GDN forwardVerify from cache)
    gdn_snapshot: zml.Exe, // model.GdnSnapshot.forward -> deep copy of the GDN state (rollback)
    slice_hidden: zml.Exe, // model.SliceHidden.forward -> extract one position of the verify hidden
    mtp_kv: model.KvCache.SelfAttnCache, // 1-layer cache template (drives buffer allocation)
    seqlen: u32,
    kv: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationParameters,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !?MtpExes {
        if (qwen_model.mtp_head == null) return null;

        const config = qwen_model.config;
        const dtype = qwen_model.text_model.embed_tokens.weight.dtype();
        const mtp_kv = model.KvCache.SelfAttnCache.initSingleLayer(config, 1, @intCast(seqlen), dtype, parameters.shardings.model);

        const prefill_exe = b: {
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start("compiling mtp prefill", 1);
            defer node.end();
            const from: std.Io.Timestamp = .now(io, .awake);
            defer Phase.prefill.logCompileDone(log, "mtp prefill", io, from);

            const prefill_tokens: zml.Tensor = .init(.{ .b = 1, .s = seqlen }, .u32);
            const prefill_hidden = mtpHidden(qwen_model, seqlen);
            const prefill_module: model.MtpPrefill = .{
                .embed_tokens = qwen_model.text_model.embed_tokens,
                .head = qwen_model.mtp_head.?,
                .target_norm = qwen_model.text_model.norm,
            };
            break :b try platform.compile(allocator, io, prefill_module, .forward, .{ prefill_tokens, prefill_hidden, parameters.token_index, mtp_kv }, .{
                .shardings = &parameters.shardings.all(),
                .program_name = "qwen3_5_mtp_prefill",
            });
        };
        errdefer prefill_exe.deinit();

        const draft_exe = b: {
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start("compiling mtp draft", 1);
            defer node.end();
            const from: std.Io.Timestamp = .now(io, .awake);
            defer Phase.decode.logCompileDone(log, "mtp draft", io, from);

            const draft_token: zml.Tensor = .init(.{ .b = 1, .s = 1 }, .u32);
            const draft_hidden = mtpHidden(qwen_model, 1);
            const draft_module: model.MtpDraft = .{
                .embed_tokens = qwen_model.text_model.embed_tokens,
                .head = qwen_model.mtp_head.?,
                .lm_head = qwen_model.lm_head,
                .target_norm = qwen_model.text_model.norm,
                .gen_options = qwen_model.gen_options,
            };
            break :b try platform.compile(allocator, io, draft_module, .forward, .{ draft_token, draft_hidden, parameters.token_index, mtp_kv, parameters.rng }, .{
                .shardings = &parameters.shardings.all(),
                .program_name = "qwen3_5_mtp_draft",
            });
        };
        errdefer draft_exe.deinit();

        // Verify: the FULL main model over s=Kv tokens against the committed cache. Compiled as a
        // prefill-phase exe (all-position argmax sampler, no token_index increment) at seqlen=Kv,
        // with verify=true so the GDN layer continues from the cached state (forwardVerify).
        var verify_exe = try KernelExe.init(allocator, io, platform, qwen_model, parameters, MTP_KV, .prefill, true, progress);
        errdefer verify_exe.deinit();

        // GDN snapshot exe: deep-copies the committed GDN conv+recurrent state before verify so a
        // rejected draft can be rolled back (the recurrent state is not invertible).
        const gdn_snapshot_exe = b: {
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start("compiling gdn snapshot", 1);
            defer node.end();
            const from: std.Io.Timestamp = .now(io, .awake);
            defer Phase.decode.logCompileDone(log, "gdn snapshot", io, from);

            const gdn_template: model.KvCache.GatedDeltaNetCache = .{
                .conv_state = parameters.kv_cache.gated_delta_net.conv_state,
                .recurrent_state = parameters.kv_cache.gated_delta_net.recurrent_state,
                .layer_index = zml.Tensor.init(.{}, .u32),
            };
            break :b try platform.compile(allocator, io, model.GdnSnapshot{}, .forward, .{gdn_template}, .{
                .shardings = &parameters.shardings.all(),
                .program_name = "qwen3_5_gdn_snapshot",
            });
        };
        errdefer gdn_snapshot_exe.deinit();

        // Slice-hidden exe: extract one sequence position from the s=Kv verify hidden ({b,Kv,d} ->
        // {b,1,d}) to feed the next MTP draft + the accept catch-up.
        const slice_hidden_exe = b: {
            const verify_hidden = mtpHidden(qwen_model, MTP_KV);
            const index: zml.Tensor = .init(.{}, .u32);
            break :b try platform.compile(allocator, io, model.SliceHidden{}, .forward, .{ verify_hidden, index }, .{
                .shardings = &parameters.shardings.all(),
                .program_name = "qwen3_5_slice_hidden",
            });
        };

        return .{
            .prefill = prefill_exe,
            .draft = draft_exe,
            .verify = verify_exe,
            .gdn_snapshot = gdn_snapshot_exe,
            .slice_hidden = slice_hidden_exe,
            .mtp_kv = mtp_kv,
            .seqlen = @intCast(seqlen),
            .kv = MTP_KV,
        };
    }

    pub fn deinit(self: *MtpExes) void {
        self.prefill.deinit();
        self.draft.deinit();
        self.verify.deinit();
        self.gdn_snapshot.deinit();
        self.slice_hidden.deinit();
    }
};

// The MTP head's per-position hidden input template ({b=1, s, d} replicated), matching the shape of
// the captured main-model hidden (ComposedKernelExe.hidden). See zml/ZML_MTP_PLAN.md.
fn mtpHidden(qwen_model: model.Model, seqlen: usize) zml.Tensor {
    return .fromShape(zml.Shape.init(
        .{ .b = 1, .s = seqlen, .d = qwen_model.config.text_config.hidden_size },
        qwen_model.text_model.embed_tokens.weight.dtype(),
    ).withPartitioning(.{
        .b = .replicated,
        .s = .replicated,
        .d = .replicated,
    }));
}

// Buffer-donation-aware swap helpers, shared by the composed exe path and the MTP path in
// session.zig. A compiled graph may donate (reuseBuffer) an input into an output, so the returned
// handle can equal the input handle; only deinit the old handle when it actually changed.
pub fn replaceBufferImpl(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandleImpl(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

pub fn releaseBufferImpl(expected: zml.Buffer, actual: *zml.Buffer) void {
    if (!sameBufferHandleImpl(expected, actual.*)) {
        actual.deinit();
    }
}

pub fn sameBufferHandleImpl(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}

pub const Inference = CompiledModel;

pub const KernelExe = struct {
    composed: ComposedKernelExe,

    pub const Runner = struct {
        exe: *const ComposedKernelExe,
        embed_args: zml.exe.Exe.Arguments,
        embed_results: zml.exe.Exe.Results,
        layers: Layers,
        sampler_args: zml.exe.Exe.Arguments,
        sampler_results: zml.exe.Exe.Results,
        // Per-layer-type profiling (ZML_PROFILE_LAYERS=1): sync + time each layer to attribute decode
        // time to int8 full-attention vs bf16+f32-scan GDN (linear-attention). Perturbs (syncs per
        // layer) -- diagnostic only. Read by session.zig after the decode loop.
        prof: bool = false,
        prof_full_ns: u64 = 0,
        prof_linear_ns: u64 = 0,

        const Layers = struct {
            args: []zml.exe.Exe.Arguments,
            results: []zml.exe.Exe.Results,
            layer_indices: []zml.Buffer,
            layer_types: []const model.LayerType,

            fn init(
                allocator: std.mem.Allocator,
                io: std.Io,
                platform: *const zml.Platform,
                exe: *const ComposedKernelExe,
                model_buffers: *model.Buffers,
            ) !Layers {
                const args = try allocator.alloc(zml.exe.Exe.Arguments, model_buffers.text_model.layers.len);
                errdefer allocator.free(args);

                const results = try allocator.alloc(zml.exe.Exe.Results, model_buffers.text_model.layers.len);
                errdefer allocator.free(results);

                const layer_indices = try allocator.alloc(zml.Buffer, model_buffers.text_model.layers.len);
                errdefer allocator.free(layer_indices);

                var initialized_args: usize = 0;
                errdefer {
                    for (args[0..initialized_args]) |*exe_args| {
                        exe_args.deinit(allocator);
                    }
                }

                var initialized_results: usize = 0;
                errdefer {
                    for (results[0..initialized_results]) |*exe_results| {
                        exe_results.deinit(allocator);
                    }
                }

                var initialized_layer_indices: usize = 0;
                errdefer {
                    for (layer_indices[0..initialized_layer_indices]) |*layer_index| {
                        layer_index.deinit();
                    }
                }

                var self_attn_layer_index: usize = 0;
                var linear_attn_layer_index: usize = 0;
                for (model_buffers.text_model.layers, exe.layer_types, 0..) |layer_bufs, layer_type, i| {
                    args[i] = try switch (layer_type) {
                        .full_attention => (exe.full_attention_layer orelse unreachable).args(allocator),
                        .linear_attention => (exe.linear_attention_layer orelse unreachable).args(allocator),
                    };
                    initialized_args = i + 1;
                    args[i].bake(layer_bufs);

                    results[i] = try switch (layer_type) {
                        .full_attention => (exe.full_attention_layer orelse unreachable).results(allocator),
                        .linear_attention => (exe.linear_attention_layer orelse unreachable).results(allocator),
                    };
                    initialized_results = i + 1;

                    layer_indices[i] = try switch (layer_type) {
                        .full_attention => b: {
                            defer self_attn_layer_index += 1;
                            break :b zml.Buffer.scalar(io, platform, @as(u32, @intCast(self_attn_layer_index)), .u32);
                        },
                        .linear_attention => b: {
                            defer linear_attn_layer_index += 1;
                            break :b zml.Buffer.scalar(io, platform, @as(u32, @intCast(linear_attn_layer_index)), .u32);
                        },
                    };
                    initialized_layer_indices = i + 1;
                }

                return .{
                    .args = args,
                    .results = results,
                    .layer_indices = layer_indices,
                    .layer_types = exe.layer_types,
                };
            }

            fn deinit(self: *Layers, allocator: std.mem.Allocator) void {
                for (self.args) |*exe_args| {
                    exe_args.deinit(allocator);
                }
                allocator.free(self.args);

                for (self.results) |*exe_results| {
                    exe_results.deinit(allocator);
                }
                allocator.free(self.results);

                for (self.layer_indices) |*layer_index| {
                    layer_index.deinit();
                }
                allocator.free(self.layer_indices);
            }
        };

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            exe: *const ComposedKernelExe,
            model_buffers: *model.Buffers,
        ) !Runner {
            var embed_args = try exe.embed_tokens.args(allocator);
            errdefer embed_args.deinit(allocator);
            embed_args.bake(ComposedKernelExe.embedTokensBuffers(model_buffers));

            var embed_results = try exe.embed_tokens.results(allocator);
            errdefer embed_results.deinit(allocator);

            var layers = try Layers.init(allocator, io, platform, exe, model_buffers);
            errdefer layers.deinit(allocator);

            var sampler_args = try exe.sampler.args(allocator);
            errdefer sampler_args.deinit(allocator);
            sampler_args.bake(ComposedKernelExe.samplerBuffers(model_buffers));

            var sampler_results = try exe.sampler.results(allocator);
            errdefer sampler_results.deinit(allocator);

            return .{
                .exe = exe,
                .embed_args = embed_args,
                .embed_results = embed_results,
                .layers = layers,
                .sampler_args = sampler_args,
                .sampler_results = sampler_results,
                .prof = getenv("ZML_PROFILE_LAYERS") != null,
            };
        }

        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.embed_args.deinit(allocator);
            self.embed_results.deinit(allocator);
            self.layers.deinit(allocator);
            self.sampler_args.deinit(allocator);
            self.sampler_results.deinit(allocator);
        }

        pub fn run(self: *Runner, args: Args) !void {
            var hidden_buf: zml.Buffer = b: {
                self.embed_args.set(.{args.tokens_buf});
                self.exe.embed_tokens.call(self.embed_args, &self.embed_results);

                break :b self.embed_results.get(zml.Buffer);
            };

            for (
                self.layers.args,
                self.layers.results,
                self.layers.layer_indices,
                self.layers.layer_types,
            ) |*exe_args, *results, *layer_index_buf, layer_type| {
                if (self.prof) {
                    const t = std.Io.Timestamp.now(args.io, .awake);
                    self.exe.runLayer(exe_args, results, layer_type, args, &hidden_buf, layer_index_buf);
                    hidden_buf.await(args.io) catch {}; // sync so the timer captures real layer time
                    const dt: u64 = @intCast(t.untilNow(args.io, .awake).toNanoseconds());
                    switch (layer_type) {
                        .full_attention => self.prof_full_ns += dt,
                        .linear_attention => self.prof_linear_ns += dt,
                    }
                } else {
                    self.exe.runLayer(exe_args, results, layer_type, args, &hidden_buf, layer_index_buf);
                }
            }

            self.exe.runSampler(&self.sampler_args, &self.sampler_results, args, &hidden_buf);

            // The sampler reads but does not donate hidden_buf, so it is still valid here. Hand it
            // to the MTP head (ownership transfers) when requested, else free it.
            if (args.capture_hidden) |slot| slot.* = hidden_buf else hidden_buf.deinit();
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        verify: bool,
        progress: *std.Progress.Node,
    ) !KernelExe {
        return .{
            .composed = try .init(allocator, io, platform, qwen_model, parameters, seqlen, phase, verify, progress),
        };
    }

    pub fn deinit(self: KernelExe) void {
        self.composed.deinit();
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        try self.composed.run(args);
    }

    pub fn initRunner(
        self: *const KernelExe,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model_buffers: *model.Buffers,
    ) !Runner {
        return .init(allocator, io, platform, &self.composed, model_buffers);
    }
};

const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    full_attention_layer: ?zml.Exe,
    linear_attention_layer: ?zml.Exe,
    sampler: zml.Exe,
    layer_types: []const model.LayerType,
    phase: Phase,

    const EmbedTokens = struct {
        embed_tokens: zml.nn.TokenEmbedding,

        pub fn forward(self: EmbedTokens, tokens_: zml.Tensor) zml.Tensor {
            const tokens = tokens_.withPartialTags(.{.s});
            return self.embed_tokens.forward(tokens)
                .withPartialTags(.{.d})
                .withPartitioning(.{ .d = .replicated });
        }
    };

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        verify: bool,
        progress: *std.Progress.Node,
    ) !ComposedKernelExe {
        const embed_tokens = try ComposedKernelExe.compileEmbedTokens(allocator, io, platform, qwen_model.text_model.embed_tokens, parameters, seqlen, phase, progress);
        errdefer embed_tokens.deinit();

        const full_attention_layer: ?zml.Exe = b: {
            const index = ComposedKernelExe.findFirstLayerIndex(qwen_model.config.text_config.layer_types, .full_attention) orelse break :b null;
            break :b try ComposedKernelExe.compileFullAttentionLayer(allocator, io, platform, qwen_model, parameters, seqlen, index, phase, progress);
        };
        errdefer if (full_attention_layer) |exe| exe.deinit();

        // verify -> the linear-attention layer CONTINUES from the cached GDN state over s=Kv tokens
        // (forwardLinearAttnVerify); prefill/decode use forwardLinearAttn.
        const linear_attention_layer: ?zml.Exe = b: {
            const index = ComposedKernelExe.findFirstLayerIndex(qwen_model.config.text_config.layer_types, .linear_attention) orelse break :b null;
            break :b try ComposedKernelExe.compileLinearAttentionLayer(allocator, io, platform, qwen_model, parameters, seqlen, index, phase, verify, progress);
        };
        errdefer if (linear_attention_layer) |exe| exe.deinit();

        const sampler = try ComposedKernelExe.compileSampler(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
        errdefer sampler.deinit();

        return .{
            .embed_tokens = embed_tokens,
            .full_attention_layer = full_attention_layer,
            .linear_attention_layer = linear_attention_layer,
            .sampler = sampler,
            .layer_types = qwen_model.config.text_config.layer_types,
            .phase = phase,
        };
    }

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        if (self.full_attention_layer) |exe| exe.deinit();
        if (self.linear_attention_layer) |exe| exe.deinit();
        self.sampler.deinit();
    }

    fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = b: {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.bake(ComposedKernelExe.embedTokensBuffers(args.model_buffers));
            exe_args.set(.{args.tokens_buf});

            self.embed_tokens.call(exe_args, &results);

            break :b results.get(zml.Buffer);
        };
        var hidden_captured = false;
        defer if (!hidden_captured) hidden_buf.deinit();

        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (args.model_buffers.text_model.layers, self.layer_types) |layer_bufs, layer_type| {
            var exe_args = try switch (layer_type) {
                .full_attention => (self.full_attention_layer orelse unreachable).args(args.allocator),
                .linear_attention => (self.linear_attention_layer orelse unreachable).args(args.allocator),
            };
            defer exe_args.deinit(args.allocator);
            exe_args.bake(layer_bufs);

            var results = try switch (layer_type) {
                .full_attention => (self.full_attention_layer orelse unreachable).results(args.allocator),
                .linear_attention => (self.linear_attention_layer orelse unreachable).results(args.allocator),
            };
            defer results.deinit(args.allocator);

            var layer_index_buf: zml.Buffer = try switch (layer_type) {
                .full_attention => b: {
                    defer self_attn_layer_index += 1;
                    break :b zml.Buffer.scalar(args.io, args.platform, @as(u32, @intCast(self_attn_layer_index)), .u32);
                },
                .linear_attention => b: {
                    defer linear_attn_layer_index += 1;
                    break :b zml.Buffer.scalar(args.io, args.platform, @as(u32, @intCast(linear_attn_layer_index)), .u32);
                },
            };
            defer layer_index_buf.deinit();

            self.runLayer(&exe_args, &results, layer_type, args, &hidden_buf, &layer_index_buf);
        }

        {
            var exe_args = try self.sampler.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.sampler.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.bake(ComposedKernelExe.samplerBuffers(args.model_buffers));

            self.runSampler(&exe_args, &results, args, &hidden_buf);
        }

        // Prefill hidden capture (all seqlen positions): hand ownership to the caller for the MTP
        // prefill (which pairs each prompt token with its per-position hidden), else the defer frees it.
        if (args.capture_hidden) |slot| {
            slot.* = hidden_buf;
            hidden_captured = true;
        }
    }

    fn runLayer(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        layer_type: model.LayerType,
        args: Args,
        hidden_buf: *zml.Buffer,
        layer_index_buf: *zml.Buffer,
    ) void {
        switch (layer_type) {
            .full_attention => {
                const layer_cache: zml.Bufferized(model.KvCache.SelfAttnCache) = .{
                    .k = args.kv_cache_buffers.self_attn.k,
                    .v = args.kv_cache_buffers.self_attn.v,
                    .layer_index = layer_index_buf.*,
                };
                exe_args.set(.{ hidden_buf, args.token_index_buf, layer_cache });
                (self.full_attention_layer orelse unreachable).call(exe_args.*, results);

                var new_hidden, var new_cache = results.get(struct {
                    zml.Buffer,
                    zml.Bufferized(model.KvCache.SelfAttnCache),
                });
                ComposedKernelExe.replaceBuffer(hidden_buf, &new_hidden);
                ComposedKernelExe.replaceBuffer(&args.kv_cache_buffers.self_attn.k, &new_cache.k);
                ComposedKernelExe.replaceBuffer(&args.kv_cache_buffers.self_attn.v, &new_cache.v);
                ComposedKernelExe.releaseBuffer(layer_index_buf.*, &new_cache.layer_index);
            },
            .linear_attention => {
                const layer_cache: zml.Bufferized(model.KvCache.GatedDeltaNetCache) = .{
                    .conv_state = args.kv_cache_buffers.gated_delta_net.conv_state,
                    .recurrent_state = args.kv_cache_buffers.gated_delta_net.recurrent_state,
                    .layer_index = layer_index_buf.*,
                };
                exe_args.set(.{ hidden_buf, args.token_index_buf, layer_cache });
                (self.linear_attention_layer orelse unreachable).call(exe_args.*, results);

                var new_hidden, var new_cache = results.get(struct {
                    zml.Buffer,
                    zml.Bufferized(model.KvCache.GatedDeltaNetCache),
                });
                ComposedKernelExe.replaceBuffer(hidden_buf, &new_hidden);
                ComposedKernelExe.replaceBuffer(&args.kv_cache_buffers.gated_delta_net.conv_state, &new_cache.conv_state);
                ComposedKernelExe.replaceBuffer(&args.kv_cache_buffers.gated_delta_net.recurrent_state, &new_cache.recurrent_state);
                ComposedKernelExe.releaseBuffer(layer_index_buf.*, &new_cache.layer_index);
            },
        }
    }

    fn runSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        switch (self.phase) {
            .prefill => self.runPrefillSampler(exe_args, results, args, hidden_buf),
            .decode => self.runDecodeSampler(exe_args, results, args, hidden_buf),
        }
    }

    fn runPrefillSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{ hidden_buf, args.rng_buffers });
        self.sampler.call(exe_args.*, results);

        var new_tokens, var new_rng = results.get(struct {
            zml.Buffer,
            zml.Bufferized(zml.Tensor.Rng),
        });

        ComposedKernelExe.replaceBuffer(args.tokens_buf, &new_tokens);
        ComposedKernelExe.replaceBuffer(&args.rng_buffers._state, &new_rng._state);
    }

    fn runDecodeSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{ hidden_buf, args.rng_buffers, args.token_index_buf });
        self.sampler.call(exe_args.*, results);

        var new_tokens, var new_rng, var new_token_index = results.get(struct {
            zml.Buffer,
            zml.Bufferized(zml.Tensor.Rng),
            zml.Buffer,
        });

        ComposedKernelExe.replaceBuffer(args.tokens_buf, &new_tokens);
        ComposedKernelExe.replaceBuffer(&args.rng_buffers._state, &new_rng._state);
        ComposedKernelExe.replaceBuffer(args.token_index_buf, &new_token_index);
    }

    fn embedTokensBuffers(model_buffers: *const model.Buffers) zml.Bufferized(EmbedTokens) {
        return .{
            .embed_tokens = model_buffers.text_model.embed_tokens,
        };
    }

    fn samplerBuffers(model_buffers: *const model.Buffers) zml.Bufferized(model.Sampler) {
        return .{
            .norm = model_buffers.text_model.norm,
            .lm_head = model_buffers.lm_head,
        };
    }

    const replaceBuffer = replaceBufferImpl;
    const releaseBuffer = releaseBufferImpl;
    const sameBufferHandle = sameBufferHandleImpl;

    fn compileEmbedTokens(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        embed_tokens: zml.nn.TokenEmbedding,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("embed tokens"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "embed tokens", io, from);

        const tokens: zml.Tensor = .init(.{ .b = 1, .s = seqlen }, .u32);

        return platform.compile(allocator, io, EmbedTokens{ .embed_tokens = embed_tokens }, .forward, .{tokens}, .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("qwen3_5", "embed_tokens"),
        });
    }

    fn compileFullAttentionLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        layer_index: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("full attention layer"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "full attention layer", io, from);

        const hidden_tensor = ComposedKernelExe.hidden(qwen_model, seqlen);
        const self_attn_cache: model.KvCache.SelfAttnCache = .{
            .k = parameters.kv_cache.self_attn.k,
            .v = parameters.kv_cache.self_attn.v,
            .layer_index = zml.Tensor.init(.{}, .u32),
        };

        return platform.compile(
            allocator,
            io,
            qwen_model.text_model.layers[layer_index],
            .forwardSelfAttn,
            .{ hidden_tensor, parameters.token_index, self_attn_cache },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("qwen3_5", "full_attention_layer"),
            },
        );
    }

    fn compileLinearAttentionLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        layer_index: usize,
        phase: Phase,
        verify: bool,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("linear attention layer"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "linear attention layer", io, from);

        const hidden_tensor = ComposedKernelExe.hidden(qwen_model, seqlen);
        const linear_attn_cache: model.KvCache.GatedDeltaNetCache = .{
            .conv_state = parameters.kv_cache.gated_delta_net.conv_state,
            .recurrent_state = parameters.kv_cache.gated_delta_net.recurrent_state,
            .layer_index = zml.Tensor.init(.{}, .u32),
        };

        // `func` is a comptime parameter, so branch on the runtime `verify` flag here.
        if (verify) {
            return platform.compile(
                allocator,
                io,
                qwen_model.text_model.layers[layer_index],
                .forwardLinearAttnVerify,
                .{ hidden_tensor, parameters.token_index, linear_attn_cache },
                .{
                    .shardings = &parameters.shardings.all(),
                    .program_name = phase.programName("qwen3_5", "linear_attention_layer_verify"),
                },
            );
        }
        return platform.compile(
            allocator,
            io,
            qwen_model.text_model.layers[layer_index],
            .forwardLinearAttn,
            .{ hidden_tensor, parameters.token_index, linear_attn_cache },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("qwen3_5", "linear_attention_layer"),
            },
        );
    }

    fn compileSampler(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("sampler"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "sampler", io, from);

        const token_index: ?zml.Tensor = switch (phase) {
            .prefill => null,
            .decode => parameters.token_index,
        };

        return platform.compile(
            allocator,
            io,
            qwen_model.sampler(),
            .sampleTokens,
            .{ ComposedKernelExe.hidden(qwen_model, seqlen), parameters.rng, token_index },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("qwen3_5", "sampler"),
            },
        );
    }

    fn hidden(qwen_model: model.Model, seqlen: usize) zml.Tensor {
        return .fromShape(zml.Shape.init(
            .{ .b = 1, .s = seqlen, .d = qwen_model.config.text_config.hidden_size },
            qwen_model.text_model.embed_tokens.weight.dtype(),
        ).withPartitioning(.{
            .b = .replicated,
            .s = .replicated,
            .d = .replicated,
        }));
    }

    fn findFirstLayerIndex(layer_types: []const model.LayerType, target: model.LayerType) ?usize {
        for (layer_types, 0..) |layer_type, index| {
            if (layer_type == target) return index;
        }
        return null;
    }
};
