const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

// libc getenv (the binary links -lc); Zig 0.16 dropped std.posix.getenv here. Used only to select
// the MTP mode at session init.
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    decode_runner: inference.KernelExe.Runner,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    generated_token_slice: zml.Slice,
    seqlen: u32,
    eos_token_id: u32,
    special_tokens: model.Model.SpecialTokens,
    think_start: ?u32,
    think_end: ?u32,
    // NEXTN/MTP speculative-decode state (null unless the model has an MTP head AND a mode env is
    // set). .measure = run drafts alongside the normal decode and report accept rate (Step 1);
    // .drive = the real spec-decode loop (Step 3, TODO). See zml/ZML_MTP_PLAN.md.
    mtp: ?Mtp,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, compiled_model.params.shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        var decode_runner = try compiled_model.decode.initRunner(
            allocator,
            io,
            platform,
            model_buffers,
        );
        errdefer decode_runner.deinit(allocator);

        // MTP mode from the environment (only if the model actually has the compiled MTP exes):
        //   ZML_MTP_MEASURE -> .measure (Step 1: draft alongside decode, report accept rate)
        //   ZML_MTP         -> .drive   (Step 3: real spec-decode loop, TODO)
        const mtp_mode: Mtp.Mode = b: {
            if (compiled_model.mtp == null) break :b .off;
            if (getenv("ZML_MTP_MEASURE") != null) break :b .measure;
            if (getenv("ZML_MTP") != null) break :b .drive;
            break :b .off;
        };
        var mtp: ?Mtp = null;
        if (mtp_mode != .off) {
            mtp = try Mtp.init(allocator, io, platform, &compiled_model.mtp.?, model_buffers, compiled_model.params.shardings.model, mtp_mode);
        }
        errdefer if (mtp) |*m| m.deinit(allocator);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .decode_runner = decode_runner,
            .kv_cache_buffers = kv_cache_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .seqlen = compiled_model.params.seqlen,
            .eos_token_id = compiled_model.loaded_model.inner.special_tokens.end_of_text_token_id,
            .special_tokens = compiled_model.loaded_model.inner.special_tokens,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
            .mtp = mtp,
        };
    }

    pub fn deinit(self: *Session) void {
        self.decode_runner.deinit(self.allocator);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.generated_token_slice.free(self.allocator);
        if (self.mtp) |*m| m.deinit(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, false);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen }, .u32);
        const prefill_tokens_slice = try zml.Slice.alloc(self.allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding: zml.Sharding = .replicated;

        var prefill_tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer prefill_token_index_buffer.deinit();

        // Capture the full-seqlen prefill hidden (h_0..h_{seqlen-1}) so the MTP head can be prefilled
        // over the prompt (each prompt token paired with its per-position hidden).
        const mtp_active = if (self.mtp) |*m| m.mode != .off else false;
        var prefill_hidden: ?zml.Buffer = null;

        try self.compiled_model.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &prefill_tokens_buffer,
            .token_index_buf = &prefill_token_index_buffer,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
            .capture_hidden = if (mtp_active) &prefill_hidden else null,
        });

        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);
        const generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        self.generated_token_slice.items(u32)[0] = generated_token;

        if (mtp_active) {
            defer prefill_hidden.?.deinit();
            try self.mtp.?.fillPrefill(self.io, self.platform, self.allocator, self.seqlen, all_tokens, generated_token, &prefill_hidden.?);
        }
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);
        const replicated_sharding: zml.Sharding = .replicated;

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
        defer current_token_buffer.deinit();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len)), .u32);
        defer token_index_buffer.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (token_id == self.eos_token_id) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (self.think_start) |think_start| if (token_id == think_start) {
                try stdout.writeAll("\x1b[2m");
            };
            try stdout.writeAll(token);
            if (self.think_end) |think_end| if (token_id == think_end) {
                try stdout.writeAll("\x1b[0m");
            };
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const mtp_active = if (self.mtp) |*m| m.mode == .measure else false;
            var captured_hidden: ?zml.Buffer = null;

            try self.decode_runner.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &current_token_buffer,
                .token_index_buf = &token_index_buffer,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .capture_hidden = if (mtp_active) &captured_hidden else null,
            });

            try current_token_buffer.toSlice(self.io, self.generated_token_slice);

            // MTP measure mode: the main model just committed token x (now in generated_token_slice)
            // at position P = all_tokens.len - 1, producing hidden h_P (captured_hidden). First score
            // the previous iteration's draft against x, then draft the NEXT token from (x, h_P, P):
            // MTP consumes the just-committed token with its producing hidden at position P (vLLM
            // alignment: input shifted, position = the hidden's position). This writes MTP-KV[P],
            // contiguous with the prompt KV filled at prefill.
            if (mtp_active) {
                defer captured_hidden.?.deinit();
                const mtp = &self.mtp.?;
                const produced = self.generated_token_slice.items(u32)[0];
                if (mtp.pending_draft) |d| {
                    mtp.total += 1;
                    if (d == produced) mtp.accepted += 1;
                }
                const pos: u32 = @intCast(all_tokens.items.len - 1);
                try mtp.draftStep(self.io, self.platform, &current_token_buffer, &captured_hidden.?, pos);
            }
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();

        if (self.mtp) |*m| if (m.mode == .measure and m.total > 0) {
            try stdout.print("\n\x1b[35m[mtp] draft accept {d}/{d} = {d:.1}%\x1b[0m\n", .{
                m.accepted, m.total,
                100.0 * @as(f64, @floatFromInt(m.accepted)) / @as(f64, @floatFromInt(m.total)),
            });
            try stdout.flush();
        };
    }
};

// NEXTN/MTP speculative-decode runtime state: the compiled prefill/draft exes (borrowed from the
// CompiledModel), their baked arguments/results, and the MTP head's OWN 1-layer self-attn KV cache.
// See zml/ZML_MTP_PLAN.md. Buffer swaps use inference.replaceBufferImpl (donation-aware).
pub const Mtp = struct {
    pub const Mode = enum { off, measure, drive };

    exes: *const inference.MtpExes,
    kv: zml.Bufferized(model.KvCache.SelfAttnCache),
    prefill_args: zml.exe.Exe.Arguments,
    prefill_results: zml.exe.Exe.Results,
    draft_args: zml.exe.Exe.Arguments,
    draft_results: zml.exe.Exe.Results,
    rng: zml.Bufferized(zml.Tensor.Rng),
    draft_token_slice: zml.Slice,
    mode: Mode,
    pending_draft: ?u32 = null,
    accepted: u64 = 0,
    total: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        exes: *const inference.MtpExes,
        model_buffers: *model.Buffers,
        model_sharding: zml.Sharding,
        mode: Mode,
    ) !Mtp {
        var kv = try exes.mtp_kv.initBuffer(io, platform, model_sharding);
        errdefer model.KvCache.SelfAttnCache.deinitBuffer(&kv);

        var prefill_args = try exes.prefill.args(allocator);
        errdefer prefill_args.deinit(allocator);
        prefill_args.bake(zml.Bufferized(model.MtpPrefill){
            .embed_tokens = model_buffers.text_model.embed_tokens,
            .head = model_buffers.mtp_head.?,
            .target_norm = model_buffers.text_model.norm,
        });
        var prefill_results = try exes.prefill.results(allocator);
        errdefer prefill_results.deinit(allocator);

        var draft_args = try exes.draft.args(allocator);
        errdefer draft_args.deinit(allocator);
        draft_args.bake(zml.Bufferized(model.MtpDraft){
            .embed_tokens = model_buffers.text_model.embed_tokens,
            .head = model_buffers.mtp_head.?,
            .lm_head = model_buffers.lm_head,
            .target_norm = model_buffers.text_model.norm,
        });
        var draft_results = try exes.draft.results(allocator);
        errdefer draft_results.deinit(allocator);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng);

        return .{
            .exes = exes,
            .kv = kv,
            .prefill_args = prefill_args,
            .prefill_results = prefill_results,
            .draft_args = draft_args,
            .draft_results = draft_results,
            .rng = rng,
            .draft_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .mode = mode,
        };
    }

    pub fn deinit(self: *Mtp, allocator: std.mem.Allocator) void {
        model.KvCache.SelfAttnCache.deinitBuffer(&self.kv);
        self.prefill_args.deinit(allocator);
        self.prefill_results.deinit(allocator);
        self.draft_args.deinit(allocator);
        self.draft_results.deinit(allocator);
        zml.Tensor.Rng.deinitBuffer(&self.rng);
        self.draft_token_slice.free(allocator);
    }

    // Fill the MTP KV over the prompt (positions 0..L-1). Input_ids are the prompt SHIFTED LEFT by
    // one with the final sampled token in the last real slot (vLLM set_inputs_first_pass); positions
    // are 0..L-1 (token_index=0); prev_hidden = the captured main prefill hidden (slot j pairs token
    // x_{j+1} with hidden h_j at position j). Padded slots L..seqlen-1 are overwritten by decode
    // before they are ever attended, so their garbage KV is harmless.
    pub fn fillPrefill(
        self: *Mtp,
        io: std.Io,
        platform: *const zml.Platform,
        allocator: std.mem.Allocator,
        seqlen: u32,
        all_tokens: []const u32,
        sampled: u32,
        prefill_hidden: *zml.Buffer,
    ) !void {
        const shape = zml.Shape.init(.{ .b = 1, .s = seqlen }, .u32);
        const slice = try zml.Slice.alloc(allocator, shape);
        defer slice.free(allocator);
        const buf = slice.items(u32);
        @memset(buf, 0);
        var j: usize = 0;
        while (j + 1 < all_tokens.len) : (j += 1) buf[j] = all_tokens[j + 1];
        buf[all_tokens.len - 1] = sampled;

        var tokens_buf = try zml.Buffer.fromSlice(io, platform, slice, .replicated);
        defer tokens_buf.deinit();
        var zero_index = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
        defer zero_index.deinit();

        self.prefill_args.set(.{ &tokens_buf, prefill_hidden, &zero_index, self.kv });
        self.exes.prefill.call(self.prefill_args, &self.prefill_results);
        var new_kv = self.prefill_results.get(zml.Bufferized(model.KvCache.SelfAttnCache));
        inference.replaceBufferImpl(&self.kv.k, &new_kv.k);
        inference.replaceBufferImpl(&self.kv.v, &new_kv.v);
        inference.releaseBufferImpl(self.kv.layer_index, &new_kv.layer_index);
    }

    // One MTP draft: consume `token_buf` (the just-committed token x_{p+1}) with `prev_hidden` (h_p)
    // at rope-position/KV-slot `pos` (= p), producing the drafted next token (into pending_draft)
    // and advancing the MTP KV cache by one slot.
    pub fn draftStep(
        self: *Mtp,
        io: std.Io,
        platform: *const zml.Platform,
        token_buf: *zml.Buffer,
        prev_hidden: *zml.Buffer,
        pos: u32,
    ) !void {
        var pos_buf = try zml.Buffer.scalar(io, platform, pos, .u32);
        defer pos_buf.deinit();

        self.draft_args.set(.{ token_buf, prev_hidden, &pos_buf, self.kv, &self.rng });
        self.exes.draft.call(self.draft_args, &self.draft_results);
        var new_token, var new_kv, var new_rng = self.draft_results.get(struct {
            zml.Buffer,
            zml.Bufferized(model.KvCache.SelfAttnCache),
            zml.Bufferized(zml.Tensor.Rng),
        });
        inference.replaceBufferImpl(&self.kv.k, &new_kv.k);
        inference.replaceBufferImpl(&self.kv.v, &new_kv.v);
        inference.releaseBufferImpl(self.kv.layer_index, &new_kv.layer_index);
        inference.replaceBufferImpl(&self.rng._state, &new_rng._state);

        try new_token.toSlice(io, self.draft_token_slice);
        self.pending_draft = self.draft_token_slice.items(u32)[0];
        new_token.deinit();
    }
};

fn tokenizeChatPrompt(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    special_tokens: model.Model.SpecialTokens,
    is_first_turn: bool,
) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse special_tokens.im_end_token_id;
    const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    if (!is_first_turn) {
        try tokens.appendSlice(allocator, &.{ im_end, newline });
    }

    try tokens.append(allocator, im_start);
    const user_tokens = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user_tokens);
    try tokens.appendSlice(allocator, user_tokens);
    const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(prompt_tokens);
    try tokens.appendSlice(allocator, prompt_tokens);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant_tokens);
    try tokens.appendSlice(allocator, assistant_tokens);

    return tokens.toOwnedSlice(allocator);
}
