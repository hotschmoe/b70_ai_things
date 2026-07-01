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

    // Stream one decoded token to stdout with the <think> dim styling (shared by runDecode + the MTP
    // drive loop).
    fn streamToken(self: *Session, decoder: anytype, token_id: u32, out_buf: []u8, stdout: *std.Io.Writer) !void {
        const token = try decoder.feedOne(token_id, out_buf);
        if (self.think_start) |think_start| if (token_id == think_start) {
            try stdout.writeAll("\x1b[2m");
        };
        try stdout.writeAll(token);
        if (self.think_end) |think_end| if (token_id == think_end) {
            try stdout.writeAll("\x1b[0m");
        };
        try stdout.flush();
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        // MTP drive mode: run the speculative-decode loop (draft -> verify Kv -> accept/commit) instead.
        if (self.mtp) |*m| if (m.mode == .drive) return self.runDecodeMtp(all_tokens, stdout);

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
                _ = try mtp.draftStep(self.io, self.platform, &current_token_buffer, &captured_hidden.?, pos);
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

        // ZML_PROFILE_LAYERS: report the decode time split between int8 full-attention (16 layers)
        // and bf16+f32-scan GDN (48 linear-attention layers) -- which lever dominates decode.
        const r = &self.decode_runner;
        if (r.prof and (r.prof_full_ns + r.prof_linear_ns) > 0) {
            const full_ms = @as(f64, @floatFromInt(r.prof_full_ns)) / 1e6;
            const lin_ms = @as(f64, @floatFromInt(r.prof_linear_ns)) / 1e6;
            const tot = full_ms + lin_ms;
            try stdout.print("\x1b[35m[prof] decode layer time: full-attn(int8x16) {d:.0}ms ({d:.0}%)  GDN(bf16x48) {d:.0}ms ({d:.0}%)\x1b[0m\n", .{
                full_ms, 100.0 * full_ms / tot, lin_ms, 100.0 * lin_ms / tot,
            });
            try stdout.flush();
        }
    }

    // MTP spec-decode DRIVE loop (K=1): each step drafts 1 token (d1), verifies [cur, d1] in ONE
    // s=Kv main forward, accepts d1 iff d1==y_0 (greedy speculative decode is exact), and commits 1
    // or 2 tokens. On a rejected draft the GDN recurrent state -- advanced by the wrong token during
    // verify, and not invertible -- is rolled back to a pre-verify snapshot and re-advanced by a
    // single normal decode of cur. The emitted token stream is byte-identical to greedy MTP-off
    // decode (the validation oracle). See zml/ZML_MTP_PLAN.md.
    pub fn runDecodeMtp(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        const mtp = &self.mtp.?;
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_buf: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_buf);
        const replicated: zml.Sharding = .replicated;

        var cur_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, replicated);
        defer cur_token_buffer.deinit();

        var cur_hidden: ?zml.Buffer = null;
        defer if (cur_hidden) |*h| h.deinit();

        // --- bootstrap: emit t0 (=x_L from prefill) and run ONE normal decode to establish cur =
        // x_{L+1} and cur_hidden = h_L. The MTP KV was filled 0..L-1 over the prompt in runPrefill;
        // the first drive draft then writes MTP-KV[L], contiguous.
        {
            const t0 = self.generated_token_slice.items(u32)[0];
            if (t0 == self.eos_token_id) return;
            try self.streamToken(&decoder, t0, out_buf, stdout);
            try all_tokens.append(self.allocator, t0);
            if (all_tokens.items.len >= self.seqlen) {
                try stdout.writeAll(try decoder.finalize(out_buf));
                try stdout.flush();
                return;
            }
            var idx = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len - 1)), .u32);
            defer idx.deinit();
            try self.decode_runner.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &cur_token_buffer,
                .token_index_buf = &idx,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .capture_hidden = &cur_hidden,
            });
            try cur_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        // Coarse phase timers (host wall-clock between device syncs) to locate the drive overhead.
        var ns_draft: u64 = 0;
        var ns_snap: u64 = 0;
        var ns_verify: u64 = 0;
        var ns_commit: u64 = 0;

        generation: while (true) {
            const cur = self.generated_token_slice.items(u32)[0]; // x_{p+1}
            if (cur == self.eos_token_id) break :generation;
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const p: u32 = @intCast(all_tokens.items.len - 1); // main committed through p

            // 1) DRAFT d1 from (cur, h_p) at MTP position p (writes MTP-KV[p]).
            const t_draft = std.Io.Timestamp.now(self.io, .awake);
            const d1 = try mtp.draftStep(self.io, self.platform, &cur_token_buffer, &cur_hidden.?, p);
            ns_draft += @intCast(t_draft.untilNow(self.io, .awake).toNanoseconds());

            // 2) VERIFY [cur, d1] at main positions p+1..p+Kv from the committed cache. Snapshot the
            //    GDN state first so a rejected draft can roll back.
            const t_snap = std.Io.Timestamp.now(self.io, .awake);
            var gdn_snap = mtp.snapshotGdn(&self.kv_cache_buffers.gated_delta_net);
            _ = try gdn_snap.layer_index.getValue(u32, self.io); // force the snapshot to complete (sync) for timing
            ns_snap += @intCast(t_snap.untilNow(self.io, .awake).toNanoseconds());
            const t_verify = std.Io.Timestamp.now(self.io, .awake);
            mtp.verify_tokens_slice.items(u32)[0] = cur;
            mtp.verify_tokens_slice.items(u32)[1] = d1;
            var verify_tokens = try zml.Buffer.fromSlice(self.io, self.platform, mtp.verify_tokens_slice, replicated);
            defer verify_tokens.deinit();
            var verify_idx = try zml.Buffer.scalar(self.io, self.platform, p + 1, .u32);
            defer verify_idx.deinit();
            var verify_hidden: ?zml.Buffer = null;
            defer if (verify_hidden) |*h| h.deinit();
            try mtp.verify_runner.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &verify_tokens,
                .token_index_buf = &verify_idx,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .capture_hidden = &verify_hidden,
            });
            try verify_tokens.toSlice(self.io, mtp.verify_tokens_slice);
            const y0 = mtp.verify_tokens_slice.items(u32)[0]; // true x_{p+2}
            const y1 = mtp.verify_tokens_slice.items(u32)[1]; // true x_{p+3} (if d1 accepted)
            ns_verify += @intCast(t_verify.untilNow(self.io, .awake).toNanoseconds());

            mtp.drive_iters += 1;
            const accept = (d1 == y0);

            // 3) EMIT cur (always correct); commit x_{p+1}.
            const t_commit = std.Io.Timestamp.now(self.io, .awake);
            try self.streamToken(&decoder, cur, out_buf, stdout);
            try all_tokens.append(self.allocator, cur);

            if (accept) {
                mtp.drive_accepted += 1;
                // GDN state through p+2 (verify) is correct -> keep it, drop the snapshot.
                model.KvCache.GatedDeltaNetCache.deinitBuffer(&gdn_snap);
                // commit d1 (=x_{p+2}) unless it is EOS (match MTP-off: never emit past EOS).
                if (d1 == self.eos_token_id) break :generation;
                try self.streamToken(&decoder, d1, out_buf, stdout);
                try all_tokens.append(self.allocator, d1);
                if (all_tokens.items.len >= self.seqlen) break :generation;

                // MTP catch-up: fill MTP-KV[p+1] for x_{p+2}=d1 with hidden h_{p+1} (verify pos 0),
                // so the MTP KV stays contiguous (the single draft only wrote MTP-KV[p]).
                var d1_buf = try tokenBuffer(self.allocator, self.io, self.platform, d1);
                defer d1_buf.deinit();
                var catchup_hidden = try mtp.sliceHidden(self.io, self.platform, &verify_hidden.?, 0);
                defer catchup_hidden.deinit();
                _ = try mtp.draftStep(self.io, self.platform, &d1_buf, &catchup_hidden, p + 1);

                // next: cur = y1, cur_hidden = h_{p+2} (verify last position).
                const next_hidden = try mtp.sliceHidden(self.io, self.platform, &verify_hidden.?, inference.MTP_KV - 1);
                if (cur_hidden) |*h| h.deinit();
                cur_hidden = next_hidden;
                self.generated_token_slice.items(u32)[0] = y1;
                cur_token_buffer.deinit();
                cur_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, replicated);
            } else {
                // reject: committed must be through p+1. Restore the GDN snapshot (verify advanced it
                // to p+2 with the wrong d1), then re-decode cur to advance GDN + full-attn to p+1
                // (the verify's full-attn slot p+2 is scratch, overwritten by a later step).
                inference.replaceBufferImpl(&self.kv_cache_buffers.gated_delta_net.conv_state, &gdn_snap.conv_state);
                inference.replaceBufferImpl(&self.kv_cache_buffers.gated_delta_net.recurrent_state, &gdn_snap.recurrent_state);
                gdn_snap.layer_index.deinit(); // committed keeps its own layer_index
                if (all_tokens.items.len >= self.seqlen) break :generation;

                var re_hidden: ?zml.Buffer = null;
                var idx = try zml.Buffer.scalar(self.io, self.platform, p + 1, .u32);
                defer idx.deinit();
                try self.decode_runner.run(.{
                    .allocator = self.allocator,
                    .io = self.io,
                    .platform = self.platform,
                    .model_buffers = self.model_buffers,
                    .tokens_buf = &cur_token_buffer,
                    .token_index_buf = &idx,
                    .kv_cache_buffers = &self.kv_cache_buffers,
                    .rng_buffers = &self.rng_buffers,
                    .capture_hidden = &re_hidden,
                });
                try cur_token_buffer.toSlice(self.io, self.generated_token_slice); // = y0 (=x_{p+2})
                if (cur_hidden) |*h| h.deinit();
                cur_hidden = re_hidden;
            }
            ns_commit += @intCast(t_commit.untilNow(self.io, .awake).toNanoseconds());
        }

        try stdout.writeAll(try decoder.finalize(out_buf));
        try stdout.flush();

        if (mtp.drive_iters > 0) {
            const iters_f: f64 = @floatFromInt(mtp.drive_iters);
            try stdout.print("\n\x1b[35m[mtp] drive accept {d}/{d} = {d:.1}%  (avg {d:.2} tok/verify)\x1b[0m\n", .{
                mtp.drive_accepted, mtp.drive_iters,
                100.0 * @as(f64, @floatFromInt(mtp.drive_accepted)) / iters_f,
                1.0 + @as(f64, @floatFromInt(mtp.drive_accepted)) / iters_f,
            });
            try stdout.print("\x1b[35m[mtp] per-iter ms: draft {d:.1}  snap {d:.1}  verify {d:.1}  commit {d:.1}\x1b[0m\n", .{
                @as(f64, @floatFromInt(ns_draft)) / 1e6 / iters_f,
                @as(f64, @floatFromInt(ns_snap)) / 1e6 / iters_f,
                @as(f64, @floatFromInt(ns_verify)) / 1e6 / iters_f,
                @as(f64, @floatFromInt(ns_commit)) / 1e6 / iters_f,
            });
            try stdout.flush();
        }
    }
};

// Build a {b=1, s=1} u32 token buffer holding `val` (for MTP draft inputs built from host tokens).
fn tokenBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, val: u32) !zml.Buffer {
    const s = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
    defer s.free(allocator);
    s.items(u32)[0] = val;
    return zml.Buffer.fromSlice(io, platform, s, .replicated);
}

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

    // --- drive-mode (.drive) state: the s=Kv verify + GDN rollback machinery ---
    verify_runner: inference.KernelExe.Runner, // main model @ s=Kv over the shared committed cache
    verify_tokens_slice: zml.Slice, // host {b=1, s=Kv}: input [t, d_1] and output [y_0, y_1]
    snap_args: zml.exe.Exe.Arguments, // gdn_snapshot exe
    snap_results: zml.exe.Exe.Results,
    slice_args: zml.exe.Exe.Arguments, // slice_hidden exe
    slice_results: zml.exe.Exe.Results,
    // Accept/verify counters for the drive-mode summary (drafts accepted / spec iterations).
    drive_accepted: u64 = 0,
    drive_iters: u64 = 0,

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

        // Drive-mode machinery (also allocated in measure mode -- cheap; only args/results, no big
        // buffers): the verify runner over the shared committed cache, and the snapshot/slice exes.
        var verify_runner = try exes.verify.initRunner(allocator, io, platform, model_buffers);
        errdefer verify_runner.deinit(allocator);

        var snap_args = try exes.gdn_snapshot.args(allocator); // no weights to bake
        errdefer snap_args.deinit(allocator);
        var snap_results = try exes.gdn_snapshot.results(allocator);
        errdefer snap_results.deinit(allocator);

        var slice_args = try exes.slice_hidden.args(allocator); // no weights to bake
        errdefer slice_args.deinit(allocator);
        var slice_results = try exes.slice_hidden.results(allocator);
        errdefer slice_results.deinit(allocator);

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
            .verify_runner = verify_runner,
            .verify_tokens_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = inference.MTP_KV }, .u32)),
            .snap_args = snap_args,
            .snap_results = snap_results,
            .slice_args = slice_args,
            .slice_results = slice_results,
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
        self.verify_runner.deinit(allocator);
        self.verify_tokens_slice.free(allocator);
        self.snap_args.deinit(allocator);
        self.snap_results.deinit(allocator);
        self.slice_args.deinit(allocator);
        self.slice_results.deinit(allocator);
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
    // at rope-position/KV-slot `pos` (= p), advancing the MTP KV cache by one slot and returning the
    // drafted next token (also stored in pending_draft for the measure path).
    pub fn draftStep(
        self: *Mtp,
        io: std.Io,
        platform: *const zml.Platform,
        token_buf: *zml.Buffer,
        prev_hidden: *zml.Buffer,
        pos: u32,
    ) !u32 {
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
        return self.pending_draft.?;
    }

    // Deep-copy the committed GDN conv+recurrent state (fresh buffers) so a rejected draft can be
    // rolled back. Caller owns the returned buffers (deinit on accept, or swap-in on reject).
    fn snapshotGdn(self: *Mtp, gdn: *const zml.Bufferized(model.KvCache.GatedDeltaNetCache)) zml.Bufferized(model.KvCache.GatedDeltaNetCache) {
        self.snap_args.set(.{gdn.*});
        self.exes.gdn_snapshot.call(self.snap_args, &self.snap_results);
        return self.snap_results.get(zml.Bufferized(model.KvCache.GatedDeltaNetCache));
    }

    // Extract sequence position `index` from the s=Kv verify hidden ({b,Kv,d} -> {b,1,d}).
    fn sliceHidden(self: *Mtp, io: std.Io, platform: *const zml.Platform, hidden: *zml.Buffer, index: u32) !zml.Buffer {
        var index_buf = try zml.Buffer.scalar(io, platform, index, .u32);
        defer index_buf.deinit();
        self.slice_args.set(.{ hidden, &index_buf });
        self.exes.slice_hidden.call(self.slice_args, &self.slice_results);
        return self.slice_results.get(zml.Buffer);
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
