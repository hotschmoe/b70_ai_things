# Plan: enable sglang mamba `extra_buffer` radix cache on Intel XPU (keep intel_xpu XMX attn @ page 64/128)

Status: PLAN (read-only investigation; nothing modified, no GPU touched). Drafted 2026-07-02.
Motivation: we proved prefix caching works via `no_buffer` + `page_size=1` + `--attention-backend triton`
(JOURNAL 2026-07-02), but page_size=1 forbids the intel_xpu XMX attention backend and may regress cold
prefill. `extra_buffer` keeps page_size 64/128 and the XMX attention path -- if we can un-gate it on XPU.

## TL;DR verdict: EASY MONKEYPATCH, not blocked on any missing kernel

The whole `extra_buffer` runtime (state tracking, checkpoint copy/ping-pong, int8 checkpoint quant) is either
Triton (already runs on XPU -- it is the basis of our working W8A8+MTP GDN serve) or pure-torch
device-agnostic tensor ops. There is NO external `fla` pip package and NO causal-conv1d CUDA `.so` on the
path we would use. The "FLA" in the blocking assert refers to sglang's VENDORED Triton module
`sglang/srt/layers/attention/fla/`, not a CUDA-only library. The image already contains the full extra_buffer
strategy AND the int8 mamba checkpoint pool (PR #28185) -- this is config+shim, not a feature port.

Exactly ONE hard gate to remove: a single assert in Python arg-validation. The two runtime scatter guards
that would fire are ALREADY stripped by our existing `mtp_tree_xpu.py` shim.

## Every CUDA gate on the extra_buffer path (paths under image /opt/venv/.../sglang/srt/)

BLOCKER 1 -- the only hard arg-validation gate:
- `server_args.py:4496-4502` `_validate_mamba_extra_buffer()`:
  - line 4497 `assert self._support_mamba_cache_extra_buffer(model_arch)` -- ALREADY passes on XPU:
    `_support_mamba_cache_extra_buffer` (4470-4485) returns `linear_attn_backend == "triton"` for
    Qwen3_5ForConditionalGeneration, and linear_attn_backend defaults to "triton" (1825).
  - line 4500 `assert (is_cuda() or is_musa() or is_npu()), "extra_buffer needs CUDA/MUSA/NPU (FLA)."`
    -- THIS is the only blocker. Verdict: trivial, drop/relax the assert (the "(FLA)" premise is false for
    the Triton linear-attn backend we run).

BLOCKER 2 -- two runtime scatter guards, ALREADY handled by our shim:
- `mamba_state_scatter_triton.py:219-221` fused_mamba_state_scatter_with_mask: `if not dst.is_cuda ...: raise`
- `mamba_state_scatter_triton.py:381` fused_conv_window_scatter_with_mask: `if not (dst.is_cuda and ...)`
  Both wrap @triton.jit kernels that run on XPU. Our `sglang/patches/mtp_tree_xpu.py:174-200` already
  re-execs both with the is_cuda guard stripped and re-binds them (source module + hybrid_linear_attn_backend).
  They fire on the spec/MTP scatter, so the daily driver already exercises + neutralizes them. Verdict: done.

Non-blockers verified benign on XPU (do NOT touch): hybrid_linear_attn_backend.py extra_buffer track/restore
(no is_cuda gates); mamba_state_scatter_triton.py:74 track_mamba_states_if_needed (no guard);
memory_pool.py MambaPool.copy_from 646-674 (pure tensor slice); memory_pool.py:1284 alt_stream=None on XPU
(benign); memory_pool.py:1689 `if not (_is_cuda or _is_hip)` is the XPU-friendly branch;
mamba_checkpoint_pool.py:344 `if device.startswith("cuda")` only a HBM free-space check (skipped on XPU),
quant is pure-torch .to(int8)+per-(head,k) scale; gdn_backend.py:29-137 is_cuda gates only guard the
ALTERNATIVE CuTe-DSL/FlashInfer GDN kernels (triton path never imports them); mamba.py:604
mamba_chunk_scan_combined is Triton (same kernel our GDN prefill runs on XPU).

Full compute set = mamba_chunk_scan_combined, track_mamba_states_if_needed, fused_{mamba_state,conv_window}
_scatter_with_mask, causal_conv1d_triton, selective_state_update -- all Triton, all proven on this box.

## Runtime shim vs rebuild -> runtime shim, no rebuild

Install hook already exists: `woq_shim.pth` (`import woq_shim`) auto-imports at interpreter startup in every
process; `woq_shim._install()` (sglang/patches/woq_shim.py:42,396) dispatches by env var
(B70_XPU_MTP=1 -> mtp_tree_xpu.install()). The .pth import runs via site.py BEFORE argparse/ServerArgs, so
patching `ServerArgs._validate_mamba_extra_buffer` at install time lands before post-init calls it.
Recommended: add a function in mtp_tree_xpu.install() mirroring its `_strip_is_cuda_guard`/re-exec pattern
(or simply `sglang.srt.server_args.ServerArgs._validate_mamba_extra_buffer = <patched>`), gated on a new
`B70_XPU_MAMBA_EXTRA_BUFFER=1` env.

## Config constraints (from _validate_mamba_extra_buffer 4503-4510)

- intel_xpu decode attention forces page_size->128 if not in [64,128] (server_args.py:4834-4845). (This is
  exactly why no_buffer, which asserts page_size in (1,None) at 4488, is incompatible with intel_xpu.)
- With MTP (speculative_num_draft_tokens set): mamba_track_interval >= speculative_num_draft_tokens,
  mamba_track_interval % page_size == 0, and mamba_cache_chunk_size must be set. E.g. --mamba-track-interval 128
  (multiple of 128, >= 11 draft tokens) and set --mamba-cache-chunk-size.
- Pass --disable-overlap-schedule to keep overlap out of the variable set (page_size>1 -> auto picks extra_buffer).

## Experiment ladder (each gated by bin/serve-sweep for coherence + warm-prefill)

1. ZERO-CODE capacity probe (do first, independent): on the current working no_buffer+page1+triton serve, add
   `--enable-int8-mamba-checkpoint`. `_handle_int8_mamba_checkpoint` (server_args.py:5002) has no CUDA gate and
   `_commit_int8_checkpoint` is strategy-agnostic (mamba_radix_cache.py:1011-1014) -> ~2x cached-prefix capacity
   TODAY with no shim. Confirms the int8 pool runs on XPU in isolation.
2. ISOLATE "does extra_buffer run on XPU at all" (removes intel_xpu variable): add the one-assert shim, serve
   `--attention-backend triton --page-size 64 --mamba-radix-cache-strategy extra_buffer --disable-overlap-schedule`
   (+ MTP track flags). Triton attn supports page 64, tests extra_buffer track/restore without intel_xpu.
3. THE GOAL: `--attention-backend intel_xpu --page-size 128 --mamba-radix-cache-strategy extra_buffer
   --disable-overlap-schedule --mamba-track-interval 128 --mamba-cache-chunk-size <N>`. Verify warm-prefix
   speedup lands AND XMX attention active, coherent under concurrent prefill+decode.

## Reuse pointers
- Guard-stripper to extend: sglang/patches/mtp_tree_xpu.py:179-200 (_strip_is_cuda_guard + re-exec).
- Install dispatch: sglang/patches/woq_shim.py:363-370 (add under B70_XPU_MTP or a new gate).
- Serve to fork: rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh (already B70_XPU_MTP=1).

Residual risk (not a blocker): extra_buffer normally rides the overlap scheduler on CUDA; we sidestep via
--disable-overlap-schedule (still valid because page_size>1 selects extra_buffer). overlap-schedule-on-XPU is
a separate investigation.

Why this matters vs the shipped no_buffer path: keeps intel_xpu XMX attention (no cold-prefill regression),
keeps page_size 64/128 (no page_size=1 side effects), and unlocks the int8 checkpoint pool for ~2x cache
capacity. If validated, this becomes the preferred caching config over triton+no_buffer.
