# sglang-XPU decode kernel-launch inventory + fusion targets (Qwen3.6-27B qwen3_5 GDN)

Static analysis of `sglang-xpu:woq` (in-image `/opt/venv/lib/python3.12/site-packages/sglang/`), TP=1, M=1 decode.
Source: B1 profiling agent, 2026-06-27. This directs the kernel-fusion thrust (task #8 / C1) and the W8A8 decode fix.

## Headline
The model is ALREADY heavily fused everywhere EXCEPT the quantized linears. Per-token launch count for the full
64 layers is ~1045 (the earlier "~256" was only the full-attn-layer or GEMV subset). The dominant lever is NOT
hand-fusing the (already-fused) elementwise paths -- it is the SYSTEMIC launch-killer: spec-decode (amortize
~1000 launches across K accepted tokens) or torch.compile/Inductor (collapse them, NO L0 graph replay). Manual
per-op fusion is incremental mop-up.

## IMPORTANT path correction
The agent profiled the AWQ scheme path (`awq_kernels.py:88-105`): `awq_dequantize(full int4->bf16)` + separate
`torch.matmul` = 2 launches/linear AND it re-materializes the whole bf16 weight to HBM every token (defeats the
int4 bandwidth win). BUT the int4 DAILY DRIVER does NOT use this -- it uses `auto_round_kernel.woqgemm` via
woq_shim (GPTQLinearScheme patch), which the journal established IS a fused int4 GEMV that realizes the 4-bit
bandwidth saving at M=1. So fusion target #1 (fused int4 GEMV) is ALREADY solved for the woqgemm driver; it only
applies if serving `--quantization awq`. VERIFY woqgemm is 1 launch/linear before chasing #1.

## Per-layer launch inventory (M=1 decode)
- Full-attention layer (~16 launches): input_rmsnorm+residual (1, fused), qkv dequant+matmul (2), qk-norm+gate
  (1, fused), RoPE (1), KV write (1), flash_attn (1), output-gate sigmoid-mul (1, fused), o_proj (2),
  post_rmsnorm+residual (1, fused), gate_up (2), SiluAndMul (1, fused), down (2).
- GDN linear-attn layer (~17 launches): input_rmsnorm+residual (1), in_proj_qkvz (2), in_proj_ba (2),
  qkvz/ba split+reshape+cat (1, fused), causal_conv1d_update (1), packed recurrence (1, fused:
  split+L2norm+gating+delta-rule), gated RMSNorm (1, fused), out_proj (2), post_rmsnorm+residual (1),
  gate_up (2), SiluAndMul (1), down (2). NOTE: decode GDN core is just conv + recurrence (2 launches); gating is
  FUSED into the recurrence on decode; NO bf16<->fp32 cast in the decode GDN path.
- Model total: 48*17 + 16*16 + embed + final norm + lm_head ~= 1045 launches/token (~304 are the awq dequant
  calls if on the awq path; ~0 extra if woqgemm).

## Ranked fusion / elimination targets
1. [if awq path only] Fused int4 GEMV (Triton W4A16: load int4 tile -> unpack+scale in-register -> dot). Saves
   ~304 launches/token + ~4x weight HBM traffic. CUDA has it (AWQMarlin); XPU falls back to unfused. N/A to the
   woqgemm driver (already fused). HIGH feasibility on intel-xpu triton (int4 analog of our W8A8 _int_mm work).
2. Fuse RMSNorm -> next GEMV: up to 2/layer = ~128/token. For W8A8 this becomes rmsnorm+int8-dynamic-quant
   fusion (the standard CUDA pattern; vLLM rms_norm_dynamic_per_token_quant) -- directly fixes our W8A8 decode
   +2-launch/layer penalty. MEDIUM feasibility.
3. Port `fused_qk_gemma_rmsnorm_rope_gate` to XPU (fold RoPE into the qk-norm+gate kernel): 16/token. CUDA+NPU
   have it; XPU uses the rope-less variant + separate rope. HIGH feasibility (direct template).
4. Fuse KV-cache write into RoPE (full-attn): 16/token. CUDA fa3 has `fused_set_kv_buffer_arg`; XPU asserts
   unsupported (`rotary_embedding/base.py:449-451`). MEDIUM.
5. Fuse `causal_conv1d_update` into the GDN packed-recurrence kernel: 48/token. MEDIUM-HARD (conv_state ring
   buffer). B70_GDN_DECODE_WARPS already env-tuned (no warm win).
6. ** SYSTEMIC (highest leverage): capture/compile the whole 64-layer decode step ** -> collapses ~1000 python
   submissions into one. torch.xpu graph CAPTURE degrades (L0/NEO accumulation, dead end). torch.compile/Inductor
   (task #12) may collapse launches via fusion + compiled wrapper WITHOUT L0 graph replay -> the launch-bound cure
   that dodges the degradation. Spec-decode (tasks #3/#11) amortizes the same ~1000 launches across K tokens.

## Already fused (do NOT redo)
GDN gating+L2norm+delta-rule recurrence (1 kernel); residual adds (folded into gemma_fused_add_rmsnorm); GDN
qkvz/ba split+reshape+cat (1); full-attn qk-norm+gate-deinterleave (1); MLP SiLU-mul (1); attn output-gate
sigmoid-mul (1); no bf16<->fp32 casts in decode GDN.

## Bottom line
On the woqgemm daily driver the elementwise/attention/GDN paths are already near-maximally fused. The recoverable
LAUNCH win is dominated by the systemic cure (spec amortization / torch.compile collapse), with #2/#3/#4 as modest
per-op mop-up (and #2 = the real W8A8 decode fix). Manual GDN-core fusion (#5) is the hardest for the least.
