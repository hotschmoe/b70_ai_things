# F05a Qwen3.8 official-FP8 32K long-context pass

Date: 2026-08-30

Status: passed synthetic C1 long-context and forced-output qualification. Two
fresh 32K-configured MTP1 lifetimes reproduced the bounded target, four actual
context points through 30,023 prompt tokens, and a 4,096-token forced output.

## CONFIG

- Harness commit: `cbb24dc`.
- F04b official FP8 W8A16 MTP1 route, TP2, P2P-off, packed serial RMSNorm,
  persistent GDN scratch, FP16 target/KV, graph-off, prefix caching off, and
  explicit deterministic compiler controls.
- `max_model_len=32768`, `max_num_batched_tokens=32768`, and one sequence.
- Two lifetimes used independent empty caches and both F04b attempts as frozen
  short-target references.
- Each lifetime ran the fixed 12-prompt suite, cold 2K/8K/16K/30K prompts with
  128 forced output tokens, and a 2K-prompt plus 4,096 forced-output trace.
- Result directory:
  `/mnt/vm_8tb/b70/results/f05a_qwen38_fp8_neural/20260830T061000Z/`.

## COMMAND

```text
STAMP=20260830T061000Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f05a.sh
```

## RESULT

The bounded target passed 12/12 across fresh lifetimes and 12/12 against both
F04b references. Class-balanced rates were 17.746417 and 17.743088 tok/s,
with a 17.744753 tok/s median and 0.019 percent spread.

Actual prompt token counts were 2,070, 8,214, 16,407, and 30,023. Every point
produced exactly 128 raw tokens, used zero cached prompt tokens, and matched
the other lifetime exactly. Attempt 1 TTFT values were 2.821, 10.878, 21.700,
and 40.151 seconds; attempt 2 values were 2.818, 10.865, 21.682, and 40.101
seconds. Decode stayed between 17.15 and 18.34 tok/s across the matrix.

Both 4,096-token forced outputs used 2,070 prompt tokens and matched as
complete raw-token arrays. Attempt rates were 19.186696 and 19.101269 tok/s
after TTFT, a 0.447 percent difference. TTFT was 2.817 and 2.811 seconds.

Target AOT key `80de0121...` and draft AOT key `be175b50...` repeated across
caches. Target compilation took 95.31 and 95.54 seconds; the larger-shape
draft compilation took 45.07 and 45.11 seconds. Both 976-file caches contained
zero `.best_config` files.

All canaries, graceful teardowns, card health, and compiled P2P-off collective
health passed. Container host RAM peaked at 9.663 GiB; minimum MemAvailable
was 111,723,052 KiB or 106.548 GiB; swap stayed zero. The 32K compile reported
15.47 GiB weights plus non-Torch, 2.91 GiB peak activation, and 5.86-5.87 GiB
KV cache per card.

The cache-manifest SHA256 values were
`5a5411b448e7140cb325d96b7dbd96768dae46b9cb9599340896524891b8a82a`
and `3bce03ed3b0529eaf7b84ddc19fbb2d1553064b8a94093d3e9757e2875b8342b`.
The primary summary SHA256 was
`014fd18be7c66bda43b0d83e11c371c5bbe5c8837948297a1b249b36ee1c194d`.

## VERDICT

F05a qualifies synthetic one-stream context through 30K and forced output
through 4K with restart exactness and clean health. It does not qualify
concurrent serving, growing tool-use sessions, or model quality. Proceed to a
concurrent batch-shape gate before any shelf entry, then run the higher
thinking-cap agent ladder. P2P-on full serving remains blocked.
