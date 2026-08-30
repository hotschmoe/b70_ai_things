# F02g Qwen3.8 official-FP8 packed-RMS target bridge

Date: 2026-08-30

Status: passed. The MTP-capable image with packed serial RMSNorm, MTP disabled,
and explicit compiler determinism reproduced the frozen F02f local target
across two independent empty caches.

## CONFIG

- Harness commit: `58baa4e`.
- Official FP8 W8A16 checkpoint; local MTP-capable image
  `sha256:338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b`.
- TP2, P2P-off, MTP0, packed serial RMSNorm, FP16 target/KV, graph-off, one
  request, 1,024 context, and prefix caching off.
- Two lifetimes used separate empty caches with the complete F02f compiler
  controls, including explicit `inductor_compile_config.deterministic=true`.
- Frozen references were both mutually exact F02f lifetimes.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02g_qwen38_fp8_neural/20260830T050400Z/`.

## COMMAND

```text
STAMP=20260830T050400Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02g.sh
```

## RESULT

Both fresh compiles matched one another 12/12 and each matched both F02f
references 12/12. Packed serial RMSNorm and the MTP-capable image therefore do
not change the explicit-deterministic MTP0 target. Both compiles used AOT key
`5001f6c4...`, took 95.41 and 95.84 seconds, and left 621-file caches with
zero `.best_config` files.

Diagnostic class-balanced rates were 11.503855 and 11.434064 tok/s, with an
11.468959 tok/s median and 0.609 percent spread. The median was 1.499 percent
below F02f, which is below the campaign's three-percent attribution threshold.
No performance preference is claimed.

Both canaries, graceful teardowns, card health, and compiled P2P-off
collective health passed. Container host RAM peaked at 7.708 GiB; minimum
MemAvailable was 113,609,756 KiB or 108.347 GiB; swap remained zero. Device
accounting reported 14.24 GiB weights plus non-Torch, 1.19 GiB peak
activation, and 8.8 GiB KV cache per card.

The cache-manifest SHA256 values were
`8ef997c35eac2eadf9d61ae4349314ad9fc47f0d5bb100f7103b6cca1c0fa8ee`
and `9798b40d0db70315901627c6e71de3090b977fd467dde8d901f413a970eabf0b`.
The summary SHA256 was
`a378bf0d71b9b4fd9ec9a62b89d460f87b0afb79cac4907661acabc1c56ef3bf`.

## VERDICT

F02g passes its bridge gate and authorizes F04b: MTP1 with the same image,
packed RMSNorm, explicit deterministic compiler controls, two empty caches,
and F02g as the frozen MTP0 target. Long-context, concurrency, P2P-on full
serving, and shelf promotion remain blocked.
