# F02e Qwen3.8 official-FP8 explicit-deterministic oracle

Date: 2026-08-30

Status: passed compile-selection exactness. Explicitly passing
`deterministic=true` inside `inductor_compile_config` collapsed the remaining
reduction tuning surface.

## CONFIG

- Harness commit: `d8f4170`.
- F02d's official FP8 W8A16, r15 Work.wait image, TP2, P2P-off, MTP0, FP16
  target/KV, graph-off, combo-off, pointwise-autotune-off, and separate-empty
  cache settings were retained.
- Added explicit `inductor_compile_config.deterministic=true`.
- Compile-only workload: one 16-token deterministic smoke per server.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02e_qwen38_fp8_neural/20260830T041300Z/`.

## COMMAND

```text
STAMP=20260830T041300Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02e.sh
```

## RESULT

Both independent compiles used AOT key `5001f6c4...`. Generated reduction
kernel metadata recorded `deterministic: True`; each final cache contained
zero `.best_config` files. The smoke outputs were exact. This is the first
fresh-cache pair in the campaign with no compiler-selected semantic target
variation.

Compilation took 92.11 and 90.91 seconds; total engine initialization took
102.78 and 101.69 seconds. This oracle did not run the performance suite, so
it makes no speed or quality claim.

Both smokes, graceful teardowns, card health, and compiled P2P-off collective
health passed. Container host RAM peaked at 7.651 GiB; minimum MemAvailable
was 113,822,200 KiB or 108.549 GiB; swap stayed zero.

Each cache contained 621 files and exactly 322,407,549 bytes. Cache-manifest
SHA256 values were
`3d2d8570cfa48b2746a2f140cbbcbedefc4d2735e2cdc80e0a15e257f6f8a251`
and `c8c9dff822e375ba8d10d2cafb349b5e4b3be90558b9456a4d02a5341a7ba221`.
Primary summary SHA256 was
`47c53fe1719f8a83515027f7f26d3de21c2de4480378e56194e441b690147f23`.

## VERDICT

F02e passes its bounded discriminator and authorizes F02f: the full 12-prompt
two-fresh-cache target gate with the same compiler controls. F02f must still
require cross-process and publisher token exactness, canaries, health,
teardown, and bounded host memory. P2P-on full serving, long context,
concurrency, and shelf promotion remain blocked.
