# F02d Qwen3.8 official-FP8 pointwise-off compile oracle

Date: 2026-08-30

Status: failed its compile-selection gate. Disabling PyTorch Triton pointwise
autotuning reduced the selection surface from 44 to 16 sites, but five
reduction sites still selected different schedules.

## CONFIG

- Harness commit: `fae7351`.
- Official FP8 W8A16, exact r15 Work.wait image, TP2, P2P off, MTP0, FP16
  target/KV, graph off, combo off, and two empty caches.
- `triton.autotune_pointwise=false`; vLLM max-autotune and coordinate descent
  were also false.
- Compile-only workload: one 16-token deterministic smoke per server followed
  by teardown and health. No speed or quality qualification was attempted.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02d_qwen38_fp8_neural/20260830T035700Z/`.

## COMMAND

```text
STAMP=20260830T035700Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02d.sh
```

## RESULT

Both attempts used AOT key `617687ab...`, produced 16 common best-config
paths, and returned exact smoke text. Five sites selected different semantic
configs; ten differed only in metadata and one was byte-exact. Every semantic
difference was the same reduction-family choice: `R0_BLOCK=2048` versus
`R0_BLOCK=8192`, with `XBLOCK=1`, 16 warps, and one stage unchanged.

The generated kernel source exposed the missing control. Its
`inductor_meta` recorded `deterministic: False` even though the launcher set
`TORCHINDUCTOR_DETERMINISTIC=1`. The environment setting did not survive into
the AOT compilation patch. PyTorch already has a deterministic reduction
filter; pass `deterministic=true` in `inductor_compile_config` explicitly
before considering a source patch.

Compilation took 96.25 and 96.33 seconds; total engine initialization took
106.95 and 107.05 seconds. Both smokes, graceful teardowns, card health, and
compiled P2P-off collective health passed. Container host RAM peaked at 7.668
GiB; minimum MemAvailable was 113,721,668 KiB or 108.453 GiB; swap stayed zero.

The two 781-file cache-manifest SHA256 values were
`51429dcec22ab8a1e2a173fec8de301b7263a111830d213a4816702963f3dda9`
and `a8dd9ecce07da9c2436b9d3f1a18afc9ac4da7d699febe77c33dcb9f2bfa655b`.
Primary summary SHA256 was
`135f482a392bb4367fafa25873e8bb1bfba33931167c02ab1e2c815e46357f58`.

## VERDICT

F02d is negative but narrows the remaining cause to five reduction schedules.
Proceed to F02e with the same bounded compile oracle and explicit
`inductor_compile_config.deterministic=true`. Only run a full token suite if
the two fresh caches become semantically exact. P2P-on full serving, long
context, concurrency, and shelf promotion remain blocked.
