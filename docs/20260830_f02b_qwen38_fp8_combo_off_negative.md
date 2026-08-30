# F02b Qwen3.8 official-FP8 combo-off negative

Date: 2026-08-30

Status: failed. Disabling Inductor combo kernels and combo-kernel benchmarking
reduced the autotune surface but did not make two fresh compilations select an
exact target.

## CONFIG

- Harness commit: `e1221c1`.
- Official FP8 W8A16 model, exact r15 Work.wait image, TP2, P2P off, MTP0,
  FP16 runtime and KV, XPU graphs off, and deterministic Inductor were retained.
- `combo_kernels` and `benchmark_combo_kernel` were both false.
- Each server used a separate empty compiler cache.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02b_qwen38_fp8_neural/20260830T024100Z/`.

## COMMAND

```text
STAMP=20260830T024100Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02b.sh
```

## RESULT

Both processes compiled the same AOT key `3b34c823...`, but their complete
token arrays matched only 7/12. Divergence began on five sensitive prompts:
`incident-retrospective` at token 392, `code-review` at 303,
`customer-email` at 124, `technical-guide` at 428, and
`performance-hypotheses` at 169.

Attempt 1 matched each publisher MTP0 reference 10/12; attempt 2 matched each
6/12. Attempt 1 differed on `technical-guide` and `risk-register`. Attempt 2
also differed on the three early-sensitive prompts and
`performance-hypotheses`.

Diagnostic class-balanced rates were 11.465029 and 11.419766 tok/s, with an
11.442398 tok/s median. Compilation took 105.74 and 105.87 seconds. These are
diagnostic values only because token identity failed.

Combo-off reduced `.best_config` sites from F02's 78 to 44. All 44 paths were
common between the new caches, but 22 selected different semantic block,
reduction, or warp configurations. Twenty-one differed only in nonsemantic
metadata and one was byte-exact. Both ranks' final AOT model binaries differed
between attempts.

The two 1,257-file cache manifests have SHA256
`22ea795f1249604897abf084a100c62b57d17188d92fca66f0158c2545258a6a`
and `53ccc5f38fed011ebd46264681f313ec72618b0a6f2ef0c61be25de80f9702f2`.
Cache-comparison summary SHA256 is
`aa731b5a29e9b03b646c42746a8a67560caa26d9a9081421e73dae3a2f3db812`;
primary summary SHA256 is
`57500b75993cfe554cef6fb87214b77447de8c513923ce5b41544efaa77b3a7a`.

Both canaries, all card checks, all compiled P2P-off collective checks, and
graceful teardowns passed. Container host RAM peaked at 7.793 GiB; minimum
MemAvailable was 113,617,324 KiB or 108.354 GiB; swap stayed zero. Memory PSI
`some` and `full` totals moved by 332.354 and 331.462 milliseconds.

## VERDICT

F02b is negative. Combo-kernel benchmarking is not the root cause. The
remaining per-kernel max-autotune and coordinate-descent choices still select
different arithmetic schedules across fresh compilations.

Proceed to F02c with combo kernels still off and both
`VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` and
`VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0`. Require cross-compile and
publisher exactness. P2P-on full serving, long context, concurrency, and shelf
promotion remain blocked.
