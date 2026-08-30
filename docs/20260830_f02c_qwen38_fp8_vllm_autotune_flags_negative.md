# F02c Qwen3.8 official-FP8 vLLM autotune-flags negative

Date: 2026-08-30

Status: failed. Disabling vLLM's Inductor max-autotune and coordinate-descent
flags did not control PyTorch's XPU pointwise/reduction tuner.

## CONFIG

- Harness commit: `8f9e9e5`.
- F02b's official FP8 W8A16, exact r15 Work.wait image, TP2, P2P-off, MTP0,
  FP16 target/KV, graph-off, deterministic-Inductor, and combo-off settings
  were retained.
- `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` and
  `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0` were added.
- Each server used a separate empty compiler cache.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02c_qwen38_fp8_neural/20260830T031700Z/`.

## COMMAND

```text
STAMP=20260830T031700Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02c.sh
```

## RESULT

The fresh processes compiled the same AOT key `eb5b1c57...` but matched only
8/12 complete arrays. Cross-process mismatches were `code-review` at token
303, `sql-debugging` at 91, `risk-register` at 127, and
`performance-hypotheses` at 169. Each attempt matched each publisher MTP0
reference only 5/12.

Diagnostic rates were 11.577039 and 11.346567 tok/s, median 11.461803 tok/s.
Compilation took 106.62 and 105.46 seconds. These values are diagnostic only
because the identity gate failed.

Both caches still contained the same 44 `.best_config` paths as F02b. Exactly
22 selected different semantic configurations, 20 differed only in metadata,
and two were byte-exact. The disabled vLLM flags therefore changed the AOT key
but did not change the offending XPU autotune surface. PyTorch source identifies
`triton.autotune_pointwise`, which defaults true, as the next direct control.

The two 1,257-file cache-manifest SHA256 values were
`4080f6d74f87acf16f58bdb89753d62b3bf192e6954cb402521836089679375e`
and `5018df4056570d226c70880f4c1c304369d1fbbb3182c677c3a87bbc05ecb5e2`.
Cache-comparison summary SHA256 was
`86134865a45f6d83ff006da881d68dac1dfd07f8e1dcaee51b9759b1beec2d63`;
primary summary SHA256 was
`d4eef66a854bba1482461e62aef11b2293adbb0ef147d369eb7c89f84e0998d1`.

Canaries, graceful teardown, card health, and compiled P2P-off collective
health all passed. Container host RAM peaked at 7.789 GiB; minimum
MemAvailable was 113,556,788 KiB or 108.296 GiB; swap stayed zero.

## VERDICT

F02c is negative. The vLLM max-autotune and coordinate-descent flags are not
the control for the selected XPU schedules. Proceed to the F02d compile oracle
with `triton.autotune_pointwise=false`. Only run the full token suite if two
fresh compile caches become semantically exact. P2P-on full serving, long
context, concurrency, and shelf promotion remain blocked.
