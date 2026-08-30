# F02a Qwen3.8 official-FP8 within-lifetime diagnostic

Date: 2026-08-30

Status: complete. The five F02-sensitive prompts are exact when repeated in
one server lifetime. A third fresh lifetime selected a prompt-specific mosaic
of the two earlier outputs, localizing the unresolved instability to fresh
compile/server initialization.

## CONFIG

- Git harness identity: `e6e3ee9`.
- Model: official `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Runtime and image were unchanged from F02: vLLM
  `0.27.2rc1.dev77+gac7509e2b`, PyTorch `2.13.0+xpu`, and local overlay image
  `sha256:dce80db0a1ad861145e88b1c565f29172641912dc75a3b50d08e370f7d58e291`.
- TP2, P2P off, MTP0, XPU Graph off, deterministic Inductor on, FP16 target,
  KV dtype `auto`, official block-FP8 weights plus W8A16 runtime, one request,
  1,024 context, prefix caching off, and a fresh compiler cache.
- The five prompts were exactly the F02 cross-lifetime failures:
  `incident-retrospective`, `code-review`, `customer-email`,
  `performance-hypotheses`, and `decision-memo`.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02a_qwen38_fp8_neural/20260829T235100Z/`.

## COMMAND

Run the tracked wrapper through its self-acquired whole-box lease:

```text
STAMP=20260829T235100Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02a.sh
```

The wrapper ran pre-health, started one fresh server with a fresh compiler
cache, requested all five prompts twice with raw streamed token IDs and zero
cached prompt tokens, gracefully stopped the server, and ran post-health.

## RESULT

All five complete output-token arrays were exact across the two repeats in the
same lifetime. Cached prompt tokens were zero for every request. The two
class-balanced first-100 interval rates were 11.224449 and 11.095187 tok/s,
with a diagnostic median of 11.159818 tok/s. No performance attribution is
made because the cross-lifetime target remains unstable.

The third lifetime did not select one earlier server's global result. It
selected this prompt-specific mosaic:

| Prompt | F02 attempt 1 | F02 attempt 2 |
| --- | ---: | ---: |
| `incident-retrospective` | exact | different |
| `code-review` | exact | different |
| `customer-email` | exact | different |
| `performance-hypotheses` | different | exact |
| `decision-memo` | different | exact |

Both repeats produced the same mosaic: 3/5 prompts matched F02 attempt 1 and
2/5 matched F02 attempt 2. This rules out ordinary request-order or mutable
request-state drift as the immediate cause. The choice is made during fresh
compiler/server initialization and is prompt-specific rather than a simple
whole-server A/B route.

All pre/post card and compiled P2P-off collective checks passed. Across 149
host-monitor samples, swap use remained zero, minimum MemAvailable was
113,335,124 KiB or 108.085 GiB, and memory PSI `some` and `full` totals did not
change. The container peaked at about 7.717 GiB of host RAM, was limited to
32 GiB with no additional swap allowance, and was not OOM-killed. The kernel
journal and server log had no configured OOM, fatal, hang, GPU-fault, reset,
or wedge marker. The persisted summary SHA256 is
`a451ab90693be76eaab82bd44812721a24d0ff0edd9638c6e14ae58e1c79d404`.

## VERDICT

F02a passes its diagnostic gate: this route is deterministic within one
server lifetime. It does not repair F02 or qualify the target for MTP, long
context, concurrency, performance attribution, or shelf promotion.

Proceed to F03 as a bounded P2P-off comparison of source-default collective
completion against the recipe's explicit `Work.wait()` route. Hold all other
runtime files, model bytes, compiler settings, prompts, and health gates fixed.
Direct-P2P full serving remains quarantined.
