# F04a Qwen3.8 official-FP8 forced-rejection positive

Date: 2026-08-30

Status: passed its discriminator. With the exact F04 target and draft AOT
artifacts, forcing every MTP1 draft to reject produced the same 12/12 arrays
as normal F04 across two fresh server processes.

## CONFIG

- Git harness identity: `f59c6d9`.
- F04 image, model, TP2, P2P-off, graph-off, deterministic-Inductor, W8A16,
  packed-RMS, persistent-GDN, FP16 target/KV, one-request, and 1,024-context
  settings were unchanged.
- The exact 3,081-file F04 cache was verified from its manifest and copied to
  a new cache root before launch.
- MTP remained enabled at depth one. The only inference-policy change was
  synthetic acceptance rate zero, which forces every draft to reject while
  retaining the two-row target-verification path.
- Result directory:
  `/mnt/vm_8tb/b70/results/f04a_qwen38_fp8_neural/20260830T020500Z/`.

## COMMAND

```text
STAMP=20260830T020500Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f04a.sh
```

## RESULT

Both server processes directly loaded target AOT key `ed4b9708...` and draft
AOT key `aa87ccb...` on both ranks. Attempt 1 reported 2.19/0.13 seconds and
attempt 2 reported 1.92/0.10 seconds for the target/draft loads. No target or
draft recompilation occurred.

Acceptance was exactly zero throughout both workloads: mean acceptance length
1.00, zero accepted tokens, and nonzero drafted tokens. Despite that, all
12/12 complete arrays matched normal F04 in both attempts and matched one
another 12/12. Therefore accepted draft tokens and rejection sampling are not
the cause of F04's target difference.

The forced-rejection diagnostic rates were 10.159880 and 10.178679 tok/s,
median 10.169280 tok/s with 0.185 percent spread. This route deliberately pays
draft and two-row verification cost while accepting no draft work; it is an
oracle, not a serving candidate. Normal F04's 18.243500 tok/s signal is caused
by useful accepted draft work, but remains unqualified against an external
target.

Additional comparisons showed:

- 5/12 exact versus both frozen local F03a MTP0 attempts;
- 8/12 exact versus both publisher P2P-on MTP0 r15 attempts; and
- 8/12 exact versus both publisher P2P-on MTP1 r32 attempts.

The publisher MTP0 and MTP1 arrays are mutually exact. F04a's four publisher
mismatches were `incident-retrospective` at token 392, `code-review` at 303,
`customer-email` at 124, and `technical-guide` at 160. This identifies F04 as
another locally selected compiled target variant, not an MTP acceptance error.

Both canaries, all pre/inter/post card and compiled P2P-off collective checks,
and graceful teardown passed. Container host-RAM use was 8.336 to 8.442 GiB,
swap stayed zero, minimum MemAvailable was 112,926,784 KiB or 107.695 GiB,
and memory PSI `some`/`full` totals moved by 677.042/673.147 milliseconds.
No configured kernel or server fault marker appeared.

The final 368 MiB cache contained 3,131 files. Manifest SHA256 was
`ecf1d795d43494631134f8bbf943d42b5e2d91a5a68b1e257f96d75dab254a6c`;
the primary summary SHA256 was
`911199dbce6e42cccd2ec7ba03e2fc7067ed0e045d3e4c82c6884d0880e7694b`.

## VERDICT

F04a passes and closes the acceptance-path question. Draft acceptance is
innocent. The remaining problem is fresh compiler/autotune target selection.

A read-only comparison of the two F02 caches found the same primary graph and
the same 78 `.best_config` sites, but 37/78 selected different semantic Triton
configurations such as block sizes and warp counts. Forty-one differed only in
recorded tuning time. F02b should disable XPU combo-kernel benchmarking and
test two separate fresh MTP0 caches. P2P-on full serving, long-context,
concurrency, and shelf promotion remain blocked.
