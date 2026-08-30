# F03 Qwen3.8 official-FP8 source-default completion negative

Date: 2026-08-30

Status: closed negatively. Restoring the pinned vLLM source-default
synchronous all-reduce did not remove fresh-lifetime target nondeterminism.

## CONFIG

- Git harness identity: `30888bc`.
- Model: official `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Runtime: vLLM `0.27.2rc1.dev77+gac7509e2b` and PyTorch `2.13.0+xpu`.
- Source-default overlay image:
  `sha256:c4fc0d651aedd8088daaf57d5de9f623f68f9066a36956fd67652d472c18c3d0`.
  It was built on the verified F02 image and changed only
  `xpu_communicator.py`.
- Source-default communicator SHA256:
  `527cbfb250760abc62096ee7cd612307b821f21b72dee1687ad866620ec89b6d`.
  The recipe's explicit asynchronous-work-plus-wait file had SHA256
  `5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d`.
- The installed W8A16, XPU-op, and GDN files remained byte-identical to F02.
- TP2, P2P off, MTP0, XPU Graph off, deterministic Inductor on, FP16 target,
  KV dtype `auto`, one request, 1,024 context, and prefix caching off.
- Two fresh servers used separate empty compiler caches and 32 GiB no-swap
  cgroups.
- Result directory:
  `/mnt/vm_8tb/b70/results/f03_qwen38_fp8_neural/20260830T004500Z/`.

## COMMAND

Run the tracked wrapper through its self-acquired whole-box lease:

```text
STAMP=20260830T004500Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f03.sh
```

The wrapper verified all 66 model files, ran card and compiled P2P-off
collective health, then ran the complete fixed 12-prompt natural corpus and
independent canaries on each fresh server. It gracefully tore down and repeated
health after each lifetime.

## RESULT

Only 7/12 complete raw output-token arrays matched across the two
source-default lifetimes:

| Prompt | First mismatch, zero based | Differing positions | Length A/B |
| --- | ---: | ---: | ---: |
| `incident-retrospective` | 392 | 110 | 512/512 |
| `code-review` | 303 | 202 | 512/512 |
| `architecture-tradeoff` | 7 | 502 | 512/512 |
| `risk-register` | 127 | 382 | 512/512 |
| `performance-hypotheses` | 479 | 33 | 512/512 |

This is not merely the F02 mismatch set. `architecture-tradeoff` and
`risk-register` were exact across both Work.wait lifetimes but changed across
both source-default lifetimes. Across all four F02/F03 fresh lifetimes, only
5/12 prompts had one unique output; seven prompts had two or three outputs.
Each source-default lifetime matched each local Work.wait reference on only
7/12 prompts.

The diagnostic class-balanced rates were 11.722245 and 11.577714 tok/s, with
an 11.649980 tok/s median and 1.241 percent attempt spread. The median is 2.442
percent above F02's 11.372225 tok/s Work.wait median, but no speed attribution
is qualified because both routes change target arrays across lifetimes.

Both independent canary files passed and again had SHA256
`f234e605954b061e7f902eb92dd96739722df5437cadd9b2aceed79b976e45f8`.
They remain insensitive to the natural-prompt failure.

All pre/inter/post card and compiled P2P-off collective checks passed. Across
293 host samples, swap use remained zero, minimum MemAvailable was
113,374,204 KiB or 108.122 GiB, and memory PSI `some` and `full` totals did not
change. Container host-RAM use peaked near 7.716 GiB. The kernel journal and
server logs had no configured OOM, fatal, hang, GPU-fault, reset, or wedge
marker. The persisted summary SHA256 is
`c7e542cafc6f095dbd9c39975a6f18e79aa9f51a988799f65cf2fc3a917debed`.

Both pairs of fresh compiler caches contained 42 AOTAutograd entries per
attempt under identical primary graph keys, but their secondary artifact keys
differed. This is not proof of causation, but it makes shared-cache reuse the
next narrow discriminator between compile-artifact selection and later
process/runtime initialization.

## VERDICT

F03 fails. Explicit `Work.wait()` is not the root cause of the fresh-lifetime
nondeterminism, and source-default completion has no qualified performance or
stability benefit. Retain the recipe Work.wait image as the control and do not
advance either route to MTP, long context, concurrency, direct-P2P serving, or
the shelf.

F03a subsequently reused one Work.wait compiler cache across two fresh server
processes and matched all 12/12 arrays. Lifetime 2 directly loaded both rank
AOT models in 1.98 seconds. Fresh compilation is therefore the target-selection
locus. Use that pinned artifact as the MTP0 control for F04.
