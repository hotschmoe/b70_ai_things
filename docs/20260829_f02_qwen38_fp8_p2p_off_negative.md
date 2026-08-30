# F02 Qwen3.8 official-FP8 P2P-off negative

Date: 2026-08-29

Status: closed at the cross-server raw-token exactness gate. The local safety
port ran cleanly but is neither exact across fresh lifetimes nor a speed
reproduction of the publisher's P2P-on MTP0 profile.

## CONFIG

- Git harness identity: `7ccff19`.
- Model: official `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Runtime: vLLM `0.27.2rc1.dev77+gac7509e2b`, PyTorch `2.13.0+xpu`.
- Local deterministic overlay image:
  `sha256:dce80db0a1ad861145e88b1c565f29172641912dc75a3b50d08e370f7d58e291`.
- Installed W8A16, GDN state, and oneCCL `Work.wait()` files matched the four
  pinned source hashes in the F01 ledger.
- TP2, P2P off, MTP0, XPU Graph off, deterministic Inductor on, FP16 target,
  KV dtype `auto`, official block-FP8 weights with the W8A16 runtime dispatch,
  one request, 1,024 context, and prefix caching off.
- Each fresh server used a new compiler cache and a 32 GiB cgroup with equal
  memory and memory-swap limits, allowing no container swap.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02_qwen38_fp8_neural/20260829T231100Z/`.

The effective engine reported `dtype=torch.float16`, `quantization=fp8`, no
speculative configuration, and `cudagraph_mode=NONE`. Both ranks logged
`CCL_TOPO_P2P_ACCESS=0`; the W8A16 kernel-selection marker fired in both
lifetimes. Model loading reported 13.85 GiB on the reporting rank, and XPU
Graph capture was explicitly skipped.

## COMMAND

Run the tracked wrapper through its self-acquired whole-box lease:

```text
STAMP=20260829T231100Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02.sh
```

The wrapper verified all model bytes through direct and ordinary reads, ran
card and compiled P2P-off collective health, then created two fresh servers.
Each server ran the complete fixed 12-prompt, six-class, natural 512-token-cap
suite with raw streamed token IDs and zero allowed cached prompt tokens,
followed by the independent semantic/repeat canaries. It gracefully stopped
each server and repeated card plus compiled collective health.

## RESULT

Both individual performance workloads passed their cold-response gates:

| Attempt | Class-balanced first-100 interval rate | Publisher MTP0 array matches |
| --- | ---: | ---: |
| 1 | 11.351052 tok/s | 6/12 |
| 2 | 11.393397 tok/s | 8/12 |
| Diagnostic median | 11.372225 tok/s | not qualified |

The attempt spread was only 0.373%, but the performance attribution is
disqualified because the output target changed. The diagnostic median is only
33.42% of the publisher's matched P2P-on MTP0 median of 34.031596 tok/s, a
66.58% shortfall. P2P is therefore not a minor transport setting in this
recipe.

Only 7/12 complete output-token arrays matched across the two local fresh
servers. The five failures were:

| Prompt | First mismatch, zero based | Differing positions | Length A/B |
| --- | ---: | ---: | ---: |
| `incident-retrospective` | 392 | 110 | 512/512 |
| `code-review` | 303 | 202 | 512/512 |
| `customer-email` | 124 | 129 | 280/272 |
| `performance-hypotheses` | 169 | 341 | 512/512 |
| `decision-memo` | 77 | 414 | 512/512 |

The publisher's two P2P-on MTP0 references match one another 12/12. Against
either reference, local attempt 1 matched 6/12 and attempt 2 matched 8/12.
Two prompts that were exact across both local attempts still differed from the
publisher at tokens 341 and 160. This separates a repeatable P2P-off numerical
route change from the additional fresh-lifetime instability.

Both independent canary files were exact and had SHA256
`f234e605954b061e7f902eb92dd96739722df5437cadd9b2aceed79b976e45f8`.
They are necessary but were not sensitive enough to detect the natural-prompt
failure.

All pre/post card and compiled collective checks passed. Both containers and
endpoints disappeared after graceful teardown. Across 300 host samples:

- swap use remained zero;
- minimum MemAvailable was 113,409,448 KiB, or 108.156 GiB;
- memory PSI `some` and `full` totals changed by zero;
- container memory was about 7.7 to 8.1 GiB while serving;
- the kernel journal had no configured OOM, hang, GPU fault, or wedge marker.

The persisted analyzer summary has SHA256
`b5b522a45ea7b1b89663f87c9b1388a70300c794ac8762214835d5af563fe0b2`.

## VERDICT

F02 fails. The official recipe can be started and torn down safely under the
local P2P-off policy, but its target is not deterministic across fresh
lifetimes and its diagnostic speed is roughly one third of the publisher's
P2P-on MTP0 result. Do not advance this target to packed-RMS MTP1, long-agent,
concurrency, or the shelf.

F02a subsequently repeated the five sensitive prompts twice inside a third
fresh lifetime. All 5/5 were exact within that lifetime, while the third
lifetime selected three outputs from F02 attempt 1 and two from attempt 2.
The instability is therefore selected at fresh compile/server initialization,
not by ordinary request-state drift, and is prompt-specific rather than a
simple whole-server A/B route. Continue with the bounded F03 source-default
completion comparison. Direct-P2P full serving remains quarantined.
