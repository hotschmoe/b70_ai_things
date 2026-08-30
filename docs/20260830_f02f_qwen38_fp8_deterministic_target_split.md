# F02f Qwen3.8 official-FP8 deterministic target split

Date: 2026-08-30

Status: passed fresh-cache local token exactness and failed the required
publisher-token gate. Explicit Inductor determinism creates a reproducible
local target, but that target is not the publisher's autotuned target mosaic.

## CONFIG

- Harness commit: `eb14d56`.
- Official FP8 W8A16 checkpoint and exact r15 `Work.wait()` image, TP2,
  P2P-off, MTP0, FP16 target/KV, graph-off, one request, 1,024 context, and
  prefix caching off.
- Two independent server lifetimes used separate empty compiler caches.
- Inductor combo kernels, combo benchmarking, vLLM max autotune, coordinate
  descent, and Triton pointwise autotune were disabled.
- `inductor_compile_config.deterministic=true` was passed explicitly.
- The full fixed 12-prompt suite used a 512-token response cap and required
  cross-lifetime and publisher raw-token exactness.
- Result directory:
  `/mnt/vm_8tb/b70/results/f02f_qwen38_fp8_neural/20260830T042700Z/`.

## COMMAND

```text
STAMP=20260830T042700Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f02f.sh
```

## RESULT

The two independently compiled servers matched all 12/12 complete token
arrays. Both used AOT key `5001f6c4...`, both caches contained 621 files and
zero `.best_config` files, and compilation took 92.19 and 92.38 seconds. The
rank AOT binaries and complete cache manifests still differed because their
build metadata differed, but no output token differed.

Both attempts matched only 8/12 arrays against each of the two mutually exact
publisher references. The required external target gate therefore failed.
This separates two findings: fresh compiler schedule selection caused the
earlier local restart drift, while the publisher artifacts encode a different
valid compiled target. The publisher output is not reproducible from a new
cache merely by requesting deterministic generation.

The diagnostic class-balanced rates were 11.637675 and 11.649289 tok/s, with
an 11.643482 tok/s median and 0.100 percent spread. No performance attribution
or promotion is authorized because publisher target identity failed. Eleven
prompts reached the fixed 512-token response cap and one stopped naturally;
this is a bounded determinism/performance corpus, not a higher-thinkcap agent
quality qualification.

Both canaries, graceful teardowns, card health, and compiled P2P-off
collective health passed after each lifetime. Container host RAM peaked at
7.696 GiB; minimum MemAvailable was 113,710,852 KiB or 108.443 GiB; swap
stayed zero. Runtime accounting reported 14.24 GiB of weights plus non-Torch,
1.19 GiB peak activation, and 8.8 GiB KV cache per card.

The attempt cache-manifest SHA256 values were
`037468e1989a2d8637860adf806305c03515d57a647d7c7a7fe4f646b8df7668`
and `8d5fe722175dd6184da0891604eb5b0acb0e8db8f7e112a182626fa833ef3139`.
The summary SHA256 was
`fc73b5bea7bb0e9c98361cd66e965591292c437fd8cee790a98e19c613703934`.

## VERDICT

F02f closes negatively against the publisher reference but positively as a
local compilation oracle. The local deterministic target is suitable for the
next mechanism discriminator, not for shelf promotion.

Run F02g as an MTP0 bridge in the MTP-capable packed-RMS image with the same
compiler controls and F02f as the frozen local target. Only if F02g is exact
may MTP1 be tested against that target. Keep long-context, concurrency,
P2P-on full serving, speed attribution, and shelf promotion blocked.
