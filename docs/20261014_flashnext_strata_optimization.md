# Flash-Next Strata optimization, 2026-10-14

Raw evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_opt_20261014`.
The 27B remains available with `bash strata/flash-next/fallback-27b.sh`.
That wrapper checks the retained prerequisites, stops Strata, and starts the
existing leased 27B serve command in a persistent user unit. Syntax and the
backend's prerequisite check passed; the new wrapper has not switched models
in this stage. Its log is `/mnt/vm_8tb/b70/results/fallback-27b.log`.

## S1: upstream head, matched serving screen

CONFIG: ISTA GSQ-RCO IQ3_XXS, both B70s, all 24576 main experts resident,
split25, two 131072 slots, int8 KV, MTP4/min-p0.5, prefill4096/no borrowing,
reserve1024 MiB, parking8192 MiB/4 entries. Stable name `hotschmoe-dd` first,
registered alias `qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata`.
Source changes from stock a0116b27/v0.1.43.1 to upstream
296b59d2f73c8f04a805df29c2fa97655adace78/v0.1.44.1, with no local patches.
Kernel 7.1.0-070100-generic, Compute Runtime 26.22.38646.4, same image
e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2,
oneAPI 2026.1, bmg-g31 AOT. Host RAM is 121 GiB; no host changes made.
The diagnostic image has PyTorch 2.13.0+xpu and oneCCL 2022.0.0;
Strata itself uses SYCL, not PyTorch/oneCCL. Host/container Level Zero and
all identities are in `identity-before.json`; binary hashes in
`binary-stock.sha256` and `binary-head.sha256`.

COMMAND: fetch origin, fast-forward the clean Strata source checkout, then
build `strata` and its static libraries in `src/build-sycl-head` using
`cmake -S /work/src/sycl -B /work/src/build-sycl-head -G Ninja
-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
-DSTRATA_SYCL_AOT=bmg-g31 -DSTRATA_GGML_DIR=/work/llama-pinned
-DSTRATA_SYCL_PARITY=ON`, then `cmake --build ... -j 12 --target strata`.
The existing pinned ggml source is 3cf03257f219afbe7334045ff7c6a06ac68c627d.
Build in the same image with `/mnt/vm_8tb/b70/build/strata-ab:/work`, no GPU
devices attached. Preserve `build-sycl-aot/strata` as the old binary.
Run `python3 strata/flash-next/bench_serving.py --output RAW/stock-a0116b27`
and `--output RAW/head-296b59d2` while each server holds the pair lease.
Three fresh prompts per case, greedy/thinking off, 1500 output tokens for
coding and 256 for document prompts; actual input counts 105/4059/32728.
Check concurrent arithmetic, then the existing S1 reasoning/tool probe with
its output directed into `RAW/head-s1` (no old evidence overwritten).
Before/after: leased `bin/xpu-health --img
b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78
--timeout 400` and `bin/xpu-collective-health --timeout 400`.
Configs, raw SSE, output hashes, build log, health and teardown logs are in RAW.

RESULT: medians of three fresh prompts per case, old -> upstream:

| Case | Prefill tok/s | Engine decode tok/s |
|---|---|---|
| Coding (105 input, 1500 output) | 119.1 -> 121.5 | 83.1 -> 83.5 |
| 4059 input, 256 output | 1317.6 -> 1238.5 | 64.5 -> 65.4 |
| 32728 input, 256 output | 1903.2 -> 1853.7 | 64.0 -> 64.7 |

No cached prompt tokens in the nine timing rows. Ten of eleven responses
(including the two concurrent arithmetic checks) are byte-identical between
builds. Coding repeat 2 first differs at character 5728, "correct position"
versus "correct index"; no claim of complete token parity. Arithmetic,
separate reasoning_content, and lookup_build_status tool round trip pass.
Both cards and ten compiled two-rank all-reduces pass before/after; head
stops cleanly with gpu-run exit0. One fdinfo sample under load is
25.32/29.37 GiB resident on cards 0/1. The old server had 10.7 hours uptime
at intake; this is observed uptime, not a multi-day stability qualification.

VERDICT: upstream head is a usable baseline for the next stage. This
sequential three-prompt screen establishes no meaningful decode improvement;
4K prefill is 6.0% lower and 32K prefill 2.6% lower in this screen. No shelf
promotion or new full quality eval. PRs #1798 and #2205 are still open and
absent from upstream head at intake. The former reports about 13-14% gain
against its matched layer-split control, not a matched 40 -> 108 tok/s gain:
https://github.com/Niko1221/Strata/pull/1798
https://github.com/Niko1221/Strata/pull/2205

## Serving and observation commands

Start a chosen stock config in a persistent user unit:

```sh
systemd-run --user --unit=strata-optimized --collect \
  --property=WorkingDirectory=/mnt/vm_8tb/github/b70_ai_things \
  /bin/bash /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/launch.sh \
  /mnt/vm_8tb/b70/build/strata-ab/serve-optimized.json
```

Stop with `docker stop -t 30 strata-ab-serve`; the owning service releases
the pair lease. GPU health must pass before starting another configuration.
The final config and soak status are recorded below when qualified.
