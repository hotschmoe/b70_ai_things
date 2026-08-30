# Qwen3.8 FP8 neural.download transfer: FULL graph and MTP8

## Outcome

The neural.download W8A16 MTP1 recipe was reconstructed against the official
Qwen3.8-27B FP8 weights and qualified with direct oneCCL P2P. Reproducing its
graph-off compiler and service envelope did not reproduce its speed on this
Threadripper 1950X host. A local FULL decode graph intervention exceeded the
45 tok/s goal reproducibly at MTP1. A clean source-only transfer of the
publisher's later MTP8 RMSNorm support then matched the publisher's strict
MTP8 speed range:

| Arm | Primary strict tok/s | Full post-TTFT tok/s | Verdict |
| --- | ---: | ---: | --- |
| F07a publisher-exact graph off, two-lifetime center | 17.373776 | about 17.2 | Compiler-envelope match did not transfer speed |
| F07b MTP1 PIECEWISE graph | 14.035667 | 16.953193 | Safe but slower |
| F07c no-MTP FULL, Triton target | 30.904190 | 30.838216 | Host-submission diagnosis confirmed |
| F07d MTP1 FULL, auto/Flash draft | 30.171360 | 40.341043 | MTP recovered material throughput |
| F07e MTP1 FULL, explicit Triton draft | 36.517513 | 42.034140 | Best bounded one-prompt screen |
| F07f strict varied suite, lifetime 1 | 46.721530 | 43.202664 median | Pass |
| F07f strict varied suite, lifetime 2 | 47.170372 | 43.129044 median | Pass |
| F08a MTP8 FULL bounded cold screen | 29.140815 | 44.262562 | Cold JIT screen only |
| F08b MTP8 strict suite, lifetime 1 | 64.965356 | 46.874718 median | Pass |
| F08b MTP8 strict suite, lifetime 2 | 67.404052 | 47.519518 median | Pass |

The two F07f primary results average 46.945951 tok/s. The two F08b results
average 66.184704 tok/s, 27.48 percent above the public MTP1 headline of
51.918757 tok/s. They sit inside the publisher's later MTP8 range of
62.432362 to 68.049727 tok/s; the local mean is 1.45 percent above the
publisher MTP8 mean. Local full-response post-TTFT medians average 47.197118
tok/s, so the result also clears 45 tok/s beyond the first-100-token metric.

## Qualified MTP1 configuration

CONFIG -> image
`neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local`,
image ID `sha256:8e0e3deb...`; official FP8 W8A16 weights; TP2; FP16 model and
KV; MTP1; deterministic Inductor; packed serial RMSNorm; persistent GDN
scratch; one serving slot; 1,024-token model/profile envelope; prefix cache
off; direct oneCCL P2P; target and draft `TRITON_ATTN`;
`FULL_DECODE_ONLY`; forced graph capture with communication; kernel
7.1.0-070100.

COMMAND -> run
`I_KNOW_P2P_WEDGES=1 bash
vllm/fp8/run_qwen38_fp8_f07f_full_decode_triton_mtp1_strict_suite.sh` under
the whole-box `bin/gpu-run` lease. The script runs card and compiled P2P-off
collective health before and after the guarded P2P1 model transaction.

RESULT -> two fresh server lifetimes passed the complete fixed 12-prompt
suite. Every prompt ran once, every `cached_tokens` value was zero, and all
responses covered the 100-event metric. Primary class-balanced rates were
46.721530 and 47.170372 tok/s. The complete raw token arrays matched 12/12
between lifetimes. The confirmation lifetime passed 32/32 exact-answer and
isolation requests across eight four-client rounds. Both server teardowns,
both card checks, and both compiled TP2 collective checks passed.

RESULT -> the graph intervention is locally deterministic but not an exact
publisher-output route. It matches 7/12 complete token arrays against each of
publisher r32a and r32b. F07a showed that the graph-off local path also failed
publisher identity and remained near 17 tok/s despite matching the publisher
target and draft AOT keys. Label F07f as a qualified local FULL graph route,
not a byte-identical reproduction of the publisher process.

RESULT -> evidence roots:

- `/mnt/vm_8tb/b70/results/f07a_qwen38_fp8_neural_publisher_exact/20260830T152100Z/`
- `/mnt/vm_8tb/b70/results/f07f_qwen38_fp8_full_decode_triton_mtp1_strict/20260830T163000Z/`
- `/mnt/vm_8tb/b70/results/f07f_qwen38_fp8_full_decode_triton_mtp1_strict/20260830T164000Z/`

The F07f performance SHA256 values are `058c26b9...` and `c9c5ce67...`.
The confirmation concurrent-quality SHA256 is `1d551218...`.

VERDICT -> the 45 tok/s single-stream goal is met and reproduced under the
publisher's strict varied-prompt metric. The causal win is full decode graph
capture, not direct P2P alone: P2P raised the earlier eager path only to about
18.34 tok/s, while FULL capture removed the dominant host submission cost.
Retain the explicit P2P risk guard and one-slot scope. A future shelf entry
still needs packaging and a deliberate serving policy decision because this
route changes attention backend and output identity relative to the publisher.

## Qualified MTP8 configuration

CONFIG -> start with the qualified image above and apply only the publisher's
103-line `layernorm.py` change from vLLM source commit
`ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`. The tracked patch has SHA256
`98c26561...`; the installed Python file has SHA256 `d911627c...`. The derived
image is `b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a`, image ID
`sha256:9ae697d4...`. No old wheel, shared object, or other ABI-specific binary
was transferred. Keep the F07f TP2, FP16 KV, direct-P2P, FULL decode graph,
Triton target/draft attention, one-slot, and 1,024-token settings, but request
eight speculative tokens.

COMMAND -> build with a new dedicated build directory using
`vllm/fp8/build_qwen38_fp8_mtp8_rms_overlay.sh`. Run the strict suite using
`I_KNOW_P2P_WEDGES=1 bash
vllm/fp8/run_qwen38_fp8_f08b_full_decode_triton_mtp8_strict_suite.sh`.
For confirmation, point `SEED_CACHE` at the first F08b cache and set
`EXTRA_SMOKE` to the publisher's `qwen38-concurrent-quality-canary.py`.

RESULT -> fresh-lifetime primary rates were 64.965356 and 67.404052 tok/s.
Full-response post-TTFT medians were 46.874718 and 47.519518 tok/s. Both strict
gates passed with zero cached tokens, and all 12 complete token arrays matched
across the two local lifetimes. The confirmation canary passed 32/32 queued
four-client isolation requests. Both teardowns, both card checks, and both
compiled P2P-off collective checks passed.

RESULT -> the local outputs match 8/12 publisher MTP8 r1a arrays and 9/12 r1b
arrays. This is a reproducible local MTP8 route, not publisher byte identity.
The publisher also withheld its original dynamic-MTP8 artifact after its own
cross-run exactness gate failed. The local source overlay closes that local
repeatability gate, but does not make the processes identical.

RESULT -> F08b evidence roots:

- `/mnt/vm_8tb/b70/results/f08b_qwen38_fp8_full_decode_triton_mtp8_strict/20260830T170500Z/`
- `/mnt/vm_8tb/b70/results/f08b_qwen38_fp8_full_decode_triton_mtp8_strict/20260830T172000Z/`

The performance SHA256 values are `100da68c...` and `21cbc97a...`. The
confirmation concurrent-quality SHA256 is `db19b6d1...`.

VERDICT -> F08b is the qualified local single-stream performance winner. It
matches the publisher MTP8 first-100-token range and clears 45 tok/s on the
full-response median. Retain F07f as the simpler MTP1 control. Do not describe
F08b as the public graph-off MTP1 recipe, and do not promote it to the serving
shelf until a deliberate max-concurrency policy and shelf packaging pass are
completed.

Publisher reference:
https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html
