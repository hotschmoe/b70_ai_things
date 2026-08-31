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

## Daily-driver and long-envelope qualification

CONFIG -> retain the F08 derived image and exact runtime hashes, official FP8
W8A16 weights, TP2, FP16 KV, direct P2P, FULL decode graph, Triton target and
draft attention, deterministic compiler settings, and prefix cache off. Raise
the service envelope to 262,144 tokens, four sequences, 32,768 batched tokens,
and 0.96 GPU memory utilization. Compare fixed MTP1 and MTP8 with identical
single-stream, c2, c4, quality, teardown, and health probes.

COMMAND -> run F09d and F09e as matched one-lifetime MTP1/MTP8 screens. Run
the 30,037-token semantic needle under the large envelope. Promote MTP1 to two
fresh F09f lifetimes using
`vllm/fp8/qualify_qwen38_fp8_f09f_mtp1_daily_driver.sh`. Every real GPU touch
ran through the whole-box `bin/gpu-run` lease and every risky P2P transaction
was surrounded by card and compiled P2P-off collective health.

RESULT -> MTP1 exposed 323,202 aggregate GPU KV tokens, or 1.23 full 262,144
windows. MTP8 exposed 303,414, or 1.16 windows. One full agent fits entirely
in VRAM under either profile; four full agents do not. Four equal active
requests share roughly 75K to 80K tokens each, subject to scheduler details.

RESULT -> MTP8 remained the speed winner. Its matched screen measured 59.07
tok/s aggregate at c2 and 101.54 at c4. A later fresh lifetime measured 61.49
tok/s strict c1, 58.44 c2, and 102.21 c4. MTP1 F09f measured strict rates of
46.610781 and 46.597152 tok/s; c2 was 45.708003 and 48.045488; c4 was
89.094019 and 88.830444. MTP1 c4 streams centered near 25 tok/s. Both MTP1
lifetimes passed 32/32 concurrent exact-answer requests.

RESULT -> the 30,037-token fresh ingest took 221.968 seconds with MTP8 and
221.961/221.837 seconds with MTP1 under the 262K envelope. The nearly identical
TTFT proves speculative depth is not the cause of the long-prefill collapse.
For comparison, the prior 32K-envelope MTP1 qualification processed 30,023
tokens in about 40.1 seconds. The regression is tied to the large configured
service/prefill shape. A 260K probe held its first 32,768-token worker
submission for more than ten minutes; it was aborted, Xe was rebound, and
card plus compiled collective health passed. Do not claim qualified near-full
prefill throughput.

RESULT -> F09f passed two-lifetime model identity, complete 12-prompt token
arrays 12/12, independent canaries, c2/c4 serving, 64/64 concurrent quality
requests, and identical 30K semantic-needle token arrays. Both lifetimes tore
down cleanly and passed post-health. No extra swap was allowed. The summary
SHA256 is `44da0a2d...`; daily-driver gate SHA256 values are `0ede55df...` and
`a998cfc2...`. Evidence root:
`/mnt/vm_8tb/b70/results/f09f_qwen38_fp8_mtp1_daily/20260830T230000Z/`.

RESULT -> V1 dynamic MTP accepted the publisher schedule but downgraded FULL
capture to PIECEWISE and fell to 27.64 tok/s strict. V2 retained FULL but
failed native GDN graph warmup because its speculative-token shape violated
the runner invariant. Keep fixed MTP profiles. Prefix caching remains disabled
in the qualified route, so repeated growing-agent cache reuse is not yet a
performance claim.

VERDICT -> promote the exact MTP1 F09f route as the conservative shelf default
at a reproducible 46.604 tok/s strict median, with MTP8 as an explicit faster
decode profile. The configured 262K window is a capacity ceiling, not a
near-window prefill-speed qualification. The next campaign should isolate the
large-envelope prefill regression and then enable and qualify prefix caching.

CONFIG -> F09g used the promoted
`rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh` default profile with its separate
daily cache and otherwise exact F09f runtime.

COMMAND -> run `vllm/fp8/smoke_qwen38_fp8_f09g_shelf.sh` under `bin/gpu-run`.

RESULT -> the shelf wrapper returned the exact served ID, completed a live
request, removed its container and listener, and passed post-run card and
compiled TP2 collective health. Evidence root:
`/mnt/vm_8tb/b70/results/f09g_qwen38_fp8_shelf_smoke/20260830T233000Z/`.

VERDICT -> the promoted MTP1 shelf wrapper is live. `PROFILE=fast` selects
the measured MTP8 decode profile without changing directories.

## Publisher graph-off and c64 profile correction

Steve's 1,091.642460 tok/s c64 result is not the graph-off r32 profile. It is
an older high-concurrency service with XPU Graph enabled, PIECEWISE mode, and
only batch size one captured. Its service envelope is 256 model tokens, 128
sequences, and 512 batched tokens. At c64 it measures 17.056913 tok/s per
stream. The scheduler can still batch 64 requests; batch shapes above the
single captured size have no configured batch-64 graph. The qualified r32
51.918757 tok/s result is the separate graph-off, one-sequence, 1,024-token
strict profile.

Graph capture is therefore not the reason the c64 service accepts high
concurrency. Its local advantages are reduced host submission overhead and
faster single-stream decode. Its costs include capture startup and cache
complexity, static-shape coverage and fallbacks, replay memory, harder
distributed ordering and debugging, and kernel compatibility limits. On this
stack FULL capture cannot contain the default FlashAttention scratch path, so
the qualified local FULL route uses Triton target and draft attention. Dynamic
MTP also either downgraded to PIECEWISE or failed a graph shape invariant.
Steve's qualification note shows why r32 stayed graph off: earlier fast paths
failed complete-output repeatability, while two deterministic r32 servers and
both MTP0 references finally matched 12/12. That is an exactness decision, not
a concurrency requirement.

CONFIG -> F10a matched the runnable r32 mechanisms and service envelope:
official FP8 W8A16, TP2, FP16 KV, MTP1, direct oneCCL P2P, graph disabled,
default FlashAttention v2, deterministic Inductor, packed serial RMSNorm,
persistent GDN scratch, publisher compilation JSON, one slot, 1,024 model and
batched tokens, 0.96 GPU-memory utilization, two empty compile caches, and the
publisher's 9 GiB memory plus 12 GiB memory-and-swap cgroup. The local image
was `8e0e3deb...` with the pinned `1e90ffa672` kernel and verified runtime-file
hashes. The documented publisher image `ba42e928...` and its `r31` tag were not
locally present and were not pullable from a public registry, so F10a is the
closest source-and-launch reproduction, not byte-identical OCI evidence.

COMMAND -> run
`I_KNOW_P2P_WEDGES=1
vllm/fp8/qualify_qwen38_fp8_neural_f10a_publisher_cgroup.sh` through the
whole-box `bin/gpu-run` lease. The qualifier ran two independent fresh servers,
the fixed 12-prompt suite, model and endpoint identity, canaries, teardown,
per-card health, and compiled TP2 collective health.

RESULT -> class-balanced first-100 rates were 17.716072 and 17.381759 tok/s;
their center was 17.548916 tok/s. This is 33.80 percent of Steve's 51.918757
tok/s result, or 66.20 percent slower, and only 1.01 percent above the earlier
F07a graph-off center. The local pair matched 9/12 complete token arrays and
each matched 7/12 against the publisher. Peak container RAM was 8.581 GiB.
Host swap did not grow above its pre-run value. Both servers tore down and all
card and compiled-collective checks passed; the kernel journal had no new Xe
fault signature.

RESULT -> evidence root is
`/mnt/vm_8tb/b70/results/f10a_qwen38_fp8_neural_publisher_cgroup/20260831T173435Z/`.
Summary SHA256 is `86bfdf34...`; attempt performance SHA256 values are
`13439b5d...` and `5317abc6...`.

VERDICT -> matching the published flags and cgroup does not put graph-off
decode within 10 percent on this host. The missing publisher OCI identity and
unpublished host CPU/runtime boundary remain material variables. Locally,
FULL graph capture is the measured intervention that removes the dominant
host-submission cost: the promoted MTP1 route is 46.603967 tok/s, 165.57
percent above F10a. Keep graph capture enabled for the local daily driver; do
not substitute the c64 service numbers for single-user or 262K evidence.
