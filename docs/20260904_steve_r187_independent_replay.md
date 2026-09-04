# Steve Qwen3.8 FP8 R187 independent replay

Date: 2026-09-04 UTC

## Outcome

CONFIG -> Fresh full-history `b70-optimization-lab` clone at
`8319e0964df12a1f0bc920301efc662ac49a949e`; Qwen3.8-27B official FP8
revision `017b9c7a...`; locally rebuilt R156 image; TP2 on the two local B70s;
1K allocation; FP16 KV and target verifier; draft-only INT4 head; prefix cache
off; strict natural-512 suite; empty compile cache per server.

COMMAND -> Run the public source and remote-asset validators; build the no-
compiler R13 -> R15 -> R31 -> R49 -> R55C -> R62 -> R139 -> R156 chain;
verify model and image identities; run the host submission and exact-shape
P2P-on collective probes; then run leased R187 whole-graph, matched R156
piecewise graph-off, and published `FULL_DECODE_ONLY` sizes `[1,2]` XPU-graph
attempts through `vllm/fp8/qualify_qwen38_fp8_steve_r187_strict.sh` with
pre/post card and compiled P2P-off collective health.

RESULT -> Public source closure now passes (`161` tracked files), remote
release validation passes, the corrected `40ca8c3f...` patch and released
libraries resolve, and every build/content contract passes. The local R156
image ID is `sha256:f46780e1...`; its installed `_xpu_C.abi3.so` is
`f912e12d...` and `_xpu_ops.py` is `6a776193...`. All 66 weight files total
`30,866,866,928` bytes and have aggregate manifest `82fb8f84...`.

The host probe measured `9.2399 us` async launch, `52.1136 us` launch plus
sync, and `200.1721 us` RMSNorm M=2. The exact P2P-on collective census was
repeat-exact and prefix-invariant at every tested row count; two-row
all-reduce was `74.6 us`. This is slower than both neural.download reference
hosts and predicts a graph-off miss.

| Local arm | Strict tok/s | Workload/cache/canary | Output comparison |
| --- | ---: | --- | --- |
| R187 whole graph, XPU graph off | 17.917452 | pass | 9/12 vs R156 graph-on; three late divergences |
| R156 piecewise, XPU graph off | 16.845797 | pass | 12/12 vs matched R156 graph-on |
| R156 `FULL_DECODE_ONLY`, sizes `[1,2]` | 49.675873 | pass | 12/12 vs matched R156 graph-off |

The matched graph-on arm is `2.948858x` the R156 graph-off rate and reaches
`90.976%` of Steve's `54.603244 tok/s` R156 MTP1 center. It also reaches
`95.439%` of the published slow-host `52.05 tok/s` full-decode graph result.
Both servers use the same R156 image and arithmetic overlays; only graph
capture changes. The R187 comparison also changes the Inductor split policy,
so its 9/12 result is not evidence against matched graph replay identity.

Every server returned its explicit model ID through `/v1/models`, retained
zero cached prompt tokens, shut down normally, and passed post-teardown card
and compiled two-rank health. Swap stayed at `646792 KiB`; no new matching Xe
fault event was found. No reset or reboot was needed.

VERDICT -> The corrected public no-compiler recipe is independently runnable
on this machine. Select the documented sizes `[1,2]` XPU-graph variant for
local MTP1 c1 use; graph-off is submission-bound on this host. This is a
single matched strict pair, not shelf qualification. Keep MTP2-MTP5, c2/c4,
long context, cache-on, and a second fresh matched pair open. Do not promote
R187 or the graph variant to `rdy_to_serve` yet.

## Exact identities

- Host: Ubuntu 26.04.1, kernel `7.1.0-070100-generic`, Threadripper 1950X.
- Host UMD: `intel-opencl-icd 26.22.38646.4-0`, `libze1 1.28.2-2`.
- Container: PyTorch `2.13.0+xpu`, vLLM
  `0.27.2rc1.dev77+gac7509e2b`, UMD `26.27.39122.11-0`, Level Zero `1.32.0`.
- oneCCL library SHA-256: `733980ab6a6eb15d2d3da0649b92052c64a9597ced48fe9188434face5298b35`.
- R55C image: `sha256:d6d817cd643f239aeb3dde6b8833acf1e2f0764e020051fef2b1278d3fe6c5e4`.
- R62 image: `sha256:8c0f0a68387000b85466ff86836deb5c98ce03dffe5e427f7d1632d795c666e0`.
- R139 image: `sha256:2bc804693d497f2dcc31dcef4644cf383d85e718fbe1f78b558589248abfffaa`.
- R156 image: `sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152`.

The guide's optional GHCR digest
`sha256:173660ec18c6e98a14b9a4f573922abe9d3414999056f07ab5c3c14b55d6ceb0`
returned `unauthorized` to an anonymous pull on this host. This did not block
the newly corrected release-binary route.

## Evidence

- Source closure log:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_r187_20260904/source-closure.log`, SHA-256
  `321214c493631fa5d99b47e990be8f9d4bead51db4dda83726bd0442c8ccfdc6`.
- Host probes:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_r187_20260904/host-probes/`.
- R187 attempt:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_r187/20260904T214127Z/`.
- R156 graph-off attempt:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_r156/20260904T220842Z/`.
- R156 graph-on attempt:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_xpugraph-r156/20260904T215712Z/`.
- Matched comparison SHA-256:
  `db8e4fc388961f9790e2feb6561afdb5ba4e1a87a4694c99372ba9a4659b3a47`.
- Unmatched R187-versus-R156 graph comparison SHA-256:
  `de0140d490a62221901c5d6f1967eab306a9ba24f838ee18f931b27777d6e4c5`.

Primary publication sources:

- https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html
- https://neural.download/learn/host-tuning.html
- https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/README.md
