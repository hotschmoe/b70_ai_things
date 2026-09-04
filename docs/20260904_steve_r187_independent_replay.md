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
| R187 MTP5 whole graph, `FULL_DECODE_ONLY`, sizes `[1,2,3,4,5,6]` | 72.245076 | pass | 12/12 vs R187 MTP1 graph-off; 9/12 vs R156 MTP1 graph-on |

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
single matched strict pair, not shelf qualification. The experimental R187
MTP5 graph arm is faster at c1 and matches the R187 MTP1 graph-off output, but
it is not a single-variable performance pair. Keep same-depth graph-off and
same-depth graph-on repeats, c2/c4, long context, cache-on, and a second fresh
matched pair open. Do not promote R187 or either graph variant to
`rdy_to_serve` yet.

## MTP5 XPU-graph follow-up

CONFIG -> Same source, image, weights, TP2 cards, 1K allocation, FP16 KV and
target verifier, draft-only INT4 head, and strict suite; R187 whole-graph
compile with MTP depth 5 and `FULL_DECODE_ONLY` capture sizes
`[1,2,3,4,5,6]`. The capture set follows Steve's R198 depth-3 precedent of
including each possible target-plus-draft row count.

COMMAND -> Run `PROFILE=mtp5-xpugraph-r187 bash
vllm/fp8/qualify_qwen38_fp8_steve_r187_strict.sh` under the two-card lease,
then compare its complete strict token arrays with the earlier R156 MTP1
XPU-graph attempt.

RESULT -> The server log confirms MTP5, empty `splitting_ops`, the exact six
capture sizes, and successful graph capture in 2 seconds using about 0.08 GiB
per card. The class-balanced median was `72.245076 tok/s`, `1.454329x` or
`45.433%` above the R156 MTP1 graph result, and `83.829%` of Steve's
`86.181722 tok/s` MTP5 graph-off center. The strict workload, cache-zero,
repeat/copy/arithmetic/JSON canaries, served-model identity, teardown, both
card checks, and compiled P2P-off post-collective health passed. Swap never
exceeded `646792 KiB`, and the kernel journal had no matching new Xe fault.

The new arm matched the same-image, same-whole-graph R187 MTP1 graph-off
reference 12/12 complete token arrays. It matched the requested R156 MTP1
graph-on comparison 9/12: divergence began at token 303 for `code-review`,
392 for `incident-retrospective`, and 127 for `risk-register`. Those are
exactly the three late divergences in the earlier R187 whole-graph versus R156
piecewise comparison, so the 9/12 cross-profile result is attributable to a
known split-policy confound rather than specifically to MTP5 graph replay.

VERDICT -> GO as a coherent experimental c1 speed arm; NO-GO for a shelf or
controlled speed claim until same-depth graph-off/on arms are run and
repeated. The direct speed comparison requested is valid as an observed
result, not as a controlled single-variable claim.

## Live long-context cache-on candidate

CONFIG -> Same R187 whole-graph MTP5 arithmetic path, FP16 KV and target
verifier, draft-only INT4 head, TP2 direct P2P, and XPU graph. Change the
serving envelope to 237,568 maximum context, four request slots, 32,768
maximum batched tokens, 0.96 GPU memory utilization, Qwen hybrid `align`
prefix caching, and capture sizes 1 through 24 for the four MTP5 decode
descriptors. Enable the `qwen3_coder` tool parser and `qwen3` reasoning parser;
keep the vLLM backend on `127.0.0.1:18124`, and expose an API-key frontdoor on
`0.0.0.0:18080` with the existing off-repository daily-driver key. Use an
explicit served ID.

COMMAND -> Start
`vllm/fp8/serve_qwen38_fp8_steve_r187_mtp5_daily.sh` inside a durable tmux
session. Keep the `bin/gpu-run` two-card lease for the entire server lifetime;
require image/model verification and pre-card/compiled-collective health.
Probe explicit model identity, a short chat completion, and a repeated prefix
longer than the 832-token hybrid alignment page. For the LAN change, drain the
loopback lifecycle, recover and recheck the cards if its TP2 trap fails, then
start a fresh lifecycle and require public health, unauthenticated rejection,
authenticated model identity, chat, and streaming gates through
`192.168.10.5:18080`.

RESULT -> The server allocated 290,188 aggregate KV tokens, enough for 1.22
full 237,568-token requests, and captured all four c1-c4 descriptors in 3
seconds using 0.12 GiB per card. `/v1/models` reports only
`qwen3.8-27b-FP8-official-W8A16-mtp5-r187-xpugraph-cacheon-ctx237568-daily`.
The chat smoke returned exactly `DAILY READY`. Two identical 3,098-token
prompts returned identical `CACHE LONG READY` output; the first reported zero
cached tokens and the second reported 1,664. Engine counters independently
reported 1,664 local prefix-cache-hit tokens. The earlier 128-token repeat
was below one 832-token align page and correctly produced no reusable block.

The controlled restart exposed one launcher live-edit edge case: the old Bash
process read the newly edited cleanup body with variables absent from its
resident pre-edit state. The model workers drained cleanly, but the lifecycle
returned 1. Fail-closed cleanup ran `xe-reset --method rebind`; both card and
compiled P2P-disabled collective post-health passed and both leases were
released. A fresh lifecycle then passed pre-health and reused the compiled
graph target. The authenticated frontdoor now listens on `0.0.0.0:18080`,
while Docker publishes vLLM only on `127.0.0.1:18124`. Direct LAN probes
returned HTTP 200 for health, HTTP 401 for unauthenticated `/v1/models`, and
HTTP 200 with the exact served ID when authenticated. Authenticated chat
returned exactly `LAN READY`, streaming reached `[DONE]`, and no API key is
present in process arguments. No matching Xe fault was present at handoff.

VERDICT -> READY for a quality test at
`http://192.168.10.5:18080/v1` using the existing daily-driver API key; not
qualified for speed, stability, or shelf promotion. The current lifecycle's
post-health remains pending until teardown. Stop through the tracked script so
the held lease performs teardown, kernel capture, and post-health.

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
- R187 MTP5 graph-on attempt:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_mtp5-xpugraph-r187/20260904T224821Z/`.
- Matched comparison SHA-256:
  `db8e4fc388961f9790e2feb6561afdb5ba4e1a87a4694c99372ba9a4659b3a47`.
- Unmatched R187-versus-R156 graph comparison SHA-256:
  `de0140d490a62221901c5d6f1967eab306a9ba24f838ee18f931b27777d6e4c5`.
- Unmatched R156 MTP1 graph-versus-R187 MTP5 graph comparison SHA-256:
  `e56ca1451a62d48a07d6050b0ef7b8e49d9fd3332226c7831841390291cd879c`.
- Same-R187 whole-graph MTP1 graph-off-versus-MTP5 graph comparison SHA-256:
  `1d77ae38a4c2be00337b904d0d922aa7d198429d03942b5e4d803db46ee3dbc1`.
- Live long-context readiness receipt:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_mtp5_daily_r187/20260904T230830Z/live-ready.json`,
  SHA-256 `741234e6d9671225d74d9bff63bc0af1cdd171fd5400338c5631a9065f25a595`.
- Controlled-restart recovery:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_mtp5_daily_r187/20260904T230830Z/recovery.log`,
  SHA-256 `bb0d3d6ed7dca056800384f3232211f5fac26d05b09f6c83fcdfa34b6e3c69d9`;
  card and collective post-health SHA-256
  `7b157e3555a8aa1d188b7377745332cbd9244e5d27c084713ecfa3124fae8718`
  and `2f3e94bca6d706216bd9c0059e2e835e5eda803e3c4db154fea86c7bcbe66d61`.
- Fresh secured LAN lifecycle:
  `/mnt/vm_8tb/b70/results/qwen38_fp8_steve_mtp5_daily_r187/20260904T232952Z/`;
  LAN receipt SHA-256
  `a2e12fbdbee37b90fd2f0f570abe0451d2eb302d68905027ae717b62740b2c0c`.

Primary publication sources:

- https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html
- https://neural.download/learn/host-tuning.html
- https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/README.md
