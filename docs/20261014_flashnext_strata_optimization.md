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

## S2: two200k, forced RAM restore, final serving rollout

CONFIG: same clean upstream binary, IQ3_XXS artifacts and runtime as S1.
Change split25 -> split27, max-context131072 -> 200000 per slot, and
parking8192 -> 16384 MiB (still four entries). MTP4/min-p0.5 remains
configured and its 950 MiB draft head resident. Upstream uses non-drafted
batch windows while requests overlap; this does not prove concurrent MTP
drafting or a decode-throughput gain. Every main expert remains in VRAM,
host mirror/wide-host KV off, direct NVMe PLE, reserve1024, prefill4096.
Final tracked config: `strata/flash-next/serve-iq3xxs-200k.json`; identical
runtime copy: `/mnt/vm_8tb/b70/build/strata-ab/serve-optimized.json`.

COMMAND: launch under the pair lease, send two simultaneous long chats
with distinct deployment codes at 10%, 50% and 95% of each prompt, then
exercise cached returns. The first lone third chat left the batch slots
intact, so that test alone is not a forced slot-eviction proof. Strengthen
it with the updated `bench_context.py --output RAW/context-200k-forced
--resume-from RAW/context-200k`: two simultaneous 4K chats with 1500-token
answers overwrite both batch slots; capture /slots, then return to both
long conversations concurrently. Run bench_serving.py again (three fresh
prompts per case) and the reasoning/tool probe into new S2 directories.
Sample HTTP metadata and per-card DRM fdinfo every five seconds during
the context tests. Normal stop, then leased per-card and compiled two-rank
health. Restart the same behavior config, validate live identity/context
and concurrent arithmetic, then start strata-soak.service at 60 seconds.

RESULT: two configured 200000 slots, both initial prompts199501 tokens,
all three needles correct. Initial client TTFT A/B111.42/302.98 s; engine
prefill1805.6/1058.5 tok/s. Prefills are serialized and the second prompt
was materially slower; no sustained 200k prefill-speed claim. Both long
contexts held in the slots (/slots199529/199504 tokens). Forced displacement
then proves /slots5521/5521, followed by both correct long-context returns:

| Return | Prompt tokens | Reused tokens | Newly read | RAM restore ms | Client TTFT s |
|---|---|---|---|---|---|
| A | 199609 | 199555 | 54 | 429.7 | 1.975 |
| B | 199609 | 199582 | 27 | 421.6 | 4.400 |

Engine log explicitly reports both conversation-cache restores. Client
TTFT also includes scheduling and prompt-tail work; batch admission copies
can delay subsequent tokens. Only the short tails were evaluated, so the
199k prefixes were restored without recompute. The two displacer answers
produce3000 tokens in54.51 s including prefill (55.0 output tok/s overall);
this is not a new matched two-stream throughput comparison.

139 samples: peak resident VRAM28.223/29.266 GiB, process RSS15.835 GiB,
container-accounted memory16.145 GiB; zero observed endpoint/native errors.
All24576 main experts resident. New config's timing medians: coding83.8
decode; 4K1337.2 prefill/64.5 decode; 32K2004.2 prefill/65.0 decode tok/s.
All nine timing prompts have zero reuse and all eleven responses match the
old a0116b27 run byte-for-byte. Separate reasoning and tool round trip pass.
Both cards and ten compiled all-reduces pass after normal teardown;
gpu-run exits0. No reset, model changes or local engine patches.

VERDICT: bounded two200k capacity and forced RAM restoration gates passed
with MTP4 configured; final short coherence checks also pass. This is a
configuration qualification, with no shelf entry or multi-day stability
claim. The 32K prefill median reaches2000 in this screen; 4K prefill and
decode remain below their targets. The changed split/context/parking are
tested together, so differences are not isolated causal speed gains.
IQ3_S, expert-parallel, draft-depth tuning and short-prompt tuning remain.

Final services: strata-optimized.service owns the leased strata-ab-serve
container; strata-soak.service observes it, without generation or restart.
URL http://192.168.10.5:18085/v1; stable name hotschmoe-dd first, IQ3_XXS
alias retained. Config, source, binary and image identity at final startup
are in `soak-start-identity.json`, with models/status and two arithmetic
responses beside it. Soak JSONL: `RAW/soak.jsonl`; native/server logs:
`RAW/soak-engine.log` and `RAW/soak-server.log`. Initial two observer samples
have no endpoint, native or server errors. Continued real use and waiting
are needed for a multi-day verdict. Stop observation with
`systemctl --user stop strata-soak.service`; stop serving with the Docker
stop command above. The retained 27B prerequisite check passed throughout;
the one-command fallback remains `bash strata/flash-next/fallback-27b.sh`.


## S3: expert parallelism, Docker IPC fix, prefill cost

CONFIG: same IQ3_XXS, image/compiler/runtime and two200k/MTP4/min-p0.5
settings as S2. Clean head296b59d2 remains untouched. Temporary detached
src-ep worktree applies PR2205's three-commit diff through
c7bb8dd5d5a8f90f2a7f1cf0d30c9307aaf25bba (includes PR1798 decode EP).
Build src-ep/build-sycl-ep/strata, oneAPI2026.1/AOT bmg-g31, pinned ggml
3cf03257f219afbe7334045ff7c6a06ac68c627d. Only topology changes:
--layer-split27 -> --expert-parallel-device1. Tracked experimental config:
strata/flash-next/serve-iq3xxs-ep-experiment.json. No shelf promotion.
Diagnostic mirror/row verification are enabled only in the separate
rowcheck config and disabled in speed measurements.

Raw root EPRAW:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_ep_mtp_20261014

COMMAND: fetch PR1798/2205 refs, git diff/apply --3way onto head in a
worktree, build CPU-only with the same image (no GPU device mount).
Leased single-card EP and original planner parity. Launch under pair lease,
with per-card and compiled collective health around EP serves. Run
bench_serving.py --repeats3 --output EPRAW/ep-mtp4-p05; separate reasoning
and tool round trip. Relaunch with STRATA_PREFILL_TIMING=1 and read one
fresh4K prompt. The native EP IPC probe is built from ep_ipc_probe.cpp
with -fsycl/-fsycl-targets=spir64_gen, AOT bmg-g31 and -lze_loader.
All native probes, serving and health are under bin/gpu-run.

RESULT: planner parity420 EP cases and641 original cases, zero failures.
Initial EP launch exits1 at IPC import with0x78000004, not a GPU crash;
post-refusal health passes. Standalone cold/warm cross-card and self-import
probes reproduce all eight failures under default Docker settings, despite
peer access=true both directions. SYS_PTRACE alone permits all eight
imports with default seccomp retained; seccomp-unconfined alone also works.
Local launcher adds the capability only for the EP flag and uses each
config's cwd so the matching PR frontend runs. No driver/kernel changes.
Exact blocked syscall has not been traced.

Fixed full-model EP places190 experts/layer on card0 (9120,14.83 GiB) and
322/layer on card1 (15456,25.13 GiB); all24576 experts resident. Two200k
slot sessions allocate2.82 GiB each on the primary and the MTP head remains
resident. This establishes startup fit, not an actual near-full EP context
and parking qualification. With diagnostic mirror,768 generated tokens
and zero peer-row mismatches. Concurrent arithmetic, reasoning and tool
round trip pass. Timing medians with diagnostics off:

| Workload | Prefill tok/s | Decode tok/s |
|---|---|---|
| coding,105 input /1500 output | 112.5 | 91.9 |
| 4K,4059 input /256 output | 578.8 | 72.3 |
| 32K,32728 input /256 output | 578.4 | 71.0 |

Every timing prompt has zero reuse. Nine output hashes are preserved;
coding repeats0/1 and all prose responses match the S2 control. Coding
repeat2 differs late from S2 but matches all new layer/MTP controls too:
no universal bit-identical claim across runs. All EP model serves stop
normally; per-card and ten compiled all-reduces pass after each teardown.

Prefill trace:4053 chunk tokens,7007 ms GPU timeline and7028 ms wall;
combine3698 ms (52.8%), gather796 ms (11.4%), QSA attention670 ms (9.6%).
Peer timeline has1205 ms GEMM and5581 ms idle. The combine bucket includes
waiting for the peer and collecting its rows, so it is not an isolated
copy-bandwidth measurement. It identifies the next optimization area.

A separate ep_transfer_probe.cpp compares uncached reads, ordinary memcpy
and a copy-engine queue with changing full-buffer patterns. All twelve
16KiB checks pass, then UR_RESULT_ERROR_DEVICE_LOST and exit139 occur
before the large-transfer timing; kernel records card0 CCS memory CAT
errors and an engine reset. This probe is INCONCLUSIVE and not a serving
patch. Do not rerun it until its memory/IPC lifetime and copy path are
reviewed. Retain source/logs, discard any speed inference. bin/xe-reset
recovers by rebind without reboot; both cards and ten compiled all-reduces
pass, boot ID unchanged. Source, hashes, configs, build logs, startup/
teardown logs and kernel/recovery evidence are in EPRAW.

VERDICT: EP gives a useful single-stream decode screen (91.9 coding,
71-72 prose) but loses most prefill throughput versus S2's1337/2004.
Keep the qualified layer-split daily driver while investigating peer-row
collection. The tested conditional Docker capability fix is submitted
upstream as https://github.com/Niko1221/Strata/pull/2468, commit
f85b6c3f191ab093310e4ae09b3faca428e70f44; only wrapper/docs changed.
It is independent preparation for the open EP PRs. No native optimization
is promoted from the failed microprobe, and no long-lived engine fork.
The long-soak observer is stopped at user request; not a stability verdict.
27B prerequisite check passes; fallback command remains available.


## S4: MTP windows3/4, confidence cutoff, concurrent output

CONFIG: same clean layer-split head and temporary upstream EP build as S3,
IQ3_XXS, two200k, int8 KV, reserve1024, prefill4096, parking16384 MiB/4.
Vary only --spec and per-request strata_tune.spec_min_p, except switching
between the recorded layer/EP topologies. Strata --spec T is the verification
window INCLUDING the seed token: T=4 offers at most three MTP draft tokens;
T=3 offers at most two. This matters when comparing other engines' settings.
No further deeper-window work after the user steered toward3/4.

COMMAND: bench_serving.py --cases coding p4096 --repeats3; --spec-min-p
0.3/0.5/0.7 on window4. Quiet window4/min-p0.5 also runs p32768. The new
bench_concurrency.py --template EPRAW/layer-mtp4-p05-quiet/coding-0.json
--output EPRAW/<arm>-concurrent sends two synchronized107-token coding
requests,1500 output each, in three fresh windows. Repeat for layer T4,
layer T3 and EP T3. Save live identity, request/output hashes, raw SSE,
TTFT, engine timings and health; reject prefix reuse over64 tokens.
Per-card health between layer variants; per-card plus compiled two-rank
health before and after EP. Exact scripts/configs in EPRAW/run-depth34.py,
run-ep3.py and serve-*.json. Roll back to the qualified layer T4 config,
then validate stable identity, concurrent arithmetic, separate reasoning,
tool round trip and one memory/HTTP snapshot. Observer remains stopped.

RESULT: quiet single-stream medians, three prompts per cell; all prompts
have zero reuse. Coding105 input/1500 output; prose4059/256:

| Topology | Verify window | min-p | Coding decode | Prose decode | Draft acceptance coding/prose |
|---|---|---|---|---|---|
| layer | 3 | 0.5 | 77.7 | 64.7 | 87.2% /76.2% |
| layer | 4 | 0.3 | 84.0 | 64.0 | 77.5% /56.6% |
| layer | 4 | 0.5 | 83.8 | 65.7 | 83.6% /67.6% |
| layer | 4 | 0.7 | 81.6 | 64.5 | 90.4% /81.8% |
| EP | 3 | 0.5 | 85.5 | 71.1 | timing screen; tool gate below |
| EP | 4 | 0.5 | 91.9 | 72.3 | S3 screen,83.0% /67.5% |

Units: tok/s. Higher acceptance alone does not maximize throughput. Window4
wins the layer coding screen by7.9% versus3; prose differences are small.
Cutoff0.3 gives no broad win,0.7 slows coding. Retain T4/min-p0.5.
Quiet control prefill1260.3 at4K and1952.2 at32K;32K decode64.9. The
previous same-behavior S2 medians were1337.2/2004.2, so2000+ prefill is
near the boundary, not established as a stable floor.

Earlier deeper screens (overlapping the CPU-only EP build) recorded
T4/0.5 coding82.8/prose65.8, T6/0.5 84.6/62.4, T8/0.5 79.3/59.0.
Acceptance falls83.3/67.6 ->74.4/59.4 ->68.0/54.1 percent. Quiet T8/0.7
recovers83.0/65.2, without beating the balanced T4 default. These are
exploratory screens; the final3/4 comparison runs without the build.

| Concurrent arm | Median total output tok/s | Engine per-stream range tok/s |
|---|---|---|
| layer T3/min-p0.5 | 63.56 | 32.4-33.8 |
| layer T4/min-p0.5 | 63.94 | 32.6-33.4 |
| EP T3/min-p0.5 | 69.99 | 35.8-37.1 |

Total output divides3000 tokens by wall time INCLUDING both prefills,
admission and tails; it is not the sum of single-stream medians. Each arm
has three windows of two requests, six outputs. All six request
hashes and outputs match across the layer3/4 and EP3 arms. Every overlap
response reports draft_n=0: these runs use ordinary batch decode, so the
window3/4 choice does not provide concurrent drafting. Upstream's opt-in
--batch-mtp is disabled by default and rejects layer splits. EP's combination
with that option has NOT been tested here. EP total is10.1% above layer3,
but still below one lone MTP-assisted coding stream. No parallel speed
qualification or150 tok/s claim.

Functional caveat: EP T3 passes concurrent arithmetic and separated
reasoning. A new forced-tool round trip emits the correct tool name/args,
but its final answer is only "Here's the build status for **b70-demo**:"
and omits17 passed/0 failed counts. An exact repeat does the same. The
IDENTICAL request body on restored layer T4 completes with the correct
counts and full answer. Explicitly requesting one sentence completes with
correct counts on both. This is an EP/T3 qualification failure, not evidence
that either EP alone or T3 alone caused it: the other two arms of that
functional2x2 remain untested. Retain the failed check-functional-v1.py,
run-ep3.log and response JSON; do not reinterpret the failed gate as a pass.
S3's separate EP T4 tool probe did pass, but used a different request.

EP T3 stops normally (gpu-run exit0); both cards and ten compiled all-reduces
pass. Its one snapshot has primary/peer resident29.83/25.76 GiB and RSS
2.94 GiB; this is startup/short-prompt evidence, not a near-full EP context
or long soak qualification. Final layer T4 restart has stable identity,
ctx200000, concurrent323/667, separated391 reasoning and a complete tool
round trip. Final snapshot has no endpoint errors; source clean296b59d2,
binary16f592e7e2f20178e4775ede7e6de306b6695b9444d1e399f68844cffb1e130b,
image unchanged, no added capabilities. Final-identity.json records these.

VERDICT: retain the qualified split27/T4/min-p0.5/two200k configuration,
which is at most three MTP draft tokens. More drafts or higher acceptance
alone do not help. Concurrent MTP and peer-row prefill collection are the
next engineering targets; EP/T3 also needs the failed tool continuation
isolated before promotion. No native patch, new shelf entry or renewed
long-soak observer. Final strata-optimized.service serves :18085/v1 with
hotschmoe-dd first. Tracked config and /build/strata-ab/serve-optimized.json
match; only log destination changes to EPRAW/final-layer-engine.log.
Final HTTP/memory evidence: final-functional/, final-memory.jsonl and
final-identity.json. 27B prerequisite check passes; fallback remains
bash strata/flash-next/fallback-27b.sh. Complete wrapper switch untested.


## S5: peer transfer bandwidth and four-conversation RAM budget

CONFIG: upstream head remains296b59d2; same image/oneAPI/UMD/kernel and
PR2205 control. No production setting changes. New raw root PFRAW:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_ep_prefill_20261014.
Preserve the S4 EP binary as src-ep/build-sycl-ep/strata-s4-control.
The transfer probe now allocates full-sized native source AND destination
once and keeps them alive across sizes/modes; it no longer replaces a
small IPC mapping with a larger one or changes the destination allocator.
These are two fixture changes, not an isolated proof of the S3 crash cause.

COMMAND: CPU-only same-image AOT build; stop daily driver normally; leased
per-card and compiled collective health; pair-leased fixed transfer probe;
post per-card and compiled health; restore clean layer split for CPU builds.
Compare uncached kernel, ordinary queue memcpy and copy-engine memcpy on
16KiB and256235520 bytes (25023 expert rows x2560 floats), four producer
updates each, full returned-buffer pattern check every time. Timings below
exclude the first epoch and are medians of the remaining three. Exact
source, build/runtime logs and hashes are in PFRAW; old failed probe/logs
remain preserved in the S3 raw root and git history.

RESULT: all24 pattern checks pass, normal exit0, no device-lost error.
For the large peer-read buffer: uncached kernel77.231 ms/3.318 GB/s;
ordinary memcpy144.676 ms/1.771 GB/s; copy-engine155.406 ms/1.649 GB/s.
These checks validate returned bytes; they do not separately prove a
subsequent cached GPU consumer's coherence for a proposed DMA path.
Both cards and ten compiled all-reduces pass before and after the probe.
The daily driver resumes healthy while the next native build runs.

The earlier two200k test logged a199502-token snapshot of3883090536 bytes
(3.616 GiB). Four similar snapshots are14.466 GiB; five during an exchange
would be18.082 GiB. ConversationCache::make_room counts the incoming image
held during exchange, and bytes() also counts retained K/V. Therefore16GiB
is tight for four DISTINCT full conversations with predictable swaps.
Two active GPU streams plus two parked conversations is technically
feasible; use a24-32GiB RAM budget for a four-conversation qualification,
with simultaneous generation still limited to two. Current production
parking remains16GiB/4 entries. No four-distinct-200k load test is claimed.
Budget evidence: PFRAW/four-conversation-budget.json and the S2 cache logs.

VERDICT: reject DMA as a prefill speed optimization on this host. Returning
individual peer expert rows consumes hundreds of MiB per layer and is the
measured bottleneck. Next test an opt-in peer reduction to one FP32 row per
token, with explicit numerical-error and model checks: regrouping changes
FP32 addition order. No native serving change is promoted by this stage,
no stability claim, no soak observer. Fallback27B prerequisites pass.


## S6: opt-in FP32 peer reduction and8192 prefill chunks

CONFIG: clean upstream296b59d2 + PR2205 c7bb8dd5 diff, same IQ3_XXS,
image/oneAPI/runtime/host, spec4/min-p0.5, two200k configured, int8 KV,
reserve1024, parking16384 MiB/4.94-line native reduction patch (three
files) adds STRATA_EP_PREFILL_REDUCE=1; default off preserves individual
rows. The peer returns one FP32 weighted sum per token instead of every
expert row. Primary adds its own routed sum and shared expert, in the
existing output buffer. Decode is unchanged. FP32 addition order changes.
Control and candidate BOTH use STRATA_PREFILL_TIMING=1. Frozen binaries:
strata-s4-control and strata-reduce-v1 under src-ep/build-sycl-ep.

COMMAND: same-image AOT CPU build; independent pair-leased numerical
probe against original router-order std::fma CPU reference: tokens1/17/256,
all-primary/all-peer/mixed ownership, two updates each; full4096 mixed
case twice (26 cases total). Save numerical max error. After health, three
fresh coding/4K/32K requests per control/candidate; concurrent arithmetic,
separate reasoning, forced-tool round trip plus the EXACT original tool
continuation that previously failed on EP/spec3. Native per-card/compiled
health after normal teardown. Then change ONLY prefill4096 ->8192 using
same candidate binary; three4K/32K repeats and functional checks, normal
stop and post-health. All native work remains under the GPU lease.

RESULT:26 numerical cases pass. Max absolute error9.53674316e-7 versus
original router-order FP32 sum; maximum error/tolerance0.0867. Tolerance
is2e-6 times(1 + contribution magnitude + shared contribution). Repeated
IPC updates and the final cached GPU consumer are exercised. No byte-exact
math or complete model-quality equivalence claim. Full-model medians:

| Arm | 4K prefill | 32K prefill | Coding decode | Prose decode4K/32K |
|---|---|---|---|---|
| original EP,chunk4096 | 569.7 | 575.6 | 91.9 | 72.3 /71.0 |
| reduced FP32,chunk4096 | 1037.1 | 1106.0 | 92.7 | 74.0 /70.3 |
| reduced FP32,chunk8192 | 1056.2 | 1265.4 | not measured | 72.0 /69.4 |

Units tok/s, three fresh prompts per cell, zero timing-prefix reuse.
Changed output text can alter decode timing; no causal decode-gain claim.
Initial prefixes agree, but later responses differ as expected from changed
FP32 summation. Concurrent323/667, separated391 reasoning and tool
name/arguments/results pass on control and both candidates. The original
EP/spec3 tool continuation completes correctly on ORIGINAL EP/spec4 AND
reduced EP/spec4. The remaining layer/spec3 arm is untested; do not blame
EP in general for that earlier functional failure.

At4K, combine falls from about3698 to622 ms; the reduced FP32 return is
about40MiB instead of roughly244MiB per layer. At8192, primary7488 plus
peer17088 = all24576 experts resident; peer cache27.79GiB and native
prefill reports2975MiB remaining after its extra buffers. Both slot0
checks pass. EP's /metrics expert_slots reports PRIMARY only (7488);
using it as a total caused one fixture assertion to stop BEFORE timing.
Native filled counts prove total residency; then the benchmark resumed.
No GPU crash or reset. This is configured capacity/short-prompt evidence,
not actual full200k candidate-context qualification.

Bulk-copy follow-up: same fixed allocations,72 changing-pattern checks
pass over16KiB/244MiB, comparing scalar64/256 workgroups and vector4/8/16
loads at64/256/1024 workgroups. Warm large GB/s: original scalar64 3.352,
scalar256 2.350; vector4 about1.64-1.65, vector8 1.647, vector16 1.938.
All wider options lose; no native wide-copy patch. Full fixture and logs
preserve the negative result. Per-card and ten compiled collectives pass
after each model arm and after this probe; clean stops exit0.

VERDICT: FP32 reduction nearly doubles matched32K prefill (575.6 ->1106),
and8192 reaches1265, still below2000. Ship as an opt-in research change,
not a daily-driver/shelf promotion or a stability claim. Next isolate
host-side peer submission, visible in the gather bucket; preserve math.
Native fix plus numerical probe/manual build docs is submitted as
https://github.com/recutita/Strata/pull/1, commit0897c76c, against the
AUTHOR BRANCH of upstream Niko1221/Strata#2205 to keep the diff small.
It is a contribution to that pending upstream feature, not a main merge.
Tracked patch: strata/flash-next/patches/ep-prefill-reduce.patch; frozen
experimental config: serve-iq3xxs-ep-reduce-experiment.json. PFRAW retains
configs, hashes, builds, profiles, raw SSE, all functional results and
health logs. The qualified clean layer-split driver resumes for CPU builds;
soak stays off and27B fallback prerequisites remain passed.
