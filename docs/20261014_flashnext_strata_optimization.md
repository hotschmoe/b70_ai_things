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


## S7 -- exact workspace and chunking; asynchronous submission rejected

CONFIG: same head296b59d2 + PR2205, IQ3_XXS, all24576 experts resident,
two200k/int8KV/spec4/min-p0.5/reserve1024/parking16GiB4. Frozen FP32
reduction v1; phase timing on in all arms. Separate v2 CPU-submission
prototype has identical math and correct thread-local device context.
COMMAND: three fresh matched prompts per arm, stable model/registry,
concurrent arithmetic, reasoning and tools; normal stop and per-card/ten
compiled collective checks between arms. Benchmark fixtures and raw results
are under PFRAW from S5. CPU builds run with no GPU device mounted.
RESULT: asynchronous peer submission is slower in the same v2 binary:
4K1046.8 ->1002.3 tok/s; 32K1261.2 ->1242.7. Client TTFT rises from
3.901 to4.076 s and26.034 to26.418 s. Rejected; source restored to v1.
With v1, STRATA_OWNED_PRICE=exact and chunk8192 reaches1267.6 at32K;
chunk12288 reaches1313.4 (median client TTFT24.999 s; native prompt
24934.95 ms). Three fresh prompts, decode70.0 tok/s. Primary155 plus
peer357 experts per layer =24576; peer17136 experts/27.87GiB, peer free
2370MiB and primary free1936MiB. Exact workspace5608MiB versus coarse
8320MiB. V1 attention at8192 is1268.5, no material gain. Functional and
normal teardown/health gates pass for all arms; no crash/reset.
VERDICT: retain exact-workspace/chunk12288 as the next experimental
control; do not promote without full-context/concurrency qualification.
Still below2000. Profiling phase buckets include queue idle/waits and must
not be described as individual kernel latency. New user direction requires
operation duration, enqueue/submit delay, host wait, bytes and direction.
Reviewed retained vLLM PUSH source118/119 and docs/P2P_GPU.md; reuse the
posted-write source idea deliberately, never its ABI-specific binary.
Reviewed neural.download shared endpoint harness (temperature0, fresh
matched shapes, repeats and concurrent answers) and Steve's retained
notes/2026-08-10-vdr2-unitrace-profile.md. Keep profiler timings diagnostic
and distinguish filtered shares from whole-request time. Intel unitrace
supports kernel timing/submission intervals plus host API and device traces:
https://github.com/intel/pti-gpu/tree/master/tools/unitrace.
The next stage tests PUSH transport with repeated freshness checks; no
PUSH full-model claim is made in this stage. Qualified layer driver is
running during CPU builds; observer off; fallback one command away.


## S8 -- topology-aware PUSH transport measured independently

CONFIG: fixed full-sized native Level Zero allocations, source owned by GPU1,
destination owned by GPU0 and imported on GPU1. Same compiler/image/AOT,
no vLLM library loaded. Add vector4 posted writes with64 workgroups and a
system release fence after each work item's stores; GPU0 reads destination
uncached into a separate local buffer before checking. Allocations stay alive.
COMMAND: leased ep-transfer-push-probe after clean serving stop and per-card/
compiled collective health. Two sizes, ten methods, four producer epochs each;
all returned words checked. Post-probe per-card/ten compiled collectives pass,
normal probe exit0, then qualified layer driver restarted for CPU builds.
RESULT:80 complete-buffer freshness checks pass. Warm medians (epochs2-4):

| Operation, GPU1 -> GPU0 | Bytes | Host enqueue-through-completion ms | GB/s |
|---|---|---|---|
| primary uncached peer read | 256235520 | 77.249 | 3.317 |
| ordinary peer memcpy | 256235520 | 144.615 | 1.772 |
| copy-engine peer memcpy | 256235520 | 155.428 | 1.649 |
| peer posted-write PUSH | 256235520 | 22.837 | 11.220 |
| primary uncached peer read | 16384 | 0.017 | 0.969 |
| peer posted-write PUSH | 16384 | 0.011 | 1.472 |

These are host wall durations INCLUDING queue submission and wait, not
isolated kernel execution. PUSH timing excludes the later local uncached
consumer copy used for verification. Full-model comparison must include
that copy and synchronization. Retained vLLM source118/119 and P2P_GPU
J/K describe the same fabric preference; its historical all-reduce timings
are context, not a current Strata all-reduce result. Strata needs only the
peer expert contribution return, not a bidirectional TP all-reduce.
VERDICT: PUSH is a measured transport candidate. Prepare optional
STRATA_EP_PREFILL_PUSH=1 on top of FP32 reduction, changing ownership of
the sum region and applying a system release fence. Preserve uncached
primary consumption and existing host rendezvous. It is not qualified in
the full model at this checkpoint. Individual-kernel/API/submission traces
will follow; Intel pti-gpu6c0d6b0b was built CPU-only into PF build root.
Runtime/compiler/driver unchanged; source idea deliberately ported.

S7 latency erratum: the initial latency extraction selected repeats1/2,
instead of all0/1/2. Throughput medians were already correct (harness uses
all three). Correct three-repeat medians: async off/on TTFT3.910/4.082s
at4K,26.014/26.401s at32K; exact12288 TTFT24.983s and native prefill
24919.1ms. Exact8192 TTFT25.882s, attention-v1 TTFT25.864s. Preserve the
original checkpoint as history and use these corrected latencies onward.


## S9 -- PUSH raises EP prefill to1531, exact transport A/B

CONFIG: same109-line opt-in peer-reduction patch on upstream296b59d2 plus
PR2205. Frozen strata-push-s9 SHA256932d2f347219edd668f60048204b6578a823df03c213a45edf043861b823184e.
Same binary, PUSH0/1 only: IQ3_XXS, all24576 experts resident, spec4/.5,
two200k/int8KV/reserve1024/parking16GiB4, exact workspace/prefill12288,
phase timing on in both arms. Sum region changes owner; producer64-group
FP32 reducer stores directly to the primary IPC mapping/system release
fence. Existing peer host wait and primary uncached local copy remain.
No math or decode-path change relative to FP32 reduction.
COMMAND: same-image CPU main-target build completes with identical live/
frozen hashes. Initial all-target build was stopped AFTER main linked,
while compiling unused parity executables; no failed GPU compilation.
Pair-leased52-case oracle (26 PUSH plus26 pull); native slot0/total expert
checks; three matched fresh105/4059/32728-token inputs,1500/256/256 output;
compare requests, text hashes and draft counts. Concurrent arithmetic,
separate reasoning/tools; normal stop and per-card/ten compiled collectives
pass on both arms. No GPU crash/reset. Native source PR updated below.
RESULT: all52 oracle cases pass; max original-order reference error9.54e-7,
max tolerance ratio0.0913. All nine matched model response pairs exactly
match (including identical MTP offered/accepted counts). Three-run medians:

| Case | Pull prefill tok/s | PUSH prefill tok/s | Native prefill ms, pull -> PUSH | Client TTFT s, pull -> PUSH | Decode tok/s, pull -> PUSH |
|---|---|---|---|---|---|
| short coding | 100.8 | 103.5 | 1042.0 ->1014.2 | 1.068 ->1.040 | 88.9 ->88.9 |
| 4K | 1047.1 | 1214.4 | 3876.4 ->3342.4 | 3.909 ->3.375 | 70.7 ->70.6 |
| 32K | 1308.4 | 1530.9 | 25013.0 ->21378.6 | 25.077 ->21.444 | 70.0 ->69.9 |

4K/32K throughput gains16.0%/17.0%; native latency falls13.8%/14.5%.
The first32K run reaches1557.2, but1530.9 is the three-run median.
Coding decode unchanged in this matched comparison. Context here is
configured capacity, NOT actual two-full200k qualification of the patch.
Native filled7440 primary +17136 peer, both slot0 verified; peer2490MiB
free after prefill buffers. Prefix cache zero in every timed request.
The original FP32 reduction still differs from full-row combine; only PUSH
versus reduced pull has measured response parity. No overall quality or
long-stability claim. Full source/commands/configs and push-s9-report.json
remain in PFRAW. The patch and standalone probe are tracked in strata.
VERDICT: retain optional PUSH atop optional reduction; below2000, no shelf
or daily-driver promotion. Update stacked author-branch PR
https://github.com/recutita/Strata/pull/1 with commit78cea7a1, preserving
small source changes and no long-lived serving fork. Keep the clean main
layer driver as qualified daily configuration,27B check passes, soak off.
Next stage collects actual kernel/API/submission latency, since phase
buckets include idle/waits. First profiler collection produced an empty
file: native successful serve shutdown uses std::_Exit(0), bypassing library
flush; model teardown/health passed. Preserve this failed diagnostic and
explicitly stop/flush the session in the next capture before shutting down.


## S10 -- individual operation latency and a complete EP decode window

CONFIG: frozen PUSH-s9 binary/runtime unchanged; pull/PUSH both have timing
on and same IQ3_XXS/spec4/.5/two200k/exact12288/all24576. Intel unitrace
2.4.0 source6c0d6b0b; binary00fc84e5, library66ec9ee2. No driver/runtime
upgrade. Leased server wraps native executable with -d -h -s plus Chrome
kernel/API logging, collection paused for load/warmup. Stable/research IDs,
Docker actual config path, native primary7440/peer17136 and slot0 checked.
COMMAND: resume -> fresh4051-input/32-output request -> pause, then a
45-input/128-output decode request. Explicit stop, post-stop arithmetic
canary to make native observe stop/flush, JSON validation, normal teardown,
per-card/ten compiled collectives. Every native touch stays under its pair
lease. Raw records, unitrace kernel-submission report, configs/wrappers,
trace windows and per-operation CSV/JSON are retained under PFRAW.

Collection repairs: first file0 bytes because successful native serve
shutdown uses std::_Exit(0), bypassing library destructors. Stop/flush is
required BEFORE shutting down. A second orchestration attempt reached a
stale server while its intended unit was still waiting for the lease; its
session controls reported already stopped. Discarded and fixed readiness
to match actual container config/native counts; all teardown/health clean.
Never treat either failed collection as a benchmark or a valid trace.
The current pti-gpu UniTimer calibrates epoch against CLOCK_BOOTTIME but
adds CLOCK_MONOTONIC_RAW timestamps. Measured trace/client offset about
9.70 seconds; record host clock calibration in trace-timebase.json and pass
--timestamp-shift-us explicitly to summarize_trace.py. This changes event
alignment, never durations. Later calibration can drift by milliseconds;
main-prefill census (48 reducers/consumers,1512 attention chunks) is checked
independently of boundary events. Per-query shapes distinguish the main
4044-token prefill from the seven-token tail and returned decode tokens.

RESULT: both155MiB traces contain about1.03million events and valid JSON;
all48 full-chunk reducers/consumers are present per arm. Median/p95 GPU
operation durations below (microseconds; exact counts in exported CSV):

| Operation | Card | Count | Pull median/p95 us | PUSH median/p95 us |
|---|---|---|---|---|
| routed peer FP32 reduction / reduction+posted write | 1 | 48 | 640.62 /700.49 | 4227.97 /4271.77 |
| FP32 sum consumer: remote read / local uncached read | 0 | 48 | 11077.66 /11154.42 | 306.61 /319.58 |
| primary routed+shared finish | 0 | 48 | 499.37 /627.71 | 499.06 /627.91 |
| inbound FP16 local uncached read | 1 | 48 | 150.78 /188.75 | 150.62 /203.04 |
| sparse attention chunk kernel | 0 | 1512 | 441.14 /494.34 | 441.04 /494.67 |
| GDN recurrence | 0 | 36 | 6664.27 /6936.25 | 6663.96 /6665.99 |
| peer expert dequantization | 1 | 11724 | 21.88 /32.19 | 21.77 /32.81 |
| primary expert dequantization | 0 | 5300 | 22.50 /32.71 | 22.50 /32.50 |

Main-chunk return41410560 bytes per layer (4044*2560*4),1.99GB across48;
inbound FP16 data20705280 bytes/layer in the opposite direction. The two
return operations' serial service costs are about11.7ms pull versus4.5ms
PUSH, though primary work can overlap reduction and kernel sums are NOT
request critical-path time. Physical trace devices:0 at0000:0b:00.0,
1 at0000:44:00.0. Sparse attention totals about612ms, GDN241ms; peer and
primary dequantization249/117ms. All operation shapes, counts, totals,
means, medians, p95 and ranges are exported, with per-device interval union
instead of adding overlapping times from the two cards.

Host API latency is different from device time:85593 launch-with-arguments
calls, median5.56/5.47us and p957.76/7.93us, summed510/513ms in the two
traces. A161760-byte device-to-host routing-table copy takes about28-32us
on the GPU while its append API call can wait about65ms for earlier work.
The copy API's1627ms total in PUSH includes queue dependencies. It does not
prove slow transfer bandwidth. Tracing roughly doubles4K wall time, so
native uninstrumented speed remains S9's matched1214/1531 figures; do not
use profiler wall/CPU totals as production latency estimates.

Graph coverage repair: capturing graphs while collection is paused hides
replayed kernels (first decode subset had no peer events and only69ms of
primary activity). Recollect PUSH with tracing ON through native load and
64-output graph warmup, then pause/resume the measured request. Complete
411MiB trace,269027 GPU events in its request window; same output hash and
108 offered/75 accepted drafts as the paused-capture request. Profiler
slows native decode73.4 ->58.4 tok/s, so again diagnostic only. A middle
steady window has40 routed entries (four verify tokens): each card48 EP
waits; primary48 sends, peer48 returns. Its bump-to-bump intervals43.708/
43.714ms include host gaps, draft/verification scheduling and instrumentation,
not a per-token graph execution latency. Peer waits19.601ms, primary waits
3.516ms; peer row-return kernels1.059ms, primary input-push kernels0.133ms.
Do not add peer wait to primary compute: they overlap. This points to the
primary dense/shared path and rendezvous before faster decode transport.
Single-window events/census are saved in decode-single-window*.json.
Post-stop arithmetic, clean teardown and health all pass; no crash/reset.
VERDICT: the PUSH prefill gain is backed by device-operation latency and
exact output parity. Attention is the next prefill target: the fallback
uses32-query batches (126 batches/layer at4044 tokens). Begin a small
same-math attention-batch experiment, retaining all measured controls.
No daily/shelf/full200k/stability promotion. Summarizer and benchmark
latency summary are tracked; full raw trace stays outside git.


## S11 -- attention launch batching is neutral; knob reverted

CONFIG: same-binary frozen strata-attn-batch-s11/dd21808c,109-line reduction/
PUSH plus5-line experimental attention-batch environment override. Only
query batching/scratch footprint varies32/128/256; default32 and per-query
math preserved. Same all24576/reserve1024/spec4/.5/two200k/int8/exact12288/
parking16GiB4/timing on; all banks still7440+17136, native slot0 passed.
COMMAND: same-image CPU main-target build; each arm three matched fresh4K/
32K requests, concurrent arithmetic, reasoning/tools; actual config/identity,
normal stop and per-card/ten compiled collectives between arms and after.
RESULT: all12 matched output pairs AND draft offered/accepted counts equal.
Three-repeat medians (native prefill ms, client TTFT seconds):

| Query batch | 4K tok/s | 4K ms / TTFT s | 32K tok/s | 32K ms / TTFT s |
|---|---|---|---|---|
| 32 | 1218.2 | 3332.1 /3.364 | 1532.9 | 21350.0 /21.414 |
| 128 | 1220.6 | 3325.3 /3.357 | 1536.7 | 21296.9 /21.360 |
| 256 | 1218.1 | 3332.3 /3.365 | 1540.7 | 21242.7 /21.307 |

Largest32K gain0.51%, no4K gain; no material launch-batch benefit or new
2000+ result. All functional/teardown health gates pass; no crash/reset.
No source/config promotion; revert the5-line knob. Source is back to the
109-line measured PUSH patch, unfrozen build output still has the rejected
experimental knob, so use the frozen strata-push-s9 control. Preserve the
failed/neutral experiment's source/config/build/report/raw responses in
PFRAW: attn-batch-s11.patch and attn-batch-s11-report.json. Nothing is
rewritten as a win. Correction to S10 attention count:1512 is the FULL
32-query kernel shape (126 full batches per layer);4044 queries also need
one12-query tail per layer, so127 batches total. Exported CSV retains the
separate tail shape. The approximately612ms figure is the full-batch row,
with a small extra tail; do not confuse it with all attention kernel time.
VERDICT: launch batching is not the main attention lever here. Next test
existing STRATA_ATTN_LANECELL=1 using frozen PUSH-s9, no source/build
change. It reproduces the original score butterfly without subgroup
shuffle reductions but loads keys more often (three heads/thread instead
of the original twelve heads sharing each load). This is a shuffle-versus-
traffic trade, not more key reuse. A commentary initially described the
reuse direction incorrectly; corrected after rereading the kernel before
running it. Exact outputs and measured times decide whether it helps B70.
Soak off;27B wrapper retained; no candidate full200k or daily promotion.


## S12 -- lane-cell score variant gives no useful gain; final daily restore

CONFIG: same frozen strata-push-s9 binary as S9; existing
STRATA_ATTN_LANECELL=1 only. Other flags, expert residency, context, quant,
MTP and RAM budget unchanged. Original per-score expression/butterfly is
reproduced by the source variant, but full math identity is not assumed.
COMMAND: three fresh4K/32K requests, compare exact requests/hashes with S9
PUSH control. Concurrent arithmetic, reasoning/tools, actual config/identity/
all24576 and native slot0 checks. Normal stop, per-card/ten compiled
collectives. Restore clean main daily driver, then repeat stable identity,
concurrent arithmetic, separate reasoning and tool round trip; memory
snapshot and27B prerequisite check. Re-fetch upstream main identity.
RESULT:4K1214.4 ->1220.9 tok/s (native3342.4 ->3324.7ms, TTFT3.375
->3.357s);32K1530.9 ->1515.4 (21378.6 ->21596.7ms,21.444 ->21.661s).
No broad gain. All six matched texts exactly equal; offered/accepted MTP
counts equal in only4/6, so no additional bitwise-math claim. Functional
and teardown/health gates pass, no crash/reset. Leave the variant off.
Main296b59d2 still equals origin/main and is clean; temporary EP source
again exactly matches the tracked109-line PUSH patch (SHA59ba0a85).
Final native serve uses clean-head binary16f592e7 on:18085 under
strata-optimized.service; split27, two200k/int8KV, spec4 (at most THREE
drafts)/min-p.5, prefill4096/reserve1024, RAM parking16384MiB/4. Config
matches the prior qualified daily config except the log path. Final
concurrent arithmetic/reasoning/tools/identity and27B check pass, observer
inactive. The one-time memory observation is not a restarted soak.
VERDICT: measured EP win remains reduced sums + PUSH (matched32K1531,
4K1214), source upstream via stacked author-branch PR
https://github.com/recutita/Strata/pull/1,78cea7a1. Individual kernel/API/
submission latency and full decode-window census are saved in S10. Query
batching and lane-cell variants did not recover2000. Next focused work is
attention value gather/accumulation and primary dense/shared decode work,
using operation latency and traffic/occupancy evidence. No shelf/candidate
full200k/daily promotion, four DISTINCT full contexts still unqualified.
Existing two-full200k forced parking/restore evidence is retained for the
qualified layer driver. Fallback remains one command:
`bash strata/flash-next/fallback-27b.sh`; full wrapper switch still untested,
prerequisites passed. No user dirty file changed. Raw S12 and final records
are in PFRAW from S5, including lane-cell-s12-report.json and final-s12*.


## S13 -- value-gather pipeline is numerically exact but neutral

CONFIG: frozen PUSH-s9 remains the serving control, SHA256932d2f347219edd6
(full hash in S9). Refreshed origin/main still296b59d2; PR2205 remains open
atc7bb8dd5, reduction/PUSH contribution recutita/Strata#1 remains open at
78cea7a1, launcher fix Niko1221/Strata#2468 remains open atf85b6c3f.
No runtime, driver, production source or serving binary change. Read the
S10 attention operation shape: INT8,32 queries,33 chunks of64 cells per
KV head,2051 actual selected cells at the selection budget. Existing
experimental CUDA code suggests transposing probabilities and fetching
four values before their dependent FMA chains. Port ONLY this value path
into an isolated translation unit, preserving score expressions and the
ascending per-head FMA order. Reuse the query SLM after its last reader's
workgroup barrier; no extra local allocation. Do not re-test the rejected
lane-cell score path. Preserve rejected source under experiments/.

COMMAND: CPU-only build in unchanged strata-ab-build:ocloc, same precise
FP model, SIMD32, bmg-g31 AOT; separately compile baseline and renamed
variant, link one numerical/timing probe. Baseline attention source SHA256
00bf1c45f15b1899e0562006aa6a410da4baec164eff1b1690316a6f2a66dcdb.
Probe SHA2567411d17bc61b49dcb516605cab4f7cde1aaaeff34892c2396d4364fad45bc4b2.
Stop daily normally; pair-leased per-card and ten compiled collectives;
`bin/gpu-run --card 0 docker run --rm --device /dev/dri
-v /mnt/vm_8tb/b70/build/strata-ab:/work -e ZE_AFFINITY_MASK=0
-e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e SYCL_CACHE_PERSISTENT=0
--entrypoint bash strata-ab-build:ocloc -c
'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1;
exec /work/attn-value-probe-s13'`.
Repeat pair health after normal probe exit. No model serving claim is made
for the modified kernel, so full model functional gates do not apply to it.
The unchanged frozen serving candidate is tested separately in S14.

RESULT: all36 INT8 cases pass (three seeds,12 shapes), both output AND
entire initialized scratch exact. Covers1/4/12/32 queries,0/1/11/63/64/65/
1999/2051 selected cells, capacity tails, shuffled page tables, page sizes
4/512, and absent pages. Small one-query cases also match an independent
double-precision dense softmax oracle, maximum absolute error7.28e-8.
Twenty warm repeats per arm per case, alternating arm order; medians across
three seeds below are HOST enqueue+wait for chunk AND merge, not a pure
kernel timer and not a production-speed measurement:

| Queries / actual cells / page | Baseline us | Value pipeline us | Change |
|---|---|---|---|
| 1 /2051 /4 | 47.200 | 47.019 | -0.38% |
| 4 /2051 /4 | 75.003 | 74.883 | -0.16% |
| 32 /2051 /4 | 530.071 | 531.112 | +0.20% |
| 32 /2051 /512 | 527.877 | 528.647 | +0.15% |
| 32 /1999 /4, missing pages | 477.019 | 478.682 | +0.35% |

Both cards and ten compiled all-reduces pass before/after; no crash/reset.
Initial build shell exited before compilation because strict shell mode
preceded oneAPI setup; corrected setup ordering, no GPU work involved.

VERDICT: reject this value-transpose/four-load pipeline for integration.
The synthetic resident-pool screen shows no useful latency gain; it does
not establish that all possible value-path changes are exhausted. Keep the
original109-line PUSH contribution, below2000 tok/s prefill, and move to
primary dense/shared decode and opt-in concurrent MTP. No source PR update,
new production benchmark or promotion is justified by this experiment.
Raw evidence: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_value_20261014
(identity-before.json, both complete source copies, build recipe/log,
probe-card0.log, probe-summary.json and pre/post health). Tracked numerical
probe: strata/flash-next/attn_value_probe.cpp; rejected diff:
strata/flash-next/experiments/attn-value-gather-s13.patch.


## S14 -- opt-in EP batch MTP improves overlapping coding

CONFIG: exact frozen strata-push-s9 SHA256932d2f347219edd6, same upstream/
PR2205/reduction/PUSH/runtime/host as S9. Spec4/min-p.5/two200k/int8KV,
exact12288/reserve1024/parking16GiB4; only STRATA_BATCH_MTP=0/1 differs.
Solo spec4 offers at most THREE drafts. Existing upstream batch MTP caps
EACH slot at ONE proposal, regardless of the solo limit. This experiment
uses that implementation unchanged. Layer splitting disables batch MTP;
EP is accepted by its guard. Keep the qualified clean-layer daily config
recoverable. No source edit, ABI-library import, driver change or long soak.

COMMAND: same frozen binary, three two-request coding windows per arm,
107 fresh input/1500 output tokens each; one solo coding request and one
mixed window each. In the mixed window a fresh32719-token prefill arrives
only after the first decoder content event and before its completion.
Save every raw SSE event, content-event arrival and output hash. Stable and
research IDs checked against registry; native slot0 and expert fill checked.
MTP private state changes residency from7440+17136 to7152+17424 experts;
all24576 remain in VRAM. Each added drafter shares weights and has242 MiB
private state/buffers. Main verifier capacity increases4 ->8 rows.
Concurrent arithmetic, separate reasoning and tool round trips pass in
both arms. Normal stops and per-card/ten compiled collectives pass after
each arm, no crash/reset. Raw orchestration: run-s14.py in RAW below.

RESULT: three-window median total output69.321 ->93.417 tok/s (+34.76%)
including prefill/admission. Per-request latency statistics over THREE
observations per requested stream (empirical quantiles, not an SLO):

| Arm / requested stream | Client decode p50 tok/s | TTFT p50 / p95 s | Completion p50 / p95 s | Median request p95 content gap ms |
|---|---|---|---|---|
| off /0 | 36.67 | 2.388 /2.428 | 43.265 /43.458 | 54.48 |
| off /1 | 35.57 | 1.084 /1.092 | 43.222 /43.414 | 54.63 |
| on /0 | 49.02 | 1.092 /2.223 | 31.782 /32.113 | 37.91 |
| on /1 | 50.44 | 2.378 /2.401 | 31.807 /32.070 | 37.90 |

Admission order varies. SSE content events can group multiple tokens,
especially with accepted drafts or detokenization; the benchmark labels
these as CONTENT-EVENT gaps, not unavailable exact per-token timestamps.
Live checks validate monotonic arrival times, first arrival=TTFT, chunk
references and gap counts. All six cross-arm coding outputs differ; both
expert placement and verification grouping change, so no numerical/token
parity or isolated kernel-speed claim. Functional checks pass in both arms.
Single solo runs are81.5/93.5 tok/s (diagnostic, not a matched3-run gain).

Native evidence resolves the misleading draft counters: ordinary batching
verifies one token per active slot and does no MTP. Opt-in batching builds
two rows per slot and accepts the proposal when it matches. On-arm native
coding logs show788/799/801 windows at3.80/3.75/3.74 emitted rows/window
with at most two active slots. More than two emitted rows proves accepted
proposals. Yet every coding HTTP response still says draft_n=0 because
BDONE carries no draft counts and Python _apply_bdone preserves the
admission's zeros. Exact offered/accepted totals are UNAVAILABLE for the
batch leg; these zeros must not be interpreted as no drafting. This does
not invalidate the earlier plain-batch zero-draft finding. A small native/
server accounting fix remains to contribute; no telemetry patch is applied
or claimed tested in this stage.

Contention remains: off/on mixed decoder completion48.356/44.463s,
maximum content gaps7.852/7.816s, incoming32K TTFT29.281/29.247s and
completion36.981/35.405s. Native mixed prompt throughput1129/1131tok/s,
compared with clean isolated S9 PUSH1531 at32K. Batching helps overlap
decode but does not remove long-prefill interruption. Mixed decoder draft
counts include the solo leg; its batch leg still lacks accounting.

The initial screens overlapped a CPU-only dense/GR AOT probe build (one
backend CPU job,32 logical CPUs, no GPU device attached). These are bounded
diagnostic screens. A separate CPU-quiet, uninstrumented confirmation is
recorded below before using a causal throughput claim. Probe binaries were
built from unchanged main; build-dense-gr-probes.log records the build,
and no tuning option from them has been run or promoted yet.

CPU-quiet confirmation: no compiler/profiler running, one fresh matched
1500-token pair per arm, normal functional/identity/residency and health
checks repeated. Total68.720 ->93.895 tok/s (+36.63%); per-stream client
rates35.34/36.39 ->48.91/50.80. Per-stream content-event p95 gaps54.79/
54.95 ->37.88/37.86ms. This confirms the configuration gain; the initial
three-window results supply the broader bounded sample. Exact within-arm
repeat0 response parity across initial and quiet runs is 4 of4;
no cross-arm token parity. Stats and all raw responses: quiet-confirm/
quiet-report.json. This is a bounded throughput improvement, not a150+
per-stream result or full-context qualification.

VERDICT: the existing opt-in batch MTP path is a promising EP candidate,
still far below150 tok/s per stream. Do not increase solo depth beyond
three/four to explain the old zero-draft overlaps. Keep frozen PUSH-s9 as
the source/build control; no extra kernel knob or source PR update from
this stage. Candidate full200k, RAM restore and four distinct full chats
remain UNQUALIFIED, as does the provisional prefix-return latency gate.
Working gate: p95 TTFT <=2s for retained~200k + short follow-up with two
returning streams, including RAM restore; separately report resident and
RAM-restored returns. User preference was requested and remains pending;
2s is a provisional working threshold, not a user-approved SLA.
A prepared bench_context_four.py samples native RSS/container cgroup RAM
at50ms and requires live32768MiB parking, correct distinct needles, all
retained history reused, two full GPU slots and measured RAM-restore
counter changes. It has syntax/help checks ONLY, not a qualification run;
restore byte measurement still needs transfer evidence. No success claim.

RAW: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_concurrent_20261014
(identity-before.json, actual configs, s14-report.json, both arms' solo/
concurrency/mixed/functional outputs, native logs, health and teardown).
CPU-quiet repeat lives under RAW/quiet-confirm. Primary operation analysis
from S10's actual middle four-token window lives in the S13 raw root:
decode-primary-window-analysis.json. Primary dense mmvq13.467ms and GR
residual mixing5.695ms are service-time sums within that diagnostic window;
peer returns1.059ms overlap primary work, not additive critical-path time.

S14 final state: clean-layer daily config is restored unchanged, binary
16f592e7e2f20178e4775ede7e6de306b6695b9444d1e399f68844cffb1e130b,
:18085/hotschmoe-dd, two200k/spec4/parking16GiB4. Final functional and
identity checks pass; soak inactive;27B prerequisite check exit0. No full
fallback wrapper switch performed. Native logging is append-only: original
S12 log prefix is preserved, and S14 restoration sessions have separate
copies under RAW; final-state.json records the current offset/container.


## S15: bound dense/shared screening and reject synchronization toggle

CONFIG: unchanged main296b59d2 builds dense_bench/gr_bench; frozen PUSH-s9
932d2f34 for serving, EP batch-MTP1/spec4/int8/exact12288/two200k/parking16GiB4.
Refresh upstream/main and contribution PRs2205/2468/recutita1: unchanged,
all open. Image/kernel/binary hashes recorded in RAW/identity-before.json.

COMMAND: leased card0/pin0 dense graph-replay microprobes,24 matrices/shape,
columns1-4, Q6_K/Q5_K; GR diagnostic host-enqueue probe tokens1-6. Separate
Q5-only control matches the RNG fixture of Q5-only variants. Repeat end
controls, per-card/ten compiled collective health. Only SH_STREAM0 receives
a full serving A/B: three solo coding and three concurrent coding windows,
identity/residency/concurrent arithmetic/reasoning/tools, normal stop and
per-card/ten compiled collectives per arm. No profiler/compiler during
serving timing. Each arm retains all24576 experts in the same placement.

RESULT: micro latency ratios to matched control (below1 is faster):

| Option | Median ratio | Decision |
|---|---|---|
| Q6 prefetch2 | 1.1281 | reject |
| Q6 rows/workgroup2 /4 | 1.1057 /1.5087 | reject |
| Q5 rows/workgroup2 /4 | 0.9461 /1.3103 | no broad candidate |
| GR unroll8 /4 | 1.1394 /1.6271 | reject |
| GR ILP /max4 | 1.2227 /0.9974 | reject /neutral |

All corresponding output hashes match. Q5 rows2 helps large3/4-column
shapes12-28% but hurts512-row shapes16-50%; this probe lacks the248320-row
vocabulary head. Projected observed-shape saving is below1ms/window,
insufficient evidence for a full serving arm. Q6 rows2 benefits the wider
out-projection but regresses other shapes. Dense end-control40 cases
median1.000, GR6 cases1.000. GR repeats hot weights and does not replay a
graph; its timings are diagnostic and not production latency estimates.

SH_STREAM0 removes the shared-expert queue fork/join but does not improve
serving: solo client93.300 ->92.236 tok/s; concurrent median93.607 ->92.805
total output tok/s (-0.856%). All six paired concurrent text hashes match.
Functional, expert residency and teardown health pass; no crash/reset.
Retain default shared-stream execution. No prefill or large-context reruns
for this losing toggle. Exact per-stream TTFT/completion/content-event gaps
and request timings live in RAW/s15-report.json and original responses.
HTTP batch draft totals remain unavailable until the accounting fix.

VERDICT: no production tuning patch or new knob. Keep PUSH-s9/batch-MTP1
as the experimental candidate and clean-layer daily recoverable. The GPU
server is stopped cleanly with post-health while the next accounting build
runs; no promotion, no soak. Two active200k and four full distinct chats
remain candidate qualifications to run next. Stage16 source work is
separate from these measured stage15 results.

Development-loop review: consulted actual claude-fable-5-1 and
claude-opus-5-5 twice; see RAW/review-decisions.txt for accepted/rejected
advice. Focused briefs cut consultation times168.5/91.6 ->43.5/38.4s.
Avoid future repeated Docker/oneAPI setup for micro arms, retain required
health checks, and repair mismatched fixtures with the smallest missing
control. Do not repeat loser matrices or infer speed from reviewer advice.

RAW: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_dense_20261014
(micro-configs/report, run scripts, actual serving configs, s15-report,
review-round1/2, review-decisions, functional and health records).


## S16: upstream accounting fix; full-context cache qualification fails

CONFIG: upstream296b59d2 plus existing EP2205/PUSH and accounting source
commit ecb1430a. Frozen control932d2f34 retained; new native binary5aaca2f5
(full identities in RAW). Same EP/spec4/.5/int8/exact12288/two200000,
all24576 experts (7152+17424). Parking32768MiB/4 entries, versus16GiB in
S15's timing control; GPU geometry/placement unchanged. No kernel/driver
change, no tuning knob. CPU-only build, no GPU device attached.

COMMAND:34 Python parallel/parser tests and4222 CPU C++ cache checks;
one native AOT link, three solo and three paired coding windows, identity/
residency/arithmetic/reasoning/tools. Four distinct199501-token prompts,
each followed by256-token cached replies to prove two active full contexts,
then forced RAM returns. Native RSS and container anon/file/swap sampled
at50ms. Normal stops/per-card/ten compiled collective health both attempts.

RESULT: all6 paired response hashes match S15 control; median aggregate
93.607 ->93.766 tok/s (no meaningful speed gain). Slot counts now reach
HTTP/metrics:783-799 offered and699-715 accepted per1500-output request.
Source adds optional accepted/offered BDONE fields, preserving legacy
five-field lines and adding admission counts. A pure host helper logs
snapshot-restore API payload by captured sizes, excluding metadata,
retained checkpoints and allocation capacity; distinct spare-row write is
counted. Both CUDA/SYCL emitters updated, CUDA inspected only on this Intel
host. Useful source fix published: https://github.com/Niko1221/Strata/pull/2511.
Native successful payloads3163725744/3164525616 bytes take355.5/357.9ms.
These are API copy payloads, NOT measured physical PCIe traffic.

First full attempt recovered all4 contexts and both two-active reply pairs.
The harness wrongly required zero RAM restores for an intended-resident
C/D pair; actual1 restore during slot handoff was valid. It aborted, and
automatic cleanup lost healthy cache state. This was a harness mistake.
Corrected classification uses observed native evidence, preserves unknown
mixed attribution, and saves phase manifests for same-PID/container resume.
Future harness failures keep the server alive for inspection. Repeated
ONLY context qualification, reusing all completed native/functional checks;
no extra source build or performance matrix. Original evidence is retained.

Retry: four distinct199501-token prompts again retrieve their three needles.
A/B cached replies overlap9.917s with155 windows/3.28 emitted rows; C/D
also passes two-active overlap. The A/B return had1 actual RAM restore;
C/D had0. Cache placement cannot be inferred from intended-resident labels.
All these returns reuse199529/199531 tokens and read53/55 new tokens, with
zero prior-tail replay. Per-request cached results before forced eviction:

| Conversation | Actual RAM restore | TTFT s | Completion s | Client decode tok/s | Content-event gap p95 ms |
|---|---|---|---|---|---|
| A | yes | 13.103 | 23.212 | 25.23 | 47.53 |
| B | no | 6.697 | 23.018 | 15.62 | 47.53 |
| C | no | 1.438 | 19.070 | 14.46 | 47.46 |
| D | no | 9.172 | 19.128 | 25.61 | 47.38 |

Empirical resident p95 over3 requests is8.925s. The two measured RAM returns
(including the failed A exchange below) give p95 about12.706s. Samples are
bounded, not an SLA qualification; working p95<=2s remains PROVISIONAL and
unmet. Per-conversation/request TTFT, completion, all content gaps and
reuse/new-token counts are in RAW/s16-report.json and original responses.

Forced exchange FAILS no-recompute: A's last prompt199584 +256 reply means
199839 tokens should already be held. Actual live RAM restore/reuse199584,
286 tokens read for new prompt199870:255 retained reply tokens replayed.
Needles are correct, TTFT5.163s, completion16.150s, but retained-history
reuse fails. B then reusesZERO and starts full199870-token prefill. Stop
redundant work after98304 tokens read via client disconnect; native reports
normal cancelled request, no crash. C/D forced returns are NOT RUN. Thus
four-chat RAM qualification, full-exchange latency and peak remain incomplete.

Measured peak during this PARTIAL FAILED exchange: native RSS18.7008GiB,
container18.7005GiB; zero swap/OOM kills. Peak container anon20027224064,
file1007616 bytes; full memory rows and lifetime/phase values retained.
Do not present the configured32GiB limit as measured usage or this partial
peak as a completed four-return peak.

Critical-path finding: native resident handoffs take~2.9-3.6s per slot
copy, and main-state RAM parking~2.35s. Source serializes slot movement
through host snapshots. These are measured phase terms from TWO requests,
not an additive decomposition of one TTFT. Correctness now precedes copy
or draft-depth tuning. Source audit finds RAM put only in park_current(main)
while slot decoding extends its own state; latest-slot parking before
reuse needs a small-context oracle. Entry-count pressure/old versions may
explain B's loss despite free byte budget, but victim attribution is not
proved. Audit re-tokenization/checkpoint matching as well as snapshot age.

VERDICT: accounting fix useful and upstream; candidate cache behavior FAILS,
no promotion. First run health and retry cancellation/teardown health pass,
no reset. Clean-layer daily16f592e7 restored and final ID/arithmetic/reasoning/
tools pass; observer off,27B prerequisites pass (no fallback switch). User's
later steering removes routine daily restore BETWEEN tests: subsequent GPU
windows queue prepared probes instead. Do not repeat four huge prefills
until a four-small-context/256-reply retained-history oracle passes.

Development loop: actual claude-fable-5-1 and claude-opus-5-5 consulted at
four more decision points (rounds3-6), with accepted/rejected advice in RAW/
review-decisions.txt. Rejected wrong token/byte units, speculative floors,
extra matrices, and unsupported nondeterminism claims. Independent tool-
enabled Fable cache-fix and Opus topology/copy-probe agents now prepare
concrete artifacts; root schedules leased GPU execution. No source fix or
new hardware speed claim from their untested candidates in this stage.

RAW: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_accounting_20261014
(s16-report.json, source/build/test identities, original context-four failure,
context-retry partial-summary/cancel/health evidence, review-round3..6,
final-state). Future candidates are under cachefix_20261014/agents.


## S17: fast small-context cache oracle (2026-10-15)

CONFIG: unchanged accounting binary5aaca2f5/EP/PUSH/spec4, parking32GiB4,
only native slot limit8192 instead of200000; all24576 experts resident
(12336+12240). Bench target4096 renders four3997-token prompts; cached
replies256 tokens. Source/quant/identity unchanged. Background CPU-only
Claude agent work is allowed: this is a CORRECTNESS oracle, not a clean
latency comparison. No new AOT link or production source fix.
COMMAND: extend bench_context_four.py with explicit --context-size and
context-aware token preflight/usage gates, defaults199600/200000 retained.
Run through leased matching server with --target4096 --context-size8192.
Normal stop, per-card/ten compiled collective health before/after. Do not
restore daily between this and the queued topology/copy probes.
RESULT: all4 fresh prompts and both two-active reply phases pass. Forced
exchange reproduces BOTH full-size failure modes, labels reversed: B
reuses4080/reads286 including255 retained reply tokens; A reuses0 and
re-reads4366. Correct code retrieval, no-recompute fails. Entire test phase
42.146s/15988 fresh prompt tokens, excluding startup/health. This replaces
full-context loops for diagnosing this bug; no new serving speed claim.
Both health checks pass; no crash/reset. Healthy cache remained available
for inspection, then stopped normally; no daily restore. Resume metadata
also binds target and configured context size.
VERDICT: concrete fast correctness gate established. Require this to pass
before repeating four full contexts. Independent Fable cache-fix and Opus
host/GPU topology-probe agents continue preparing source/artifacts. All
real GPU work remains root-coordinated under leases, with CPU-quiet timing
windows where required. No full-context promotion or source-fix claim.
RAW: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/small-baseline
(result-summary.json, config/residency, context request/phase data and health).


## S18: snapshot copy paths across cards and CPU placements (2026-10-15)

CONFIG: standalone SYCL component probe, no model/server. CPU-only AOT build
in image e41d59c0; binary6d3ff9cd/source41a43db4; kernel7.1.0-070100.
Threadripper1950X: one OS NUMA node, four8MiB LLC groups, separate PCI root
complexes/Gen3x16 host links. No proven die-local CPU mapping. Default
schedutil governor, no builds/serves/CPU-heavy agents during timing, load<1.
COMMAND: leased and pinned card0/CPU2, card1/CPU10; 42 odd-size/offset/guard
cases each, then64MiB10/256MiB8 samples with fresh patterns and poisoned
destinations. One card0/CPU10 control,256MiB8 with relevant subset only.
Record enqueue/wait, event service, allocation/free, faults and exact bytes.
Normal process exit, per-card and ten compiled collective post-health.
RESULT: all byte/guard checks and sample freshness pass; physical PCI IDs
match pins. At256MiB (seven later samples), card0/card1 respectively:

| Path | Card0 ms | Card1 ms |
| --- | ---: | ---: |
| Warm pageable H2D | 43.60 | 42.38 |
| Persistent host-USM H2D | 18.90 | 18.93 |
| Warm pageable D2H | 61.25 | 61.05 |
| Persistent host-USM D2H | 57.74 | 57.73 |
| Same-device D2D | 1.03 | 1.03 |
| Fresh vector allocation + segmented save | 176.83 | 179.37 |
| Uninitialized allocation + segmented save | 142.66 | 145.29 |
| Persistent host-USM segmented save | 57.87 | 57.87 |

Fresh vector allocation alone111.98/114.02ms; paired restore including
free64.78/64.91ms. Persistent segmented restore19.24/19.23ms, pool creation
excluded. Host-to-host queue copies~80ms are NOT PCIe traffic. Alternate
CPU keeps transfer medians within2.7%; fresh-vector save+allocation rises
to202.72ms with65505 vs53216 minor faults. Do not attribute this allocation
difference to NUMA or generalize no placement effect beyond this window.
No crash/reset; post-health passes. All servers remain stopped, no daily
restore. These are hardware component results, not model gains.
VERDICT: actual claude-opus-5-5 source review identifies fresh per-layer
temporary ConversationKv in both slot-copy lambdas. Prepare the smaller
pageable-image reuse candidate before introducing pinned storage. Default
unchanged_tokens=0 must keep full recopy, even across layers/conversations.
Retain geometry validation, stream/ring restore and MTP behavior. Correct
slot preservation first; small no-recompute gate before full contexts.
Direct same-device copy may remove the bounce later, with separate state
and ring validation. API payload3164MB may mix host/device-authoritative
pools; do not derive physical traffic or serving speed from this table.

Development loop: CPU/source agents prepared work while root used the GPU
gap. Skip more placement/P2P/NOINIT matrices and repeated small/full runs.
Timed probe includes its own42case oracle, so standalone oracle-only runs
were redundant here and are unnecessary next time. Source review rejected
CPU warm-copy extrapolation: that arm repeats one16MiB pair, not a multi-GiB
footprint. Prefer persistent streaming Claude sessions and narrow concrete
deliverables; initial35turn Opus cap required a focused continuation.
Fable correctness candidate still in progress; no cache-fix claim in S18.
RAW: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/agents/opus
(s18-report.json, root-summary/identity, topology, commands, samples, source
reviews, build and health). Source: strata/flash-next/latency_probe/.


## S19: preserve completed batch slots before overwrite (2026-10-15)

CONFIG: upstream296b59d2 still current (GitHub rechecked), existing EP/PUSH/
accounting plus batch-slot-cache-preserve.patch. Binaryb09c9bf3, frozen
accounting5aaca2f5 unchanged. Same8K slots/spec4/parking32GiB4/all24576;
STRATA_SNAPSHOT_VERIFY=1 adds draft-KV readback. No performance patch yet.
COMMAND: actual claude-fable-5-1 prepares factored slot/main snapshot path,
checkpoint carry and coverage helper;4271 CPU checks/SYCL syntax pass.
Root freezes identical final source, one CPU-only AOT link (489.39s from
log birth to last write, ~1s resolution), then existing4x3997/256-reply
oracle, observed two-active proof, identity/math/reasoning/tools, normal
stop/per-card/ten compiled post-health. Reuse S18 post-health before start:
no GPU workload intervened. No duplicate new agent harness or cold baseline.
RESULT: all four forced RAM returns reuse4335 held tokens,31 new tokens
read, zero retained-tail replay; all needles and no cross-chat leak pass.
Four restored draft-KV readbacks pass. All10 available S17 response hashes
match exactly; C/D forced returns had no completed baseline for comparison.
Four-small-chat qualification passes; full200k remains NOT TESTED here.
Small-context sampled exchange peak native3755134976/container3648733184
bytes,0cgroup swap/OOM. Diagnostic p95 TTFT (four samples each) RAM1.758s,
resident2.161s; this does not qualify the provisional2s target at200k.
Functional checks, identity/residency and teardown health pass; no crash/
reset. Servers stopped, no daily restore or shelf promotion.

Source finding: idle slot reply state was overwritten without RAM capture;
the main snapshot held only the prompt-read state. Returning slots also
lost their checkpoint chain, allowing old RAM versions to consume entries.
Fable reconstructs this eviction sequence from full logs and cache policy;
the small native run now verifies the complete remedy. Both overwrite
callers park the outgoing idle slot with its own sessions, cvec/checks and
independent draft K/V, using existing budget/admission/snapshot logic.
Successful giveback carries the slot checkpoints back to main. No captured
address/ownership swap, new entry cap or snapshot format. Existing fed ids
are used unchanged: the last sampled un-fed token was already excluded.
CPU rendering preserves the full4336-token canonical history prefix; actual
sampled IDs were not retained, so segmentation was not independently proved.

VERDICT: cheap latest-conversation cache gate passes. Text-only EP with
batch-MTP on is measured. Images and MTP-without-batch-MTP skip slot parking;
pipeline stages and CUDA are unexecuted. Coverage of a longer continuation
does not guarantee arbitrary shorter-branch endpoint restoration. Do not
generalize this qualification beyond its actual scope.

Development loop: actual claude-opus-5-5 reviews frozen source; reject its
invented acceptance floor and resolve checkpoint questions against code.
Fable coding task produced concrete artifacts but hit35turn cap after2095s;
use persistent streaming/narrow tasks. Follow-up decision briefs take25.83s
(Fable) and14.96s (Opus). Both accept a separately built8line buffer-reuse
candidate, cheap gate, then one full4x200k final qualification for absolute
targets. Skip extra full cachefix-only baseline and unmatched relative speed
claims. Reject invented vision scope, incorrect32GiB cgroup limit (90GiB
actual), and redundant large prefill probes without a concrete failure.
Next required build can test experimental parallel AOT flag supported by
installed icpx; current build was serial, no flag change in S19.
RAW: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014
(cache-candidate/s19-report.json, small-first requests/bytes/health, agent
findings/reviews, frozen patch/identities, next-stage-review-decisions.json).


## S20: four long conversations recover from RAM (2026-10-15)

CONFIG: frozen `strata-slot-reuse-s20`, SHA256
`97bad88a5f059e5f077c6429d8b166d68208b50d8b64ee193e3827d0ad425dd6`.
Upstream296b59d2 + EP/PUSH/accounting/S19 cache preservation, plus Opus's
8line image-buffer reuse: one ConversationKv per stage rather than per QSA
layer in both slot copy directions. Full save still overwrites all bytes
and metadata. Same spec4, batch-MTP depth1, two200000 slots, INT8 KV,
32768MiB parking/four entries. All24576 experts resident (7152+17424).
Stable served ID and research alias unchanged. Full run omits snapshot
readback instrumentation; small oracle keeps four draft-KV readbacks.

COMMAND: CPU-only incremental offline AOT with installed experimental
`-fsycl-max-parallel-link-jobs=8`; no GPU devices in build container.
Existing4x3997 small oracle first: four forced returns,12/12 S19 output
hashes equal, draft readbacks and functional/teardown health pass. Then
ONE4x199501 full-context run using bench_context_four.py, two overlapping
256-token replies per pair, followed by forced AB and CD RAM exchanges.
Run identity/concurrent arithmetic/separate reasoning/tool roundtrip,
normal stop and per-card/compiled collective post-health. No daily restore.

RESULT: full correctness PASS. All four forced returns retrieve each
chat's needles, reuse199839 held tokens, read31 new tokens, replayZERO
retained reply tokens. Native records prove two actual full RAM restores
per forced pair,419.4-430.3ms each; payload3168427056 bytes each is snapshot
API payload (host+device pools), not physical PCIe bytes. Two-active proof
passes for both resident reply pairs and both exchanges. Four chats end
at199899 tokens each. Full-run peak sampled native RSS17387425792 bytes
(16.19GiB), container17376952320, zero cgroup swap/OOM, host available at
least102299648000. Configured32GiB is a cache budget, not measured usage.

Forced-return TTFT2.758-6.423s, median4.592/p95 6.398 (four samples);
resident-only diagnostic p95 6.250 (two); the first mixed resident/RAM
pair remains unattributed. User explicitly demotes2s to diagnostic only.
Native copy components at200k are about1.60-1.65s; no matched relative
serving speed claim. Fresh199501-token prefills1153.5-1174.4 tok/s, below
2000 goal. Long-context reply streams remain far below150 tok/s.
Identity, concurrent arithmetic, reasoning and tool roundtrip PASS.
Final post-health status is recorded in s20-report.json and stage journal.

VERDICT: true eviction-to-system-RAM recovery without retained-history
recompute is demonstrated for these four latest text conversations with
two active slots. This is a bounded correctness qualification, not broad
sampling/image/CUDA/pipeline coverage or a daily shelf promotion. General
shorter-branch restoration retains S19 limitations. Prioritize concurrent
MTP/decode next; direct-copy/main-park latency ideas stay queued. Skip a
second full-context baseline and further latency-only matrices.

Dev-loop: S20 build122.177s versus S19 approximately489.39s, both1600 AOT
success lines. Recompiled target counts differ (2 versus6), so this is not
an isolated4x claim for the parallel-link flag. Actual Fable5.1 prepared
the depth candidate source during the full run; actual Opus5.5 reviews its
state/acceptance integration before bounded GPU screens. Cheap CPU checks
and short decode windows replace another four expensive prefills.

Raw: cachefix_20261014/slot-reuse/{small,full,s20-report.json,build.log,
build-comparison.json}; tracked patch:
strata/flash-next/patches/batch-slot-image-reuse.patch.


## S21: bounded concurrent MTP depth screen (2026-10-15)

CONFIG: S20 stack plus opt-in batch-depth host packing/acceptance and the
context-end guards; frozen `strata-batch-depth-s21`, SHA256
`db53efb23790f682478f8469eab5737cc0e9b569a0bdf6c7a98ea86c6272fba3`.
Same all24576 experts, two200000 slots, spec4, INT8 KV,32GiB/four entries,
EP/PUSH, draft cutoff0.5. STRATA_BATCH_MTP_DEPTH=1/2/3 changes proposals
per active slot (seed is additional); default remains1. Target sampling
is temperature0, not draft min-p sampling. No runtime/device-stack changes.

COMMAND: actual Fable5.1 source implementation, Opus5.5 commit/drafter
review; root adds proposed-row clamp to context acceptance bounds and ports
CUDA drafter's max_cells step cap to SYCL. CPU packing/acceptance/EOS/cancel/
length/context-end checks PASS in CPU-only build container (host g++ absent,
use installed container compiler). Incremental AOT125.391s. First build
cancelled121.834s after review found the boundary hazard; its logs retained.
No GPU run used the unguarded candidate. Boundary tests replace proposed
API probe: API/native admission's8-token margin cannot reach the bad cells.
QsaState.max_cells is full context, independently of ring length; cap does
not silence later drafts. Default1 now omits unacceptably late/unavailable
draft rows; regular short request geometry/output is unchanged.

One paired coding window per depth,107input/1500output per stream. Depth2
wins screen; depth3 loses, so no extra depth3 windows. Confirm depth2 and
control with two more fresh prompt indices (--start-repeat avoids cache
reuse/redundant repeats), then one9707-token/256output pair each for
indexer-selection parity. Same frozen binary and aliases in all arms;
CPU agents/builds idle during timing. Normal stops between arms, no daily
restore. Functional checks at depth2; final per-card/ten compiled collective
post-health recorded in s21-report.json and journal.

RESULT:

| Batch proposal depth | Aggregate output tok/s, individual windows | Median | Median client decode tok/s per stream |
|---|---|---|---|
| 1 | 93.903,93.064,92.408 | 93.064 | 49.701 |
| 2 | 99.351,98.886,98.901 | 98.901 | 52.737 |
| 3 | 93.156 (screen only) | 93.156 | 49.566 |

Depth2 median gains6.272% over matched depth1. Bounded one coding workload,
three pairs each, not a broad throughput guarantee. All six response hashes
PER depth1/depth2 match S16; depth3's two also match. Both9707-token output
hashes match depth1/depth2. Offered/accepted counts now real: depth1 offers
783-799/accepts699-715 per1500-token stream; depth2 offers1120-1132/accepts
933-939. Identity, concurrent arithmetic, separate reasoning and tool
roundtrip PASS. Long-context cached recovery remains the separately tested
S20 depth1 result; no repeated4x200k or depth2 daily-shelf promotion.

Native first-window summaries (averages over each two-request run):

| Depth | Windows | Emitted rows/window | Run ms | Commit ms | Emit/draft ms | Total ms/window |
|---|---|---|---|---|---|---|
| 1 | 788 | 3.80 | 32.28 | 1.43 | 3.55 | 37.26 |
| 2 | 561 | 5.34 | 40.67 | 1.56 | 7.03 | 49.27 |
| 3 | 456 | 6.57 | 52.90 | 1.65 | 10.40 | 64.95 |

Rows/window are emitted rows, not offered drafts. More accepted proposals
reduce window count, but verifier work and serial per-slot drafting grow.
Native window rate omits parts of admission; use client aggregate above
for matched serving comparison. This is not yet150tok/s per stream.

VERDICT: keep depth2 as an opt-in experimental candidate, default1 unchanged;
reject depth3 for this workload. Next investigate main verifier work and
serial drafter submission, guided by actual Fable/Opus source reviews.
Do not repeat the full RAM qualification, prior losing knobs, solo test
whose code is unchanged, CUDA/pipeline/penalty/coupled-sampling matrices,
or synthetic unreachable API end tests. No cache-latency2s gate.

Raw: cachefix_20261014/batch-depth-s21/{identity.json,s21-report.json,
depth1*,depth2*,depth3,long-parity-verdict.json,build.log,aborted-build.log};
agent sources/reviews/root resolutions in agents/fable-depth/. Tracked
patch: strata/flash-next/patches/batch-mtp-depth.patch, after accounting;
adds CPU header test plus matching CUDA/SYCL host path, only SYCL executed.


S21 review resolution: internal STRATA_VERIFY_PROFILE explicitly disables
shared-stream forks (verify.cpp guards require !prof_on_), so the proposed
profile-only pair would observe an altered branch. Prefer hardware tracing
with the env off, or label altered-path attribution and first use a short
canary. Zero-doorbell logs' zero expert counters are omitted accounting,
not evidence of zero routed traffic. More rows can read more distinct
expert weights; do not infer non-bandwidth-bound execution from row scaling.
Keep hash checks for traced requests. Opus found per-instance drafter
queues; shared mutable scratch under borrowed loads is the remaining source
gate before an asynchronous submit/finish prototype. No overlap gain measured.


## S22: production-path trace and rejected async drafting (2026-10-15)

CONFIG: S21 frozen db53efb2 for production-path control tracing. New frozen
strata-async-s22 SHA256
487eb67ac661a921fa0212b330c392222aa72bfa61f86e42e6eb3a0bbaf4c8b0
adds opt-in STRATA_BATCH_MTP_ASYNC: unset0 unchanged,2 own-queue residual
copy only,1 submit all slot drafts then finish all. Same spec4/depth2,
two200000 slots,32GiB4,all24576 experts, EP/PUSH,greedy. CUDA untouched.
Private per-drafter scratch/host buffers/queues checked; only immutable
weights/head shared. SYCL create_queue's bool enables exception handling;
in-order comes from its implementation, not that bool (USM_LEVEL_NONE
absent). Own-queue copies follow completed verifier/commit and all drafters
finish before the next overwrite. Partial-submit errors drain without
reading incomplete outputs; finalization fixes early pending-state issue.

COMMAND: actual Opus5.5 source candidate, actual Fable5.1 independent
review; one CPU-only incremental AOT136.132s. No duplicate build tree or
unchanged CPU acceptance tests. Frozen same-binary0/2/1 one paired coding
window each (107in/1500out). All source agents/builds idle during clean
benchmarks. Hardware unitrace control/async each one107in/64out pair,
collection enabled before batch capture, explicit stop/flush and arithmetic
canary before normal shutdown. Internal STRATA_VERIFY_PROFILE stays OFF:
it would disable shared-stream forks. CPU source preparation allowed only
during diagnostic traces, not clean timings. Per-card/ten compiled
collective post-health PASS; no crashes/resets; all servers stopped.

RESULT: all two response hashes per mode match the same S16 repeat0 pair
(six comparisons total; not six distinct prompts). Mode0/2/1 aggregate
99.462/99.422/99.357 tok/s. Native561windows/5.34rows each; total49.12/
49.16/49.16ms, run40.54/40.60/40.61,commit1.56each,emit7.02/6.99/6.99.
Neutral screens; no repeat triples or long/cancel/functional/full-cache
qualification for this rejected candidate. Four traced output prefixes
match the reference; post-stop canaries pass.

Both valid traces contain24 batch windows, about303-305MB each. Cold
capture inflates the control average to129.97ms/window, so exclude it.
Eight interior cycles (full-vocabulary six-row target-head completion to
next completion) give control/async medians59.174/58.988ms. Source-calibrated
trace-to-client shift uses BOOTTIME minus MONOTONIC_RAW; raw clocks retained.
Each8-cycle census has384 EP waits on EACH card (48/cycle),32 one-row draft
heads (4/cycle),8 six-row target heads. Draft heads total/union both about
3.981ms/cycle in both modes; no overlap above0.001ms timestamp tolerance.
That is a finding about these profiled executions, not a hardware-wide
claim that concurrency is impossible. Clean benchmark remains authoritative.

Control per-cycle service examples: full draft heads3.981ms, target head
2.259; hyper-connection GR up3.561/down2.397; QSA attention2.501; primary
EP wait3.043/peer wait27.207, peer send1.399ms. Eight-cycle device busy unions
are386.331ms primary/295.405peer over473.075ms wall; divide by8 for a cycle.
Do not sum overlapping device times or interpret the peer's waits as
transfer bytes. Entry counts validate replay coverage; whole-request trace
prefill/dequant/snapshot copies are excluded from these interior cycles.
Full operation median/p95/counts, raw SSE/TTFT/gaps/completion/reuse timings,
and payload names are retained. Physical PCIe bytes are not inferred from
launch grids or cache ownership fractions.

VERDICT: reject the async/copy modes; no observable serving benefit. Remove
ONLY S22 source delta; exact pre-S22 diff hash restored (frozen binaries and
patch retained). No qualified daily-server restoration. Keep S21 depth2
experimental and S20 RAM recovery evidence. Skip more async-depth knobs or
repeat matrices without a new mechanism. Fable prepares a bounded GR
root-sync-property candidate; Opus prepares the successful cache fix for
upstream review. No2s latency work. Agent first15turn cap left usefulcode;
focused4turn finalize and8turn independent review completed.

Raw: cachefix_20261014/draft-profile-s22/{s22-report.json,profile*,mode*,
build,source-restored.json,async-*,root-review-decisions.json}.
Rejected source evidence: strata/flash-next/patches/
rejected-batch-mtp-async-s22.patch. Do not activate without new evidence.


## S23: submit independent slot-cache preservation fix upstream (2026-10-15)

CONFIG: upstream main 296b59d2 plus S19 slot preservation/checkpoint fixes
and S20 snapshot-buffer reuse, independent of accounting PR #2511.
Branch batch-slot-cache-preservation-main, commit ab2c9fc705f5ec5f38c3b78fe7fb69373671801f.
COMMAND: apply independent four-file patch; CPU-only C++20 cache test and
SYCL generate syntax check; source/dependency review by actual Claude5.5;
push fork branch and create draft [Strata PR #2526](https://github.com/Niko1221/Strata/pull/2526).
RESULT: 4,271 CPU checks and SYCL syntax PASS. Independent branch changes
349 insertions/48 deletions in four files. Full GPU evidence remains S20
on the matching EP/PUSH/accounting overlay; no independent-branch GPU
rebuild or CUDA compile/run is claimed. Draft PR discloses both limits.
VERDICT: useful bounded fix submitted, no runtime/shelf change. Avoid a
redundant full-model rebuild merely to repeat the already measured overlay.
Raw: cachefix_20261014/cache-upstream-s23/{validation.json,pr-state.json}.


## S24: operation map and bounded per-card readiness probes (2026-10-15)

CONFIG: actual Claude Fable5.1 and Claude5.5 prepare isolated CPU-built
probes; servers stopped, no daily-server restoration. Existing S22 traces
remain the production-path operation evidence. Standalone AOT readiness
binary a1dceb2c73359896f4be72f31d086e20b79de9badd58c88f1a7738ddbf6accf7.
COMMAND: root leases/pins each card; query metric capabilities; one GR
32-case oracle/timing screen plus tiny T6 unitrace; one integrated readiness
run/card (680 samples each), CPU2 pin, no duplicated smoke/full matrix.
Source and metadata audit, then both-card health. No new two-card workload;
S22 collective-health evidence remains separate.
RESULT: metric groups fail ERROR_DEPENDENCY_UNAVAILABLE on both host
queries; device identity/timestamps work. GR default versus no-root-sync
passes all32 bitwise cases but is neutral: T6 old54.9-56.7 us, candidate
54.8-56.6 us; rejected before whole-model build. Tiny trace is perturbed
(about112 us median chain versus about55 us clean), retained for geometry
and gap attribution only. Readiness all1360 byte/guard/checksum samples
PASS. At614400-byte synthetic D2H, warm median producer-end/copy-start
gaps pageable fresh121.20/116.93 us (cards0/1), pinned prealloc1.41/1.41 us,
11warm samples/arm. These are single-card stand-ins, not EP P2P or serving
gains; synthetic pack/consumer costs are not model pack/dequant costs.
command_submit/start differences are inconsistent (some negative): exclude
ALL queued-time values, retain separate host clocks and device start/end
service/gaps. Initial removable producer busy loop caught before GPU run,
replaced by observable nonlinear checksum and host verification.

Delivered [offline interactive data map](../strata/flash-next/data-map/index.html)
and [Markdown companion](../strata/flash-next/data-map/README.md):40 stages,
170 operation types, six lanes, phase/status filters, clickable sources/
shapes/quant/tile/median/p95/call counts and sortable operation table.
JSON is embedded for offline use; checked synchronization/ASCII and actual
DOM execution across40 detail views,5phase filters,13sort columns and
text filtering. Browser layout has not been visually inspected. Raw GPU
service sums overlap and do not define a latency budget. Peer wait includes
upstream compute/publication/transport; no causal split or recoverable
27ms wall gain is claimed. Real GGUF geometry confirmed, includingK10,
N2560,FF640 and Q5_K head. PLE enabled, second GGUF resolves to NVMe
/mnt/cache, primary IQ3_XXS GGUF remains SATA. Actual SYCL PLE path is
8-way host row cache -> page dedup/alignment/sort -> queued worker-pool
O_DIRECT pread (default64), not io_uring. Joined PLE readiness unmeasured.

VERDICT: retain map and reproducible probes; reject neutral GR variant.
Both-card post-health PASS; no crash/reset, probe containers gone, all
model servers stopped. No serving/default/shelf change. Skip full-engine
rebuild, repeated huge prefills, unchanged collective suite and server
restoration for these single-card diagnostics. Next useful work is joined
CPU/input-ready/EP publication timing and actual-head C1/2/3/4/6/8 replay,
with measured kernel shapes and a numerical oracle before serving tests.
Every-op production CPU and hardware-counter coverage is incomplete.
Raw: cachefix_20261014/lowlevel-s24/{s24-report,readiness-summary,model-geometry,storage-identity}.json.
Sources: strata/flash-next/lowlevel_probe/. Rejected GR patch retained.

S24 final review: actual Claude5.5 caught host-mapped PLE/HC/drafter input
copies mixed into runtime glue; these now have a dedicated measured stage.
Root verified embedding call sites, residency log counts, logical return
upper bound and head-parity hypothesis scope. Source/storage audits
override a reviewer assumption that the second GGUF was SATA: realpath
and findmnt prove its NVMe symlink. Metric-query timer-unit display was
fixed against installed ze_api.h (properties stype, not driver API version);
raw values retained, no conclusion depended on the original label.


## S25: real vocabulary-head curve and production PLE waits (2026-10-15)

CONFIG: upstream296b59d2 and accounting/cache PR heads unchanged at stage
start. Frozen S21 db53efb2 for one PLE diagnostic, same EP/PUSH/spec4/
depth2/two200k slots/32GiB4/all24576 experts. STRATA_PLE_TRACE=1 adds
timestamps/logging only; STRATA_VERIFY_PROFILE stays unset. Actual Fable5.1
prepares existing head replay; actual Opus5.5 audits trace and independently
reviews same-depth head batching. Probe frozen SHA256
21e4beb3fbe8dbedfc94362a1cf2b725410d7e85b949f7cce1be2d08348f9c03.
COMMAND: reuse prepared standalone source and existing runtime archives;
CPU-only AOT build, no engine rebuild. Derive head offset with GGUF header
reader and cross-check existing reader. Root leases/pins each card for one
C1/2/3/4/6/8 curve (three30-launch means/case); tiny card0 unitrace with
20launches/round. One native pair107input/64output for PLE logging,
arithmetic canary, normal stop, both-card and10compiled collective health.

RESULT: actual output.weight is Q5_K,2560x248320,437043200bytes at offset
11024256 in served first GGUF, SHA256
93f1c7cb7a7cf8134ce9f8cf80c5416df968f7f1ddcc16f2159a9521080826a5,
FNV1a64 da8746c3dcc2469e. Inputs are deterministic normal activations,
not captured model hidden states. All six C values pass full bitwise
columns versus independent C1 rows, argmax/finiteness and inactive-row/
tail guards on both cards. All C1 reference1986560floats finite.

| Head rows C | Card0 us/call | Card1 us/call |
|---|---|---|
| 1 | 1006.0 | 1006.6 |
| 2 | 1322.0 | 1307.9 |
| 3 | 1721.7 | 1676.5 |
| 4 | 2043.2 | 1978.0 |
| 6 | 2700.5 | 2620.7 |
| 8 | 3376.6 | 3302.6 |

These are minimum-of-three mean event-barrier spans over30 launches,
including queue/kernel/inter-launch gaps; quantization measured separately
at4.2-4.6us. Tiny unitrace confirms individual kernels, grid62080x1x1/
workgroup32x4x1/SIMD32, C1 median1016.666us/C2 1309.010/C6 2681.146.
All traced oracles also pass;1.27MB trace, not a full-model trace.
Micro replay differs from serving cache/frequency/context; no physical
weight traffic or bandwidth claim. C2 versus2xC1 suggests1.38-1.41ms
ideal saving per depth2 window before additional graph/coordination cost.
Only same-depth slots can pair; autoregressive depths cannot form oneC4.

Production PLE: both64-token reply prefixes match S16, both offer46 and
accept40 drafts; canary323 and teardown health PASS. Between first batch
capture and24-window summary:21 miss tickets,776jobs total,8-64jobs/ticket.
Collect wait median667.9us,p90 817.4us,max876.8us,sum13.2942ms; issue->submit
median41us, submit->last631.2us. The24 native windows average51.90ms
(run43.65,commit1.51,emit6.73). Trace omits jobs0/cache hits and jobs200+,
contains no absolute anchors, and cannot exactly join GPU events. Worker
reaping/scheduling are included, so submit->last is not pure NVMe latency.
Do not rank startup/canary traces as steady decode;125total log records
versus21 batch-phase records. Observed waits are about1.1% of this phase's
24*51.9ms, not a global PLE bound. Client metrics retained separately:
TTFT2.423/1.124s,completion3.785/3.724s; SSE gap p95 49.866/49.798ms,
max277.571/1316.801ms (admission/capture included, multi-token events).
Logging run is diagnostic, not clean throughput qualification.

VERDICT: C2 same-depth head batching merits one small opt-in candidate,
with per-slot state/graphs/context/EOS/cancel guards and in-serve parity.
Independent Opus audit estimates200-300 source lines; extra launches can
consume the ideal saving. Fable prepares source in an isolated S21-overlay
worktree; runtime/defaults unchanged at S25 end. PLE waits are measured but
lower priority here; skip new PLE engine rebuild or counter-package detour.
Both-card health and10compiled collectives PASS, no crash/reset; servers
stopped. No fullRAM repeat, soak or daily-server restoration. Fresh focused
agent review reduces repeated large-context setup. No150/stream claim.
Raw: cachefix_20261014/head-profile-s25/{s25-report.json,head-runs,ple,fable,opus-head-review}.
Source replay/parser: strata/flash-next/lowlevel_probe/. Data map refreshed.

## S26: head-pair prototype held for EP freshness diagnosis (2026-10-15)

CONFIG: same S21 base, optional same-depth two-slot Q5_K head pairing,
SYCL-only and default off (STRATA_BATCH_MTP_PAIR_HEAD). Actual Claude
Fable5.1 implemented in an isolated tree; actual Opus5.5 reviewed. Root
corrected reversed active-slot binding, retained the allocation queue for
partial initialization/teardown, propagated pair capture failures and added
activation counts outside the intrusive verifier profiler. No backend,
driver, quantization or serving default changed.

COMMAND: CPU-only syntax checks of both changed translation units; host
fixture extracting the actual caller binding and eligible/steps bodies;
one incremental no-DRI AOT build, 141.101s, eight link jobs. Frozen prototype
strata-head-pair-s26 SHA256
264f221e2c1b02d4bdd9b332c46f91e91bf4f6301afbdabe563df5ff2a16e44a.
Then coordinated pair-leased standalone EP tests of real push17520B and
upper-bound return614400B, fresh salts/poison/guards each sample. Native
kernel archives reused; payload variant compiled only in probe-local TU.

RESULT: CPU checks pass forward/reverse slot order, distinct bindings,
owner rejection and unequal context/proposal fallback. These do not prove
GPU graph, draft-KV or serving correctness. No S26 model serve was run.
Copy-control passes21 samples including first use. Native flag-mode push
fails with1068 bad words at sample10 using owner poison. Producer poison
also fails (1158 at sample3); distinct staging poison localizes later
failures to current INBOX poison (1296 at sample8). Waiting on completed
host events still fails (720 at first use). Changing both ep_push payload
store sites to uncached hints also fails (440 at sample9), so that variant
is rejected. Return/guards/spin status pass in these failed samples, and
later source/inbox readbacks are clean. Matching flags/counters AFTER drain
do not establish the epoch observed inside ep_wait. Neither the exact
cause nor production serving-output corruption has been demonstrated.
All failed-arm timings are excluded from performance conclusions.

Every arm drains queues and exits normally with an oracle failure; no
crash, timeout or reset. Final both-card health and ten compiled two-rank
collectives (4x5120, P2P0) pass. Servers remain stopped. The runtime EP
kernel source is unchanged (SHA164ea229); only the CPU-built head prototype
is present in the external live source. Frozen S21 remains available.

VERDICT: hold the head prototype unqualified. Preserve the failed
uncached-payload-only patch as negative evidence. Next discriminators are
observed wanted/seen wait epochs, consecutive receiver reads and actual
BMG fence/cache-control ISA; Fable/Opus prepared CPU-only work independently.
Stop losing arms on first failure, reuse standalone objects, skip full
model/cache/soak repeats and daily-server restoration. No speed or shelf
promotion. Tracked probes and patches are under strata/flash-next; raw
cachefix_20261014/head-pair-s26/s26-report.json includes hashes, commands,
root review dispositions, failures and final health logs.

## S27: intended epoch observed, first read stale (2026-10-15)

CONFIG: actual Fable5.1 prepared a bounded dynamic probe; actual Opus5.5
prepared static source/SPIR-V analysis. Frozen source26e26f96, binary
49cfed4f7b018fa3af8a0e0681abe8af20e41a2eecc91169e765ee1529c2fa44;
native EP source/archives unchanged. Producer poison,17520B push,614400B
return; distinct stage poison, guards and finite spin/host/outer caps.

COMMAND: coordinated worker audits all11 manifest entries and queue/mapping
lifetimes, then pair-leased native and debug waits with two extra immediate
same-queue receiver reads, maxfail2/reps64, outer120s. Both in-order queues
fully drain before readback/free. Required per-card and ten compiled4x5120
P2P0 collectives after EACH arm. No model serves or engine rebuilds.

RESULT: native first use passes, two failures consume956/1428 current
inbox poison words; second/third reads both zero bad. Debug failures consume
1404/1406 first-read poison words, both later reads clean. Device side
channel records wanted3/4, actual flag transitions2->3/3->4,198/203 spins,
no spin error. The intended epoch really was observed in these debug
samples. Post-drain source/inbox views clean; return and guards pass.
Normal drained exit5, removed containers, no timeout/crash/reset. Both-card
health and all ten compiled collectives pass after each arm.

VERDICT: exclude wrong wanted epoch for THESE debug samples, not all native
runs. First consumption after the current flag can still be stale while
immediate later reads are clean, without a host wait. Incoming-write
completion and receiver visibility remain unresolved. Extra reads are
instrumentation, not a production remedy; failed-arm timings are invalid.
No full-model corruption claim or runtime change. Opus output is source/
SPIR-V only: native disasm folders contain sections.txt, not decoded BMG
ISA; initial work hit turn cap, and static visibility is unproven.
Next single discriminator: same native payload assignment in one producer
kernel, flag publication in a separate same-queue kernel. CPU-only prep
continues separately while backend strategy is discussed.

Dev-loop: reused the audited binary, two bounded arms only; no duplicate
prehealth after unchanged S26 final closure, full-model/cache/soak/daily
restoration. Root independently read wait/triple-copy bodies. Tracked probe
source/README/runner updated; adapted runner syntax checked without another
GPU repetition. Raw ep-ordering-s27/dispatch/{s27-report.json,findings.md,
provenance.json,sha256-dispatch.txt,native.log,debug.log} preserves commands,
hashes, sample diagnostics and health. Servers stopped, both leases free.

## S28: scoped EP publication passes bounded oracle (2026-10-15)

CONFIG: unchanged two-B70 stack, real EP17520B push/614400B return,
producer inbox poison/distinct stage poison/guards, finite bounds. Actual
Fable5.1 prepares mirror/scoped producer; actual Opus5.5 reviews native
ISA. Runtime adds one functional line: explicit system memory scope on
EP push's workgroup barrier, retaining release fence and one kernel.
Head-pair option stays OFF; cache and return-path source unchanged.

COMMAND: pair-leased native versus split payload/flag kernels, then
same-binary default/scoped mirror, then patched actual native EP TU with
original S27 probe object and native wait/no extra reads. CPU-only full
engine build87.888s; frozen152055a6 C2 small serving canary. See the
[dated evidence](20261015_strata_ep_publication_scope.md) for full identities,
protocol, commands, source delta, all six arms and native ISA qualifications.

RESULT: old native/debug first-use PASS then first reads fail1316/1374;
split debug/native each65/65 PASS. Default mirror fails112/648 including
first use; scoped65/65 PASS. Actual patched EP TU/native wait65/65 PASS.
All guards/returns/spin codes pass; queues drain normally, no crash/reset
or timeout. Native BMG ISA IS available: ocloc emitted hidden .text.*.asm
files missed in S27. This explicitly corrects S27's no-native-ISA claim.
Preflag UGM fence none.group becomes invalidate.gpu; two post-EOT null
moves also differ,10 rawbytes total. No compiler-bug/legality proof.

Frozen engine152055a6 serves stable/research identities, all24576 experts
resident7152/17424. Both concurrent64-token coding replies match S16
prefixes, each46 offered/40 accepted drafts. Concurrent math323/667,
reasoning391 and tool roundtrip PASS. Normal stop/lease exit0 and final
both-card/ten compiled4x5120 BF16/P2P0 collective health PASS.

VERDICT: keep small EP-scope correctness fix and submit to author EP
branch, independent of PUSH. Actual native-TU proof avoids trusting only
a probe-local clone. No speed/stability/production-corruption/compiler-bug
or new fullcache claim. Preserve old binaries/archives; don't relabel a
fresh fixed native archive as the failing historical control. S26 headpair
remains GPU-unqualified, now ready for its separate OFF/ON canary.
Loop: recovered existing ISA, reused objects, stopped failures early,
batched drained numerical arms before health, one small model canary.
No repeated four200k test, soak, framework setup or daily restoration.
Tracked probe exact S28sourcea73dc02e; adapted wrapper bashsyntax only.
Raw ep-ordering-s28/s28-report.json and sha256-s28.txt preserve provenance.

Contribution submitted and API-verified:
https://github.com/recutita/Strata/pull/2, head69dee3c3, onefile/+1/-1,
target sycl-ep-prefill-main (upstream2205), independent of PUSH1.
Current main296b59d2 lacks EPfile. Full checkpoint276b099 pushed.

## S29: same-depth head pairing improves bounded C2 decode (2026-10-15)

CONFIG: same frozen strata-ep-scope-s28 SHA152055a6, EP scoped barrier,
all24576 experts7152/17424, two200k slots,32GiB/four-entry parking,
spec4/batchdepth2. Toggle only STRATA_BATCH_MTP_PAIR_HEAD=0/1, separate
log paths. No source/build/framework/quant change. Profile/PLE tracing OFF.

COMMAND: corrected S26 pair source finally runs on GPUs after S28's EP
oracle closure. ON then OFF, two fresh servers. Each gets the same S16
repeat0 requests capped64 and math/reasoning/tool functional suite, then
fresh107-input/1500-output coding pairs repeat1/2/3. Reuse bench_concurrency;
no duplicate microprobe, full-context or daily restore. Both normal stops
and both-card/ten compiled4x5120 BF16/P2P0 health close after EACH arm.

| Bounded metric | OFF | ON |
|---|---|---|
| C2 aggregate tok/s, median of3 pairs | 97.881 [97.100,98.407] | 102.279 [101.560,102.714] |
| Client decode tok/s, slot0 median | 51.019 | 53.399 |
| Client decode tok/s, slot1 median | 53.452 | 55.620 |
| Native run/commit/emit ms per window, weighted printed means | 40.347/1.563/7.033 | 39.810/1.550/5.380 |
| Per-request SSE content-event gap p95, median across6 | 49.715ms | 47.530ms |

RESULT: ratio of aggregate medians+4.493%; paired-repeat differences
3.759-5.334% (three observations). All six exact ON/OFF request bodies
and output hashes match. All four available S16 repeat1/2 references
match in each arm; repeat3 has no retained S16 reference. ON1675 windows
paired/19deferred-solo; ordinary short-canary fallback observed. Both
stable/research identity+registry, expert residency, functionals, normal
stop/leaseexit0 and health PASS. First-use graph capture/replay succeeds.

Actual claude-opus-5-5 and claude-fable-5-1 consulted independently;
responding IDs verified. Opus identifies actual107-vs108-token handoff
behind a short-canary one-draft delta. Root does NOT adopt its per-window
burst-alignment proof:55content events are not64tokens, and 5ms grouping
cannot establish token keep counts. Fable correctly counts logical depth2
graph replays4->6 and explicit drafter waits2->1; those are not measured
L0-call counts. Native run also varies, so don't attribute all wall gain
solely to head math or equate the standalone1.38ms ideal with a threshold.

New pre-existing accounting defect: OFFrepeat3slot1 reports28offered/21
accepted for1500tokens. Native log shows exactly those counts for its final
31-token solo tail. serve/server.py carries decode/token totals through
batch->solo but carry_leg/add_carry omit draft totals, losing the preceding
batch leg. Full-request acceptance comparison is INVALID; raw counts stay
in the report. This does not invalidate wall timing or exact text parity.
No accounting source change applied in S29; small optional-field fix and
CPU regression are prepared separately for the next stage.

VERDICT: retain head pairing as the winning bounded C2 candidate; no daily
promotion. Still far below150/stream. Depth2/headpair full RAM recovery,
long-context performance, C4, mixed32K arrival and boundary/cancel coverage
are not newly qualified. S20's full RAM proof remains depth1. Queued tiny
ON trace requires a new instrumented start and is not executed here.
Loop: two starts, no rebuild or sweep, extend the promising arm in place;
review artifacts precede timing. Four-turn high-effort consultations took
213-218s: overlap future review with teardown/control loading and use
narrow source briefs. Own review workers stopped before benchmarks;
other session's source research wasn't isolated, run order isn't reversed,
and clocks aren't controlled. Three pairs are not stability evidence.
Raw head-pair-s29/{s29-report.json,analysis/analysis.json,sha256-s29.txt};
prepared tracing/candidate steps are under head-pair-s29-prep/.

## S30: carry correction and full paired-head cache qualification (2026-10-15)

CONFIG -> Same frozen152055a6 engine as S29, paired headsON/depth2,
CLI spec4, two200000 slots, INT8 KV, 32GiB/4 parking entries,
all24576 experts7152primary/17424peer. Seven Python helper lines now
carry optional draft totals across batch-to-solo handoff. Native argv,
build, kernels, stack and scheduling are unchanged. Add snapshot draft-KV
readback and a buffered completion observer for correctness; this is not
a clean throughput run. Actual INFO spec6/mtp_max4/lookup3 reflects
existing suffix expansion of the solo verifier, not extra batch depth.

COMMAND -> Actual-helper old/new regression; six existing/project CPU
tests in the serving image without DRI/network; one leased server for
64/512-token handoff, functionals, cheap4k four-chat oracle, full four-chat
oracle and one fresh32k arrival during decode. The cache harness now
requires mean emitted rows above native depth+1 plus content overlap:
at depth2 a single slot can emit3, so the old above2 test was too weak.
Normal stop, leaseexit0, per-card and ten compiled two-rank collectives.

RESULT -> Old helper regression reports28/21 instead of constructed
1134/935; new helpers pass six cases including unknown/None and zero.
Six project tests pass. Live long native BDONE40accepted/46offered plus
solo DONE304/371 equals finalserver/API344/417; short39/46 also matches.
Both output prefixes match S29. Stable ID/alias/registry, concurrent
math323/667, separate reasoning391, tool continuation and residency pass.
Fix pushed/API-verified on existing
[Strata PR2511](https://github.com/Niko1221/Strata/pull/2511),
head1fce5591a0ede4d5451316f462ad8acd528a21e6. No new native build.

Full oracle: four distinct fresh199501-token prompts; all needles correct,
no cross-conversation codes. Each resident reply emits256 tokens. AB/CD
content intervals overlap8.677/8.719s; native means4.21/4.24 kept rows
per window prove some windows contain both slots. Both resident pairs
are genuinely resident in this run. Four forced RAM returns each reuse
199839 held tokens, read31 new and replay0 retained-history tokens.
All four draft-KV restored-byte comparisons pass (199840 cells).

| Return phase | Native restore ms, observed order | Client TTFT s, named streams | API payload bytes each |
| --- | ---: | ---: | ---: |
| AB | 685.1 / 673.8 | A3.189 / B6.918 | 3168427056 |
| CD | 670.1 / 663.3 | C2.992 / D6.842 | 3168427056 |

Native restore records are phase-attributed; their observed order is
not joined to individual client streams. API bytes are snapshot restore payload, not physical
PCIe traffic. Client RAM-return p50/p95=5.015/6.907s on four samples,
diagnostic only. Peak sampled nativeRSS16.255GiB/container16.244GiB;
exchange-only15.161/15.166GiB. 15030 samples at50ms; zero container swap
and OOM delta; minimum host available95.103GiB. These are bounded peaks,
not a multi-day stability claim.

Mixed arrival: fresh32719-token prefill arrives during1500-token decode;
both finish, cache_n0 each. Decoder max content-event gap7.753s,
completion43.900s; incoming TTFT29.564s/completion35.360s. Retain the
interference bottleneck; SSE events can contain several tokens. Snapshot
verification/observer/current cache population prevent a clean speed
comparison against S14 or S29. Normalstop/leaseexit0 and final both-card
plus ten compiled4x5120BF16/P2P0 collective checks all PASS; no reset.

Existing full-prefill timers reused without more GPU work: two ordinal
sequences each16x12288+2886 timed tokens. Equal full blocks1->16 show
QSA select224/225->6278/6284ms while attention2291/2303->2630/2636ms.
The199494 timed tokens omit seven prompt tokens; no A/B names or exact
context offsets are assigned. Value gather is not separately isolated;
peer idle is on its own timeline, not hardware utilization or additive
CPU/GPU budget. S22's170 operation rows stay the reference; no new C2
per-kernel trace is claimed.

VERDICT -> Winning depth2/headpair candidate now passes the requested
bounded two-active/four-distinct/32GiB RAM recovery qualification. Keep
it recoverable at raw draft-carry-s30/config.json; no daily shelf change.
Clean S29 coding remains102.279 aggregate,53-56clienttok/s perstream,
well below150. Full-context correctness-run client decode18.44-29.38
tok/s is diagnostic and a different workload. Prefill targets, mixed
interference and paired-head critical-path profiling remain open.

Actual Opus5.5/Fable5.1 IDs verified. Initial narrow source reviews hit
their turn caps; resumed tool-free verdicts complete in15.6/12.1s.
Opus approves bounded telemetry proof. Root rejects Fable's alternating
single-slot explanation of mean>3 using actual bt_rows/bt_windows source;
readback and phase RAM requests were already enabled. Its GR-down lead
is distinct from S24 but not ready: direct retains four partials, changes
reduction order, and source records sliced faster. No unqualified default
arm or redundant numerical run. Loop: one server start/no build/no second
old GPU arm/no1500 rebenchmark/no daily restore; full200k run only after
the cheap oracle passes. Useful source/timer work overlaps necessary
prefills. Raw draft-carry-s30/{s30-report.json,full-cache-verdict.json,
prefill-phase-diagnostic.json,model-review-closure.json,post-health.json}.

## S31: paired-head replay operations and readiness gaps (2026-10-15)

CONFIG -> Same frozen152055a6/headpairON/depth2, CLI spec4, two200k slots,
32GiB/4 parking, all24576 experts7152/17424; S30 Python carry fix. Only
unitrace wrapper/log paths change; ordinary Intel entry, no S30 observer
or snapshot readback. S30 full-cache/functional evidence retained.
COMMAND -> Collection ON before capture; one32+32 coding pair; named
pause/stop, normalserverstop/leaseexit0, bothcard+10compiledhealth. Reuse
S22 parsers with fresh bracketed BOOTTIME-RAW clock correction. Exclude
capture cycles0..4 and conservative tail10; review5..9, selected by API
construction evidence, not fastest durations. No build/cache rerun.
RESULT -> Both output prefixes and exact32-token caps PASS,9paired/2solo.
Valid270756565B trace;18C2heads=2x9nativepaired. Per reviewed cycle2C2/
0C1,4privateMTPselect,48primaryEPwait,9AppendExp/2Copy/11EventHostSync.
API-start HostSync319/324/334/324/349 are invocations, not logicalbarriers;
arguments/CSV omit handles,timeouts,results. Most take<=5us; long waits
overlapGPUwork. Pairedheadservice2.416ms/mainC6head2.262ms, dense13.899ms,
HC7.199ms percycle; overlapping sums are not latencybudget. GRsliced->
reduce gapmedian0.209us/p950.438us, not oldstandalone9.5us. Largest five
primary recorded-activity gaps followMTPselect,3.300-3.941ms; nextfive
barrier->mappedinputcopy1.555-1.648ms. Associations notcause/physicalidle
or savings; resets/barriers instrumentation-sensitive. Normalstop and
finalbothcard+10compiled4x5120BF16/P2P0health PASS, noreset.
VERDICT -> Current C2 execution/ON API costs measured; no OFF delta or new
speed claim. No direct-GR arm justified; source/untraced boundary audit
next. Fable's preliminary runtime-prefill quantum lead needs state/callback
validation; root read run_impl/PLE persistence/drains, rejects arbitrary
bitwise cross-chunk hash gate. Opus completed review of supplied trace
facts; use narrow observedC2 claim and skip OFFtrace solely for wording.
One start/64totaloutputs/no build/no fullcache repeat/no dailyrestore.
All159 current operations exported to data-map/20261015_s31_replay_operations.csv;
old170 S22 operation rows preserved. Full analysis and limits in
[dated S31 report](20261015_strata_paired_head_profile.md).
Raw head-pair-s31/analysis-prep/{result/final-analysis.json,
primary-quiet-gaps.json,reviewed-summary/}; s31-report.json/post-health.json.


## S32: CPU boundary diagnostic without unitrace (2026-10-15)

CONFIG -> Frozen595f2ff5, S31headpairON/depth2/two200k/32GiB, samePython;
opt-in bufferedCPUsteady_ns marks, no newGPUcalls/events/waits or perwindow
prints. Removingdiagnosticlines recovers all3oldTUs byte-for-byte.
COMMAND -> OfflineCPU-onlySYCLbuild/noDRI, one128+128pair, functionals,
normalstop/leaseexit0/bothcard+10compiledhealth. No fullcache rerun.
RESULT -> Bothprefixes/caps/residency/functionals PASS;45paired/2solo.
Exact1684records/dropped0,48windows;40replay/38adjacentjoins aftercoldroles.
PLEgathermedian0.829ms/p951.085; verifier/pairsubmit55/45us; pairedmapped
stores0.316us. Pairwaitreturn->nextsubmit0.866ms/p951.112. Existingwaits
andcommit includeGPUprogress, not CPUoverheadbudget. S31devicecompletion
andS32CPUwaitreturn unjoined; no physicalidle/traceroverhead/savingclaim.
VERDICT -> No multi-ms postwaitpacking/stagingdelayobserved; skiptuning
thosefromoldertrace. Prepare smaller runtimeprefillquantum retaining12288
allocationfor7.753smixedgap, subjectstatecontinuityproof. No daily/speed
promotion. ActualOpus/Fablecompletedpatchreviews; rootfixeswindowreset/
emptyframebug, rejectsstrongerGPUgapcausality. OneincrementalTUrebuild;
initialsnapshotracepartial rejected/recoveredfromstoppedlog, noGPUrepeat.
DatedS32report anddata-mapCPUquotes; raw host-boundary-s32/s32-report.json.


## S33: fixed-allocation prefill quantum tradeoff (2026-10-15)

CONFIG -> Samefrozen32ff11a2 forQ0/4096; C12288, FIRST0, trace+prefilltimers,
headpairON/depth2/two200k/32GiB/all7152+17424experts; S32markersOFF.
COMMAND -> ActualOpus/FablepatchreviewsBEFOREsingle122.7sCPU-onlybuild;
one32769-token5anchorarrivalduringretained1500codingdecodeperarm, cached
followup, functionals/normalstop/leaseexit0/bothcard+10compiledhealth.
RESULT -> Bothfull1500decoderrefsMATCH, anchorsinitial+followupPASS,
59tokens, returnscache32827/read32. ActualQ4096eligible1,7x4096+4090 vs
12288/12288/8186; contiguouspart0..32762,7tokensoutside. Maxcontentgap
7.824->4.059s, incomingTTFT29.526->37.581s, decoderwall41.619->46.612s.
Prefillcalls21.117->25.639s plus measureddecode7.718->11.255s explain
batchedread28.835->36.897s; sequentialwalls, notsummedwithGPUtimelines.
Primarygatherregion2.348->5.174s, dequant.472->1.056s; NOTisolatednative
weightgather: MMQnotbuilt, regionincludesactivationgather/hostpeerenqueue.
Functionals/normalstop/allfinalhealthPASS botharms,noreset; noRAMtest.
VERDICT -> Fairnessoptiononly, retainQ0/nodaily/newfullcontextpromotion.
ActualOpus/Fable bothstopmidpointsweep; rootrejectsunsupportedclaims6144
mustinterpolateor1-2sdifferenceisnoise. No hardgaprequirement/selecting
utility; actualcostsourceleadbetter. BYIELDcheckcadencechanges, 0taken/
0declined, waitersunexercised. CPUfixtureoneunitestimate13vssteady12fixed
with64unitamortization; failedcounttranscriptpreserved, final21calls0.29s,
noGPUrepeat. Sourcecoldpathnoabsolutealignmentgate; purecheckpointbranch
unknown. DatedS33doc/optionalpatch/data-map; rawprefill-quantum-s33/.


## S34: historical peer dispatch attribution (2026-10-15)

CONFIG -> Retained S10 PUSH trace, S9 binary 932d2f34, 4044-token main
chunk. Historical ownership 155/357 experts per layer differs from current
149/363. This is a CPU-only evidence audit, with no new runtime change.
COMMAND -> Parse the old trace once, reuse compressed selected events,
and link GPU launch IDs through host H2D flows to append calls. Read the
current actual SYCL route, S7 rejected async arm and GGUF metadata.
RESULT -> Across all 48 layers, primary activation gather has median
0.149 ms service; following primary intervals without recorded positive
GPU operations have median 52.987 ms. Matching host dispatch spans have
median 53.146 ms and 58,716 peer kernel appends. Host API union covers
361.786 ms of 2395.300 ms elapsed. These are overlapping, historical,
profiler-sensitive scopes, not physical idle or a removable latency budget.
VERDICT -> Current gather phase labels include host peer submissions;
MMQ weight gathering is not built. Do not rerun the already slower S7
per-layer async mechanism. Opus/Fable source reviews suggest narrower
dispatch and conditional dense fusion leads; actual header checks limit
same-format K/V eligibility to 7/12 layers. Prepare a real-weight,
bitwise one-gather/reverse-alias oracle before any serving patch. Current
host markers can share the next planned serve. No new performance or
health result; no build/GPU test of the new candidates yet. See the
[S34 report](20261015_strata_ep_prefill_dispatch.md) and raw
`ep-prefill-dispatch-s34/{s34-report,review-closure,historical-gaps}.json`.


## S35: one-gather reverse-alias numerical gate (2026-10-15)

CONFIG -> One-card standalone 4f2b42fb,363 real mixed-format experts and
synthetic routing. Exact production gather/dequant/FP16 GEMM/SiLU calls;
reference per-expert gathers versus one FP16 gather into existing FP32
output allocation and reverse-offset expert processing. No production edit.
COMMAND -> Actual Opus/Fable source reviews, one135s offline CPU-only AOT
build; two quiet card-1 windows with pre/post health and normal teardown.
RESULT -> T4096/12288, rows29040/87120, largest641/1921 plus ne1 cases PASS.
All297,369,600 FP32 values match bitwise and are finite; gathered bytes,
nine buffer guards and complete unused output tail PASS. Four health
checks and both wrapper/container closures PASS, no reset.
VERDICT -> Prepare guarded actual IPC serving integration. No speedup
claim: reference-first/candidate-second timings have cold/warm order bias.
No imported-memory/concurrent-reader/reduction or new full-cache proof.
One build/two finite cases; no model-server reload or new trace. S36
buffered marker attribution fixed before compilation; new K/V fusion
deferred after measured small service-pool ranking. See the
[S35 report](20261015_strata_peer_gather_alias.md), tracked
`strata/flash-next/lowlevel_probe/peer_gather_oracle.cpp`, and raw
`peer-gather-oracle-s35/{s35-report,build-closure}.json`.


## S38: guarded one-gather alias in real IPC serving (2026-10-15)

CONFIG -> Same frozen9d30ad8b, aliasOFF/ON only, C12288/Q0, depth2/headpairON,
all24576 experts resident, two200k slots/32GiB parking. First diagnostic has
buffered CPU markers; later timed samples capture no timestamps or dumps.
COMMAND -> Exact actual Opus/Fable reviews, one183s CPU-only build, two finite
servers. One4049-token diagnostic then3x4K/3x32K and3C2 pairs/arm; no new
per-kernel trace, mixed/cache/full200k campaign, broad sweep or daily restore.
RESULT -> Actual48/48 IPC activation, identical routes; all13 exact request/
output/token/cache matches, four retained S16 hashes, functionals PASS.
Prefill medians1303.0->1326.2 and1546.1->1556.1 tok/s; C2 aggregate
104.516->104.017. Peer-loop CPU span median15.826->14.830ms/layer in the
first diagnostic, not GPU service/physical idle or an additive budget.
Normal container/wrapperexit0 and both-card+compiled10 health PASS eacharm.
VERDICT -> Keep aliasOFF/qualified152055a6; optional patch retained. Small
prefill gains/no decode benefit do not merit another full-context campaign.
Existing GRF-resource numerical adapter is source preparation only. See
[dated report](20261015_strata_peer_gather_serving.md), raw
`peer-gather-serving-s38/{s38-report,serving-comparison,host-spans}.json`.


## S39: unchanged-kernel HC register-allocation discriminator (2026-10-15)

CONFIG -> One frozen7365c788 adapter, current production kernel archives,
N2560/HC4/LR320 BF16 fixture, GRF0/1 fresh card1 processes, slicedDOWN/
cols16UP/NORMSPLIT1, all unrelated modes fixed. No engine/kernel edit.
COMMAND -> Actual Opus/Fable exact-source plus narrowv2 reviews before one
29s offline build. Existing unitrace on the same two numerical processes;
quietCPU outside card1lease, pre/posthealth, normalcontainer/wrappers0.
RESULT -> All48 fullsnapshot pairs35,893,248 bytes match, activefinites/
guards/untaken/inactive tails/readonly inputs and changed-input graph PASS.
1260DOWN/UP executions eachmode prove default versus grf256; actual reported
registers128/256, private/spill0 both. Whole-read instrumented medians are
4.62-17.62% slower in all12cases, not a clean or serving result.
VERDICT -> Keep default. No extra serving arm, main-engine build or fullcache
repeat for this knob. Card1health/teardown PASS both; no reset. Tiny MTP
Q4-sharing fix independently confirmed/reviewed but still source-only;
current Q8dense pool0.744ms/cycle gives no reason for a broad quant sweep.
Larger Q6dense packing source audit next; no new candidate measured. See
[dated report](20261015_strata_grf_resource_probe.md), raw
`grf-oracle-s39-prep/{s39-report,build-closure}.json` and `run-profiled/`.
