# Live Pi bang investigation, 2026-09-10

CONFIG -> User reports frequent bang-guard recoveries during the last hour
from a LAN Pi harness. Neighbor table confirms 192.168.10.56 (reported
address had a typo). Live lifecycle: qwen38_int4_fp8kv_trial/20260909T224703Z.
Primary ID hotschmoe-dd, secondary ID
qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276-mtp3-fp8-e4m3-calkv-cache800k;
both match evals/configs/models.yaml. Exact image 521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad.
INT4 weights, calibrated FP8 E4M3 KV, MTP3, prefix on, FULL_DECODE_ONLY,
200K context, four sequence slots, 860000 logical KV tokens, no RAM offload.

COMMAND -> Read /v1/models and /metrics, snapshot server/frontdoor logs and
manifest, inspect installed cache manager/coordinator and local bang-guard
source. Compare both rank load records against the frozen scale artifact.
Inspect kernel journal for the reported window. No inference, restart,
cache flush, live patch, or serving setting change.
Raw root: /mnt/vm_8tb/b70/results/bang_investigation_20260910/20260910T013217Z/.

RESULT -> Window 00:32:17-01:32:17 UTC has 278 engine samples: maximum
three running requests, 94 samples with more than one, zero sampled waiting,
maximum sampled KV usage 19.8 percent. Lifetime preemptions are zero.
Ten-second sampling can miss shorter overlap or usage peaks. Normal finish
and zero repetition/error counters cannot establish coherent output.
Kernel journal 00:30-01:33 contains a perf sample-rate adjustment, no Xe
fault/reset/OOM. Both ranks loaded all 17 attention layers with exact
artifact scales and hash d53fb6565c485658b345d0b48aaf3ea4f5e0a48748e09a24bc6a29234c026a82.
Loading correctness does not prove runtime finite values or absence of FP8
clipping on real conversations; scale_audit is false in this lifecycle.

Installed MambaManager.find_longest_cache_hit accepts drop_eagle_block but
uses it in neither the aligned nor partial lookup branch. The coordinator
excludes MambaSpec from the speculative lookahead margin. Source hash
1a0dedb76ed07c64fa780e10a89ff718c37975804bf70594a9577c6ecebd5787 matches
the September 8 base regression. Upstream report:
https://github.com/vllm-project/vllm/issues/53912
Candidate fix: https://github.com/vllm-project/vllm/pull/48375
These describe a compatible hypothesis, not a diagnosis of this episode.
The earlier local backport failed the MTP4 oversized-history qualification
with a timeout, without captured corruption; do not present it as a proven
repair or as proof that the boundary issue is irrelevant.

The older FP16-KV MTP3 workload already produced 4/132 raw bang attempts
in concurrent shared tool conversations, with recovery. A serialized
32-check control was clean; private per-session salts still produced a bang.
See 20260908_prefix_campaign.md, September 8 overlap investigations. These
small samples support a concurrency/history-state lead but do not establish
that today's frequency increased or that FP8 caused it. The prior matched full-feature
trial's 32 shared-tool checks were clean; they are not a long-run rate estimate.

Local bang-guard d341fe6c9a3b53061e9936b0cb78b6f08e6f6438 trips at 32
consecutive exclamations in text, thinking, or tool arguments. For the default
hotschmoe-dd/openai-completions route it rotates a session cache salt and
filters corrupted context on recovery. LAN installation/version/settings
are not verified. This guard mitigates output corruption; it does not repair
recurrent state. The frontdoor relays response bytes without rewriting them
and logs neither request bodies nor generated text nor client addresses.
Consequently these logs cannot attribute/count the user's bang attempts.

VERDICT -> Strongest existing lead is concurrency-sensitive MTP/recurrent
prefix-state handling. New FP8 KV precision is an additional unisolated
variable. Cache eviction is unsupported by the zero-preemption evidence.
Missing scales are ruled out at load time; runtime saturation and NaNs are
not measured. Root cause remains unconfirmed. Preserve the serving process
and obtain the affected Pi JSONL: timestamps, first bad content block, model,
usage and retry records allow a raw-attempt denominator and timeline. SSH as
hotschmoe to 192.168.10.56 was denied by public-key authentication; client
transcript or existing SSH access details requested from the user.

Next controlled reproduction should replay the affected history with raw SSE
and request metadata retained, including cache salt/hash, cached tokens,
sampling settings and actual overlap. Compare concurrent versus serialized
requests, then isolate prefix reuse, speculation, and FP8 KV one factor at a
time in separate leased lifecycles. Correlate scheduler row order, accepted
speculative counts and recurrent-state slots on both ranks before naming a
kernel. Logging/logprobs can perturb timing. Any TP2 relaunch must honor
current P2P safety rules, bounded recovery, and pre/post health; the running
P2P-enabled control is not authorization for arbitrary new P2P serves.

CONFIG -> Exact serving image, separate CPU-only container, no GPU devices,
network disabled, 2 CPUs/2 GiB, 60-second outer bound.
COMMAND -> vllm/int4/test_mamba_eagle_drop.py --expect-bug.
RESULT -> Exit0, all six aligned lookup cases match the expected defect.
With five16-token blocks and drop requested, full attention returns64 tokens
and Mamba80. With one block, full attention returns0 and Mamba16. Zero-block
and no-drop controls agree. Log: cpu-mamba-regression.log in the raw root.
VERDICT -> Reconfirmed installed aligned-boundary behavior without GPU
execution. Does not demonstrate that a failing live request used that boundary
or test the partial lookup branch dynamically. No serving repair claimed.

## Pi diagnostic bundle: 2026-09-10 follow-up

CONFIG -> User supplied pi-corruption-20260910T022712Z.tar.gz from
/mnt/storage/StrongSync/junk/agent-world-diagnostics/. Extracted under
/mnt/vm_8tb/b70/results/bang_investigation_20260910/pi-corruption-20260910T022712Z/;
bundle-sha256.txt records exact input identity. The nested directory contains
three sessions, RPC event streams, manifest and events.json. No gateway source,
wire request bodies, actual cache-salt values or installed extension source.
Manifest: release c41b639b55a0d0ce7658cd303290e46f848f69cb, clean;
hotschmoe-dd through 192.168.10.5:18080, temperature0.7, reasoning enabled
(Pi session level medium), max output8192, context200K, cache_salt=true.
World paused by the other team. No contained command was executed.

COMMAND -> Validate archive members and extract data only. Run
vllm/int4/audit_pi_corruption_bundle.py against the extracted directory;
compare independently detected session bang runs, RPC trip notifications and
events.json timestamps. Preserve audit.json and correlated.json alongside
fresh server-at-bundle-analysis.log. No inference or serving changes.

RESULT -> Counts agree across all three sources:

| Agent | Assistant attempt records | Local 429 | Bang attempts | Non-429 records |
| --- | ---: | ---: | ---: | ---: |
| agent_1 | 189 | 14 | 15 | 175 |
| agent_2 | 105 | 14 | 16 | 91 |
| agent_3 | 59 | 8 | 10 | 51 |
| Total | 353 | 36 | 41 | 317 |

Observed fraction is41/317=12.93 percent in selected affected sessions, not a
representative long-run rate or a matched comparison to earlier short tests.
Non-429 records include three streams missing finish_reason. All41 bang
attempts are aborted, with missing/zero final usage; zero does not mean zero
GPU work or zero prefix hits. Bang runs occur in33 thinking blocks,5 tool
argument blocks and3 text blocks. Some outputs degrade through gibberish or
number loops before bangs. One thinking block begins with bangs; most contain
preceding generated material. Bang-guard is detecting actual stream content,
not merely tripping on its own recovery notice. Five retry-limit exhaustion
notices are separate from the41 trip notifications and must not be double
counted. Initial audit caught this distinction via count disagreement; the
final audit matches exact trip notice text and all three sources agree.

There are36 recovery custom messages and36 local429s. Every queued guard
recovery's first attempt is rejected with
429 "one inference request per agent at a time". Each next attempt starts
0-3ms after the aborted session message is recorded. Pi's ordinary retry
logic then uses a2-second delay. This is strong evidence that immediate
recovery races the gateway's previous-request cleanup/admission release.
It is separate from initial model corruption. Exact cancellation propagation
and lock-release timing need gateway source/logging to locate the defect.
Do not attribute these429s to vLLM capacity or tune its sequence limit for them.

The first12 bangs occur before agent_2 is born;16 total occur while only one
listed world agent is alive. Nearby10-second server samples for the first12
show only0 or1 running request. This revises the earlier concurrency lead:
concurrent world agents are not necessary for this observed pattern. Short
unobserved overlap, other clients and lingering cancelled requests are not
excluded. Repeated single-agent history with reasoning is now a relevant
reproducer; do not insist on four-way concurrency to reproduce it.

First concrete incident:
- agent_1 message204e2197, responseId chatcmpl-bdde136af4bb9bc6.
- Request/message start2026-09-09T23:54:51.485Z.
- Aborted message recorded23:54:55.533Z, thinking bang offset59 characters.
- Recovery attempt starts23:54:55.536Z and gets local429.
- Adjacent engine samples23:54:48 and23:54:58 show running1 then0,
  KV8.8 then0 percent, waiting0.
- Previous successful response reports62400 cacheRead tokens and2116
  noncached input tokens. This proves nearby reuse, not this bad request's
  hit length, which was lost on abort.

Pi sessions DO retain responseId fields even though gateway request IDs were
not captured. These are preserved in the audit for future correlation; current
vLLM access logs do not contain them. events.json uses start timestamps, not
abort times. The audit also records session persistence timestamps as a close
client-side completion observation, not an exact network arrival timestamp.
Recovery records say cacheSalt=true but do not include actual wire salts.
Therefore effective salt rotation through the gateway remains unverified.

VERDICT -> Two distinct problems are established at the observed-behavior
level: corrupted model output and rejected immediate recovery attempts.
The cache-boundary CPU defect remains a plausible generation mechanism,
but these transcripts do not prove it caused the first bad token. FP8 KV,
MTP/recurrent state and long reasoning remain unisolated. The observed
single-agent failures weaken a concurrency-required explanation. No NaN,
particular kernel, FP8 regression, or sufficient patch is established.

Concrete next steps:
1. Gateway owner should trace client disconnect -> upstream cancellation ->
   request finally/slot release. A recovery must wait for confirmed cleanup
   before readmission, while preserving the one-request rule. Do not blindly
   clear the lock while old work remains active. Include request/response IDs,
   upstream end reason and a hash of the effective forwarded cache salt.
2. Capture the actual first-failure payload including system prompt/template,
   sampling parameters and salt; session JSONL alone is not an exact wire
   replay. Keep sensitive payloads outside Git.
3. Replay the single-agent growing history with reasoning in a bounded leased
   experiment. Separate MTP-off, prefix-off and FP16-KV controls one factor at
   a time; no automatic production restart or unqualified P2P launch.
4. Check corruption beyond bangs: this bundle shows broader degeneration
   before the guard fires, so a bang count is not a complete quality metric.

No gateway fix was deployed: its source and runtime are not in the bundle.
The current serving runtime remains unchanged.
