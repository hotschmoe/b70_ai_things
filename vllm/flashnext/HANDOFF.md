# Flash-Next on vLLM: session handoff (2026-10-21)

Latest measured checkpoint: the final section below and reports/3D1_decode.md.
The retained route/Q8F pair closes cleanly with shutdown30/Docker45; canonical
63.534/63.861tok/s, strict repeats still fail. Strata DD remains down.

Written after an office power outage, around 15:38 UTC on 2026-10-21. The
host rebooted. All agents were killed mid-run: grok, Codex, Opus, Fable.
Nothing was lost on disk except /tmp, which is tmpfs, so any agent scripts
kept under /tmp are gone.

Read this file first. Then read BRIEF.md (the rules for agents) and the
last ~40 entries of JOURNAL.md.

## 1. Goal and standing user direction

- Serve Qwen3.8-Flash-Next (ISTA GSQ-RCO IQ3_XXS GGUF) on vLLM across two
  Intel Arc B70 cards. The aim is to beat our own Strata stack, which is
  the current daily driver, on the same model and cards.
- Latency is the top metric. Motto: make it work, make it right, make it
  fast. Mindset: there is no software moat; build any kernel or feature
  that is missing.
- Targets: prefill 2000+ tok/s, decode 150+ tok/s per stream, 2x200K
  context, KV evicted to host RAM and restored without recompute.
- The localmaxxing leaderboard ranks only NON-speculative runs. No-MTP
  decode is therefore the headline number.
- Keep the vLLM fork thin and rebaseable on upstream.
- The Qwen3.6-35B-A3B GGUF is a proxy for testing single pieces.
  Flash-Next is the real target.
- Ornith-1.5-122B is PARKED. Revisit it only if it beats Flash-Next on
  terminal benchmarks AND survives 3-bit quantization. See
  reports/2S1_ornith122b_fit.md.
- The user approved localmaxxing no-spec submissions for three configs:
  Flash-Next, 35B Q4_K_M and 35B UD-IQ3_XXS. MTP submissions are NOT
  approved yet. Ask the user before any other external submission.

## 2. Bars to beat (Strata, same model, 2x B70)

| metric | Strata |
|---|---|
| decode C1 no-spec | 66.2 / 65.9 tok/s on lmx reasoning-v1 / code-v1 (~15.1 ms) |
| decode C1 with MTP | 141 tok/s (7.09 ms) |
| decode C2 with MTP | 92 tok/s per stream (10.87 ms) |
| prefill | ~2200 tok/s |

## 3. Where vLLM stands (all measured; evidence in JOURNAL and reports)

| item | vLLM now | notes |
|---|---|---|
| Flash-Next coherent serve | yes | greedy 17*19 -> 323 and 23*29 -> 667 match Strata |
| Flash-Next decode C1 no-spec, 64-token bench | 15.20 ms (65.8 tok/s) | merged recipe, max_len 131072 |
| Flash-Next on the lmx suites (512 tokens) | 61.3 / 61.5 tok/s | NOT submitted; trails Strata's 66.2 |
| Flash-Next decode C2 no-spec | 19.5 ms per stream | |
| Flash-Next MTP | k=3 C1 9.19-9.28 ms; k=2 C2 12.89 ms per stream | O5; k=2 on/off parity passed |
| Flash-Next prefill (sidecar) | 8K 2448, 12K 2397, 32K 2349, 128K ~1587 tok/s | F1; 128K pays dense attention growth |
| KV capacity | ~356K tokens on card 0 in the merged recipe | X2: 2x200K bf16 fits |
| MoE decode kernel | 0.53-0.58x Strata R2 layer time | O3; we are faster |
| KV parking to RAM | 200063-token restore in 0.30 s (13.9 GB/s) | Ornith stand-in, not yet Flash-Next |
| 35B proxy, localmaxxing (SUBMITTED, verified) | UD-IQ3_XXS 120.9 / 122.1 tok/s; Q4_K_M 97.0 / 97.2 tok/s | IDs in localmaxxing/submitted-20261021/SUBMITTED.txt |

The layout of record is the EP sidecar. It runs as one process with world
size 1:
- Card 0 runs the trunk, the shared expert and about 138 experts per layer.
- Card 1 holds about 374 experts per layer (plugin b70_ep_xpu).

The DP2/EP2 layout is retired.

## 4. Recipes (start here)

- Merged best recipe: decode from O4 plus prefill from F1.
  - harness/o4_recipe.env
  - harness/f1_prefill_delta.env
  - harness/f1_merged_serve.sh (max_model_len 131072, chunk 12288)
  - The full flag list is in reports/2O4_flashnext_decode.md and
    reports/2F1_prefill.md.
- Decode-only recipe (O4 s4): harness/o4_serve.sh, port 18147. It waits for
  its own alias and for free RAM.
- MTP: harness/o5_window.sh and reports/2O5_mtp.md.
- 35B proxy: reports/1X5_proxy35b.md (X5 recipes) and o3/serve_35b.sh.
- localmaxxing tooling:
  - localmaxxing/ (lmx_vllm.py, run_dry.sh, validate_dry.sh)
  - L1 scripts in localmaxxing/l1/
  - report reports/2L1_localmaxxing.md
- Promotion gate: harness/promo_gate.sh, from X4 (reports/1X4_gate_lmx.md).
  It checks identity, the 323/667 gates, exact repeat, and 8K/32K needles.

Costs:
- The pinned PLE table uses 26.8 GiB of host RAM per Flash-Next process and
  takes about 90-120 s to load.
- Run only one Flash-Next serve at a time.

## 5. Code state (IMPORTANT)

- Campaign repo: master at 2d5ff2a1 plus this handoff commit. About 25
  campaign files written after 2d5ff2a1 were committed with the handoff.
- vLLM fork: /mnt/vm_8tb/b70/build/vllm-flashnext/vllm, branch
  b70/flashnext, tip 0a6170da8c.
  - It carries about 20 UNCOMMITTED files from many agents.
  - The 16:00 snapshot ref is b70/wip-snapshot-20261021-1600. It is a
    stash-style commit made with `git stash create`; the working tree was
    left untouched.
  - Earlier snapshots: b70/wip-snapshot-20261020-1940 and
    b70/wip-snapshot-20261021-0200.
  - Frozen localmaxxing engine: branch b70/lmx-20261021, commit cfd2b6ff,
    worktree /mnt/vm_8tb/b70/build/vllm-flashnext/vllm-lmx-20261021.
  - TODO: split the WIP into small, upstream-rebaseable commits per
    feature. All new behaviour sits behind default-off B70_* flags.
- Plugins live in vllm/flashnext/plugins:
  - b70_gguf_xpu: the current _C.so / _C.graph.so is O3's o3g build, sha
    38f7176a. O4 has its own variants under build/vllm-flashnext/o4/.
  - b70_ep_xpu: the F1 build is f1c, sha 249cf94e.
  - b70_qsa_xpu, b70_lowbit_xpu, b70_kvpark, b70_o4_xpu.
- The venv, wheel and build outputs are under /mnt/vm_8tb/b70/build/vllm-flashnext/.
- The vLLM base is upstream main 5fd62688 (0.32.1rc1.dev94), XPU build.

## 6. Open threads, in priority order

1. Beat 66.2 tok/s on the lmx suites (O4 owned this).
   - Per-token time grows inside the decode graph by ~1.6 us per KV token
     per step (15.3 -> 16.2 ms over 512 tokens). The FA split count is NOT
     the cause (JOURNAL O4 04:52).
   - O4 kept route + fused Q8 HC, which saves 0.54 ms (lmx-shape mean
     15.70 ms).
   - Next was "Ld": a unitrace over one 512-token lmx request to find the
     kernel(s) that grow with KV length.
   - Other queued levers: EP dispatch/top-k fusion (0.2-0.3 ms), streamed
     HC GEMVs (~0.5 ms), the remaining PLE kernels.
   - Once a dry run on the merged recipe beats 66.2, submit Flash-Next.
     Use the L1 procedure: freeze the engine, run the promo gate, dry-run,
     submit.
2. Nondeterminism (O5, JOURNAL 05:00). The GDN b|a BF16 F.linear in
   prefill returns different bits for the same input. It runs on oneDNN at
   M=20; decode at T<=8 uses bf_gemv. The fix is a batch-invariant GEMV/GEMM
   for that projection, like O4's bf_gemv. This blocks strict exact-repeat
   gates. O5 also found that dynamic MTP k (k=3 at C1, k=2 at C2) gives the
   best of both.
3. Wire QSA into a real serve (X6, JOURNAL 04:33-05:23).
   - The QSA paged KV spec now binds.
   - The component graph gate passed on card 1 (X6_QSA_RUNNER_GRAPH_OK).
   - Next: a one-card synthetic integration test, then a full QSA serve
     compared against Strata long-context references.
   - Dense FA matches QSA only up to ~2051 tokens, so this matters for
     200K fidelity. It also matters for 128K prefill, since dense attention
     grows from 1.9 to 7.9 s per chunk.
   - Generating Strata references needs hotschmoe-dd or Strata on the GPUs.
     Coordinate this; agents must not start the daily driver themselves.
4. Card-1 prefill experts (F1). They take 36-42 ms per layer against an
   estimated 10-15 ms floor. The stage-profile build f1/_C.f1bucket2.so is
   ready. Its run q2 was interrupted by the outage.
5. KV parking on Flash-Next (G4 spec, reports/2O2 and 1G4). It is proven
   on Ornith but not yet on the Flash-Next sidecar. Budget host RAM:
   26.8 GiB PLE pin plus a 32 GiB parking pool.
6. MTP to beat 141 tok/s. Today 9.2 ms is about 108 tok/s. The host gap is
   ~1 ms per step; the verify graph at T=3 is ~19.8 ms. MTP submission
   needs user approval.
7. Housekeeping:
   - Split the fork WIP into commits (section 5).
   - Retire agent-specific .so copies once the recipes pin the final builds.

## 7. Agents at the outage (all dead now; none need killing)

| agent | model | owned | last state |
|---|---|---|---|
| O4 | Opus | no-MTP decode | lmx-shape "Ld" unitrace queued |
| O5 | Opus | MTP | nondeterminism located; dynamic k measured |
| F1 | Fable | prefill | q2 card-1 stage profile starting |
| X6 | Codex | QSA serve | synthetic integration test prepared |
| G2b | grok | auditor | pass 34 |
| L1 | Opus | localmaxxing | standing by for Flash-Next |

Finished earlier: O3 (MoE kernels), X1-X5, C7b, H1e, all of wave 2.

The Opus/Fable agents and their transcripts belong to the old session and
cannot be resumed from a new one. Relaunch fresh agents with a pointer to
their report and to the JOURNAL. Launchers:
- vllm/flashnext/grok-agent.sh NAME PROMPT. It now passes
  --background-wait-timeout 86400; the 600 s default killed H1d.
- vllm/flashnext/codex-agent.sh NAME PROMPT. It runs gpt-6.1-sol at high
  effort with sandbox bypass.
- Prompts live in vllm/flashnext/prompts/.

Lessons learned:
- Give one owner per area. Write orchestrator directives as ORCHESTRATOR
  entries in the JOURNAL, because Codex agents cannot take messages
  mid-run.
- Set an explicit card priority. Last order: L1 > O4 > O5 > F1 > X6.
- Check claims against raw JSON/logs. The G2b auditor helps here.
- Never edit a window script while its shell is still running it.
- Codex can briefly hold .git/index.lock. Retry the commit.
- `pkill -f PATTERN` can match your own shell. Kill by pid.

## 8. First steps for the new session

1. hotschmoe-dd (Strata) and its frontdoor AUTO-STARTED after the reboot
   and are active. Before any GPU research, run
   `systemctl --user stop hotschmoe-dd`.
2. Run bin/xpu-health and bin/xpu-collective-health first. This is a fresh
   boot after a power loss.
3. Requalify the merged recipe: run harness/f1_merged_serve.sh, then the
   gates and the bench. Confirm C1 is ~15.2 ms and 8K prefill ~2450 tok/s.
4. Relaunch owners for threads 1-4 in section 6, Opus or Fable for the
   critical ones. Point each at its report and the last JOURNAL entries.
5. Restart hotschmoe-dd when the GPU session ends.

## 9. Resumed session (2026-10-21, after 16:00 UTC)

The user resumed this campaign with Sol 6.1 agents and explicitly authorized
startup serving down for research. AGENTS.md now reflects active vLLM work.
Owners are D1 decode, N1 GDN determinism, and Q1 QSA. Their current reports
are reports/3D1_decode.md, reports/3N1_determinism.md, reports/3Q1_qsa.md;
orchestrator recovery evidence is reports/3R1_restart.md. Consult the latest
campaign JOURNAL entries for live GPU ownership and subsequent results.

Post-outage native and compiled collective health passed. The unchanged
merged recipe reproduced C1 15.17 ms and C2 about 19.6 ms, passed identity,
arithmetic and 8K/32K needles, and failed strict exact repeat. No promotion.

Several runs marked pending above actually finished before the outage:
Ld's trace and F1 q2 stage profile completed with post-health; the first
QSA tiny engine test failed, then worker-config binding was repaired at
05:28. The resumed Q1 tiny test passes all 12 real pooled cache bindings,
FULL graph initialization and C1/C2 generation. Full-model QSA remains
unqualified. Read the new reports before repeating any recovered work.

The merged serve harness now keeps hotschmoe-dd first, defaults to maxlen
131072, supports optional promotion gates and an integrated pre-teardown
callback, and propagates failed checks. For canonical suites, use the
metadata shim (harness/d1_api.py); ordinary vllm serve does not expose the
collector's required /server_info endpoint on this pinned version.

Resumed matched control: 61.642817 / 61.800831 tok/s on the two 512-token
suites, no MTP, complete health/source/teardown evidence, strict-repeat
FAIL. D1's compact-plan16 component is numerically valid but its corrected
full serve (r2) failed with level_zero OUT_OF_RESOURCES40 before canonical
measurement. Keep it disabled; the PLE traceback is a surfacing site, not
a proven cause. Post-health recovered cleanly. No new submission or speed
promotion. N1 full-model validation uses the original attention path.

N1 original-attention full validation subsequently stalled after prefill
and required a forced EngineCore shutdown; strict C2 repeat also failed.
Keep N1 off as well. Its primitive remains valid, not a serving fix.
Both-card health and source checks passed after teardown. A bounded low-VRAM
replay/PLE stress test did not reproduce either full-model liveness issue.
Next liveness work needs loaded-context/EP evidence, not a blind buffer fix.

P1 selected-expert dequant passes20 exact component cases and reduces
measured mode0 gather+DQ latency6.9-23.3 percent. Reproducible sources are
plugins/b70_p1_xpu and harness/p1_*. It has no serving hook yet.
Canonical no-draft Strata4K/16K/32K references are now on disk; first full
QSA serving comparison is the remaining stage. See the new reports.

The first full QSA attempt now completed. All 12 real pooled bindings,
identity, arithmetic, graph initialization, graceful EngineCore teardown,
unchanged sources and both-card post-health passed. Exact continuation
and repeat gates failed: at 4K both solo outputs match Strata, but one
C2 output differs; at 16K even solo repeats differ. At 32K all four vLLM
texts agree with each other but differ from Strata. These are 64-token
reasoning prefixes, not completed-answer quality scores or token-ID parity.
A large 32K C2 stream timing gap also prevents a stability claim.

Next bounded work, after reviewing reports/3A1_audit.md and 3Q1_qsa.md:
1. Run these same fixtures with QSA disabled before attributing drift to
   QSA. Preserve per-output comparisons and individual arrival times.
2. Localize loaded-context/EP liveness and repeatability on the original
   recipe. Keep N1 and D1 plan16 disabled pending concrete evidence.
3. Integrate P1 selected-expert dequant behind a default-off flag, then
   measure a matched serving control and treatment with full lifecycle
   checks. Component timing alone is not a serving improvement.
No new speed promotion or external submission was made this session.

User steering at 17:19 UTC: keep the GPUs on the vLLM testing campaign;
do not restore Strata between tests. A daily-driver restoration had just
started and was stopped again before readiness. Continue the matched
QSA-disabled control, P1 serving integration and loaded-context liveness
localization under coordinated GPU leases.

The QSA-disabled original-FA long-fixture control subsequently completed
with normal teardown, unchanged sources and both-card post-health. It
also fails solo exact repeats and 16K/32K Strata text parity. This prevents
attributing those failures uniquely to QSA. Individual stream arrivals
now show a long C2 gap while the other stream awaits first content; the
cause remains unproven. See 3Q1_qsa.md for the per-output comparison.

P1's real-weight bucket integration gate passes44 bit-exact outputs:
original repeat, overlay flag0 and overlay flag1, including T12283.
Body51s, normal exit, unchanged sources/SOs, native pre/post HEALTHY.
The same-binary full-serving flag0/1 pair is now queued/running; keep
all campaign harness/plugin source frozen until BOTH arms close.

P1's matched full-serving pair has now PASSED and released both GPUs.
Observed first-content TTFT, warmup excluded and three samples per size:

| Actual prompt tokens | Flag0 ms | Flag1 ms | Reduction |
|---:|---:|---:|---:|
| 8189 | 3341.403 | 3198.826 | 4.27% |
| 12285 | 4876.350 | 4678.923 | 4.05% |
| 32765 | 13949.494 | 13509.475 | 3.15% |

Both arms pass identity, content/usage checks, long arithmetic, actual
no-speculation/prefix-off configuration, normal engine teardown and native
health. All 431 source files, SOs, fork snapshots and fixture identities
match; only selected-dequant flag differs. Short decode is effectively
unchanged. This is one fixed-order diagnostic pair; both arms still have
nonrepeatable continuations. Keep P1 opt-in, no daily-driver promotion.
Sources: plugins/b70_p1_xpu and harness/p1_*; report3P1_prefill.md.

Next: instrumented N1-original reproduction via d1_n1_diagnostic.sh, then
the isolated QSA top-k/whole-chain profile via q1_topk/window.sh. Both
source ports are ready; shared harness/plugins/registry stay frozen during
each active serve. DD remains down, as explicitly directed by the user.

The instrumented N1 run failed earlier than the historical stall point:
EngineCore SIGSEGV during the promotion repeat, last scheduled batch
T6=[1 decode,5 prefill]. Saved /proc wait status11 proves signal11;
container exit143 is later cleanup. No native PC or cause is available.
Hooks were installed/armed without metadata errors, but no >=8s phase
witness occurred; F1 and the historical callback were never reached.
Native post-health and all445 source/SO/private checks pass. The original
classifier mislabeled shell-body RC0 as successful requests; preserve the
raw output and use the corrected offline classification. N1 stays off.

QSA profile-r1 was rejected before execution by an invalid unitrace
session label. After the label-only fix, profile-r2 passes63 adversarial
key cases, six M1/M2/context shapes, canaries, changed-input graph replay,
normal exit/source checks and native pre/post health. At32K M1, top-k is
about4.02ms of a4.64ms chain (~87%). Replacing its reduction tree saves
only about4-5us versus an identically built copied tree. Native-vs-copy
differences are build-context confounded; no serving gain is established.
Next QSA candidate is exact parallel selection using upstream Torch int64
topk/sort, preserving packed-key ties, dynamic width and ascending IDs.

The Torch selector component now passes exact IDs, canaries, gated outputs
and changed-input graph replay at six real shapes. At32K M1 its chain is
about1.50ms versus4.66ms native, but it regresses the4K case. A first
fused pack/serial-compaction variant also passes correctness but loses
the speed gain to its serial scan. Preserve both measured variants under
harness/q1_torch and q1_fused; neither has a measured serving gain.
The next component candidate uses parallel bitonic ordering of selected
IDs, under a separate q1_fused_r2 path. See reports/3Q1_qsa.md.

The N1 native-capture attempt failed BEFORE READY and GDB attachment:
kernel card0 CCS memory CATerror18, followed by DEVICE_LOST20 during EP
initialization. It provides no native PC for the earlier SIGSEGV. The
container is down; all471 source files, N1 SO and debug-image guards pass,
and both-card post-health passes. The Xe dump requires root authentication
and could not be copied; kernel events are saved. Root then ran bin/xe-reset:
rebind recovered both cards, per-card and compiled two-rank collective
health passed, boot unchanged,66s total. Keep N1 off; no further blind retry.

Before full QSA selector integration, fix two CPU-review findings: keep
prefill on the native selector to avoid retaining288MiB per transient
prefill cache entry, and make the progress watchdog tolerate the previously
observed139s healthy32K fixture phase. Use a bounded, decode-owned cache;
do not evict workspaces referenced by captured graphs. Independent review
is in reports/3A1_audit.md. DD stays down throughout this campaign.

The fresh native/r2 QSA serving pair now closes operationally, with normal
engine teardown, both-card HEALTHY, no new kernel faults and unchanged
486-file source/SO/fixture identities. Actual r2 activation is12 persistent
decode owners, B4/66048 score columns/cap2051, warmed before capture.
Configured65536/chunk12288/4/no-spec/prefix-off is matched. Realized auto KV
capacity changes234->233 blocks (567978->565551 tokens), so effective memory
capacity is not identical. This is one fixed-order diagnostic pair.

| C1 median ITL | Native ms | R2 ms |
|---|---:|---:|
| Short benchmark | 18.15 | 22.75 |
| 4K | 30.173 | 28.780 |
| 16K | 47.574 | 28.828 |
| 32K | 70.833 | 28.918 |

At32K C2 medians improve about76.94->34.64ms, but first-stream p99 remains
about9s and mean387.35->346.68ms. Both arms fail strict repeat/reference
gates; arithmetic and short coherence pass. Native prefill is unchanged,
and full long-client duration only drops216->207s. Keep r2 opt-in: no
universal speed, fidelity, stability or daily-driver qualification.
See reports/3Q1_qsa.md and independent3A1_audit.md for full stream evidence.

Next no-spec work: the retained default lacks both B70_EP_ROUTE and
B70_DECODE_HC_Q8F. Earlier Lc enabled them (plus older FA/HC settings) and
measured a gain; it is not today's canonical61.64tok/s recipe. Current
F1EPc249cf94e already contains the route operation, so retest ONLY those
two flags with original FA, F1 prefill and plan0/N1off before integrating
new HC kernels. A separate M1 subgroup HC component is CPU-built and
queued under root coordination; it has no serving qualification.

HC component probe-r1 then fails before timing: existing_fused differs
from original on a large_gate synthetic case, and its fail-fast assertion
masks the new candidate comparison. Body17s, no leftover container,
card1 post-health HEALTHY/source hashes unchanged. Keep this raw reject.
Pause full route/Q8F serves until a bounded partial-report repair records
all variants independently, mismatch magnitude and affected outputs.
No kernel rebuild, relaxed exactness gate or speed result is authorized
by this failure. Reports/3D1_decode.md owns the subsequent evidence.

The reporting-only probe-r2 completes all nine numerical comparisons:
only one signed-zero encoding differs, original0x0000 (+0) versus both
existing-fused and candidate0x8000 (-0), large_gate/weight1/h386. Saved
CPU storage bytes independently confirm zero magnitude; no FTZ inference.
Every other output bit matches; existing-fused and candidate match all
23040 elements. Strict COMPONENT_FAIL and skipped timing remain intact;
no graph/96-weight/M2 qualification follows. Body16s, source hashes
unchanged, card1 post-health HEALTHY, no container leftover. No kernel
rebuild or relaxed gate. Cause of the zero sign remains unproven.
This does not establish a nonzero arithmetic error or runtime invalidity;
the retained route/Q8F diagnostic serving retest can proceed after its
bounded client and branch-observation CPU checks, without a bitexact claim.

Retained-default control d1-existing-control-r1 now completes all clients
in134.307s: canonical reasoning61.528796/code61.791001 tok/s, warmup+3x512
per suite; identity/math/8K32K retrieval pass, exact repeat fails. Actual
ordinary w8_gemv returns before full capture in the same worker; fused/route
calls are absent. Actual config,502 sources, private helpers/four SOs and
both-card pre/post health pass; kernel log has no new fault/reset.

Operational closure nevertheless FAILS: SIGTERM19:09:51, worker cleanup
done19:09:52, manager force-kills remaining EngineCore at19:09:56. Container
RC0 does not clear ENGINE_ERROR_RC1. No treatment arm was authorized/run.
The effective5s grace comes from vllm/platforms/xpu.py converting config0
to5; source log and live config agree. Worker cleanup is not final process
exit. Next bounded control uses supported --shutdown-timeout30 plus outer
Docker45s to distinguish slow finalization from an exit hang. Preserve the
original reject; no runtime-fault, cause, stability or timeout-fix claim.

Shutdown30 control d1-shutdown30-control-r1 closes operationally:
canonical61.514492/61.836306, all eight warm+measured requests complete,
identity/math/retrieval pass, exact repeats fail. Source507/private16/SO4,
both-card health and kernel checks pass. Host sampler actually captures
EngineCore Z with waitstatus0, no forced kill. About5.6s is the watcher
span, NOT exact SIGTERM-to-exit; request logs have second resolution.
This is one finite grace discrimination, not a universal timeout cure.
Root then grants treatment d1-shutdown30-routeq8f-r1 with identical frozen
sources, clients, libraries and30/45s budgets, adding only route/Q8F flags.
The treatment is running; no result or promotion is yet claimed.

## Latest measured checkpoint (2026-10-21 19:34 UTC)

The final route/Q8F treatment now closes operationally and is independently
audited. Same507 sources, fork snapshot, private serve source, pinned SOs,
fixtures and configured workload match the shutdown30 control. Actual
runtime config and231 GPU blocks/356207 planned KV tokens match; instance
IDs, object repr addresses and small allocator byte recommendations differ.
Both use maxlen131072/chunk12288/4, TP1 EP sidecar, original FA, no spec,
prefix off, shutdown30/Docker45. The only inference change is adding
B70_EP_ROUTE=1 and B70_DECODE_HC_Q8F=1. Actual native route/down_act/
up_collapse returns precede full capture in the same worker; fallback
w8_gemv remains in use where the fused branch does not apply.

Canonical medians, one excluded warmup plus three measured512-token
requests per suite, independently recomputed from raw content arrivals:

| Suite | Control tok/s | Two flags tok/s | Gain | Historical Strata tok/s |
|---|---:|---:|---:|---:|
| reasoning-v1 | 61.514492 | 63.534291 | 3.28% | 66.2 |
| code-v1 | 61.836306 | 63.861133 | 3.27% | 65.9 |

The remaining historical gaps are4.03% reasoning/3.09% code. This is one
fixed-order combined-flag pair, not separate kernel attribution or a fresh
matched Strata run. All16 warm/measured requests have512 usage tokens,
512 content arrivals, correct counters and valid dirty diagnostic payloads.
Identity/math/8K32K retrieval and short coherence pass; exact repeats fail.
HARNESS_RC3 preserves that strict reject while operational_outcome is true.

Both engines are actually observed in host /proc as Z with waitstatus0,
without force kill. Sampler spans are about5.6s, not exact SIGTERM-to-exit.
Both-card pre/post native health, source/private/SO and kernel checks pass.
The prior5s forced-control teardown remains a reject; these two finite
successes do not prove a universal shutdown cure or global stability.

Reproduce only under root GPU coordination with a fresh TAG:

    ARM=routeq8f TAG=fresh bash vllm/flashnext/harness/d1_shutdown_grace/window.sh

This finite research wrapper does not deploy a long-lived server. Default
O4/F1 recipes, DD service and shelf are unchanged. P1 selected dequant and
QSA selectors were measured separately and are NOT combined into this pair.
Historical MTP results have not been requalified after the outage.

Next decision points:
1. Optimize the existing fused HC path with the subgroup candidate only
   after an explicit numerical contract: its original strict component
   gate remains rejected for one signed-zero bit; no timing/graph claim.
2. Measure a combined P1 plus retained fusion recipe and requalify MTP;
   do not assemble a performance claim from separate configurations.
3. Close repeatability/long-C2 tails and qualify actual Flash-Next2x200K
   plus RAM eviction/restore without recompute. These remain open.

No external submission or daily-driver promotion. All research containers
are down and leases released; leave Strata DD down per user campaign mode.


## 2026-10-21 20:26 -- Matched C4 PP/TG comparison complete

User requested the same~20K prompt/output512 test across Strata Flash-Next,
vLLM Flash-Next IQ3_XXS, and the retained27B INT4 recipe. See
reports/20261021_c4_compare.md and .json, harness/c4_compare, and the last
four JOURNAL stages. All original cold batches are identical79192 input/
2048 output tokens, zero cache hits, warmup plus two measured repeats.

Steady C4 TG: Strata188.5,27B161.4,vLLM Flash121.4tok/s. Strata uses an
explicit cached replay supplement because its cold512 scheduling never
overlaps all four decoders. Effective cold PP: Strata1739.5,27B722.4,
vLLM Flash2262.3. Cold walls52.74/122.29/51.92s respectively. Definitions
and raw arrival checks are in the report; do not conflate burst TG with
steady TG or substitute Strata's13s cached wall for its53s cold result.

All finite runs close normally and pass per-card/compiled pair pre/post
health; source SO pins unchanged, no GPU faults observed. Servers and
leases are down/free; DD stays down. No promotion/external submission.
This is MTP-on C4 serving evidence, not a change to the retained canonical
no-spec checkpoint above or a new200K/RAM-restore/quality qualification.
