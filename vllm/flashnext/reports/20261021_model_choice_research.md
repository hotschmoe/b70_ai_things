# Repository agent choice: 27B AutoRound INT4 vs Flash-Next IQ3_XXS

Research date: 2026-10-21. Read-only research; no model run, server change,
build, commit, or push. Numeric audit and compact public-source extract:
[supporting JSON](20261021_model_choice_research_evidence.json).

I would hand a substantial feature or an unfamiliar repository bug to
Flash-Next IQ3_XXS on the qualified Strata recipe. It has the stronger
direct local task results, and current public Terminal-Bench evidence
supports the model-level preference. I would keep the 27B for small,
well-specified fixes where its lower latency wins, and as an independent
reviewer. Confidence is moderate: there is no trustworthy matched
Terminal-Bench result for these exact two quantized serving recipes, and
no new C4 agent-task quality/time comparison.

The claim that Flash-Next universally needs fewer reasoning tokens is
unverified. Local longer coding tasks support fewer total completion
tokens, but simple troubleshooting and public harder-task measurements
provide counterexamples. Time to a correct, verified result is the metric
that should decide the deployment.

The identities are literal, verified against local locks and public cards:

- A: `Qwen/Qwen3.8-27B`, quantized by
  [devan-carlin](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound),
  revision `bce40cacab0a4535b92fb3d57615c2bea9adf3d1`, AutoRound W4A16
  group128. Local serving uses the GPTQ-routing relabel
  `int4-autoround-gptq-relabel-r212`, not a new GPTQ quantization.
- B: `Qwen/Qwen3.8-Flash-Next`, unpruned ISTA GSQ-RCO GGUF IQ3_XXS,
  revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The
  [local lock](../../../strata/flash-next/model-lock-ista-iq3.json) fixes both
  shards. This is mixed tensor precision with a separate IQ4_NL n-gram
  shard; similarly named Unsloth, Swift, Coder/pruned, IQ3_S, and NVFP4
  builds are different artifacts.

The exact27B publisher card offers handpicked math/logic/code answer
parity and throughput47.8 vs BF1630.2tok/s on four B70s, while warning
that INT4 reasoning is more verbose. It does not supply a standard
Terminal-Bench/agentic quality score for this artifact. Neither those
speed numbers nor another publisher's AutoRound result qualify the
present two-card recipe.

The benchmark evidence falls into distinct levels:

| Evidence, date | 27B | Flash-Next | What it establishes |
|---|---:|---:|---|
| Exact local quants, abfast langs, Oct14 | 15/19 | 18/19 | Pi1.2.0, C2, Rust/Dart/Go commit-replay tasks; B wins3/A wins0 |
| Exact local quants, abfast sysadmin, Oct14 | 18/24 | 23/24 | Same agent/policy; troubleshooting, B wins5/A wins0 |
| Exact Flash IQ3_XXS, publisher LCBv6 | Not measured in that protocol | 86.29 vs same-protocol BF16 87.43 | Small loss on code generation; no agentic-quality guarantee |
| Public AA Terminal-Bench2.1, retrieved Oct21 | 79.78% xhigh | 86.14% | Model/API prior; serving quant not disclosed in extracted model data |
| Public AA Terminal-Bench4.0, retrieved Oct21 | 5.56% xhigh | 25.25% | Different, harder benchmark and harness; model/API prior |
| Qwen publisher DeepSWE1.1 | 42.2 | 58.7 | Full-model prior; best across two harnesses for Flash |
| Qwen publisher SWE-bench Pro | 61.7 | 62.5 | Much smaller separation on this task distribution |
| NVIDIA Flash Terminal-Bench2.1, Aug31 card | Not tested | FP8 83.3 vs NVFP4 82.9 | Quantization can preserve terminal ability; not evidence for IQ3_XXS |

The exact-quant local rows come from the four `meta.json` and per-task
`result.json` trees listed in the supporting JSON. Both arms used thinking
`xhigh`, context131072, output cap32768, parallel2, same task order, and
hidden tests after the agent phase. Langs cap2700s, sysadmin cap1800s.
A was vLLM TP2/MTP3/calibrated FP8 KV; B was stock Strata0.1.43.1,
two131072 slots/MTP4/parking. End identity and health were recorded as OK,
with no endpoint errors. This compares deployed recipes, not isolated
weight intelligence. Each arm ran once; there was no order randomization.

The earlier base-tier 18/19 tie used Pi0.85.1, despite originally being
reported as1.2.0. It is a matched comparison but must not be merged as a
Pi1.2.0 replicate. See the
[amended brief](../../../docs/20261013_flashnext_iq3_vs_27b_ab_brief.md).
The local Terminal-Bench calibration did not establish a useful head-to-head
score: harness problems, length stops, and time limits contaminated it.
Zeroes from that exercise are not evidence that both models lack terminal
ability.

The quant publisher reports IQ3_XXS GPQA91.41 vs its BF16 baseline91.92,
and AIME25 100 vs100 alongside the LCB numbers. Its reported 99.4% recovery
is the average of three short-form reasoning/generation scores, not
Terminal-Bench recovery. Evaluation seed, exact thinking policy, and
agentic trajectory details are not supplied for an exact local recipe
comparison. The locally downloaded locked README has the same results.
[ISTA results](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF#results).

Public AA scores were extracted from its actual model-page JSON, not an
aggregator. The Flash page exceeded the web reader's4MB limit, so a direct
HTTP read decoded its Next.js payload and retained exact model IDs, fields,
URL, retrieval time, and HTML hash in the supporting JSON. The same page
contains both model entries. Individual serving precision/provider and
Flash effort level were not specified there. Its TB4 output tokens per
task are235646 Flash vs148327 27B: Flash uses58.9% more, while passing more
tasks. These are not successful-task-only counts, and AA time-per-task
fields are not local measured task wall times.
[Flash model](https://artificialanalysis.ai/models/qwen3-8-flash-next),
[27B model](https://artificialanalysis.ai/models/qwen3-8-27b).

AA's current methodology separates TB2.1:89 tasks, Terminus2, E2B,
pass@1 over three repeats,250 episodes, at least7200s per task; from
TB4:66 tasks, mini-SWE-agent, three repeats,500 steps, upstream timeouts,
full transcript without compaction. The public board now defaults to4.0.
Neither is the pre-registered local TB3-AB24/Pi test. These scores should
never share one ranking column or be treated as our deployment's expected
pass rate.
[AA methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking),
[current Terminal-Bench](https://www.tbench.ai/).

Qwen's DeepSWE comparison selects the best result across Claude Code and
mini-SWE-agent; Flash does best with mini-SWE-agent. SWE-Pro uses Claude
Code, temp1/top_p0.95/context256K and a refined task set. The large DeepSWE
lead therefore supplies a directional prior rather than a matched Pi
prediction. No task-completion token or wall-time comparison is supplied.
[Qwen Flash card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#benchmark-results),
[Qwen27B card](https://huggingface.co/Qwen/Qwen3.8-27B#benchmark-results).
NVIDIA's different-quant TB2.1 comparison uses xhigh, temp1/top_p0.95,
max_new_tokens131072; its card does not identify enough harness detail to
merge with AA or the local Pi runs.
[NVIDIA evaluation](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4#evaluation).

An independent Sep1 BF1627B reproduction supplies another check on the
publisher's73% TB2.1: raw61/89=68.54%, audited64/88=72.73% after task/infra
corrections. It used SGLang, Harbor0.21, Terminus2,16A100s, context262144,
output65536, temp1/top_p0.95 and a6x agent timeout. Its7h23 batch and8.27M
output tokens are not a matched Flash timing. The audit shows why raw and
corrected scores and infrastructure errors must remain visible.
[Reproduction](https://blog.konakona.dev/en/posts/qwen38-terminal-bench-21-reproduction/).

The raw token/time audit changes how the local advantage should be stated:

| Local Pi1.2.0 metric, A=27B / B=Flash | A | B | Interpretation |
|---|---:|---:|---|
| Langs all19 completion tokens | 818277 | 587442 | B28.2% fewer; includes failed runs |
| Langs same15 both-passed completion tokens | 578862 | 411692 | B28.9% fewer on identical solved tasks |
| Langs both-passed median agent wall | 1134.5s | 991.0s | B12.6% faster, conditional on both passing |
| Sysadmin all24 completion tokens | 150325 | 100036 | B33.5% fewer; failures drive part of saving |
| Sysadmin same18 both-passed completion tokens | 43083 | 61109 | B41.8% MORE tokens |
| Sysadmin both-passed median agent wall | 72.65s | 116.85s | B60.8% slower on common successes |
| Langs sum(all agent wall)/passes | 1757.9s | 1392.9s | Failure-inclusive campaign effort per success |
| Sysadmin sum(all agent wall)/passes | 226.2s | 172.6s | Same metric, different task distribution |

The last two metrics charge failed-task time to completed work. They are
not serial latency forecasts: agent times include shared endpoint waits,
tool execution and builds, and the experiment ran C2. Observed whole-run
wall/pass was916.0 vs717.8s langs and155.0 vs94.7s sysadmin. No retry or
human review time is included. Success-only medians select survivors;
all-task tokens mix efficiency with avoiding expensive failures. Both
views are needed.

`run_arm.py` counts assistant `usage.output` as completion tokens. Sample
traces contain thinking blocks but `usage.reasoning=0`; the runner does
not separately tokenize reasoning. Consequently, these totals do not
prove that the reasoning pass itself is shorter. Langs had2 A timeouts
and3 B timeouts, with some capped trajectories still passing hidden tests;
test success and explicit agent completion are separate facts. The 27B
markdown task stopped on length after4 turns and33122 aggregate output
tokens without changing a file, despite32768 being a per-turn cap.

One local preservation failure is narrowly verified: on
`scheduled-task-xml-fix`, A changed the existing `InteractiveToken` logon
mode to `S4U`. Functional checks passed, but
`safe_logon_interactive=false`; B preserved it and passed all checks.
A retained the user account and least-privilege mode. This supports
reviewing unnecessary configuration changes, not a general claim that
the model escalates privileges or destroys data.

Public experience is mixed and heavily dependent on engine and harness:

- A substantive feature case from deepu105, posted Oct20/21, compares
  twelve public PRs implementing LlamaStash KV save/restore. Flash used
  Halogen4.16bpw on a Strix Halo laptop. Core end-to-end tests passed, but
  reviews found edge-case/test/doc gaps; medium was faster than xhigh
  without a clear Flash quality loss. Reviewers were models, it is one
  task, and there is no27B arm. It supports feasibility and effort tuning,
  not our exact quant or universal autonomy.
  [Post and PR links](https://www.reddit.com/r/LocalLLaMA/comments/1xbipsl/qwen38flashnext_vs_opus_55_sixway_benchmark_on_a/).
- In the Sep30/Oct2026 IQ3 discussion, Jagerius reports better village-sim
  output than several27B variants using a Swift IQ3_XXS build/Hermes;
  Most-Trainer-8876 favors IQ3_S for client code, while IntelVEVO reports
  IQ3_XXS tool-call failures. DiscipleofDeceit666 describes Flash as the
  stronger planner and27B as the stronger implementer at different quants.
  These are useful hypotheses, not controlled evidence; artifacts and
  sampling are inconsistent or missing.
  [Discussion](https://www.reddit.com/r/LocalLLaMA/comments/1wuark9/how_smart_is_the_iq3_family_of_qwen_38_flash_next/).
- A Sep2 first-person local MLX4bit Flash/Terminus2 run graded only70/89
  tasks, passing36;13 failures were timeouts and four server wedges
  required restarts. Its reported51.4% is conditional on graded tasks,
  not a full89-task score. Hardware/backend limits can dominate practical
  trust despite favorable full-model benchmarks.
  [Overnight report](https://blog.brainosi.com/en/2026-09-02-qwen38-terminal-bench-overnight/).
- There is a public ISTA-linked IQ3_XXS Airbench/DSH report from Oct6:
  33/49 passed,15 unanswered,1h39m. It is a partial, different benchmark
  with no matched27B and no pinned model revision. It adds little weight
  to a repository-handoff decision.
  [Report](https://airbench.ai/checkup/80707d24-3c1c-452e-8efb-399727e5bf8f/report).

For substantial features, my choice is Flash-Next/Strata because local
repository task success improved without sacrificing common-success time.
For unfamiliar repository diagnosis I would also start there, while
recognizing that sysadmin tests are not repository-debugging tests. For
an obvious one-file edit or known procedural fix, the27B can be the faster
first attempt. All unreviewed repository changes still need the project's
ordinary tests and review; published scores do not measure every edge
case a feature request requires.

The fresh [C4 serving comparison](20261021_c4_compare.md) supplies current
capacity, not task intelligence: Strata Flash188.5 aggregate steady TG,
vLLM27B161.4, vLLM Flash121.4tok/s. On cold four20K+512 requests, walls were
52.74/122.29/51.92s respectively. Thinking was off and output fixed512;
Strata steady TG needed a cached replay. These numbers cannot simply
multiply Oct14 agent times to predict present C4 task completion.

A minimal next decision test can reuse the existing fixed TB3-AB24 and
external-endpoint Harbor/Pi1.2 adapter. First complete two unscored
trajectory/payload canaries outside the24 on both models. Explicitly set
`MAX_TOKENS=32768`: the external runner's default still16384. Keep the
same131072 context, thinking policy, agent, tools, timeout and concurrency;
start C1 for quality/latency, then repeat representative tasks at C4 for
the actual deployment. Preserve the fixed task selection, alternate recipe
order, report failures and length/time caps separately, and rank passes
per hour plus paired time/tokens on common successes. Do not score a
broken endpoint/adapter as failed intelligence. A newer TB4 experiment is
useful, but requires a separately frozen dataset/harness and new label;
do not silently replace the existing TB3 subset.

CONFIG -> Exact local artifact locks and four historical Pi1.2 C2 runs;
current public primary model cards, AA model JSON/methodology and
attributable experience reports. COMMAND -> Read-only `rg`, JSON sums and
paired intersections; HTTPS fetch and JSON decode of AA's embedded model
objects. RESULT -> Exact local41/43 vs33/43; public model priors favor
Flash, exact-quant Terminal-Bench comparison missing; token efficiency is
task-dependent. VERDICT -> Prefer qualified Strata Flash IQ3_XXS for
substantial repository handoffs, keep27B for quick bounded work/fallback,
and measure time-to-verified-success before choosing the current C4
deployment or changing reasoning effort.
