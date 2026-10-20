# X4 - promotion gate and localmaxxing preparation

The read-only promotion gate and dry-run tooling are implemented. Nine
CPU HTTP/stream failure-injection tests pass. Four real lmx dry-run plans
(reasoning-v1/code-v1, no-spec/MTP2) return planned, canSubmit=false.
The live proxy gate correctly rejects promotion: the second C1 repeat
times out, and both long prompts exceed that server's 4096-token limit.
No localmaxxing measurement, submission, key disclosure, service change,
GPU reset, or commit was performed. No X4 job remains running.

The journal's wave3 directive transfers new MTP work from X1 to O5. X1's
existing implementation/evidence remains in reports/1X1_mtp.md; X4 does
not alter that report or restart the superseded DP2 MTP campaign.

## Promotion gate

Sources: harness/promo_gate.py and harness/promo_gate.sh. The client uses
bench_latency.py HTTP helpers and COHERENCE_PROMPT, and reproduces the
four prompts/settings from x1_compare_greedy.py. It has no lifecycle code
and does not import Torch, touch devices, stop servers, or take a second
GPU lease against a server that already holds one.

Checks, with explicit boolean pass in one ASCII JSON:

- /v1/models: first ID hotschmoe-dev or hotschmoe-dd; require a secondary
  research alias registered in evals/configs/models.yaml. --expect-alias
  pins the intended experiment. Identity is checked again at the end.
  Compare ordered id/root/parent/owned_by/max_model_len fields. vLLM's
  ModelCard.created is freshly generated on every GET (fork
  vllm/entrypoints/serve/engine/protocol.py:108); it is retained as raw
  evidence but cannot be treated as a reload. Root/context changes fail.
- Greedy chat, thinking disabled: Return only the integer: 17 * 19. and
  23 * 29.; require stripped strings 323 and 667, without explanations.
- proxy35b only: raw capital-of-France prompt, 16 forced tokens, byte
  exact C4 text from reports/1C4_e2e_gguf.md:27. Flash-Next does not
  inherit a different model's golden Paris continuation.
- Four raw prompts x 96 forced tokens, temperature=0, seed=0,
  ignore_eos=true, return_token_ids=true. Two complete C1 passes,
  then two complete C2 passes, each containing two synchronized pairs.
  Every result must match its first C1 result, including across C1/C2.
  Require all 96 token IDs; no missing-ID/text-only fallback and no EOS
  truncation exception. IDs are serialized uint32 little endian and
  compared exactly. UTF-8 text must also match. Save IDs, text, hashes,
  first mismatch positions, usage and response IDs. Client concurrency
  is proven; engine scheduling overlap is not inferred from HTTP alone.
- Exact 8192 and 32768 input tokens, constructed using /tokenize and
  token-ID prompt input, with needles at the start, middle and end.
  Require violet cedar copper in order and server usage equal to the
  requested prompt length. These are bounded recall/coherence checks,
  not long-context performance or full 200K qualification.

Unsupported context length, HTTP failure, missing tokens, or any wrong
answer fails the check and exit status is 3. Identity failure prevents
inference and leaves remaining required checks false/not run. Partial
results are saved atomically after each check/complete repeat round.
The proxy C4 check is intentionally strict: a valid but different
quantization/dtype can fail it; do not silently regenerate its reference.

Usage against a server in its owner's agreed request window:

```bash
bin/cpu-run --quiet vllm/flashnext/harness/promo_gate.sh \
  --base-url http://127.0.0.1:18125 --profile proxy35b \
  --expect-alias qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0 \
  --out /mnt/vm_8tb/b70/build/vllm-flashnext/x4/proxy-live-gate.json
```

For Flash-Next select --profile flashnext and its actual research alias.
Passing the long checks requires server max_model_len >= 32800. The
wrapper never relaxes them for a server configured with 4096 context.

## Localmaxxing method and provenance

Sources: localmaxxing/lmx_vllm.py, run_dry.sh, validate_dry.sh, README.md,
prompts.json, canonical text files, and hardware_card1/pair.json.
Read the Strata run scripts, payloads and lmx_verified.py first. Canonical
hashes fetched via lmx context match the existing Strata payloads:

| Suite | SHA256 |
| --- | --- |
| reasoning-v1 | 9000edaadb16fc77ab78b39917245a03d4d7585c3ba770cb045a47a0e0683445 |
| code-v1 | a737db738bd4e37881d1737f8dc51cae0295eff9ece6551d39826ce8ed43a690 |

Installed CLI v0.1.54 is commit 8bb36c3eb8fcd95d363beb44725d1992e80fe404.
Fetched the matching source into scratch x4-lmx-source. Primary source:
[lmx measurement code](https://github.com/LottoLottoLotto/localmaxxing-cli/blob/8bb36c3eb8fcd95d363beb44725d1992e80fe404/cmd/lmx/main.go#L6435).
The CLI uses one warmup, three timed requests, a leading cache-bust nonce,
authoritative endpoint usage, and the first-to-last content arrival window.
Reasoning deltas count as output. The scripts preserve that method, use
temperature 0, 512 maximum output tokens, C1, and backend=xpu.

Decode rate = (completion_tokens - 1) / (last content - first content).
TTFT includes network/API/scheduler/prefill delay until first nonempty
reasoning/content. Effective ITL = 1000 / decode rate. SSE chunk intervals
are stored separately: a speculative chunk may contain multiple tokens.
The same finite-window formula has boundary bias when multiple tokens
arrive in the first chunk; this matches lmx/Strata's client method and
is not a per-token device timestamp claim.

Strata's verified client selects the median decode iteration and carries
its matching TTFT/output/acceptance evidence. The official CLI reports
independent medians of each metric and leaves the last output text in
its result (source main.go:6715). This collector follows Strata's
correlated median sample; validation.json also records the independent
CLI metric medians so the difference can be audited.

vLLM does not normally provide Strata's per-request timings object.
engineTimingsRaw therefore contains actual usage, native timings only
if present, and isolated Prometheus counter deltas with explicit
provenance. No predicted_ms/prompt_ms or engine tok/s is invented.
Raw requests, every SSE event plus monotonic offset, text, usage, and
before/after metrics are retained. No retokenization or character-based
estimate substitutes for endpoint output usage.

Attribution requires idle request gauges, success delta exactly one,
generated-token delta equal to usage, no counter resets or prefix cache
hits. Periodic export lag is polled outside the HTTP timing window, up
to 20 seconds. These checks detect contamination; an agreed exclusive
request window is still required. A new leading nonce prevents shared
canonical-prefix reuse; disabling prefix cache is preferred.

No-spec additionally requires live structured /server_info speculative
config=null and zero draft activity. MTP requires method/depth agreement
and positive draft counters; acceptance=accepted/draft, mean accepted
length=1+accepted/num_drafts. Global counters are attributed only after
the isolation checks pass. Model identity and config must remain equal
after each suite. Config snapshots omit the server environment and redact
credential fields. --engine-version must match live /version. The plan
examples use installed-wheel metadata; the fork's generated version lacks
the .xpu suffix, so replace it with the actual live version before measuring.

Ranking comparisons use NON-spec runs. MTP payloads remain marked MTP
and cannot become headline no-spec results. Current comparison references
are the same canonical Strata no-spec payloads (~66.2 tok/s), with MTP
122.2 tok/s retained separately; the 141 tok/s DD reference is a different
measurement configuration. No vLLM ranking result is claimed here.

The live LocalMaxxing context requires prompt/output hashes and samples,
batch one, >=256 output tokens, engineTimingsRaw, and speculative counts
for verification. Whether the server accepts this explicit vLLM evidence
object as a Verified run remains untested; it must be checked using the
dry-run validator once a real measured payload exists. No badge is
claimed and no dummy performance numbers were generated for validation.

## Exact dry-run commands and evidence

Executed from the repository root; all four plan files are under
/mnt/vm_8tb/b70/build/vllm-flashnext/x4/plans. The launch command files
describe proposed proxy configurations and are not qualification evidence.
X5 changed its evolving stack while this work ran; replace launch identity
with its final measured stack before collecting a real result.

```bash
L=vllm/flashnext/localmaxxing
B=/mnt/vm_8tb/b70/build/vllm-flashnext/x4
COMMON=(--base-url http://127.0.0.1:18125 --served-model hotschmoe-dd
  --hf-id unsloth/Qwen3.6-35B-A3B-MTP-GGUF
  --quantization Unsloth-Dynamic-IQ3_XXS --hardware "$L/hardware_card1.json"
  --engine-version 0.32.1rc1.dev94+g5fd62688f.xpu
  --engine-commit 0a6170da8ca9991972322fa1ca1d27674f0e75cd
  --engine-build b70-flashnext-shared-dirty-plan --out "$B/plans")
"$L/run_dry.sh" "${COMMON[@]}" --mode no-spec \
  --alias qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0 \
  --command-file "$B/proxy-launch-plan.txt"
"$L/run_dry.sh" "${COMMON[@]}" --mode mtp --spec-tokens 2 \
  --alias qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp2 \
  --command-file "$B/proxy-mtp-launch-plan.txt"
```

run_dry.sh always adds --dry-run; --measure-local conflicts at argparse.
The saved official CLI output says no speed-test command or API request
ran. backend=xpu, draft depth, CLI argv and canonical hashes are retained
in the plans. The scripts have no submit path; authentication/config
handling stays inside the CLI. X4 did not inspect or disclose the key.

After the orchestrator chooses a qualified server, local collection is
available by invoking lmx_vllm.py with the same arguments and
--measure-local instead of --dry-run, under cpu-run --quiet. This path
was NOT executed by X4. A real result can then be validated without
submission using localmaxxing/validate_dry.sh <suite>/payload.json. That
wrapper requires local validation=true and calls only validate-local
and speed-test dry-run. lmx owns its saved-config authentication.

For Flash-Next change hf-id to
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, quantization to IQ3_XXS,
hardware to hardware_pair.json, and use the actual sidecar server alias,
launch command/config and fork/kernel identities. No proxy substitution
is hidden in the metadata. Preserve dirty source diffs/hashes alongside
real measurements because the current fork commit is not the whole build.

## Validation

```bash
python3 -m unittest discover -s vllm/flashnext/harness -p test_promo_gate.py -v
bash -n vllm/flashnext/harness/promo_gate.sh \
  vllm/flashnext/localmaxxing/run_dry.sh vllm/flashnext/localmaxxing/validate_dry.sh
```

Nine tests pass: a full fake HTTP pass with changing created timestamps;
changed model root; C2-only token mutation; missing
IDs and short forced generations; wrong long-prompt usage; wrong model;
reasoning-first TTFT and multi-token chunk accounting; truncated SSE;
DP-label metric summing and counter reset detection. Raw CPU results:
scratch x4/cpu-tests.txt. These verify client behavior, not GPU/model
correctness or serving speed. ASCII and Python/shell syntax checks pass.

## Live gate result (2026-10-20 21:27:40 to 21:28:47 UTC)

CONFIG: X5's stock proxy, physical card1/logical0, BF16, O3g kernel
38f7176a, shared expert fused, stock GDN, installed FlashAttention,
FULL_DECODE_ONLY, reclaim500, maxlen4096, maxseq2. Model identity:
hotschmoe-dd first, qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0 second. X5
holds the GPU lease. Its native card1 pre-health passes. X4 holds only
CPUquiet during its HTTP requests, after X5's first greedy pass.

COMMAND: the promo_gate.sh command above, adding --timeout 60, under
an outer 180-second client timeout. Completed in about 67 seconds,
exit 3. No other server was stopped or reconfigured. Earlier X5 startup
had failed before HTTP readiness; that attempt supplied no gate result.
F1's later long-context HTTP window closed without a confirmed X4 slot;
X4 sent no inference requests to it.

RESULT:

| Check | Observed result |
| --- | --- |
| Identity / registered alias | PASS before and after |
| Stable model fields unchanged | PASS; response timestamps/permission IDs differ normally |
| 17*19 / 23*29 | PASS: 323 / 667 |
| C4 proxy Paris continuation | PASS: exact 16-token text |
| Four prompts x 96 forced tokens, first C1 pass | Complete; four 96-token ID arrays retained |
| Exact repeat across C1/C2 | FAIL: request timeout during second C1 pass; C2 not reached |
| 8192 input tokens | FAIL: HTTP 400, maximum model context 4096 |
| 32768 input tokens | FAIL: HTTP 400, maximum model context 4096 |
| Overall | FAIL, exit 3; no promotion |

Server metrics log one running request and zero generation at 21:28:05.
The timeout is an observed request stall, not a located graph/kernel
cause or hardware-health failure. The saved first pass is not evidence
of repeatability. The missing C2 result is explicit, not silently passed.
Subsequent tokenization, length validation and /v1/models remained
responsive, allowing the client to finish its JSON and release CPUquiet.
X5 was notified via JOURNAL to inspect its owned server; teardown and
post-health are X5's responsibility. X4 performs no reset or new serve.

Evidence root: /mnt/vm_8tb/b70/build/vllm-flashnext/x4/

- proxy-live-gate.json: single complete check result and raw completed
  first-pass token arrays.
- live-gate.log: check statuses, timeout, HTTP 400 details and exit 3.
- source-manifest.json: all gate/localmaxxing source hashes; verified
  unchanged after the live run.
- cpu-tests.txt, dry-guard.txt, plans/*-plan.json: nine passing CPU
  tests, rejected measurement override, four official dry-run plans.

Matched server evidence belongs to X5 under
x5/ud-iq3_xxs-mtp0-gdn0/: serve.log, window.log, health-before.txt,
source-before.json. No ITL or stability claim is made from this gate.

VERDICT: both X4 deliverables are implemented and exercised within
their authorized scope. The gate exposes a real serving failure and
enforces long-context limits. A successful full C1/C2 repeat and long
recall run remain qualification work after the serving stall is fixed
and a sufficiently large context is configured. Localmaxxing remains
dry-run only; no measured or verified payload is claimed.
