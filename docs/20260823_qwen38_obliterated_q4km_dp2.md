# Qwen3.8-27B OBLITERATED V3 Q4_K_M DP=2 on B70

Status: PAUSED 2026-08-23 after the operator rejected its practical output
quality. Its performance, equivalence, concurrent soak, real 152289-token
request, and shelf-restart gates remain valid historical evidence. The live
daily driver moved to stock Q4_K_M TP=2.

Peer performance comparison, including the corrected status of Steve's old
101.922 tok/s claim:
`docs/20260823_obliterated_q4km_peer_comparison.md`.

## Artifact identity

Source: `OBLITERATUS/Qwen3.8-27B-OBLITERATED`.

- Repository revision: `2648a6231b82328c601ba27b9ffd5029057d0e33`.
- Q4_K_M fixed-merge commit: `736efa47dfad6ccd5d4b4d51d55cee46078dc00f`.
- File: `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`.
- Bytes: `16810714400`.
- SHA256: `c5e4fe705883e244a468c9e445c8d6ba37fd310b0113e25d2b8a7f2d6f1243e8`.
- GGUF name: `Qwen3.8 27b S99 Merged Fixed`.
- GGUF architecture: `qwen35`, 65 blocks, native context 262144.
- Tensor inventory: 866 tensors, including `blk.64.nextn.*`; embedded MTP is present.

The pre-V3 Q8_0 file was deleted at the operator's request. It is not a fallback.

Sources:

- https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED
- https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative-simple/README.md

## Runtime design

DP=2 means two independent TP=1 replicas, not tensor parallelism:

```
client :18080
  -> nginx round robin
     -> 127.0.0.1:18181 -> B70 card 0 -> full Q4_K_M replica
     -> 127.0.0.1:18182 -> B70 card 1 -> full Q4_K_M replica
```

Both backends expose the same OpenAI model id, `hotschmoe-dd`. This avoids
cross-card collectives and gives two independent request lanes. Each replica has
one 245760-token slot. The target KV cache uses Q8_0 because the 16 full-attention
layers would need about 15 GiB of f16 KV at this context, versus about 7.5 GiB at
Q8_0. The other 48 layers use recurrent GDN state rather than a full KV history.

Runtime:

- Image: `qwen38-b70:latest`, image id
  `sha256:8c6dc0462011e7d4596882009fc7fb1128fbe656cb17a998999cd1e720a2b4de`.
- llama.cpp: mndodd fork plus the B70 optimization-lab TP2/Q4K patch stack.
- OneAPI 2025.3 SYCL JIT, BMG G31 detected on both replicas.
- `LAB_DOORS=1`, Q4_K reorder and fused MMVQ family enabled.
- `CTX_SIZE=245760`, `KV_TYPE=q8_0`, `BATCH=1024`, `UBATCH=256`, `PARALLEL=1`.
- Temperature 0, repetition penalty 1.15, thinking off, empty system prompt.
- API key file mounted read-only; the key is not placed in Docker environment metadata.

## Experiment 0: launch wrapper

Config -> DP=2 wrapper, same runtime as above.

Command ->

```
./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q4km_dp2.sh start
```

Result -> first attempt exited before llama.cpp because oneAPI `setvars.sh`
references optional unset variables while Bash nounset was already enabled.
Moving `set -u` after `setvars.sh` fixed the wrapper. Both cards remained healthy.

Verdict -> wrapper bug fixed; no model or context failure occurred.

## Experiment 1: no-MTP 245760 baseline

Config -> DP=2, embedded MTP disabled, Q8 KV, Q4K lab doors on, 245760 ctx.

Command ->

```
./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q4km_dp2.sh start
```

Result -> both replicas became healthy and returned exact Paris. Two proxy calls
also returned exact Paris. `/v1/models` reports:

```
id=hotschmoe-dd
n_ctx=245760
n_ctx_train=262144
n_params=27320697856
ftype=Q4_K - Medium
```

Four new proxy connections alternated upstreams 18182, 18181, 18182, 18181.

Phase bench used unique cold prompts, target p512/g128, five measured runs after
warmup, forced 128 output tokens. The generated prompts tokenized to about 1150.

| path | median post-first tok/s | median prefill proxy tok/s | median TTFT |
|---|---:|---:|---:|
| nginx, serial | 23.93 | 591.4 | 1.982 s |
| card 0 while both cards active | 23.84 | 567.6 | 2.037 s |
| card 1 while both cards active | 23.85 | 589.1 | 1.964 s |
| DP=2 aggregate | 47.69 | n/a | n/a |

The simultaneous card runs retained the per-card serial rate, so DP scaling is
effectively 2.00x for two independent streams.

Raw evidence:

- `results/logs/qwen38_obliterated_q4km/nomtp_proxy_p512_g128.json`
- `results/logs/qwen38_obliterated_q4km/nomtp_card0_concurrent_p512_g128.json`
- `results/logs/qwen38_obliterated_q4km/nomtp_card1_concurrent_p512_g128.json`

Verdict -> GO baseline. The requested model identity, DP=2 topology, one endpoint,
and large context are real. Do not promote until embedded-MTP A/B and sustained
concurrent coherence complete.

## Experiment 2: embedded MTP3 A/B

Config -> identical 245760 context, Q8_0 target and draft KV, batch/ubatch
1024/256, one slot per card. Only embedded MTP changed from off to
`--spec-type draft-mtp --spec-draft-n-max 3`. No external sidecar was loaded.

Command ->

```
ENABLE_MTP=1 MTP_SIDECAR=0 MTP_DRAFT_MAX=3 \
  ./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q4km_dp2.sh start
```

Result -> both replicas fit at full context and passed four Paris gates. The
main GGUF's `blk.64.nextn.*` head was used. Same phase bench as the baseline:

| path | no-MTP tok/s | MTP3 tok/s | change |
|---|---:|---:|---:|
| nginx, serial | 23.93 | 41.25 | +72.4% |
| card 0, simultaneous | 23.84 | 40.35 | +69.3% |
| card 1, simultaneous | 23.85 | 41.51 | +74.0% |
| DP=2 aggregate | 47.69 | 81.86 | +71.7% |

Observed per-request draft acceptance ranged from about 0.42 to 1.00, with
mean speculative lengths from 2.26 to 4.00. It was workload-dependent rather
than a fixed or obviously bogus counter.

Verdict -> GO. Embedded MTP3 is a large controlled speed win and fits beside
the full 245760-token context on both 32 GB cards.

## Experiment 3: MTP output equivalence and coherence

Config -> seven deterministic prompts on both direct replicas: Paris,
multiplication, modular arithmetic, Fibonacci, sorting, syllogism, and an exact
24-line square table. Save full text, then restart without MTP and repeat the
same prompts at temperature zero and seed 1.

Command -> `llamacpp/qualify_qwen38_obliterated_q4km.py` on ports 18181 and
18182, first with MTP3 and then without MTP using `--reference`.

Result -> the two MTP replicas were byte-identical on all seven completions.
MTP and no-MTP were byte-identical on six of seven prompts on both cards. The
seventh was semantically identical and differed only in punctuation:

```
MTP3:   No. Since all zorps are blue ...
no-MTP: No; since all zorps are blue ...
```

Both modes made the same model-quality miss on the harder arithmetic prompt,
answering `(12345*6789) mod 97` as 0 instead of 71. That miss is not introduced
by speculative decoding. Paris, 391, Fibonacci, sorting, logic, and every line
of the 24-line square table were coherent and correct.

Verdict -> GO for MTP coherence. The speedup is not accompanied by garbling or
an MTP-only correctness regression in this deterministic gate. Keep the
modular miss recorded as a Q4 model-quality limitation.

## Experiment 4: sustained concurrent mixed-load soak

Config -> MTP3, c4 through nginx for 300 seconds. Six cases cycle across short,
medium, and long prefills: Paris, 391, Fibonacci, hash-map prose, a long exact
marker, and a square list. Every answer is validated and every response is
checked for empty output, repeated-character runs, and character dominance.

Command ->

```
OPENAI_API_KEY=... ./bin/gpu-run python3 -u \
  llamacpp/soak_qwen38_obliterated_q4km.py \
  --base http://127.0.0.1:18080 --concurrency 4 --duration 300 \
  --out results/logs/qwen38_obliterated_q4km/mtp3_proxy_c4_soak_300s.json
```

Result -> 338/338 requests passed, 9488 completion tokens, zero coherence
failures, zero degenerate outputs, and zero request errors. nginx routed 171
requests to card 0 and 167 to card 1. Each of the six cases ran 55-58 times,
including 57 long exact-marker prefills.

Verdict -> GO. MTP3 remains coherent under sustained concurrent proxy load and
both replicas carry traffic evenly.

## Experiment 5: real long-context requests

Config -> MTP3 candidate, unique cold entropy prompt, one request through
nginx, forced output. The first sizing run targeted 65536 harness tokens.

Command -> `phase_bench.py --prompt-tokens 65536 --gen-tokens 16 --n 1
--skip-warmup --ignore-eos --timeout 1200`.

Result -> actual prompt usage was 133551 tokens and output was coherent. The
server reported 938442.75 ms prompt eval, 142.31 prompt tok/s, 16 output tokens,
940207.26 ms total, and `truncated = 0`. MTP accepted 7/21 draft tokens on this
entropy-heavy generation. The phase harness's printed 30.327-second TTFT is not
valid for this request because its timer starts after the response headers;
use the server timing above.

The scaled run used `--prompt-tokens 75000 --gen-tokens 8 --timeout 1800`.
Actual prompt usage was **152289 tokens**. It returned coherent text and the
server reported 1184699.71 ms prompt eval, 128.55 prompt tok/s, 1185582.65 ms
total, MTP acceptance 4/7 with mean length 2.33, and `truncated = 0`. No context
shift occurred. As above, ignore the phase harness's header-excluding TTFT and
prefill proxy figures for this request.

Verdict -> GO. A real request crossed the requested 150k lower bound, while the
live API and allocated slot expose 245760 of the model's native 262144 context.

## Backend choice and DSpark

GGUF is a llama.cpp artifact, so two llama.cpp replicas are the direct DP=2
serve path. DSpark is a vLLM/transformers hidden-state drafter and cannot be
attached to this GGUF runtime. The fixed V3 file already carries a compatible
MTP head, and llama.cpp can create its MTP context directly against the target
model, avoiding a second full draft model. Embedded MTP is therefore the
applicable speculative path.

## Shelf and service

The measured default is now:

```
rdy_to_serve/llamacpp/qwen38-27b-obliterated-q4km/serve.sh
```

This was the measured default at promotion time. It remains a reproducible
paused shelf entry, not the current daily driver. The tracked systemd unit now
points to `rdy_to_serve/llamacpp/qwen38-27b-q4km/serve.sh`.

The tracked systemd unit is `llamacpp/deploy/b70-daily-driver.service`. Install
it after the final gate with:

```
sudo install -m 0644 llamacpp/deploy/b70-daily-driver.service \
  /etc/systemd/system/b70-daily-driver.service
sudo systemctl daemon-reload
sudo systemctl restart b70-daily-driver.service
```

Non-interactive sudo is unavailable in this agent session, so installing the
unit requires operator authentication. The current three Docker containers use
`--restart unless-stopped` and remain live independently of that installation.

## Experiment 6: final shelf restart

Config -> no environment overrides; start only from the promoted shelf entry.

Command ->

```
./bin/gpu-run bash \
  rdy_to_serve/llamacpp/qwen38-27b-obliterated-q4km/serve.sh start
```

Result -> exact SHA check passed; the wrapper selected MTP3, Q8_0 KV, and
245760 context. Both replicas, both direct Paris gates, and both proxy Paris
gates passed. Final `/v1/models` reports `hotschmoe-dd`, n_ctx 245760,
n_ctx_train 262144, 27320697856 parameters, and Q4_K Medium. Four proxy
connections routed 18182, 18181, 18182, 18181.

Verdict -> GO live shelf. The default handoff command is self-contained and the
single endpoint is balanced across both cards.

## Remaining operator action

1. Install the tracked systemd unit when sudo authentication is available.
