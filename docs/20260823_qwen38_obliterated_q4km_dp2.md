# Qwen3.8-27B OBLITERATED V3 Q4_K_M DP=2 on B70

Status: baseline live and measured; embedded-MTP and soak gates in progress.

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

## Next gates

1. Same-context embedded MTP at draft lengths 2, 3, and 4; keep only a coherent
   speed win on representative generation, not an easy counting-only win.
2. Mixed-length concurrent proxy soak and upstream distribution proof.
3. Long-context allocation/request probe beyond the API metadata check.
4. Shelf recipe, systemd daily-driver update, final restart, and post-restart gate.
