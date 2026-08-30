# Qwen3.8 FP8 F05b/F05c concurrency boundary

Date: 2026-08-30 UTC

## F05b: old GDN kernel rejects mixed MTP batches

CONFIG -> Official Qwen3.8-27B-FP8, W8A16, FP16 KV, TP2, P2P off,
MTP1, graph off, deterministic Inductor, packed serial RMSNorm, persistent
GDN scratch, 32,768 model/batch limits, and four service slots. Image ID was
`338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b`;
its `vllm-xpu-kernels` version was 0.1.12.3. Result root:
`/mnt/vm_8tb/b70/results/f05b_qwen38_fp8_neural/20260830T065500Z/`.

COMMAND -> Run the normal 12-prompt and independent-canary gates, then four
serial 2K-prompt/512-output controls followed by synchronized C4 requests.
The whole run used `bin/gpu-run`; failure cleanup included card and compiled
P2P-off collective health.

RESULT -> The normal suite completed at 17.511970 tok/s and retained the F05a
target. The first C4 transition mixed a new prefill with active speculative
decode rows. Both ranks then raised:

```
causal_conv1d does not support spec-decode and non-spec (prefill + decode)
tokens in the same invocation
```

The engine exited rather than hanging. Both cards and the compiled two-rank
collective passed immediately afterward. F05b is a closed software-path
negative, not evidence of a host wedge, device poison, or RAM spill.

VERDICT -> Do not serve concurrent MTP1 with the image's 0.1.12.3 GDN
kernel. The failure exactly matches the old-kernel boundary disclosed by the
Neural.Download recipe.

## F05c: corrected mixed-path kernel survives C4

CONFIG -> Add only the recipe's pinned XPU kernel wheel at commit
`1e90ffa672ba02f17a909da11838a4c55b199783` to the qualified local image.
The wheel SHA256 is
`f3d999060c11ad6db5b4033d50d19c6b665492380075480d041ec4ee58fdfeb6`.
The composite image ID is
`8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81`.
All F04b/F05a vLLM integration hashes remained unchanged. The runtime stayed
P2P off, graph off, MTP1, deterministic, 32K-capable, and C4. Result root:
`/mnt/vm_8tb/b70/results/f05c_qwen38_fp8_neural/20260830T073000Z/`.

COMMAND -> Compile two independent empty caches, run a short serial/C4 oracle
with 2K prompts and 64 forced output tokens per stream, tear each server down,
and run per-card plus compiled P2P-off collective health. Compare complete raw
arrays while separately recording the serial and concurrent batch shapes.

RESULT -> Both attempts used target AOT key `80de0121...` and draft key
`be175b50...`; each cache contained zero `.best_config` files. Every serial
and concurrent request completed: 8/8 serial streams and 8/8 C4 streams
returned all 64 tokens. Neither server logged the mixed-path exception or an
engine error. Aggregate post-first-token rates for the bounded C4 oracle were
18.334303 and 18.192014 tok/s. These are diagnostic, not promotion rates.

RESULT -> All four serial arrays matched across restarts. At C4, streams 0
and 2 matched while streams 1 and 3 changed; their first differences were at
tokens 29 and 1. This confirms the publisher's disclosed batch-history
dependence. Exact C1/C4 or asynchronous C4/C4 byte equality is not a valid
concurrent quality contract. Complete responses, semantic correctness, task
isolation, and repeatable latency/throughput are the appropriate gates.

RESULT -> Teardown and all card/collective health checks passed. Peak
container process memory was 9.816 GiB and minimum host MemAvailable was
111,388,204 KiB. The container had memory and memory-swap set to the same
32 GiB limit, so it had no swap allowance. Global host swap rose from zero to
28,652 KiB while more than 106 GiB remained available; no OOM or kernel GPU
fault was logged. Do not describe this run as host-swap-zero.

VERDICT -> The pinned 1e90 kernel closes the fatal mixed spec/non-spec GDN
failure under the local P2P-off policy. F05c intentionally fails the old
restart-byte-exact gate because that gate is too strong for asynchronous
continuous batching. F05d must run full 512-token completion batches plus the
publisher's concurrent exact-answer semantic canaries before shelf work.

## Recipe transfer boundary

The recipe is directly buildable on this host. Its artifact helper currently
downloads the wheel under an extra `dist/` directory; the local fail-closed
builder accepts either layout and retains the published digest check. The
publisher's 51.918757 tok/s strict profile uses direct oneCCL P2P. That setting
is not transferred to full local serving because the local vLLM queue-handoff
failure remains open. The currently qualified local P2P-off MTP1 headline is
17.649601 tok/s from F04b, not an under-15 result.

Sources:

- https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html
- https://github.com/steveseguin/b70-optimization-lab/tree/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70
