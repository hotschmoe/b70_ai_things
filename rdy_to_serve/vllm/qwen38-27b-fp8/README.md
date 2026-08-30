# Qwen3.8-27B FP8 daily driver

This shelf entry serves the official Qwen3.8-27B FP8 checkpoint as W8A16 on
both B70 cards with vLLM TP2, FP16 KV, direct oneCCL P2P, Triton target and
draft attention, and FULL decode graph capture.

The default `PROFILE=daily` uses MTP1. F09f qualified it across two fresh
server lifetimes at a 46.604 tok/s median strict rate, 45.71 to 48.05 tok/s
aggregate at c2, and 88.83 to 89.09 tok/s aggregate at c4. Complete outputs
matched 12/12 across lifetimes, both 32-request concurrent quality gates
passed, a 30,037-token semantic needle matched exactly, and every teardown,
card check, and compiled collective check passed.

`PROFILE=fast` uses MTP8. It is the decode-speed profile: the strict varied
suite measured 64.97 and 67.40 tok/s in F08b, and the 262K/c4 screen measured
59.07 tok/s aggregate at c2 and 101.54 tok/s at c4. Use the default MTP1 when
the extra 19,788 aggregate KV tokens and the fully completed c4 qualification
matter more than peak decode.

Start the default profile:

    ./bin/gpu-run bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh start

Start the fast profile:

    PROFILE=fast ./bin/gpu-run bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh start

Stop either profile from another shell:

    bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh stop

The configured context limit is 262,144 tokens. Capacity was measured at
323,202 aggregate GPU KV tokens for MTP1 and 303,414 for MTP8, so one full
window fits but four full windows do not. At c4, requests share the aggregate
pool.

Do not interpret the configured limit as a near-window prefill-speed claim.
Under the 262K envelope, both MTP1 and MTP8 took about 222 seconds to ingest a
fresh 30,037-token prompt. A 260K probe exceeded ten minutes in its first
32,768-token worker submission and was aborted; recovery and post-health
passed. Prefix caching is disabled in this exact qualified recipe. Large
envelope prefill and prefix-cache enablement are the next campaign, especially
for agents that repeatedly reuse a growing context.
