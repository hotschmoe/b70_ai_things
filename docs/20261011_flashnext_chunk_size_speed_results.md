# Matched chunk-size diagnostic results

CONFIG -> Selected original Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL, source40,
single B70 card0, context2048, fp16 KV, PLE mmap65536, cacheOFF, temperature0/
seed1, no MTP/suffix/lookup drafts, no borrowing/adaptation, segmented1024
mirrors. Same four54/524-token prompts and192 output cap. New V2 controller
239f41838885d83e2784c81b3b5788e1922834b15e65534fdc5c31d07ea64d90 and
current173 registry d4242070dff786a5977254f1ff0fa4deae6487ec308765aa2b4e5a4607234167.
Complete configs differ only in prefill64 versus512 and the declared alias.

COMMAND -> Chunk64 session90356 completed terminal0/656s. Parent receipt
f17/speed-chunk64-registry173-prepared-v2/speed-parent.json SHA256
9e018f96804bec7b017862bb9db7fa2c8211ff0a21af3b9dd566d07e898860a8.
Root independently rehashed all nine parent metadata/source/screen/stop/client/
health/full4/supervisor-log/command edges. Paired compiled health uses both
cards under lease; actual inference is pinned to GPU0.

RESULT -> Ordinary six-request identity/coherence screen and both repeated
timed prompt pairs passed. All timed requests produced192 tokens, cached0,
finish length. Native rates are engine host request intervals; client first
text and usage arrival are separate clocks. No device-only timing or true
per-token-gap claim follows from SSE text chunks.

| Request | Prompt tokens | Native prefill tok/s | Native decode tok/s | Client first text s | Client usage arrival s |
| --- | --- | --- | --- | --- | --- |
| 0 | 54 | 4.7 | 4.8 | 11.482 | 51.397 |
| 1 | 524 | 28.4 | 6.8 | 18.504 | 46.849 |
| 2 | 524 | 33.6 | 8.8 | 15.679 | 37.540 |
| 3 | 54 | 12.6 | 8.7 | 4.372 | 26.424 |

Actual startup admission:1897MiB free,8092 resident experts,16484 mirrored
experts,51660083200 mirror bytes; complete24576 roster, no fallback, first
chunk override absent/default0. These are startup allocation observations,
not dynamic expert-hit or transfer measurements. Owned shutdown/removal was
normal; post strict/compiled health, kernel and fresh full4 passed. Post full4
SHA256 cc797f9f1c0209bcba1e7953c1a4c8bf46e94dcb1456816f8b24432d06edfbcb.

VERDICT -> Verified scoped chunk64 baseline with substantial warming. It
reproduces earlier low rates. Original-model fidelity, natural completion,
concurrency, physical placement, critical-path attribution and shelf remain
unqualified. Chunk512 session83338 acquired the lease after chunk64 retired
and is live; no chunk comparison or speedup result yet. Source41 fresh ABI
build32690 is queued after the comparison, with actual new oracle/model/cache
qualification still required. The full original campaign remains active.

CONFIG -> Same matched configuration, now prefill512.
COMMAND -> Original83338 completed terminal0/688s. Parent SHA256
e57b0a9b77ec994289e3b2b9b65acdd5d044cdfc34c14b31e1155f2dea5793bd.
Root rehashed all nine parent edges and compared the actual request bodies,
outputs, reasoning, usage and finish across both completed arms. Independent
comparison f17/chunk64-512-root-matched-comparison-v1.json SHA256
c9f833debad9469b115808ccd20e75aec3d31bf8806bd81ba13140ae856afde8.

| Request | Prompt tokens | Native prefill tok/s | Native decode tok/s | Client first text s | Client usage arrival s |
| --- | --- | --- | --- | --- | --- |
| 0 | 54 | 6.3 | 4.9 | 8.597 | 48.092 |
| 1 | 524 | 57.0 | 6.4 | 9.273 | 39.396 |
| 2 | 524 | 85.7 | 8.6 | 6.187 | 28.367 |
| 3 | 54 | 12.3 | 8.7 | 4.448 | 26.505 |

RESULT -> Per-arm repeat/coherence and normal owned teardown, post-health,
kernel and fresh full4 gates passed. Startup512:2016MiB free,7992 resident,
16584 mirrored experts,51971379200 mirror bytes. Increasing chunk capacity
reduced resident experts by100 under the unchanged automatic cache policy.
Actual request bodies, reasoning, usage and finish match across arms; short
answers match, both long answers differ. All requests remain192-token caps.

VERDICT -> Long-prompt prefill and first-text latency improve in these scoped
samples, while warm decode remains near8-9tok/s. Changed answers preclude a
qualified matched-output optimization or original-model fidelity claim. These
are one32GiB-card experiments, not a measurement of the user's two-card target.
Next diagnose the token critical path, including host/queue gaps and waits,
before assigning the remaining cost to expert traffic or individual kernels.
Synthetic inputs/no-op/resident-only substitutes are permitted as clearly
labeled component diagnostics; they cannot qualify real-model speed/fidelity.
