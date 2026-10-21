# R1 -- post-outage campaign restart (2026-10-21)

CONFIG: unchanged host kernel 7.1.0-070100-generic and runtime. User
resumed the Flash-Next vLLM campaign and authorized startup serving down
for testing. Sol 6.1 high owners: D1 decode, N1 GDN determinism, Q1 QSA.
Strata remains the daily driver; no model or shelf promotion.

COMMAND: stop startup serving; `bin/gpu-run bin/xpu-health`, then
`bin/gpu-run bin/xpu-collective-health`. Run merged O4+F1 with
`TAG=restart-r1 MAX_LEN=131072 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
F1_LENGTHS=8192,12288 F1_COHERENCE_LEN=8192 F1_REPEAT=1 F1_DECODE=0
RUN_PROMO_GATE=1 SKIP_HEALTH_BEFORE=1 bash
vllm/flashnext/harness/f1_merged_serve.sh`.

RESULT: native per-card health and ten compiled two-rank collectives
passed with P2P disabled. Baseline ready 16:15:33 UTC, normal teardown,
post-health HEALTHY. Raw evidence is under
`/mnt/vm_8tb/b70/build/vllm-flashnext/restart-20261021/` and
`f1/merged-restart-r1/` in the same build root. Effective O4 GGUF, F1 EP,
F1 bucket and O4 helper libraries retained their SHA256s. Tracked fork
diff before/after the serve was identical. The fork remains dirty;
this is not a frozen submission engine.

| Baseline check | Result |
|---|---|
| First ID / registered research alias | hotschmoe-dd / qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4 |
| Arithmetic / coherence | 323, 667, repeated Paris pass |
| 8K / 32K needles | pass / pass |
| C1 ITL p50, 64 outputs | 15.1669 ms |
| C2 ITL p50 per stream, 64 outputs | 19.5596 / 19.6108 ms |
| 8186-token prefill, first measured request | 4327 ms, 1892 tok/s |
| 12283-token prefill, request-completion proxy | 5048 ms, 2433 input tok/s |
| Strict exact-repeat promotion gate | FAIL |

All C1 exact repeats match. C2 differs from C1 for multiple prompts;
the hash-table prompt also differs between the two C2 rounds. This is
both batch-shape variance and a repeatability concern. N1 addresses one
identified projection source, not every possible source of divergence.
The single 8K timing is not a matched warm regression measurement.
The 12K row emitted no content event (`n_chunks=0`); its `ttft_ms` is the
client's request-wall fallback, not observed first-content TTFT. The 8K
row did observe content. Neither rate isolates engine-only prefill time.

The attempted 512-token canonical collector stopped before measurement:
stock `vllm serve` lacks its required `/server_info` endpoint. No suite
rate was produced or submitted. A shim and an integrated callback are
prepared for the next matched window. The original harness printed a
zero shell exit despite its recorded gate RC3; it now propagates requested
gate failure after collecting diagnostics and completing teardown/health.

VERDICT: the short decode baseline reproduced the handoff after reboot;
strict promotion remains rejected. Run bounded component gates before
another full model load. Future suite collection must run inside the
serve window with the metadata shim and before teardown.

## Recovered work and setup cost

The handoff understates the completed work. D1 recovered Ld's successful
trace and post-health; Q1 recovered a synthetic integration failure and
a subsequent CPU binding repair. See their individual reports.

F1 q2 also completed before the outage: exit0, pre/post HEALTHY, arithmetic
and long coherence pass. Parsing real T12283, E374 rows (48 layers) gives
median dequant13.35 ms, combine9.80, bmm1 9.43, bmm2 5.405, gather4.415;
mean stage sum47.271 ms. T8186 card1 mean40.542 ms. These instrumented
stages synchronize after every operation and do not establish clean
end-to-end latency. Synthetic T12288 profile rows are excluded. Raw
summary: `restart-20261021/f1-q2-recovered-summary.json`. No repeat of q2
is necessary; dequant and combination are concrete prefill candidates.

Consultations with `claude-fable-5-1` and `claude-opus-5-5` each timed out
after35s (exit124, no responding model ID). No advice is attributed to
them. All new owners actually use gpt-6.1-sol.

## Matched control and rejected attention treatment

The integrated metadata shim and callback produced a complete control
on the same unchanged recipe: reasoning61.642817 tok/s, code61.800831,
three measured 512-token requests per suite. Source/SO checks, isolated
request counters, zero draft activity, identity, coherence, normal engine
teardown and both-card post-health pass. Strict C1/C2 repeatability fails.
The independent audit recomputed rates from raw event arrival times.

The first plan16 serving attempt was discarded after an integration guard
was found to reject the real server's scale tensors. After a corrected
wrapper gate passed and activation was proven in the real server, r2
failed with native level_zero error40/OUT_OF_RESOURCES during a mixed
20-token prefill plus one decode token. The traceback surfaced at a PLE
ring gather; this does not establish the cause. No candidate canonical
suite completed. Shell SERVE_RC0 describes cleanup and does not negate
the engine failure. Both-card post-health and source/SO checks passed.

Plan16 stays disabled and unqualified. There is no new headline speed
result. Full-model GDN validation proceeds on the original attention path.
See reports/3A1_audit.md and reports/3D1_decode.md for the finite evidence.
