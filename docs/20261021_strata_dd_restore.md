# Strata daily service with four streams and 200K context

The user requested daily Strata serving on 2026-10-21 after the C4 research
comparison. `hotschmoe-dd` and its API-key frontdoor are active and enabled.
The service supports four active batch slots, a 200000-token maximum context
per request, vision and RAM-backed KV. Grafana on 3001, Open WebUI on 3000
and Prometheus are running; Grafana and WebUI health endpoints return 200.

## Configuration

CONFIG -> Retain the night-final-v3 release binary, two B70 expert-parallel
layout, ISTA GSQ-RCO IQ3_XXS shards, CPU vision encoder and BF16 mmproj.
Update both the vision daily config and text fallback:

- `parallel`: 2 to 4.
- `STRATA_BATCH_MTP_DEPTH`: 2 to 1; solo `--spec 4` remains.
- Add `--kv-resident 65536`; retain `--max-context 200000`, INT8 KV and
  `--conversation-cache-mib 32768 --conversation-cache-slots 4`.

The systemd units and launcher are unchanged. They load the tracked config
and hold both GPU leases. KV streaming stores full K/V in pinned RAM and
stages needed cells into the resident pool; idle-conversation parking is
separate. Four whole 200K sessions in VRAM cannot fit this expert placement:
the cleanly refused first launch left 60 experts/layer on card 0 and 452 on
card 1, requiring 35.28 GiB there with 31.84 GiB free. With RAM-backed KV the
successful launch places 171/341 experts/layer and keeps all experts in VRAM.

Identity:

- Binary: `/mnt/vm_8tb/b70/build/strata-ab/releases/strata-night-20261016`.
- Binary SHA256: `52d1778c909b90123a8947a3a2a9700370b76a2f64d293765f0e2510526fdad2`.
- Container image: `sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
- Vision config SHA256: `ac29d1cbd4e868e135c98a1be0c011b6317338d74b8d2c483ba67c1556eb2f25`.
- Kernel: `7.1.0-070100-generic`; API engine version: `0.1.44.1`.
- Primary model: `hotschmoe-dd`; research alias:
  `qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata`.

## Live checks

COMMAND -> `bin/gpu-run` wrapped per-card and compiled-pair P2P0 health
before serving. `systemctl --user start hotschmoe-dd` uses the retained
leased launcher and another per-card precheck. Through the API-key
frontdoor, run the retained `strata/flash-next/dd/dd_functional.py` plus
bounded image, concurrent-stream and long-context probes. Raw scripts,
requests, outputs, configuration, identity and logs are saved under:
`/mnt/vm_8tb/b70/results/hotschmoe-dd/restore-20261021/`.

RESULT -> Per-card and compiled-pair health pass. The fully resident
capacity refusal reports no device failure, and the next unit precheck
passes. Live `/v1/status` reports concurrency 4 requested/serving; `/slots`
has four 200000-context entries; `/health` reports loaded/images true;
`/v1/models` lists `hotschmoe-dd` first.

- Vision: identify the blue circle in a generated image containing a blue
  circle and red square; answer `blue`.
- Four simultaneous 512-token code streams: all coherent, all 512 tokens,
  four slots observed busy together and 10.13s of overlapping content.
- Retained functional checks: thinking arithmetic, tool call and tool
  result round trip, four concurrent arithmetic answers and 2048-token
  streamed code response all pass.
- Long context: 198036 input tokens, exact beginning-of-prompt marker
  returned correctly, 12 output tokens, 207.28s wall. This exercises KV
  streaming beyond the 65536-cell resident pool.
- After an intervening request, the same long prompt returns the marker
  again in 2.38s. Usage records 198029 cached tokens, leaving 7 tail tokens
  to read; engine evidence records a 198029-token checkpoint restored from
  the conversation cache in 908ms.
- No matching GPU fault/wedge in the saved kernel window. Both service
  units remain active/running/enabled. Live client traffic resumed during
  checks, so the service was left running without a further teardown or
  post-health run. Timings are functional observations under live traffic,
  rather than controlled throughput measurements.

VERDICT -> Functional readiness for the requested daily settings passes.
These checks cover four short simultaneous streams and one near 200K
conversation with RAM restore. They do not benchmark four full 200K
conversations simultaneously or qualify long-term stability. Keep the
[matched C4 rates](../vllm/flashnext/reports/20261021_c4_compare.md) labeled
with their original configurations: its Strata baseline used four 64K,
fully resident slots, rather than this 200K RAM-backed daily config.

Stop the daily service before resuming the
[queued vLLM C4 investigation](20261021_flashnext_c4_next.md); keep it down
between experiment arms. The latest user request to restore daily serving
supersedes the earlier instruction to leave it down after research.
