# hotschmoe-dd on Strata (night-final-v3, updated 2026-10-21)

The office daily driver: Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS on two B70s, Strata expert-parallel, batch MTP
depth 1 for concurrent batches, 4 slots, 200000-token maximum context per
request, 65536 resident KV cells per slot backed by pinned system RAM,
and 32 GiB / 4-entry idle-conversation parking. Solo requests retain
`--spec 4`. Engine and original flags: docs/20261016_night_lab.md.

The Oct21 update raises `parallel` from 2 to 4 and changes batch MTP depth 2
to 1, matching the retained eight-row C4 verify path. It adds
`--kv-resident 65536`: allocating four whole 200K histories in VRAM leaves
too little room for the expert-parallel complement on card 1. KV streaming
keeps the full history in RAM and stages needed KV without recomputing it;
idle-conversation parking is a separate existing cache. The release binary,
vision encoder, quantization, API key and systemd units are retained.
Live image, four overlapping streams, thinking/tools and 198K-context
cached-restore checks pass; see [Oct21 deployment evidence](../../../docs/20261021_strata_dd_restore.md).

| Piece | Where |
|---|---|
| Engine binary | /mnt/vm_8tb/b70/build/strata-ab/releases/strata-night-20261016 (src-ep branch night-20261016, sha256 52d1778c...) |
| Config | strata/flash-next/dd/hotschmoe-dd.json (four slots, 200K max, RAM-backed KV, vision); text fallback matches slot/KV settings |
| Images | on since 2026-10-16: the config's "vision" section (CPU strata-vision, build-vision-cpu/bin, mmproj BF16) + `--vision`; text-only fallback hotschmoe-dd-text.json |
| Backend | 127.0.0.1:18124 (Strata server, container `hotschmoe-dd`, holds the GPU pair lease) |
| Public endpoint | 0.0.0.0:18080/v1, API key /mnt/vm_8tb/b70/secrets/dd_api_key (frontdoor: vllm/fp8/openai_key_frontdoor.py) |
| Model ids | `hotschmoe-dd` (first), `qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata` |
| Units | ~/.config/systemd/user/hotschmoe-dd.service, hotschmoe-dd-frontdoor.service (enabled, linger on) |
| Logs | /mnt/vm_8tb/b70/results/hotschmoe-dd/engine.log, /mnt/vm_8tb/b70/logs/hotschmoe-dd/{server,frontdoor}.log |
| Monitoring | Prometheus job `hotschmoe_dd_strata` (127.0.0.1:18124/metrics/prometheus); Grafana :3001 home = "hotschmoe-dd Strata" |
| UI | open-webui :3000 (v0.12.3) -> http://127.0.0.1:18080/v1 |

Install (already done): `cp strata/flash-next/dd/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload &&
systemctl --user enable --now hotschmoe-dd-frontdoor hotschmoe-dd`. Start runs per-card xpu-health first, then loads in ~40 s.

Operations:
- Text-only fallback: `DD_CONFIG=.../hotschmoe-dd-text.json` in the unit, or copy it over hotschmoe-dd.json, then restart.
- Status: `systemctl --user status hotschmoe-dd`, `curl -s localhost:18080/health`.
- Research mode (frees both GPUs): `systemctl --user stop hotschmoe-dd` (the frontdoor can stay; it answers 502 meanwhile).
  Restart: `systemctl --user start hotschmoe-dd`.
- Fallback to the 27B INT4 vLLM driver: `systemctl --user stop hotschmoe-dd hotschmoe-dd-frontdoor`, then start
  vllm/int4/native5802_trial/serve.py (it brings its own frontdoor on :18080 and the same key).
- Functional smoke test (thinking, tools round trip, 4 concurrent, long stream): `python3 strata/flash-next/dd/dd_functional.py`.
- Engine crashes inside the container restart on the next request (Strata server); a dead container restarts via
  systemd (Restart=on-failure, 30 s).

Measured at the original two-slot Oct16 deploy (fixed prompts): C1 ~141 tok/s (2048-token code reply 152.6), C2 ~92/stream, prefill 4K ~2200 /
32K ~2150 tok/s; functional checks pass through the frontdoor and through open-webui. Feedback from real use goes in
JOURNAL.md / docs/20261016_night_lab.md follow-ups.

## Matched C4 reference and next tests

Oct21 comparison, two B70s, four cold 19798-token prompts,512 output tokens
each, thinking off. Rates aggregate tok/s; median of two measured batches.

| Recipe | Cold effective PP | Steady C4 TG | Cold output per wall second | Cold batch wall s |
|---|---:|---:|---:|---:|
| Strata Flash-Next IQ3_XXS, MTP depth1 | 1739.5 | 188.5 | 38.84 | 52.74 |
| vLLM Flash-Next IQ3_XXS, MTP1 | 2262.3 | 121.4 | 39.44 | 51.92 |
| vLLM Qwen3.8-27B AutoRound INT4, MTP3 | 722.4 | 161.4 | 16.75 | 122.29 |

Effective PP includes scheduling and overlapping decode; steady TG uses
only the all-four decode interval. Strata required a separate cached replay
for steady TG; its cold PP/wall remain cold. This Strata reference used
four 64K slots with fully resident KV and is not a new benchmark of the
four 200K RAM-backed daily config.
[Full comparison](../../../vllm/flashnext/reports/20261021_c4_compare.md).
[Next research item and 2-3 hour budget](../../../docs/20261021_flashnext_c4_next.md).

## Soak notes (from 2026-10-16: weeks of office use)

What recovers by itself: an engine crash (the server starts the engine again on the next request), a dead container
(systemd restarts it after 30 s), a host reboot (both units, open-webui, Prometheus and Grafana come back; the DD start
runs per-card health first). A reply stuck on one token is ended after 256 repeats. There is no content watchdog (the
old dd-watchdog stays retired by operator decision).

What needs a hand:
- Replies turn to garbage (`!!!!` or nonsense) and stay that way: `systemctl --user restart hotschmoe-dd` (about 45 s).
  The SYCL port has no non-finite-logits guard (upstream's #879 guard is CUDA/HIP only), so a poisoned conversation
  cache survives until the engine restarts.
- The unit fails to start (its xpu-health step fails, or DEVICE_LOST in engine.log): `bin/xe-reset`, then start it
  again; reboot only if the reset ladder fails.
- Replies cut off while thinking: the defaults are medium effort with a 16,384-token thinking budget; the open-webui
  "think harder" preset allows 32,768. Raise reasoning_budget_tokens in hotschmoe-dd.json (and restart) if needed.
- Logs: engine.log carries per-window trace lines (a few MB per hour of generation); server.log and frontdoor.log
  grow about 1 MB a day. Plenty of disk; look at them monthly.
Report odd behaviour with the time and the client (open-webui, Pi, API) so it can be matched in the logs and the
Grafana dashboard (:3001).
