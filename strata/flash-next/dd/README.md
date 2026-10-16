# hotschmoe-dd on Strata (night-final-v3, 2026-10-16)

The office daily driver: Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS on two B70s, Strata expert-parallel, batch MTP
depth 2, 2 slots x 200k context, 32 GiB / 4-entry RAM parking. Engine and flags: docs/20261016_night_lab.md.

| Piece | Where |
|---|---|
| Engine binary | /mnt/vm_8tb/b70/build/strata-ab/releases/strata-night-20261016 (src-ep branch night-20261016, sha256 52d1778c...) |
| Config | strata/flash-next/dd/hotschmoe-dd.json (= night/configs/night-final-v3-c2.json, release exe, log path) |
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
- Status: `systemctl --user status hotschmoe-dd`, `curl -s localhost:18080/health`.
- Research mode (frees both GPUs): `systemctl --user stop hotschmoe-dd` (the frontdoor can stay; it answers 502 meanwhile).
  Restart: `systemctl --user start hotschmoe-dd`.
- Fallback to the 27B INT4 vLLM driver: `systemctl --user stop hotschmoe-dd hotschmoe-dd-frontdoor`, then start
  vllm/int4/native5802_trial/serve.py (it brings its own frontdoor on :18080 and the same key).
- Functional smoke test (thinking, tools round trip, 4 concurrent, long stream): `python3 strata/flash-next/dd/dd_functional.py`.
- Engine crashes inside the container restart on the next request (Strata server); a dead container restarts via
  systemd (Restart=on-failure, 30 s).

Measured at deploy (fixed prompts): C1 ~141 tok/s (2048-token code reply 152.6), C2 ~92/stream, prefill 4K ~2200 /
32K ~2150 tok/s; functional checks pass through the frontdoor and through open-webui. Feedback from real use goes in
JOURNAL.md / docs/20261016_night_lab.md follow-ups.
