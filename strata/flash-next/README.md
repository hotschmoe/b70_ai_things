# strata/flash-next

Live: model-lock-ista-iq3.json (ISTA GSQ-RCO IQ3_S and IQ3_XXS artifact hashes).
The serving build and configs are outside the repo under
/mnt/vm_8tb/b70/build/strata-ab; launch notes are in JOURNAL.md (2026-10-13).

Current measured config: serve-iq3xxs-200k.json, stock upstream
296b59d2/v0.1.44.1. Split27, two200k, MTP4, int8 KV, all main experts in
VRAM, parking16384 MiB/4 entries. Bounded capacity, forced RAM restoration,
concurrent coherence, reasoning/tool checks and teardown health passed.
Evidence: docs/20261014_flashnext_strata_optimization.md. Multi-day soak
is started separately and remains unqualified.

Start persistently after the previous endpoint has stopped and GPU health
has passed:

```sh
systemd-run --user --unit=strata-optimized --collect \
  --property=WorkingDirectory=/mnt/vm_8tb/github/b70_ai_things \
  /bin/bash /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/launch.sh
```

Stop: `docker stop -t 30 strata-ab-serve`. Endpoint: :18085/v1,
`hotschmoe-dd` first, detailed IQ3_XXS alias retained.
Switch to 27B: `bash strata/flash-next/fallback-27b.sh` (:18080/v1).
The wrapper passes the existing prerequisite check and shell syntax check;
its complete model switch has not been exercised in this stage.

Probe scripts operate through the already-leased server:

- bench_serving.py: raw SSE, live identity, matched fresh prompts and timings.
- bench_context.py --parking: near-full concurrent retrievals, overwrite both
  GPU slots with short chats, then verify concurrent cached-prefix returns.
- soak.py: HTTP metadata, container RAM, per-card DRM fdinfo, and error lines;
  no test prompts, GPU queries or automatic restarts. Stop its persistent
  observer with `systemctl --user stop strata-soak.service`.

The paused UD-Q4_K_XL RAM-spill campaign code that used to live here was
moved to the gitignored archive/flashnext-udq4xl-paused-20261014/ on
2026-10-14 and remains in git history before that commit. Its docs are in
docs/flashnext_udq4xl_paused/.
