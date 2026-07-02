# deploy/ -- running the daily-driver as a managed service

## How the serve runs today (manual launch)

`daily_driver_serve.sh start` launches the serve **fully detached**: `nohup setsid <gpu-run lease> bash -c '<recipe
serve.sh> && docker wait'` in a NEW session, and the vLLM itself is a `docker run -d` container owned by the Docker
daemon. So it **survives this shell / SSH / Claude session closing** -- it keeps running until you `stop` it.

```
# start (int4 DP=2 NONE -- the SHIPPED wedge-proof weekend serve, API-key enforced, 96k context):
DD_MODEL=qwen36-27b-int4 DD_REPLICAS=2 DD_MAXLEN=98304 DD_ENV="GRAPH=1 CGMODE=NONE UTIL=0.88" \
  DD_API_KEY="$(cat /mnt/vm_8tb/b70/secrets/dd_api_key)" \
  ./daily_driver_serve.sh start
# (UTIL 0.88 is THE safety lever -- dialed back from 0.95 after the 2026-06-26 ~7h "!!!!" incident (0.95 left
#  ~1-2 GiB headroom; the fp16 KV cache poisoned a persistent buffer after hours). max-model-len does NOT change
#  VRAM: vLLM sizes the KV pool from UTIL (~280k tok/replica at 0.88 single-card), so 98304 (96K) only caps a
#  single request's length and fits with room to spare. Run the watchdog alongside, below.)

./daily_driver_serve.sh status     # model, GPU lease, replicas/proxy, served id, web ui
./daily_driver_serve.sh logs       # follow the serve container log
./daily_driver_serve.sh stop       # stop container(s) + release the GPU lease
tail -f /mnt/vm_8tb/b70/dd-logs/daily_driver.log   # the launch/health log
```

What the manual launch does NOT do: survive a **box reboot** (no restart policy; the gpu-run lease wrapper does
not re-run). A reboot only happens here on a TP=2 BCS wedge -- which needs a manual reboot anyway -- so for an
unattended weekend the detached launch is sufficient. For real production, use the systemd unit below.

## systemd unit (boot-start + managed lifecycle)

`deploy/b70-daily-driver.service` runs the daily-driver under systemd: `systemctl start/stop/status`, and
**auto-start on boot** (`enable`). The API key is read from `/mnt/vm_8tb/b70/secrets/dd_api_key` at start (not
baked into the unit). Install (needs sudo -- one-time):

```
sudo install -m 0644 -o root -g root deploy/b70-daily-driver.service /etc/systemd/system/b70-daily-driver.service
sudo systemctl daemon-reload
sudo systemctl enable --now b70-daily-driver      # start now + on every boot
sudo systemctl status b70-daily-driver
journalctl -u b70-daily-driver -f                 # follow
sudo systemctl stop b70-daily-driver              # stop
```

The unit defaults to the **int4 DP=2** serve (`DD_MODEL=qwen36-27b-int4 DD_REPLICAS=2`,
`DD_MAXLEN=98304`, `DD_ENV="GRAPH=1 CGMODE=NONE UTIL=0.88"`). To switch to the faster **w8a8 +MTP TP=2**
(`DD_MODEL=qwen36-27b-w8a8-sqgptq-mtp DD_REPLICAS=1`, drop the `DD_ENV` line), edit the `Environment=DD_*` lines,
then `daemon-reload` + `restart`. NOTE (2026-07-02): the TP=2 BCS/GuC DEVICE_LOST wedge is CURED on kernel 7.1
(the 70.54.0 pin is retired; AGENTS.md GPU Discipline), so w8a8 TP=2 is fine unattended -- and on sglang it is
already the production daily driver. If the manual launch is already running, `stop` it first so the unit owns the lease.

### Caveats / levels of resilience

- **oneshot + RemainAfterExit** (this unit): gives managed start/stop + **boot-start**. It does NOT supervise the
  detached container, so it will not auto-restart the container if it crashes mid-run (RemainAfterExit keeps the
  unit "active"). Good enough for boot-persistence + clean management.
- **Full crash-restart** (container auto-restarts on crash): the cleanest way is a foreground-supervised unit
  (`Type=exec` running the gpu-run lease + `docker wait` in the foreground, `Restart=on-failure`) -- this needs a
  small "foreground" mode added to `daily_driver_serve.sh` so it does not self-detach. Ask if you want that.
- **A BCS hardware wedge cannot be fixed by any restart** -- the card is hung; it needs a `sudo reboot` (GuC
  70.54.0 firmware fix makes it rare). systemd boot-start does bring the serve back automatically *after* a reboot.
- **Secret hygiene:** `VLLM_API_KEY` is passed as a container env, so it is visible in `ps` / `docker inspect` to
  local users. Rotate the key in the secret file if it leaks; restart to apply. The file is `chmod 600`, off-repo.

## Watchdog -- content probe + self-heal (closes the "!!!!" diagnostics gap)

On 2026-06-26 the int4 DP=2 serve ran ~7h then returned `!!!!` on every prompt. The containers did **not** crash
and `/health` stayed 200 the whole time -- it was soft in-process corruption (a NaN poisoning a persistent KV
buffer -> argmax falls to token 0 = `!`). A liveness probe and nginx `max_fails` BOTH miss this (the upstream
answers 200 with garbage), and nothing captured `docker logs`/dmesg/VRAM at the moment it turned. `bin/dd-watchdog`
closes both gaps: it probes each replica's **output** and, on confirmed garbage, snapshots diagnostics then bounces
ONLY that replica (`docker restart` = same card pin, same API-key env, no GPU-lease re-acquire, **no reboot** --
single-card ops cannot BCS-wedge).

```
bin/dd-watchdog            # loop: probe each replica (output, not just /health) -> heal on garbage -> sleep 60s
bin/dd-watchdog --once     # single pass, print verdicts, exit  (smoke test against the live serve)
bin/dd-watchdog --no-heal  # detect + write incidents, but NEVER restart (forensics-only)
```

Per pass, per replica (dp0:18091/card0, dp1:18092/card1): if the container is not running, or has been up less
than `BOOT_GRACE=360s`, it is **skipped** (a cold start = model load + JIT is ~120-280s of health-down -- never
mistake a still-loading replica for a dead one). Otherwise `GET /health` -> `GET /v1/models` (discovers the served
id) -> a known-answer generation (temp 0, thinking off). Verdict: `OK` (reply contains the expected answer),
`GARBAGE` (empty / single-char flood / >=70% one char -> the `!!!!` signature), or `WARN` (wrong-but-coherent ->
logged, NOT healed, so a verbose answer never triggers a false restart).

**The watchdog restarts ONLY on confirmed GARBAGE** (health 200 + degenerate output -- the one failure nothing
else catches). Booting, not-running, health-down, `/v1/models`-down, and gen-fail are all **observe-only**: nginx
`max_fails` already ejects a hard-down upstream, and the serve launcher / systemd own restarting a dead replica.
(Healing on health-down was the original bug -- on 2026-06-26 it bounced both replicas mid-cold-start and tripped
the launcher's `EXITED EARLY` check; heal-on-garbage-only + boot-grace is the fix.) After `DEBOUNCE=2` consecutive
`GARBAGE` it writes `dd-logs/incidents/dd-incident-<ts>-<replica>.log` (docker-logs error grep + dmesg xe/GuC tail
+ docker stats + xpu-smi) and `docker restart`s that replica. Rails: at most one heal per pass (staggers a both-bad
event), a 7-min cooldown after each restart (the replica JIT-compiles ~5min on boot), and a cap of 4
restarts/replica/hr -- past that it **ESCALATEs** (keeps logging, stops auto-bouncing) since a persistent fault is
likely hardware, not a config bug. Logs to journald + `dd-logs/dd-watchdog.log`.

**Order of operations matters:** because the watchdog is harmless to a cold start (it skips booting replicas), it
is safe to run continuously, including during a `daily_driver_serve.sh start`. But the FIRST buggy version healed
on health-down -- if you ever see it restart a loading replica, `sudo systemctl restart b70-dd-watchdog` to pick up
the current heal-on-garbage-only logic.

Run it as a service alongside the serve (works with the manual launch OR the unit above -- it does not own the
serve lifecycle):

```
sudo install -m 0644 -o root -g root deploy/b70-dd-watchdog.service /etc/systemd/system/b70-dd-watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable --now b70-dd-watchdog
journalctl -u b70-dd-watchdog -f
```

Note: dmesg capture in incident files needs the watchdog user to read the kernel ring buffer (`kernel.dmesg_restrict=0`
or `CAP_SYSLOG`); otherwise that section records the restriction and the rest of the incident still captures. The
watchdog only ever `docker restart`s a single card's container -- it never reboots and never touches the TP=2 path.
