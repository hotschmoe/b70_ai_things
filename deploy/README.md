# deploy/ -- running the daily-driver as a managed service

## How the serve runs today (manual launch)

`daily_driver_serve.sh start` launches the serve **fully detached**: `nohup setsid <gpu-run lease> bash -c '<recipe
serve.sh> && docker wait'` in a NEW session, and the vLLM itself is a `docker run -d` container owned by the Docker
daemon. So it **survives this shell / SSH / Claude session closing** -- it keeps running until you `stop` it.

```
# start (w8a8 +MTP TP=2 NONE, API-key enforced, MAXLEN 65536):
DD_MODEL=qwen36-27b-w8a8-sqgptq-mtp DD_REPLICAS=1 DD_MAXLEN=65536 \
  DD_API_KEY="$(cat /mnt/vm_8tb/b70/secrets/dd_api_key)" \
  ./daily_driver_serve.sh start

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

Change the model/config by editing the `Environment=DD_*` lines (e.g. `DD_MODEL=qwen36-27b-int4 DD_REPLICAS=2`
for the wedge-proof int4 DP=2), then `daemon-reload` + `restart`. If the manual launch is already running, `stop`
it first so the unit owns the lease.

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
