# hotschmoe-dd on TensorFold Flash-Next: cutover prep (owner: worker INT)

Nothing here is installed, enabled or started until the owner runs the cutover. Strata 0.2.16
(hotschmoe-dd.service, strata/flash-next/dd/hotschmoe-dd-0216.json) serves production on both cards until then.
Plan and status: ../SERVE-READINESS.md. The GPU window and its gates: ../GPU-TESTS.md.

| File | Role |
|---|---|
| build_dd.sh <commit> | frozen ReleaseFast xpu-native build in a detached worktree /mnt/vm_8tb/b70/build/tensorfold-dd/<commit8> (cpu-run, MemoryMax 90G, -j4); prints the sha256 |
| tf-dd-prod.env | production candidate: frozen b70/dd-serve build, 200000 context x 4, reasoning medium, 16384 thinking budget, item flags (KV/SAMP/VIS/API) |
| tf-dd-common.env | engine env shared by all variants (int-rf2 bench env) |
| tf-dd-b1.env, tf-dd-b2.env | older 372bf1b2 greedy-only variants (desc windows 20480; int8 paged 131072) |
| dd_launch_tf.sh | ExecStart body: sources the env file, checks the frozen sha and the port, execs the server (TF_PORT, TF_LISTEN, TF_EXTRA_ARGS, TF_DD_DRYRUN=1) |
| dd_poststop_tf.sh | ExecStopPost: xe-reset only if the last launch logged a device loss, then xpu-health under the lease |
| hotschmoe-dd-tensorfold.service | the TF unit (Conflicts=hotschmoe-dd.service, gpu-run holds both cards, Restart=on-failure) |
| tf_clamp_proxy.py + hotschmoe-dd-tf-clamp.service | fallback only for a greedy-only build (TF_LISTEN=unix): clamps sampling params; not used once SAMP lands |
| soak_monitor.py + hotschmoe-dd-soak.service | passive per-request record + alerts only (--backend strata / tf / auto); never logs prompt or response text |
| gpu_window.sh, gpu_window_steps.tsv, gpu_window_checks.py | the one GPU window: preflight, regress, prod steps, cutover, rollback |

## Freeze (CPU, before the window)

    tensorfold/flashnext/dd/build_dd.sh <b70/dd-serve head>        # ~builds under cpu-run; prints sha256
    # edit tf-dd-prod.env: TF_COMMIT=<full sha>, TF_BIN=.../tensorfold-dd/<commit8>/zig-out/xpu/bin/tensorfold-native,
    # TF_BIN_SHA256=<sha256>; then:
    TF_DD_CONFIG=tensorfold/flashnext/dd/tf-dd-prod.env TF_DD_DRYRUN=1 bash tensorfold/flashnext/dd/dd_launch_tf.sh
    tensorfold/flashnext/dd/gpu_window.sh preflight               # must end rc=0

## Window, cutover, rollback (owner)

    CONFIRM=yes tensorfold/flashnext/dd/gpu_window.sh window       # stops Strata; G1 regress + G2 prod; ~25 min GPU
    # HARD failure -> Strata restarted automatically, no cutover. All PASS -> it prints the cutover command:
    CONFIRM=yes WIN_DIR=/mnt/vm_8tb/b70/results/hotschmoe-dd/gpu-window/<tag> tensorfold/flashnext/dd/gpu_window.sh cutover

cutover does exactly this (manual equivalent in brackets):

1. Install the unit [cp tensorfold/flashnext/dd/hotschmoe-dd-tensorfold.service ~/.config/systemd/user/ &&
   systemctl --user daemon-reload].
2. Stop Strata, start TF [systemctl --user stop hotschmoe-dd; systemctl --user start hotschmoe-dd-tensorfold].
   The unit runs xpu-health, then serves 127.0.0.1:18124 behind the unchanged frontdoor :18080.
3. Wait for :18080 /v1/models with the API key (<= 900 s), then
   [python3 strata/flash-next/dd/dd_functional.py --image-fixture /mnt/vm_8tb/b70/results/hotschmoe-dd/restore-20261021/vision-fixture.png].
4. PASS: [systemctl --user disable hotschmoe-dd; systemctl --user enable hotschmoe-dd-tensorfold] and point the
   passive record at TF (drop-in ~/.config/systemd/user/hotschmoe-dd-soak.service.d/backend-tf.conf, --backend tf).
   FAIL at 3 or 4: rollback runs automatically.

Rollback (any time): `tensorfold/flashnext/dd/gpu_window.sh rollback`, i.e.
[systemctl --user stop hotschmoe-dd-tf-clamp hotschmoe-dd-tensorfold (ExecStopPost runs xpu-health);
systemctl --user disable hotschmoe-dd-tensorfold; systemctl --user enable --now hotschmoe-dd; remove backend-tf.conf;
wait for :18080; dd_functional.py with the image]. If Strata does not come back: bin/xe-reset, then start it again.

## After cutover

- Users are the soak. Only alerts are reported (errors, crashes, restarts, TTFT slowdowns): soak_monitor ALERTS.log.
- Logs: /mnt/vm_8tb/b70/logs/hotschmoe-dd/tf-server.log (server), soak.log (record).
- A crash restarts the unit (Restart=on-failure, 30 s, at most 4 in 30 min); a device loss triggers xe-reset in ExecStopPost.
