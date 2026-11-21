# GPU test list for the cutover window (run in one window, in order; each <= 3 min unless noted)

Owner of ordering and gates: worker INT. Each item: command, expected time, PASS gate, rollback.
Sections per item: KV, SAMP, VIS, API, INT. Workers write their section below; INT transcribes each step into
tensorfold/flashnext/dd/gpu_window_steps.tsv (the file the runner executes) and orders it here.

## How the window runs (owner)

Before the window (CPU only, any time, Strata keeps serving):

    tensorfold/flashnext/dd/gpu_window.sh preflight     # frozen sha, clean worktree, steps file, fixture, port 18150 free

The window (both cards; Strata down for about 30 min; GPU total about 25 min, budget 45):

    CONFIRM=yes tensorfold/flashnext/dd/gpu_window.sh window
    # = write PRIORITY.hold (hotschmoe-dd-tf), stop hotschmoe-dd, G1 regress, G2 prod (all steps below), remove hold.
    # Any HARD gate FAIL -> Strata is restarted automatically and the window ends with no cutover.
    CONFIRM=yes WIN_DIR=<window dir printed above> tensorfold/flashnext/dd/gpu_window.sh cutover
    # = install + start hotschmoe-dd-tensorfold, dd_functional.py through :18080 with the image fixture,
    #   then enable TF / disable Strata. FAIL -> automatic rollback.
    tensorfold/flashnext/dd/gpu_window.sh rollback      # any time later: TF unit down (+xpu-health), Strata up, functional

Results: /mnt/vm_8tb/b70/results/hotschmoe-dd/gpu-window/<UTC tag>/gates.tsv (one line per gate), window.log, per-step dirs.
Every GPU touch is under bin/gpu-run (both cards); timed parts hold bin/cpu-run --quiet outside the lease. Port 18150
only; 18080/18124 are touched only by cutover and rollback. No prompt or response text is logged beyond gate summaries.

## Order and budget

| # | Step | Owner | Kind | GPU time | PASS gate | Rollback if it fails |
|---|---|---|---|---|---|---|
| 0 | preflight (CPU) | INT | hard | 0 | frozen binary sha == env, worktree clean at TF_COMMIT, no @FREEZE, steps file valid, fixture, :18150 free | fix and rerun; Strata untouched |
| 1 | stop Strata | owner/INT | - | 0.3 min | both card leases free within 120 s | systemctl --user start hotschmoe-dd |
| G1 | regress: int-rf2 config quick bench on the frozen binary | INT | hard + soft | 3 min | HARD: health pre/post HEALTHY (cards 0 1), p0 128-token greedy hash f8caa42cc5b4 bitwise == int-rf2, every C-level stream == solo p0; SOFT: C1/C2/C4/C8/PF2k >= 0.95 x int-rf2 (109.5/139.5/168.0/209.0/1560) | runner restarts Strata; regression bisect on CPU |
| G2.0 | health pre + production server load on :18150 (tf-dd-prod.env: 200000 x 4) | INT | hard | 2-3 min | xpu-health HEALTHY; /v1/models within 600 s; load line recorded (GiB/GPU, load s) | session teardown + post-health; Strata restarted |
| P0 | identity | INT | hard | 0.1 min | first id hotschmoe-dd, aliases listed, context_length 200000, WIN_EXPECT_CAPS (vision, greedy_only=false, penalties) | as G2.0 |
| P1 | quick serve bench at the production config | INT | soft | 1 min | completes (oracle recorded or == TF_PROD_EXPECTED, concurrency == solo); C1/C2/C4/C8/PF2k >= 0.90 x int-rf2 (C8 queues on 4 lanes) | owner decision; no automatic rollback |
| KV* | 90K prompt + decode, 200K x 4 admission, park/restore bitwise | KV | hard | <= 8 min | KV section | as G2.0 |
| SAMP* | sampling sanity + seeds + MTP acceptance at temp 0.6 | SAMP | hard/soft | <= 3 min | SAMP section | as G2.0 |
| VIS* | image request | VIS | hard | <= 2 min | VIS section | as G2.0 |
| API* | tool call + streaming, reasoning, tool_choice, errors | API | hard | <= 3 min | API section | as G2.0 |
| P9 | dd_functional.py direct to :18150 with image (no key) | INT | hard | 2 min | thinking ok + reasoning chars, tool call get_weather + round trip, 4 concurrent exact, vision blue, long stream >= 512 tokens with code | as G2.0 |
| G2.9 | server log scan + stop + health post | INT | hard | 0.5 min | no DEVICE_LOST/panic/error.X/OOM since launch; xpu-health HEALTHY | xe-reset in teardown; Strata restarted |
| C1 | cutover: start hotschmoe-dd-tensorfold (stops Strata via Conflicts) | INT | hard | 2-3 min | :18080 /v1/models answers with the key within 900 s | gpu_window.sh rollback (automatic) |
| C2 | dd_functional.py through :18080 with the key and image fixture | INT | hard | 2 min | same as P9 through the frontdoor | gpu_window.sh rollback (automatic) |
| C3 | enable TF unit, disable Strata unit, soak monitor -> --backend tf (drop-in backend-tf.conf; rollback removes it) | INT | - | 0 | units as stated | gpu_window.sh rollback |

Estimated GPU time: G1 3 + G2 load 3 + P0/P1 1.5 + KV 8 + SAMP 3 + VIS 2 + API 3 + P9 2 + teardown 1 + cutover 5 = about 31 min.

## INT

Frozen build (CPU, before the window): `tensorfold/flashnext/dd/build_dd.sh <b70/dd-serve head>`, then put TF_COMMIT,
TF_BIN and TF_BIN_SHA256 into tf-dd-prod.env and run preflight. The window tests exactly the binary the unit serves.

- G1 regress -- `gpu_window.sh regress` (inside window). Runs the unchanged tools/flashnext_serve_bench.sh from the
  frozen worktree with S_SKIP_BUILD=1, FN_BENCH_EXPECTED = workers/S/expected-current.json, in a clean environment
  (no XPU_FN_KV etc.), so it is the int-rf2 config: ctx 16384, --parallel auto, greedy, thinking off. 153 s at int-rf2.
  PASS: rc 0 (bench asserts oracle_equal and stream == solo; outer asserts HEALTHY pre/post); speed soft gate via
  gpu_window_checks.py compare (>= 0.95 x int-rf2).
- G2 prod -- `gpu_window.sh prod` (inside window): cpu-run --quiet + gpu-run (both cards) + timeout 2400 around one
  session: xpu-health, dd_launch_tf.sh with TF_PORT=18150 and tf-dd-prod.env, then every line of gpu_window_steps.tsv
  in order (each under its own timeout <= 180 s), log scan, stop, xpu-health (xe-reset first if the log shows a device
  loss). A HARD step failure ends the session at once.
- P0 identity, P1 bench, P9 functional: lines in gpu_window_steps.tsv. TF_PROD_EXPECTED (optional) is the KV worker's
  expected p0 token file for the production KV mode; without it P1 records p0 and checks concurrency == solo only.
- CPU self-test of the runner (mock server, no GPU): /mnt/vm_8tb/b70/results/tf-flashnext/workers/INT/selftest/run.sh
  -> SELFTEST PASS (hard failure stops the loop, soft failure recorded, teardown + post-health always run).

## KV

(worker KV: add step lines here -- id, hard|soft, timeout <= 180 s, exact command using $TF_URL, PASS gate. Needed:
90K-token prompt + 256-token decode coherent with TTFT recorded; 4 concurrent long requests admitted at 200000;
park/restore bitwise; the expected p0 token file for the production KV mode -> TF_PROD_EXPECTED.)

## SAMP

(worker SAMP: step lines -- temperature/top_p/top_k/min_p/penalties distribution sanity, seed repeatability, greedy
still bitwise under the sampling build, MTP acceptance rate at temperature 0.6 with a floor.)

## VIS

(worker VIS: step lines -- image request through the serving path with the fixture $TF_FIXTURE; encoder time; the
WIN_EXPECT_CAPS entry vision=true.)

## API

(worker API: step lines -- tool call + streamed tool-call deltas, tool_choice auto/none/required, reasoning_content
default on at effort medium, usage incl. cached tokens, /health, over-context error shape like Strata.)
