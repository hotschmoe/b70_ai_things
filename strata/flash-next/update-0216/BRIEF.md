# Strata update to upstream v0.2.16 + DD patch port (brief for Codex, 2026-11-19)

Orchestrator: Claude (Opus 5.5). Worker: you (Codex gpt-6.1-terra, xhigh).
Read first: /mnt/vm_8tb/github/b70_ai_things/AGENTS.md (= CLAUDE.md standing rules), strata/flash-next/dd/README.md,
strata/flash-next/night/pr020/README.md, docs/20261016_night_lab.md (skim), docs/20261021_strata_dd_restore.md.

## Goal
Move the daily-driver Strata engine (Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS, two B70s,
expert-parallel, batch MTP, 4 slots x 200K RAM-backed KV, vision) from our old night branch to
current upstream, carry our custom patches forward, and measure how Flash-Next looks on the new
version. The user's next priority after this is CONCURRENCY (C2/C4), so the result must leave a
clean, profile-ready v0.2.16 build. Do not start concurrency optimization yet.

## Current state (facts)
- Old engine: /mnt/vm_8tb/b70/build/strata-ab/src-ep branch night-20261016 (HEAD e753e9cb),
  merge base with upstream = v0.1.44.1. Release binary used by the DD:
  /mnt/vm_8tb/b70/build/strata-ab/releases/strata-night-20261016. Night commits are listed by
  `git log --oneline main..night-20261016` there; extra patch files in strata/flash-next/patches/.
- Upstream: origin = github.com/Niko1221/Strata, origin/main = v0.2.16-4-g58ee32277 (~5660 commits
  past our base). Upstream dropped DeepSeek after 0.2.13 (irrelevant here).
- Our open upstream PRs (rebased on 0.2.0 on 2026-10-19, branches pr/* in src-ep): #2802 IQ decode,
  #2804 KREG mmvq, #2806 decode attn, #2808 prompt-attn union, #2810 HC Q8, #2812 HC fused read,
  #2526 slot-cache preserve, #2511 draft accounting, draft #3273 EP prompt reduce + PUSH. Check with
  `gh pr view N -R Niko1221/Strata` whether upstream merged, superseded or reworked any of them.
- DD config: strata/flash-next/dd/hotschmoe-dd.json (args + a long STRATA_* env list). Only the
  features that config actually turns on are "required carries"; everything default-off in the night
  branch is optional and should be dropped unless it measures as a win on v0.2.16.
- Build recipe for 0.2.x: PR_BUILD_IMAGE=strata-sycl-dev:020 (IGC 2.41.5); the old runtime image
  sha256:e41d59c0 (IGC 2.36.3) ICEs on int8 prompt-expert kernels. If v0.2.16 needs a newer
  oneAPI/IGC, build a new dev/runtime image from upstream's own Docker/CI recipe and record identity.
- Model files: packs under /mnt/vm_8tb/b70/build/strata-ab/packs/, GGUFs under
  /mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS, MTP at /mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt.
  The pack/MTP formats may have changed upstream: regenerate with upstream tools if required
  (keep new artifacts beside the old ones, never overwrite the old ones).
- hotschmoe-dd.service is STOPPED (user direction). Both GPUs are free. Frontdoor :18080 stays up and
  points to :18124: do NOT run test servers on 18124; use 18134.

## Work plan
S0 (no GPU). Record identity (host kernel, UMD/Level Zero, images, upstream sha). Create a new
   worktree off origin/main: /mnt/vm_8tb/b70/build/strata-0216/src on branch dd-0.2.16 (build
   outputs under /mnt/vm_8tb/b70/build/strata-0216/). Map every DD-enabled feature/env flag to
   (a) already upstream (maybe under another name), (b) needs porting, (c) obsolete. Write that map
   to /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/update-0216/feature_map.md before porting.
S1 (GPU). Build STOCK v0.2.16 SYCL. Serve Flash-Next IQ3_XXS with upstream-only flags as close to the
   DD config as upstream allows (EP, MTP, parallel 4, KV streaming/resident, vision). Also re-measure
   the OLD night-final-v3 release with the same harness in the same session (matched baseline; reuse
   hotschmoe-dd.json via a copy with port 18134). Metrics: coherence (fixed prompts, deterministic
   check), C1 decode, C2 and C4 per-stream + aggregate decode, prefill at 4K and 32K, TTFT,
   a long-context (>=128K) cached restore, and teardown + post-health. Harnesses already exist:
   strata/flash-next/bench_concurrency.py, bench_context_four.py, bench_serving.py,
   strata/flash-next/dd/dd_functional.py. Fix harness drift rather than writing new harnesses.
S2 (GPU). Port the required carries onto dd-0.2.16 as small default-off commits (one feature per
   commit, upstream style, byte-identical default path). Prefer upstream equivalents over our code.
   Measure each group against stock v0.2.16; keep only what helps or is needed for the DD config.
S3 (GPU). Assemble a candidate DD config for v0.2.16 (new file
   strata/flash-next/dd/hotschmoe-dd-0216.json; do not edit hotschmoe-dd.json), run the full
   functional check (thinking, tools round trip, 4 concurrent, images, long stream) on :18134 with
   served names hotschmoe-dd (first) + qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216.
S4 (no GPU). Prepare (locally only) refreshed PR branches for our still-open upstream PRs on
   v0.2.16 where the code still applies. List which are moot.

## SCOPE UPDATE (user, 2026-11-19 21:50 UTC) - supersedes S1/S2/S3 above where they differ
- Do NOT measure each ported patch or each feature group. The user does not care about per-step data.
- Port the daily driver's ACTUAL config (hotschmoe-dd.json: EP via --expert-parallel-device, batch MTP,
  4 slots x 200K, --kv-resident 65536 RAM-backed KV, conversation parking, vision, and the STRATA_* env
  switches it sets to 1/non-default) onto v0.2.16. Where upstream already has the feature (some of our
  PRs/patches may have been upstreamed or superseded, e.g. #4185 fast IQ helpers), use upstream's.
  Carry the rest as needed (EP from our #3273/#2205 stack is required).
- Get it running at the daily-driver level, then measure ONCE: the ported 0216 build vs the old
  night-final-v3 baseline (which you have already been measuring - keep that data). A stock-0216
  measurement is optional; skip it unless it is nearly free.
- Then do S3 (candidate config + functional check) and S4 (local PR branch refresh). Same rules and
  stop conditions as below.

## MODEL NOTE (user, 2026-11-19 22:30 UTC)
For consultations use the newest models at high effort: `claude --model fable --effort high` and
`claude --model opus --effort high` (aliases resolve to the latest, currently Fable 5.2). Record the
responding model IDs.

## Rules
- Every GPU touch via /mnt/vm_8tb/github/b70_ai_things/bin/gpu-run (whole box for EP/two-card runs). bin/xpu-health before the
  first and after the last GPU run; after a crash/DEVICE_LOST run bin/xe-reset, then xpu-health.
- Builds through /mnt/vm_8tb/github/b70_ai_things/bin/cpu-run, MAX_JOBS<=4 / docker --memory 90g (host has 121 GB RAM). Timed
  measurement windows under bin/cpu-run --quiet. Host /tmp is tmpfs: scratch on disk.
- Do NOT touch: src-ep night branch, releases/strata-night-20261016, hotschmoe-dd.json, the systemd
  units, archive/. Do not start hotschmoe-dd. Do not push to GitHub, comment on PRs, or force-push
  anything. Do not commit in the b70_ai_things repo (the orchestrator commits); committing in the new
  Strata worktree's dd-0.2.16 branch is fine. Preserve the dirty user files in b70_ai_things.
- ASCII only. Evidence: /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/update-0216/JOURNAL.md, one entry per stage as CONFIG -> COMMAND -> RESULT ->
  VERDICT, short. Raw logs under /mnt/vm_8tb/b70/results/strata-0216/. No speed claim without a
  matched config, coherence check, served identity (/v1/models), health and teardown.
- Light evidence, no verification loops; do not re-run a passing check without a reason.

## Stop conditions (report and stop)
- Stock v0.2.16 cannot load Flash-Next on SYCL after reasonable fixes, or two consecutive
  DEVICE_LOST/health failures: stop with diagnosis.
- Finished S3 (or S4): stop.
- Budget about 8 hours wall. If you run long, stop at a stage boundary with state written down.

## Final report (last message)
Table old-night vs stock-0216 vs ported-0216 (C1, C2/stream, C4/stream+agg, prefill 4K/32K, TTFT,
long-ctx restore, coherence pass/fail), the feature map summary (ported / upstream-equivalent /
dropped), open issues, exact binary + config paths, and the ranked list of concurrency bottlenecks
you observed (inputs for the next profiling stage).
