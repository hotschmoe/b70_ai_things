# abfast: small paired agentic-coding eval (Pi 1.2.0, no Harbor)

Tiers: base (19 commit replays, this section), hard, langs (Rust/Go/Dart), sysadmin,
longrun (built, not yet run). Pi is called as /opt/pi/node_modules/.bin/pi: the image
PATH also carries a Pi 0.85.1 from the host Node tree, and runs before 2026-10-14 used
that older Pi by mistake (see JOURNAL.md).

19 validated "commit replay" tasks: repo at the parent commit, instruction written from
the real commit's intent, hidden verifier = the real commit's tests (withheld files are
overwritten from the real sha; Zig inline tests are appended from the real file).
Every task was validated: hidden tests FAIL at parent, PASS at real, inside the image.
Data (clones, runs) lives in /mnt/vm_8tb/b70/evals/abfast/ ; nothing touches owner checkouts.

## Setup (once)
  docker build -t abfast:1 -f evals/abfast/Dockerfile \
    --build-context node=$HOME/.nvm/versions/node/v24.17.0 --build-context zig=$HOME/.local/zig \
    --build-context pi=/mnt/vm_8tb/b70/tooling/pi-1.2.0 evals/abfast
  Source clones: /mnt/vm_8tb/b70/evals/abfast/src/<repo> (git clone of the owner repos / upstreams).
  Re-validate tasks (no model): python3 evals/abfast/run_arm.py --validate --parallel 6 --write

## Run an arm (no GPU use; talks to an OpenAI-compatible endpoint)
  python3 evals/abfast/run_arm.py --base-url http://192.168.10.5:18080/v1 \
    --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd \
    --arm-label armA --parallel 4 --timeout-s 900 --thinking xhigh
  Options: --tasks id,id  --max-output-tokens (16384; 8192 if off)  --context-window 131072

## Hard tier (tasks_hard.json)
Separate, non-saturated tier: multi-file / multi-commit changes, instructions give the goal and the public API
the hidden tests use, not a recipe. Selected with --tier hard, which switches the tasks file to tasks_hard.json and
the protocol defaults to --timeout-s 3600, --max-output-tokens 32768 (thinking xhigh, context 131072). The base
tier is unchanged. Each task carries its own "image" (abfast:1, or abfast:2 for extra python deps).
  python3 evals/abfast/run_arm.py --tier hard --base-url http://192.168.10.5:18080/v1 \
    --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --arm-label hardA --parallel 2 --thinking xhigh
  Validate (no model): python3 evals/abfast/run_arm.py --tier hard --validate --parallel 4 --write
  abfast:2 = abfast:1 + flask/httpx/anyio/urllib3/rich/pydantic/... :
    docker build -t abfast:2 -f evals/abfast/Dockerfile.2 evals/abfast
Extra task fields: "pysrc": true (python repo made importable + package metadata so subprocess-spawning test
fixtures work), "hidden_decls" {zig file: [names]} (file-scope test-only helpers appended with the hidden tests).
Harness guards (both tiers, both arms): PYTEST_ADDOPTS=--timeout=120 for every agent command, so a hung test
fails instead of blocking the shell until the wall cap (cause of the werkzeug timeouts: the full werkzeug suite
spawns dev servers whose child process cannot import the uninstalled package; the fixture then spins forever).

## Report
  python3 evals/abfast/report.py RUN_A [RUN_B]     # RUN = /mnt/vm_8tb/b70/evals/abfast/runs/<arm>-<stamp>

## Pi config (same for both arms)
Pi 1.2.0 `--print --mode json`, provider harbor-endpoint, models.json identical to
evals/terminalbench/harbor_pi.py (qwen-chat-template, xhigh -> enable_thinking true,
contextWindow 131072, maxTokens 16384), prompt = pi_concise_prompt.j2 wrapper. Keep
pi_models_json() in run_arm.py in sync with harbor_pi.qwen_model_definition.

## Outputs per task (runs/<arm>-<stamp>/tasks/<id>/)
result.json, pi_transcript.jsonl, pi_session.jsonl, agent_diff.patch, hidden_test.log.
End reasons: completed, timeout, endpoint_error, agent_error, harness_error.

## Langs tier (tasks_langs.json): Rust, Go, Dart
Commit replays from recent open-source projects (commits dated 2026-07-01 or later) in the languages the
owner actually uses. Same semantics as the other tiers (hidden test files are overwritten from the real sha;
validated parent FAIL / real PASS), each task has "lang", "difficulty" (medium|hard) and "image": "abfast:3".
Selected with --tier langs, which switches the tasks file to tasks_langs.json and defaults to
--timeout-s 3600 --max-output-tokens 32768 (thinking xhigh, context 131072).
  Image (abfast:1 + stable Rust 1.99, Go 1.27, Dart 3.13; toolchains live in /mnt/vm_8tb/b70/tooling/langs):
    docker build -t abfast:3 -f evals/abfast/Dockerfile.3 \
      --build-context rust=/mnt/vm_8tb/b70/tooling/langs/rust --build-context go=/mnt/vm_8tb/b70/tooling/langs/go \
      --build-context dart=/mnt/vm_8tb/b70/tooling/langs/dart-sdk evals/abfast
  Dependencies are not in the image. Per task, `langs_prep.py prep` (needs network, no model, once) extracts the
  parent commit in an abfast:3 container, runs the task's "prep_cmd" (cargo test --no-run / go mod download +
  warm test build / dart pub get) and saves the caches (~/.cargo registry + warm target/, ~/go modcache +
  go-build cache, ~/.pub-cache + .dart_tool) to /mnt/vm_8tb/b70/evals/abfast/langprep/<id>.tar. run_arm.py injects
  that tar into every task container (both arms identically) right after the repo is extracted, so the
  agent's edit-test loop is an incremental build with CARGO_NET_OFFLINE=true, GOPROXY=off, GOFLAGS=-mod=mod,
  and a populated PUB_CACHE. A task with "prep" refuses to start without its tar.
    python3 evals/abfast/langs_prep.py prep --parallel 2            # all tasks (or --tasks id,id)
  Validate (no model; ABFAST_NETWORK=none runs the containers with no network at all):
    ABFAST_NETWORK=none python3 evals/abfast/run_arm.py --tier langs --validate --parallel 2 --write
  Run an arm (the agent container needs the network only to reach the model endpoint):
    python3 evals/abfast/run_arm.py --tier langs --base-url http://192.168.10.5:18080/v1 \
      --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --arm-label langsA --parallel 2 --thinking xhigh
  Test parsers: parse "cargo" (sum of `test result:` lines), "go" (`go test -json`, leaf tests), "dart"
  (`dart test -r expanded` final +N -M line). Hidden-file mtimes are set to "now" because cargo compares mtimes.
  Source clones for this tier: /mnt/vm_8tb/b70/evals/abfast/src/{hyperfine,bat,tokio,regex,ripgrep,gojq,
  samber-lo,go-toml,dart-tools,dart-test}.

## Longrun tier (tasks_longrun.json, evals/abfast/longrun/)
Long-horizon tier: 3-6 hour agent budget per task, each task graded in 10-20 hidden MILESTONES (feature groups),
so the score is a fraction and progress over time can be compared between arms. Task types: black-box reverse
engineering of stripped binaries / obfuscated bundles (reimplement in Rust, Go or Python), build-from-spec, and one
autoresearch-style optimisation task. Reference sources, generators and hidden suites live outside the agent
container; built artifacts are under /mnt/vm_8tb/b70/evals/abfast/longrun/<task>/{agent,target,hidden,ref_sub,ref_partial}.

  Image (does not touch abfast:1/2):
    docker build -t abfast-long:1 -f evals/abfast/longrun/Dockerfile.long \
      --build-context rust=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu \
      --build-context dl=/mnt/vm_8tb/b70/evals/abfast/src/dl evals/abfast/longrun
  (dl/ = ghidra.zip 12.1.4 and rea/rea-agents-6.4.0.tgz; image has JDK 21, Go 1.22, Rust stable, gdb/strace/ltrace/radare2.)
  Build task artifacts:  python3 evals/abfast/longrun/tasks/<id>/build.py ; python3 evals/abfast/longrun/assemble_tasks.py
  Validate (no model):   python3 evals/abfast/longrun/validate_long.py [--tasks id,id] --runs 2
  Run an arm:
    python3 evals/abfast/run_arm.py --tier longrun --base-url http://192.168.10.5:18080/v1 \
      --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --arm-label lrA --parallel 2
  Options: --timeout-s 14400 --max-output-tokens 32768 --tools task|rea|plain (RE tasks default rea) --snapshot-s 1800
           --idle-rounds 2 --tasks id,id --no-score-snapshots
  Report:  python3 evals/abfast/report.py RUN_A [RUN_B]   (milestone fractions, paired table, hygiene, curves)
  Harness self-test with no model: longrun/tools/mock_openai.py (scripted OpenAI-compatible server).

How a run works (identical for both arms):
- Rounds: Pi 1.2.0 `--print --mode json --session-dir /logs/pi` runs round 1 with the task prompt. When Pi exits (the
  model stopped calling tools), the harness re-invokes `pi --continue` in the SAME session with a fixed neutral
  message (CONTINUE_MSG in run_long.py: keep going, re-read the task, run the public tests, fix what is missing)
  until the wall budget ends or `--idle-rounds` (2) consecutive rounds changed no source file. End reasons:
  timeout, agent_stopped (model repeatedly declared done), endpoint_error (6 failed rounds), max_rounds, harness_error.
  first_stop_s records when the model first stopped by itself.
- Context: Pi auto-compaction is on (threshold contextWindow - reserveTokens, summarizer = the same model);
  settings.json is written identically for both arms (reserveTokens = min(32768, ctx/4), keepRecentTokens 20000).
  Compaction events are counted from the transcript. TASK.md in the repo lets the agent re-read the task after a summary.
- Snapshots: every --snapshot-s (default 1800 s) a poller outside the container tars the workspace (source-sized files,
  no target/.git) if it changed; nothing is scored during the run. After all agent runs finish, every snapshot is
  scored in a fresh network-less container (curve.json per task); the final tree is scored at the end of each task.
  last_change_s = time of the last source-file modification.
- Scoring: build in a clean container WITHOUT /opt/target, then /hidden/check.py. fraction = milestones_passed/total
  (a milestone needs all its tests); tests_fraction = hidden tests passed/total. Submissions containing bytes of the
  original binary are scored zero (cheat_flagged).
- Hygiene metrics (result.json): turns, tool calls/errors, unknown-tool and bad-argument calls, empty turns,
  compactions (and failures/overflow), longest run of identical tool calls, tokens, rounds, stop reasons.
- Tools switch for reverse-engineering tasks: `--tools rea` (default) gives the agent REA (rea-agents 6.4.0, MCP server
  `rea` over Pi's native MCP support, Ghidra 12.1.4 provider, offline) plus the reverse-engineer-anything skill;
  `--tools plain` gives objdump/gdb/radare2/strace only. Same switch value for both arms.

## Sysadmin tier (tasks_sysadmin.json, --tier sysadmin)
"Fix my machine" tasks: office and family users asking the local model, through Pi, to repair Linux (and
pwsh-on-Linux "Windows-flavoured") problems. Each task = a container of image abfast-sys:1 put into a broken
state by a per-task setup script (then every trace of the setup is deleted), a short non-expert problem report
with no hint of the cause, and a hidden verifier run after Pi stops. The verifier has functional checks and
`safe_*` checks (data kept, no chmod 777, auth/firewall-like protections kept, unrelated service untouched);
a task passes only if all checks pass. The report lists "symptom fixed but unsafe" separately.
Layout: sysadmin/tasks/<id>/{task.json,setup/,verify/,shared/,fix.sh,lazy.sh}; verify/, fix.sh, lazy.sh are copied
into a container only after the agent phase (or in --validate). sysadmin/sysrun.py = runner hooks (run_arm.py calls
them for tasks with "sys": true); sysadmin/gen_tasks.py rebuilds tasks_sysadmin.json from the task.json files.

Environment facts (identical for both arms): Pi runs as root (the agent plays the local admin), HOME=/root, cwd /root;
container on the default docker bridge network (not --network host) with --init, 12g/6 cpus; some tasks add
`--tmpfs /srv/data:size=40m` so "disk full" is real. No systemd: `systemctl`, `journalctl`, `service` are small
shims (sysadmin/shim) that run units from /etc/systemd/system; Restart= is not honoured. There is no CAP_NET_ADMIN,
so no firewall tasks. The container has outbound NAT networking (as the Pi endpoint requires); tasks are written to
be solvable offline. PowerShell 7.4.6 (pwsh) is installed; the windows_flavour tasks say so in their text.

  # build (once; does not touch abfast:1/2; needs network for apt, pip download of two wheels, pwsh tarball)
  docker build -t abfast-sys:1 -f evals/abfast/Dockerfile.sys evals/abfast
  # validate with no model (setup -> verify FAILS; fix.sh -> PASSES; lazy.sh -> FAILS; twice each)
  python3 evals/abfast/sysadmin/gen_tasks.py
  python3 evals/abfast/run_arm.py --tier sysadmin --validate --parallel 4 --write
  # run an arm (defaults for this tier: --timeout-s 1800, --max-output-tokens 32768)
  python3 evals/abfast/run_arm.py --tier sysadmin --base-url http://192.168.10.5:18080/v1 \
    --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --arm-label sysA \
    --parallel 2 --thinking xhigh
  python3 evals/abfast/report.py RUN_A [RUN_B]    # adds difficulty/category breakdown and the unsafe-fix count
Per-task extra outputs: setup.log, verify.log, verify_checks.json (check name -> bool) next to result.json.
Validation data: /mnt/vm_8tb/b70/evals/abfast/runs/validate-sys/<id>/ .
