# abfast: small paired agentic-coding eval (Pi 1.2.0, no Harbor)

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
