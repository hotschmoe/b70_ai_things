# hotschmoe-dd (Strata night-final-v3, Flash-Next ISTA IQ3_XXS) coding/agentic eval, 2026-10-16/17

Harness: evals/abfast (Pi agent in docker, hidden verifiers). Pi 1.3.0 (containers; images bake 1.2.0, overlay via
ABFAST_PI_DIR=/mnt/vm_8tb/b70/tooling/pi-1.3.0). Host pi: login shell 1.3.0; nvm copy 0.85.1 (unused by the harness).
Pi upgrade: 1.2.0 -> 1.3.0 in containers; confirmed `pi --version` = 1.3.0 inside abfast:1 with the overlay, 1.2.0 without.
Identity: /v1/models lists hotschmoe-dd (alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata, n_ctx 200000); /health ok.
evals/configs/models.yaml entry "hotschmoe-dd" is STALE (describes the 27B INT4 vLLM driver); live model is Flash-Next IQ3.
Settings: thinking xhigh (chat-template enable_thinking; server default effort medium, 16384 thinking budget), context 131072,
max output 32768, single run per task.

## Results
| Tier | N | Pass | Notes |
|---|---|---|---|
| sysadmin (full) | 24 | 22 | fails: reg-file-fix, logrotate-not-rotating |
| langs (subset) | 6 | 5 | fail: gotoml-omit-empty-super-tables (2872/2874) |
| total | 30 | 27 | single run, small N |

Langs subset: yaml-edit-flow-trailing-comma, gojq-match-empty-after-nonempty, hyperfine-shell-operator-guard,
fd-sanitize-error-messages, markdown-linear-link-definitions, gotoml-omit-empty-super-tables. Chosen to include the hard
tasks and every task a prior arm failed; NOT the full 19. Longrun tier (7 tasks) not run; no terminal-bench run.

Load: first 9 sysadmin tasks ran at parallel 2, the rest at parallel 1 (after the coordinator reported another user on the
server). All timings are CONTENDED, not representative; no speed claims. No server errors; end_health=ok every run.

## Failure modes
- reg-file-fix (also failed in the 27B arm A, passed in prior B): verifier KeyError on every HKCU key, i.e. the written .reg
  did not parse under the verifier; agent claimed it validated with its own parser. Cause of the format difference unconfirmed.
- logrotate-not-rotating: unsafe/incomplete fix: made /var/log/webapp world-writable (safe_ check FAIL) and lost old log
  contents (6/8 checks). The only "unsafe fix" in this run.
- gotoml-omit-empty-super-tables: 2 hidden tests short; same task failed in both earlier arms (A and B).
- gojq-match-empty-after-nonempty: agent hit the 3600 s wall (end_reason timeout, 44 turns, 76k completion tokens) but the
  hidden tests PASS on the final tree (1095/1095), same as the earlier B run at 2700 s. It keeps working past a done state
  or verifies at length. Scored as pass-by-final-state, flagged as timeout. This is the only timeout; no format errors,
  no endpoint errors, no looping crashes observed.
- markdown-linear-link-definitions: passed but long (95 turns, 6.5M prompt tokens, 1349 s, contended).

## Comparison (baselines Pi 1.2.0, parallel 2, different engine build; NOT strictly matched: Pi version, server
build (stock Strata v0.1.43.1 vs night-final-v3), load, and langs subset)
- sysadmin 24: this run 22/24; earlier same-quant arm B 23/24 (stock Strata); 27B INT4 arm A 18/24.
- langs, same 6 tasks: this run 5/6; arm B 5/6 (gotoml fail); arm A 2/6 (passed only hyperfine and fd).
- Conclusion: no evidence of quality regression from the speed work (night-final-v3 vs stock Strata, same IQ3 weights);
  within noise at N=30 single runs. Not evidence of absolute capability on harder benchmarks (abfast is near-saturated).

## Files
Per-run results.json/meta.json and logs here; full transcripts under /mnt/vm_8tb/b70/evals/abfast/runs/{sysC,sysC2,langsC,langsD}-*.
Harness edits: evals/abfast/run_arm.py (PI_MOUNT: ABFAST_PI_DIR bind-mount over /opt/pi, + docstring), evals/abfast/sysadmin/sysrun.py
(same mount in SysBox), evals/abfast/README.md (note). Dockerfile comments still say 1.2.0 because the images are not rebuilt.
