# bin/ -- shared TOOLS (the stable, depended-on tier)

Re-run-constantly infrastructure, separated from the append-only lab notebook (`scripts/NN_*.sh`) and
from the golden path (`rdy_to_serve/`). See `../ORGANIZATION.md` for the full layout + contracts.

| tool | what it is |
|---|---|
| `gpu-run` | per-card flock GPU lease. Default locks BOTH cards (TP=2/DP/PP); `--card N` locks ONLY card N (run on one card, leave the other free for `--card <other>`). `--status` shows per-card holders. |
| `30_serve_w4a8_graph.sh` | the generic env-driven serve engine (all quants; eager / PIECEWISE capture / TP). |
| `35_sweep_bench.sh` | concurrency sweep vs a running server (`docker exec`s `vllm bench serve`). |
| `64_dataparallel_2rep.sh` | dual-B70 data-parallel bench (2 replicas + proxy). |
| `dp_nginx.conf` | round-robin proxy config for the data-parallel daily driver. |
| `serve-sweep` | the SWEEP GATE harness (`--smoke` / `--bench`) across all rdy_to_serve models. |

## [!] Contract (STABLE tier -- biggest blast radius in the repo)
These are depended on by the lab notebook, `daily_driver_serve.sh`, AND `rdy_to_serve/`. A break here
breaks every serve path. Any change requires `serve-sweep --smoke` GREEN across all rdy_to_serve models
before commit (and `--bench` if it could move perf) -- same gate as `rdy_to_serve/_common/`.

## [!] bin/ (repo) <-> host flat root (runtime)
The GPU host (`/mnt/vm_8tb/b70/`) runs these tools FLAT at its root, NOT under `bin/`
(e.g. `/mnt/vm_8tb/b70/gpu-run`, `/mnt/vm_8tb/b70/35_sweep_bench.sh`). This repo is the tracked
source of truth; the host is hand-synced (the host is not a git repo). Filenames here intentionally
MATCH the host flat names so the mapping is `bin/<f>` -> `/mnt/vm_8tb/b70/<f>`. Since the 2026-06-23
migration we work locally ON the box (b70s4dayz), so after editing a tool just copy it into place:
```bash
tar czf - -C bin <file> | tar xzf - -C /mnt/vm_8tb/b70
Open follow-up (ORGANIZATION.md): drop the leading `NN_` numbers (these are tools, not lab entries) and
script the host sync. Both need the host flat layout reconciled, so they are deferred.
