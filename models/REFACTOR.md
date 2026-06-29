# Serve-path refactor for the models/files layout

Tracks the serve-script changes that go with moving weights into `models/files/<family>/<scheme>`
(de-rooted, no links, complete). Done in the same commit as this doc unless marked TODO.

## Run order (the two sudo steps are yours; serve edits are already in the repo)

1. `sudo APPLY=1 bash models/reorg.sh`          -- de-root + materialize + drop. ~5-15 min copy.
2. `sudo APPLY=1 bash models/graft_w4_complete.sh`  -- complete w4a16/w4a8 (needs step 1 first).
3. `./bin/serve-sweep --smoke`                   -- gate the shelf on the new paths.

Until steps 1-2 run, the serve scripts point at paths that do not exist yet. That is expected.

## Container path map (old dir -> new /models path)

| old (host dir under /models or /models_w8a8)        | new container path                  |
|-----------------------------------------------------|-------------------------------------|
| Qwen_Qwen3.6-27B                                    | /models/qwen3.6-27b/bf16            |
| Qwen_Qwen3.6-27B-FP8                                | /models/qwen3.6-27b/fp8             |
| Lorbus_Qwen3.6-27B-int4-AutoRound                   | /models/qwen3.6-27b/int4-autoround  |
| Lorbus_Qwen3.6-27B-int4-mtp                         | /models/qwen3.6-27b/int4-autoround  |
| Qwen3.6-27B-W4A16  /  -W4A16-mtp-graft              | /models/qwen3.6-27b/w4a16           |
| Qwen3.6-27B-W4A8-sqgptq-prepacked                   | /models/qwen3.6-27b/w4a8-sqgptq     |
| Qwen3.6-27B-W8A8-sqgptq-mtp-graft                   | /models/qwen3.6-27b/w8a8-sqgptq     |
| (models_w8a8) Qwen3.6-27B-W8A8-sqgptq-vision-mtp    | /models/qwen3.6-27b/w8a8-sqgptq     |
| Intel_Qwen3.6-35B-A3B-int4-AutoRound                | /models/qwen3.6-35b-a3b/int4-autoround |
| Qwen3.6-35B-A3B-Quark-W8A8-INT8                     | /models/qwen3.6-35b-a3b/quark-w8a8-int8 |

Note: several shelf entries now resolve to the SAME ckpt (they differ only in serve flags):
- `qwen36-27b-w4a16` and `-w4a16-mtp`           -> qwen3.6-27b/w4a16   (complete build has the MTP head)
- `qwen36-27b-w8a8-mtp` and `-w8a8-sqgptq-mtp`  -> qwen3.6-27b/w8a8-sqgptq
- `qwen36-27b-int4`, `-int4-mtp`, `-int4-graph`, `-w4a8-graph` -> qwen3.6-27b/int4-autoround

## Repo changes APPLIED in this commit

- `rdy_to_serve/_common/lib.sh`:
  - new knob `MODELS_FILES` (default `/mnt/vm_8tb/github/b70_ai_things/models/files`).
  - mount changed: `-v "$ROOT/models:/models:ro"` -> `-v "$MODELS_FILES:/models:ro"`.
- Per-`serve.sh` CKPT/TOK updated to the new `/models/<family>/<scheme>` paths (table above).
  Sglang scripts (int4-graph, w4a8-graph, w8a8-mtp, int4-mtp) also had their inline
  `-v "$ROOT/models..."` mount switched to `-v "$REPO/models/files..."`.
- `qwen36-27b-w8a8-mtp`: dropped the second `-v "$ROOT/models_w8a8..."` mount (the build is
  now materialized under files/, no cross-mount symlinks).
- Retired shelf entries `qwen3-14b-w4a8` and `qwen3-14b-w8a8` (their 14B models are dropped).

## TODO (not done here)

- [ ] **Coherence-gate w4a16 + w4a8** after graft_w4_complete.sh: serve each, run a few text
      prompts + one image, confirm no "!!!!"/garbage, before trusting them. The w4a16 build in
      particular got a spliced multimodal config (its quant was authored text-only) -- verify
      the vision tower actually loads.
- [ ] **`rdy_to_serve/README.md`**: refresh any host paths / model dir names to the new layout.
- [ ] **Active tooling** that hardcodes `$ROOT/models/<OldDir>` (e.g. some `bin/` helpers,
      `sglang/graft_*.py` defaults): audit and point at the new layout. The append-only
      `scripts/NN_*.sh` lab notebook is historical -- do NOT rewrite (per AGENTS.md); copy to a
      new number if a path-updated rerun is needed.
- [ ] Consider collapsing the duplicate shelf entries listed above once the new paths are gated.
