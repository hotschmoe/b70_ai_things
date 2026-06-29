# models/ -- the model registry

Single source of truth for every model weight we keep. Tracked in git: this README,
`manifest.yaml`, and the helper scripts. NOT tracked (see `.gitignore`): `files/`, the
actual weights.

## Layout

```
models/
  manifest.yaml            # registry: every kept model, its source, vision/MTP flags
  fetch.sh                 # reprovision a fresh box from the manifest (HF downloads)
  reorg.sh                 # ONE-TIME migration: de-root + materialize + move + drop (sudo)
  graft_w4_complete.sh     # restore vision+MTP to the W4A16/W4A8 builds (GPU-free)
  REFACTOR.md              # serve-script path fixes for the new layout
  files/                   # GIT-IGNORED weights, de-rooted, no links, nothing stripped
    qwen3.6-27b/{bf16,fp8,int4-autoround,w4a16,w4a8-sqgptq,w8a8-sqgptq}/
    qwen3.6-35b-a3b/{bf16,int4-autoround,quark-w8a8-int8}/
```

Container path stays `/models` (lib.sh mounts `models/files` -> `/models:ro`), so a
checkpoint is addressed as `/models/<family>/<scheme>` (e.g. `/models/qwen3.6-27b/w8a8-sqgptq`).

## Rules for `files/`

1. **De-rooted.** Everything owned by `hotschmoe`, not `root` (root ownership was a
   migration artifact from the old box).
2. **No symlinks, no hardlinks.** Each `<family>/<scheme>` dir is a self-contained set of
   real files. (The old box used symlink/hardlink "graft" dirs to save space; that is gone.)
3. **Nothing stripped.** Every checkpoint carries the vision tower (333 `visual.*` tensors)
   and the MTP head when the architecture has them -- the full stock the creators provide,
   plus our grafts. The `manifest.yaml` `vision:`/`mtp:` flags assert this.

## Reprovisioning a fresh machine

```
bash models/fetch.sh            # downloads every source:hf entry into files/<path>
```

`source: hf` entries pull straight from HuggingFace. `source: custom` entries were
**quantized on this box** and are not downloadable yet -- see the TODO at the top of
`manifest.yaml`. After `fetch.sh`, run `bash models/graft_w4_complete.sh` to rebuild the
W4A16/W4A8 vision+MTP grafts.

## TODO (custom quants reprovisioning)

The `source: custom` quants (`w4a16`, `w4a8-sqgptq`, `w8a8-sqgptq`) exist only on disk.
Before the next migration, pick one:
  (a) push them to HF under our namespace and flip `source: custom` -> `source: hf`, or
  (b) add a rebuild recipe (base HF model + quant script) so `fetch.sh` can regenerate them.
