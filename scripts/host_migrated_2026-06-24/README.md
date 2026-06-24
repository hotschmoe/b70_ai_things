# host_migrated_2026-06-24/ -- uncommitted host scripts rescued during the repo consolidation

On 2026-06-24 the "two-repo split" (git checkout at ~/github vs a non-git flat
runtime copy at /mnt/vm_8tb/b70) was consolidated into a single clone on the 8TB
SSD at `/mnt/vm_8tb/github/b70_ai_things`. During that move, these files were found
ONLY on the old flat runtime root (`/mnt/vm_8tb/b70/`) with no copy in git -- they
are ad-hoc / pre-numbering-convention one-offs that never got committed.

They are preserved here VERBATIM so the work is not lost. They are NOT curated and
NOT part of the numbered lab notebook (`scripts/NN_*.sh`). Curate/renumber or delete
in a later focused pass. Provenance: every file's content was confirmed absent from
git (by sha256) before rescue; everything else on the old flat root was already in
git (identically or under a numbered name) and was removed during cleanup.

- `m0_mtp_gate.sh` -- the missing first step of the MTP m-series (m1..m5 are committed
  as scripts/78..83); the gate that precedes them.
- `build_int8*.sh` -- int8 kernel build scripts (superseded by the baked `int8g` image,
  kept as the build record).
- `dl_q35.{sh,py}` -- Qwen3.6-35B download helpers.
- `serve_{int4,w8a8}_webui.sh`, `w8a8_eval_serve.sh` -- ad-hoc serve/eval drivers.
- `inspect_*.py`, `b1_validate.py`, `pp1_validate.py`, `bytemap.py` -- one-off
  model/checkpoint inspection + validation probes.
- `44_gdn.sh` -- GatedDeltaNet probe (numbered, but its content was not in git).
- `patches/` -- `loadtest_35b.py`, `quark_v0230.py`, `sitecustomize.py`,
  `scheme_path.txt`: patch/loadtest fragments that lived in the old flat `patches/`.
