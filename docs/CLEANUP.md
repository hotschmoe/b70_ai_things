# Cleanup Ledger

## 2026-08-26 clean-slate pass

Config:

- Retain Qwen3.8-27B BF16, NVFP4, and W8A8 GPTQ.
- Retain Qwen3.6-27B NVIDIA NVFP4.
- Retain the Ornith-1.5-35B-A3B model family.
- Retain Steve reproduction trees and the newest topology/runtime findings.
- Quarantine first; permanent deletion remains a separate decision.

Command:

- Move retired weights, clones, caches, raw logs, histories, and stale backend
  source under `archive/to-delete-20260826/`.
- Export off-target Docker images to a compressed `docker save` archive.
- Trim `JOURNAL.md` at a complete entry boundary.

Result:

- Live model registry contains eight retained artifacts.
- `JOURNAL.md` contains 3,478 lines; the full prior journal is quarantined.
- Live runtime root contains only stable tools, health caches, monitoring data,
  secrets, lease files, and the two Steve trees.
- ZML, llama.cpp, old model families, stale shelves, historical script trees,
  raw result trees, build trees, and backend clones are quarantined.
- Docker now has 15 retained images, three monitoring/UI containers, and no
  build cache; 23 experiment containers were removed after metadata capture.
- The quarantine is about 1.3 TB. Disk space is reclaimed only when that
  review buffer is permanently removed.
- About 256 MB of August 23-26 raw evidence cited by current findings is kept in
  a separate non-deletion archive and linked from ignored `results/` paths.

Verdict:

- The active repo and runtime root now describe the current SGLang-first work.
- Restore instructions and the keep set are recorded in the quarantine README.
- Do not remove the quarantine until its manifest and compressed Docker archive
  have been reviewed.
