# Storage layout

Updated 2026-08-26.

## Live paths

- Repository: /mnt/vm_8tb/github/b70_ai_things
- Model weights: models/files/
- Runtime root: /mnt/vm_8tb/b70
- Shared tools: bin/
- Verified shelf: rdy_to_serve/
- Cleanup quarantine: archive/to-delete-20260826/
- Retained recent evidence: archive/research-evidence-20260823-26/

models/files is gitignored and contains only entries in models/manifest.yaml.
The retired /mnt/vm_8tb/b70/models tree is quarantined and must not be used.

## Runtime root policy

Keep /mnt/vm_8tb/b70 small and rebuildable:

- GPU lease files and stable symlinks;
- secrets;
- collective health caches;
- current backend/compiler caches created after the clean refresh;
- preserved Steve trees.

Do not accumulate duplicate model roots, numbered kernel clones, old PyTorch
checkouts, or unbounded server logs there.

## Quarantine policy

Cleanup uses same-filesystem moves into the gitignored archive tree. Each batch
must record original paths and restoration instructions. Permanent deletion is
a separate operator action and is the point at which disk space is reclaimed.

The 2026-08-26 batch is described by:

- archive/to-delete-20260826/README.txt
- archive/to-delete-20260826/MANIFEST.tsv
- archive/to-delete-20260826/runtime/README.txt
- archive/to-delete-20260826/docker/README.txt

The separate recent-evidence archive is intentionally excluded from permanent
cleanup deletion. Ignored links under `results/` keep active journal paths
resolvable without restoring the historical result tree to Git.
