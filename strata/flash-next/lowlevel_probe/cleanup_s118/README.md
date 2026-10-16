# S118: bounded cleanup inventory and one stale container removal

CONFIG -> Keep live Strata source/current build, all S107/S112/S114/S116
artifacts, experiment evidence, model/fallback payloads, user dirty paths,
running infrastructure, archive/quarantine and pinned/unknown images.

COMMAND -> Read-only worktree/Docker metadata, eight explicit TMP/cache roots
sized under a one-second CPU lease, then direct metadata of five historical
private build roots and ten empty TMP/cache directories. No whole-R/8TB walk.
Root separately inspected and removed ONE exact never-started health container
with normal `docker rm`, then verified the same full CID absent.

RESULT -> One main worktree only; ten unrelated dirty/untracked paths retained.
The eight sized roots had0 allocated bytes. Ten additional historical private
build TMP/cache directories were empty, with no .bc/.spv/.ll/.tmp candidate
at those build roots. No meaningful reclaimable intermediate was found.
Docker snapshot:91 tag/digest rows for83 IDs; three running infrastructure
containers protected. The exact old created container was removed, with no
GPU operation, running-process kill or image removal. No freed-GB claim.

VERDICT -> Scoped cleanup CLOSED. Retain unknown images and all protected
evidence. Do not broaden this into Docker pruning or whole-build removal.

`inventory-before/` preserves the original read-only snapshot, including its
then-pending wording about the old container. `private-temp-followup/` records
the later bounded empty-directory check and small recipe reference search.
`root-removal/` is the authoritative actual mutation receipt. Root's first
post-removal string check expected uppercase `No such object`; the daemon
returned lowercase. Rechecking absence involved no second removal or force.
The receipt preserves that limitation and notes raw rm stdout was not saved.

`record-copies.json` gives exact source paths, bytes and SHA256 for every
small copied record. Full Docker Config/environment was excluded from the
inventory; root's exact old health-container inspection has no credential
environment names. All source/exe/object/archive/image/trace/selected ISA
payloads remain external and untouched. See the
[dated note](../../../../docs/20261016_strata_scoped_cleanup.md).
