# S118: scoped cleanup, 2026-10-16

S118 removed one stale, never-started health container and found no meaningful
unambiguous temporary-file reclaim in a bounded inventory. No GPU operation,
running-process termination, image deletion, binary/source change or evidence
removal occurred. Pinned and unknown images, models/fallback, current builds,
user dirty state and archive/quarantine remain protected.

CONFIG -> Read-only inventory first. Protect all S107/S112/S114/S116
executables/objects/archives, live source/build-sycl-ep and experiment evidence.
Only explicit TMP/cache/tool-intermediate locations were considered.

COMMAND -> Git worktree/status and Docker metadata; eight exact S112/S114
TMP/cache roots sized with `du -s -B1` under bin/cpu-run (one second, released).
A later direct metadata pass covered five historical private build roots:
S85 original/v2, S98, S103v2 and S104. Their ten TMP/cache directories were
empty; the build roots had no .bc/.spv/.ll/.tmp files. Small build recipe
references were checked. No recursive whole-results/disk/model/trace scan.

RESULT -> Only the main registered master worktree exists. Ten unrelated
dirty/untracked paths are recorded and preserved. The first eight directories
reported0 allocated bytes; the ten later directories were empty. Deleting
these empty paths would recover no useful space. Existing build recipes name
them as TMPDIR/XDG_CACHE_HOME/SYCL_CACHE_DIR intermediates; parents and named
artifacts are not cleanup candidates. No S116/current build directory was
scanned. Visible open-fd checks are permission-limited and momentary.

Docker metadata showed91 tag/digest rows for83 distinct image IDs and three
running infrastructure containers plus one old created health container.
Root independently checked full CID
`abc6220232176de0c93901e34e7d66f78e91ae40a51df0152bf1c9f306884d5a`,
name `xpu-health-1203363-17874-11440-card0`: created2026-10-13, statuscreated,
Running=false, Pid0, never started, ExecIDs=null and no active health unit.
Normal `docker rm` returned0; subsequent same-CID inspect returned1/absent.
No force or second mutation occurred. The first verification predicate was
case-sensitive (`No such object` versus actual lowercase `no such object`);
root rechecked absence without repeating removal. Raw rm stdout was not
saved; this is explicit in the authoritative receipt.

The removed container's image `d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067`
is pinned by the native5802 fallback and was retained. Strata e41 compiler,
adc915 health identity and all other pinned/unknown image layers stay.
Untagged-looking RepoDigest-addressable images are not proved stale; their
displayed sizes include shared layers and are not reclaimable-byte estimates.
The removed container's writable layer displayed0B, so no freed-GB claim is
made. Running Grafana, Prometheus and OpenWebUI were protected.

VERDICT -> Close this bounded cleanup. No worktree, image or generated-file
deletion is justified by the remaining evidence. Further inventory should
name a specific candidate and preserve the CPU/GPU lease discipline; do not
prune broad directories to manufacture a storage saving.

[Exact small records and index](../strata/flash-next/lowlevel_probe/cleanup_s118/README.md)
preserve the original pre-removal snapshot, the private-build followup and
root inspection/removal/absence limitation. Root performed the sole deletion;
the inventory/curation agent performed none.
