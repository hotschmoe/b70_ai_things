# Read-only cleanup inventory, 2026-10-16

CONFIG -> Protect the live Strata checkout/build-sycl-ep, all S107/S112/S114/
S116 binaries/archives/objects and experimental evidence, current model and
fallback payloads, user dirty files, current containers, archive/quarantine.
No deletion, prune, container stop, build, GPU operation or model call.

COMMAND -> Git worktree/status, Docker container/image metadata and an exact
eight-root TMP/cache `du -s -B1` under bin/cpu-run. CPU lock acquisition was
bounded to1s, workload completed/released in1s, no quiet GPU run was active.
No whole-disk/result-tree/model/archive scan or binary/content hash. Visible
/proc fd symlinks and container bind mounts were checked for these eight roots.
Commands/results and selected metadata are retained alongside this note.
Docker full Config/environment fields are deliberately not retained.

RESULT -> One registered worktree: the primary master clone. No spare
worktree deletion candidate. HEAD at snapshot af597e4b. Ten unrelated dirty/
untracked paths are listed exactly in worktree-dirty.json and remain protected.

All eight scoped task-created temporary/cache directories are empty and
reported0 allocated bytes, with no visible open-fd or container-mount reference:

- R/greedy-top1-s112-oracle-prep-v2/build/tmp
- R/greedy-top1-s112-oracle-prep-v2/build/cache
- R/greedy-top1-s112-run-v2/tmp
- R/greedy-top1-s112-run-v2/cache
- R/mtp-parallel-argmax-main-s114-build/build/tmp
- R/mtp-parallel-argmax-main-s114-build/build/cache
- R/mtp-parallel-argmax-main-s114-audit/tmp
- R/s114-default-isa-addendum/tmp

R is /mnt/vm_8tb/b70/results/
flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014.
Build recipes identify the first two build/cache pairs as compiler TMPDIR,
XDG_CACHE_HOME and SYCL_CACHE_DIR with persistent cache disabled. They are
expendable intermediate locations rather than named executables/receipts,
but deleting empty directories recovers no meaningful space and may remove
expected recipe directories. Leave them. No S116 directory was scanned.
The /proc check is permission-limited and momentary, not a guarantee that an
unseen process or future run cannot use a path. Do not delete whole parents.

Docker snapshot has91 tag/digest rows for83 distinct image IDs and4 containers:
Grafana, Prometheus and OpenWebUI are running and protected. The fourth is
created/unstarted health container
`abc6220232176de0c93901e34e7d66f78e91ae40a51df0152bf1c9f306884d5a`,
name `xpu-health-1203363-17874-11440-card0`, created2026-10-13, State=created,
Pid0, no StartedAt, writable layer displayed0B. This is a concrete stale-
looking root-review candidate, NOT an automatic removal authorization: current
containers were explicitly protected, ownership/control is not independently
proved here, and removing it does not reclaim its referenced image.

Its image `d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067`
is explicitly pinned by vllm/int4/native5802_trial/serve.py:21, so KEEP it
regardless of whether the old created container is removed. Strata compiler
image e41d59c0 is pinned by current builds and KEEP. Health scripts explicitly
pin the adc915d2 RepoDigest; keep that identity and its backing image/layers.

Unused/untagged-looking image IDs are in selected-image-identities.json,
including vLLM1db27a8b/2ac07cf8/f01e24f6, Intel SGLang1a13d3d2 and oneAPI
e9db5183. Their RepoDigests remain addressable; `Tag=<none>` does NOT mean
unreferenced. No comprehensive historical recipe/digest/ancestry audit was
performed, and f01e24f6 is itself a retained vLLM/UMD baseline reference.
KEEP all pending deliberate owner/pin review. Displayed16-27GB image sizes
include shared layers and are not reclaimable-byte estimates. Other unused
evaluation/legacy tags are not established as this task's property.

VERDICT -> No meaningful unambiguous expendable bytes found in this bounded
pass. Do not run broad Docker pruning or remove experiment/source/build
parents. Root may separately resolve the exact old created health container;
images, unknowns, user dirty state, fallback and all experiment evidence stay.
Further disk-size investigation should target a specific agreed path under
cpu-run, not a recursive8TB inventory or repeat measurement disturbance.
