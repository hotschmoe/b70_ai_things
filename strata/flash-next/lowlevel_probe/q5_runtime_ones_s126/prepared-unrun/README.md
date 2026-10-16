# S126 one-card private runtime-ones runner - FROZEN SOURCE ONLY

CONFIG -> One physical card0, mask0/level_zero:*, native PCI0000:0b:00.0.
Exactly frozen sourceV3 cbc8aedaa29c2852cff60f1eb51c6f4da638fd7fd2076816dd0d774bf6fb3d86.
Actual EXE SHA is UNKNOWN until root CPU build; required argument, never guessed.
Source/header/fixtures remain in their frozen source directory, unchanged here.

COMMAND -> ROOT ONLY after build and full selected compiled audit GO, announced
card0 window (~60s including fresh health). No automatic launch/retry:

  bash ROOT_RUN.sh NEW_RUN_DIR \
    --source R/q5-c2-runtime-ones-s126-prep-v3 \
    --source-manifest R/q5-c2-runtime-ones-s126-prep-v3-manifest.json \
    --manifest-sha256 cbc8aedaa29c2852cff60f1eb51c6f4da638fd7fd2076816dd0d774bf6fb3d86 \
    --exe-sha256 ROOT_ACTUAL_SHA \
    --audit ROOT_COMPILED_GO_JSON --audit-sha256 ROOT_ACTUAL_AUDIT_SHA

Replace R with the absolute shared result root. Audit must contain GPU_GO:true
and binary_sha256 equal to actual pinned executable. It must resolve live
hostbindings, original/public8128-EOT control identity and candidate payload/ISA;
this runner does not manufacture that decision or decode more images.

RESULT -> Prepared lifecycle, not a runtime result.

- bin/cpu-run --quiet OUTSIDE bin/gpu-run --card0 owns the whole window.
  Fresh card0 xpu-health BEFORE leaf and AFTER owned removal; exact health image
  adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78.
- e41 image, UID1000, no network, readonly source/build/fixtures/NVMe weights,
  diskTMP, persistent runtime cacheOFF. Prospective ldd/libsycl/libstdc++/ZE
  dependency identities, effective entry env, image/actual argv/driver/native
  physicalPCI logged. Ldd is not actual process-map/dlopen proof; no LD_DEBUG
  instrumentation or separate GPU identity process is added.
- Before health: source+model-lock+fixtures/exe/audit hashes; selected437MB head
  slice SHA once, never75GB model. Leaf verifies actual consumed slice FNV and
  full device weights only twice. Post identity rechecks small files/exe and
  shard stat continuity; it does not reread437MB slice. FP32 input is a readonly
  witness; actual projection operands are Q8 packets and weights.
- Owned non--rm docker create/start-a/wait. FullCID/NAME/image/user/binds checked;
  final stopped state/ExitCode/OOMKilled retained, normal rm, authoritative
  fullCID+NAME absence BEFORE health. A live/unknown container or unresolved
  removal holds the SAME leases for root; no forced stop/reset/alternate/retry.
- Known numeric exit5 only (raw INT64 or full-logit mismatch) may receive normal
  posthealth. V3 raw validation tail now precedes INT64 comparison, and cleanup
  failures force2; guard/tail/readonly/runtime/unknown failures require root
  recovery review without blindly touching GPU health. Failure summaries NULL.
- All native logs/final state/removal captured before health. Successful native
  exit0 also requires parser/DOT/full-artifact validation before posthealth.
  Normal analyzer:49 snapshot phases/2 wholeweight checks/4 raw gates(656cells),
  all74 full binary artifacts,5 bound DOTs/43nodes/38edges and common3pointers.
  Original vs candidate8 completed20-call samples, exact balanced2round order,
  paired means recomputed from raw samples with none excluded.
- Root wrapper/native/wait/rm/analysis/health/identity/tee exits preserved.
  One process/nested timing samples only; no serving, population, bandwidth,
  physical-idle or universal FP-equivalence claim.

VERDICT -> FROZEN/UNEXECUTED; await root read, actual build pins and GPU GO.
No source/build/GPU/model calls, configs, profiler, full-cache request or restore.
