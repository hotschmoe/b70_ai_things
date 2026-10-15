# S80 grouped prefill accuracy record copies

These are EXACT small historical records. record-copies.json preserves original
file names, source and resolved absolute paths, bytes and SHA256. No executable,
object, archive, model/compact payload, plane, reference array, full readback or
large linker/symbol log is copied. Never execute these copies in place or
reuse their already-completed output directories.

CONFIG -> S42 compact16expert/10format/1748row wave, real quantized weights and
saved synthetic half activation rows. New preregistered FULL GU reference target;
unchanged actual scalar anchor plus strict same-algorithm whole-path coherence.
COMMAND -> CPU-only helper precompute; known adapter AOT build using8frozen
archives; one quiet physical-card1 window; eager full pipelines then4paired
AB/BA/BA/AB completed-wave samples, with metadata included and no stage copies.
RESULT -> CPU v1 stops before GPU because optional down type42 Q2_0 uses block64,
not assumed32. Versioned v2 changes only those geometry checks/row strides, with
no GU target/header/math change. Full100MiB GU CPU/GPU byte identity and50MiB
scalar/grouped GPU down-DQ coherence PASS. All4GU targets,2,237,440values each,
pass; max absolute reference error8.242204785e-8, worst bound ratio0.000406271.
Own repeats/changed/restored eager stages, same-GU production SiLU producer,
S42 scalar anchor, guards/tails/readonly/metadata, health and removal PASS.
Sixteen large artifacts totaling309432320B remain external, referenced by hashes.
VERDICT -> Numerical target PASS independently of timing. All4grouped pairs
slower: completed-wave medians1.051034ms scalar versus4.404047ms grouped,
median PAIRED change+314.594292percent. Reject this exact mapping; keep production
scalar, no serving arm, retry or promotion. Four nested pairs in one process
are not independent process trials, p95, kernel service or hardware-cause proof.
No ideal SiLU/down accuracy or model qualification; S42 strict failure remains
FAILED_UNCHANGED. Actual production prefill is eager; no graph attempt/gate.

Two GPU build recipes and path assumptions

The gpu/ build scripts are the ORIGINAL frozen source-preparation recipe.
The actual root-build/ recipe changes the final link to include
-fp-model=precise and adds declared source/copy identity checks. This is a
conservative flag choice, not a claim about the default compiler FTZ policy.
Root's actual map/link audit finds no crtfastmath object; the actual leaf's
host RNE/FTZ/DAZ gate passes. Actual build CPU lease139s, adapter12.373s and
AOT link112.725s; these phase observations are not a matched build-speedup.

Actual executable0e5c75c9 was built under the external root-build/build path.
The original external gpu preparation/build was an explicitly audited SYMLINK
to that new actual build for the unchanged runtime controllers. The copied
root-build-closure records both paths. That symlink is NOT created here, and
copying a controller does not supply its binary, archives or valid root closure.
Do not confuse the retained pre-S79 engine77a18ca2 archive with current mutable
main build outputs or link from the latter without a new reviewed identity.

CPU v2 controllers expect their own external prep/build and run-cpu parent,
container /prep,/build,/inputs,/out bindings, and pinned compact S42 exports.
finalize.py expects preflight.json under the run parent and reference/
under it, not directly inside this repository record folder. Its historical
reference.json/core status says GPU identity PENDING because CPU alone did not
measure that premise. The subsequent root-result/ closes full GU identity for
S80; no copied CPU historical status is silently rewritten. Down CPU plane is
diagnostic-only and is not an independent down-accuracy target.

The shared header468471b8 is byte-identical in GPU and CPU v2 records. The CPU
v1-v2 diff is an exact whitespace-clean copy. Original findings/source pins
are retained, not replaced by this explanation. No original raw evidence is
edited. Source-review IDs/conditional decisions are recorded in root-result;
actual post-result claude-opus-5-5 (24.734s) and claude-fable-5-1
(22.687s) both ACCEPT, with root full reads. Their exact records and whole-stage manifest copy are included. Completed external whole manifest
SHA256: 5f0f2262cef1699098066fb71f5e8bbabecf4842631bdcb800f9d211fa5f728f.
Post-review closure SHA256: ba36f6b6ca21654c255d2dcb2e34ee093e60500f384bb1fb215e2e7945575d3d.
