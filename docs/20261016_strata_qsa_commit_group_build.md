# S143: default-OFF three-row QSA commit source and build

CONFIG -> After S141's strict private contract passed, apply only a verifier
host-selector patch. STRATA_QSA_COMMIT_GROUP_STEPS is process-constant before
capture and defaults OFF. ON replaces three single appends only for contiguous
owner groups with length exactly3. Every other length retains the old loop;
accepted-prefix1/2/3 tests are not evidence for n_steps2/4/5/6. The existing
steps entry gets raw first row, contiguous commit positions, stride1,n_steps3,
position-base0 and unchanged gamma/epsilon/buffers/geometry/capacity/RoPE/queue.
Device tail restoration remains before the branch. Source inverse restores
all original bytes, and dedenting the OFF else body reproduces the original.
GDN/conv, PLE, KV, snapshots, waits and eviction contracts are source-unchanged.
This is a source projection, not a main compiled/numerical theorem.

COMMAND -> Root applied the exact reviewed patch successfully after S142 closed:
verify.cpp e97ce93a ->
`c28de288061beae19949982ff3de7cf0a54fcae4d23f61e3a0374f90aeb435c6`.
The committed actual-model source/post-private-pass decision is5c9aa822,
claude-opus-5-5 and claude-fable-5-1; full replies are referenced rather than
copied. The tested analytic RoPE domain requires common STRATA_ROPE_TABLE0 in
both later serving arms, with no map/IMRoPE text requests. No extra test or ISA
certification was added to this source/build stage.

One existing-config CPU-leased incremental build used pinned e41, no DRI or
network, UID1000, four CPUs/jobs/threads, disk TMP and the stable offline cache.
Actual dry plan passed exactly3 actions: verify TU, engine archive and main link.
No configure, broad rebuild, backup restore or retry. Applied/unchanged source
and configured build-state pins were checked before/after. Exact scripts, guard,
plan, source hashes, cache counters, ownership observations and exits are retained.

RESULT -> Apply, build and tee each exited0. Root execution65.029740755s;
Ninja49.830s. Actual last-three Ninja component durations recorded in the root
receipt: verifier compile16.660s, archive0.078s, main link33.065s.

| Frozen completed output | SHA256 |
| --- | --- |
| Main executable | b5104c309877cf242b153075609c3386a1683c9cc388dea414fdd6522192f779 |
| Verifier object | 726df48c67069c1295168a47aa442a3074224c88ea639e78fa4dc6709047497c |
| Engine archive | 902f583325aba6a04a9c7f439c0876bdc9917cfe93bdb7199901e902246e3ccf |

Persistent offline cache counters were2923hits/1903misses before and
4532hits/1903misses after:1609 additional hits, no additional misses. This is
actual cache reuse evidence for this build, not an isolated causal build-time
benchmark or forecast. Other TU/source, CMake/cache policy and compile flags
remain guarded by recorded identities. Curator did not rehash larger outputs.

The exact unique HOST marker is:
strata QSA commit grouped steps: enabled=1 host-selection-only

Its source identifier is unique and emission occurs only at the first selected
three-row capture, startup allowed. This stage did not audit the compiled host
marker or device ISA/images, and does not assert actual graph execution/counts.

VERDICT -> Source apply and exact incremental CPU build passed. Default remains
OFF. S144 must independently use fresh control/ON processes on the same NEW
binary, all other candidates OFF, explicit common RoPE0, exact output/work/
capture/marker/functionals and normal teardown/full-health gates. Its runtime
is pending at this cutoff and was not read for this curation. S141's private
whole-chain timing is not a main serving forecast. No full200k/four-conversation
RAM-restore qualification, promotion or hardware-overlap claim follows here.
No live source edits, compiler, GPU or model invocation were made by curator.

[Compact source/build receipts](../strata/flash-next/lowlevel_probe/qsa_commit_group_build_s143/README.md).
