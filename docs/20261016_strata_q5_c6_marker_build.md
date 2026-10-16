# S138: rebuilt C6 marker binding

CONFIG -> Apply only the two C6 block-local identifier edits reviewed in
[S137](20261016_strata_sycl_sibling_static_marker.md), committed a214916e.
Native source changes from e25708bf to
`dd08daf548eab9ab69375ed1702dfa85214bd1431e9957126d12eea51ed569d1`.
The C2 branch, parser/guards, marker literals, launch arguments and device math
remain source-exact under the inverse projection. Both actual source reviewers
accepted a main build conditioned on a narrow rebuilt-host marker check; their
full responses and filtered identities remain in S137 rather than duplicated.

COMMAND -> One existing-config incremental CPU build in e41, no DRI/network,
UID1000, four CPUs/threads, disk TMP and the existing persistent offline cache.
The actual dry plan contains exactly native TU, kernels archive and main link.
No configure or other source rebuild is claimed. Applied/unchanged source and
build-state pins were checked before and after. The exact actual root wrapper,
plan, compiler, cache counters, ownership observations and execution are retained.

Before execution, root saw that the proposed V1 checker assumed LC symbols,
where the actual object uses section-plus-addend string relocations. The
manual-host wrapper omitted that automatic checker, retaining a separate
required host check before GPU. V2 resolved the actual section/addend form.
The unused V1 helper and exact source delta remain as lineage. This was a
preexecution source/setup correction, not a failed build or repeated attempt.

RESULT -> Build and tee exited0 with no retry. Total execution99.832075363s;
Ninja90.933s. Root reported the last three Ninja durations: native57.886s,
archive0.299s and link32.713s. Offline cache counters moved1314hits/1903misses
to2923hits/1903misses:1609 additional hits, no additional miss. Those are actual
counters for this build, not a controlled cache timing or population estimate.

| Frozen output | SHA256 |
| --- | --- |
| Main | 6b37aef71607a4217192165523f413063c03d5a3cf042c6dd1031800ca2a6eb2 |
| Native object | 98d502efe4007fbcf82ba5199cdb10c0e0de425da9a8f90ea9be16ca962c5dd6 |
| Kernel archive | a1652f48f406acf1823cec8d7881be1b49881434e0fd342dff5b85dedee60227 |

The separate required V2 host check exited0 in a root-reported0.249s. Root read
the complete facts and both actual logging blocks. C2 guard0x68 references
C2 text at string-section offset0x2a42 with fwrite length99. C6 guard0x78 now
references C6 text at0x2b9a with length102. Parser guards remain distinct at
0x368/0x388, with kernel-info records at0x10e08/0x11020. This corrects the
specific compiled host marker symptom seen in S134. The actual selected host
function and root-read disposition are preserved with their exact hashes.

VERDICT -> CPU build and required narrow host-marker binding passed. No new
device ISA/full-image certification or GPU arithmetic/dispatch proof is claimed.
S134 remains its original failed marker/null comparison. A later same-NEW-binary
control/ON finite serving stage must stand on its own identity, exact output/work,
marker and normal teardown/health evidence; its outcome is not consumed here.
The marker is host selection evidence only. No automatic serving, retry,
compiler flag sweep or old-evidence rewrite follows from this build record.

Compact source, actual build and required host-check evidence: [S138 payload](../strata/flash-next/lowlevel_probe/q5_c6_marker_build_s138/README.md).
