# S147 GDN layer commits: guarded source and completed main build

CONFIG -> S147 V2 adds default-OFF `STRATA_GDN_COMMIT_LAYERS` to the existing
Strata SYCL main. Two new ordinary-launch classes call the existing convolution
and state-only helpers directly; the original helpers and public per-layer
launchers are unchanged. Eligible layout is exactly two three-row owner groups,
MT8, full36-GDN-layer geometry, in-order queue and the tested native modes.
All other layouts/modes retain the old loop. QSA grouping remains OFF/parked.

The [S145 private oracle](20261016_strata_gdn_commit_layer_oracle.md) passed
complete235MB state comparisons, accepted-state byte continuation, finite/
guard/read-only/sentinel checks and normal health. Its private26.04% whole
bundle improvement includes batching, ordinary launch and private compilation;
it forecasts no main serving gain. The separate
[actual-model decision](20261016_strata_gdn_commit_integration_decision.md)
records Opus/Fable CPU_BUILD_GO and root corrections. Source GO is not runtime
qualification.

COMMAND -> Root applied exact V2 after full review, then ran the existing
configured incremental controller under cpu-run: pinned e41, UID1000, noDRI,
networknone, four CPU jobs, O3/precise/SG32/per-kernel BMG-G31 policy and the
existing owned offline cache. No configure, old-main/archive/model rehash,
cache warm, extra ISA certification or GPU workload was part of this build.
The exact wrapper and six source arguments are in
`strata/flash-next/lowlevel_probe/gdn_commit_layers_main_s147/build-prep/`.

V1 was unused. V2 explicitly keeps old validation: null z/y or null norm data
falls back; a missing norm WeightRef retains its existing need error. A host
in-order property guard covers external-stream ambiguity. Before recording,
seven bounded/aligned/nonoverlapping spans cover both complete owner states and
qkv/h/gate/beta/device keep buffers. Every eligible GDN norm weight is checked.
The existing QSA/PLE/KV capture tail and commit wait remain byte-unchanged;
GDN operations move before their independent buffers. No extra y_dummy memory
is allocated: selected state-only math does not receive y/z/gamma. Source
patches, full inverses, V1/V2 lineage and exact projection pins are retained.

RESULT -> Actual first attempt26211 stopped at the ownership preflight,
BEFORE Ninja/dryplan/compilation: one existing public verify-kernel object was
root-owned/nonwritable. Root retained that attempt, then atomically replaced
ONLY that named503792B file with identical bytes, mode and mtime, changing
ownership0->1000 in its writable parent. No sudo/broad chown/source change.
This setup correction is not a kernel/runtime failure or repeated experiment.

Corrected actual build3421 exited0 (tee0). Root's wrapper receipt reports65s
as an integer; Ninja reports52.846s. The actual dryplan was exactly FIVE actions:

| Action | Recorded duration seconds |
| --- | ---: |
| verify-kernel TU compile | 11.531 |
| kernel archive | 0.313 |
| verifier core TU compile | 16.828 |
| engine archive | 0.128 |
| final SYCL/AOT/backend plus host-link action | 35.865 |

Compiles overlap; their durations must not be summed as build wall time.
The five target interval span is52.822s. Final link is not an isolated AOT
measurement. Offline-cache counters4532hits/1903misses ->6087/1959
(+1555/+56) are descriptive; they establish no per-kernel cause or cache speedup.
All source/build-system pre/post guards passed. CMake, native MMVQ, MTP and other
source inputs remained pinned; two TUs, two archives and main were rebuilt.

Frozen new main SHA256:
`98f0357dd01d176476edcb101ec489280deb2925399a64d31b44ef3abfe17200`.
Verifier source: `f669057242387e85ba3d433e8be0acacfff0c835a28af34f60fc278ccd88410b`.
Verify-kernel source: `929813472cd1ee9710c811e9c05fe48e8e459daa9fb62527f4cedb0839b86afb`.
Existing build receipts bind output/object/archive identities; curator did not
reread those binaries. Full source duplicates, binaries and full noisy Ninja
log remain external; exact last-five target rows and small receipts are indexed.

VERDICT -> Source application and CPU build PASS; default remains OFF. S148
fresh same-NEW-binary OFF/ON serving is a separate stage and is not claimed
here. Main numerical/work/capture/marker/functionals/normal-health gates remain
required. The reused health attestation is explicitly composite: retained S144
card1/collective with later normally closed single-card0 health, not a new GPU
health run performed by this CPU build. No promotion, stability claim or RAM
restore requalification follows from compilation.
