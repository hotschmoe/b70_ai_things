# S104: Q6 C6 aligned-load primitive

CONFIG -> Two real C6 Q6_K shapes, 6144->2560 and2560->10240. A4 candidate
changes only ql/qh load reconstruction; canonical signed centering, DP4A,
FP accumulation and subgroup reduction remain. No main integration.

COMMAND -> Actual source Opus41.562s/Fable53.569s; root512case source-byte
check0.032s. One25s CPU adapter build, three static host/image bindings,
public/private-original16384B prefixes exact S85 original. One10s selected
candidate decoder; actual compiled Opus38.755s/Fable62.821s/root read.
One40s quiet CPU/card0 window, physical pin, fresh pre/posthealth, no retry.

RESULT -> All114 strict snapshots and10 graph DOTs PASS. Public/S44/private
original/candidate anchors, full changed/restored outputs, canonical Q8,
readonly/guards/tails/finiteness and single/batch20 graphs pass on both
shapes before timing. Four balanced rounds per shape have median paired
completed latency changes -4.852871% and -2.204638%. Original/candidate
median paired-mean call times are63.096850/60.052813us and
73.827700/72.207338us. All four paired changes favor candidate per shape.
These are nested warm observations in one process, not serving speed.

VERDICT -> Bounded private mechanism passes; keep separate from production
graph comparison. No automatic integration, combined flags, default/shelf
or full-cache qualification. Native/driver/tee/analysis/identity and
pre/posthealth exit0; exact created CID+NAME absent. Existing --rm lifecycle
has no final Docker OOM-field observation, so no such field is claimed.

Actual candidate removes30 static selects (38->8),32 moves and6 compares,
retains48 DP4As,20FP MUL/6MAD/36ADD, SIMD32/GRF128. Missing resource fields
are not measuredzero. Static counts are not traffic or latency budgets.
Controls are computational-prefix exact; raw whole image/text equality is
false and retained. Candidate identity is established by its own binding,
ISA and bounded runtime anchors, not universal equivalence.

record-copies.json binds exact text artifacts. Raw compiler/ISA whitespace
is preserved. Binary, selected device images, model slices and saved binary
snapshots remain on disk, pinned in these records. No75GB model rehash or
extra primitive run. See [dated report](../../../../docs/20261015_strata_q6_aligned_loads.md).
