# S101 next low-level kernel candidates, 2026-10-15

Actual Claude Opus5.5 and Fable5.1 prepared independent next experiments.
Root accepts source audits only; no new kernel build, fusion qualification
or GPU result follows. [Exact packets, answers and root decisions](../strata/flash-next/lowlevel_probe/next_kernel_designs_s101/README.md)
retain the actual responding IDs and corrections separately.

CONFIG -> Current S95 primary-GPU operation census and exact current SYCL
source, pinned source packets25.0/26.0KB. S100 routed IQ2_XS work is separate.
Largest named all-nonvocabulary MMVQ service is12.680488ms per reviewed
instrumented cycle, disjoint from routed GU/down7.504495ms. Service spans
are non-additive; no CPU/GPU or cross-card clock join establishes a budget.
CPU handoff2.36us does not justify another host-handoff experiment.

COMMAND -> Actual `claude-opus-5-5`104.737s and
`claude-fable-5-1`111.676s, independent packets in their own resumed sessions,
with tools disabled. Root read both complete answers and final frozen packets.
Two earlier SHA-only preflights stopped before any model invocation because
packets were still being shortened. Explicit author freeze and final hashes
resolved that coordination issue; no GPU retry or repeated model response.

RESULT -> Opus recommends an existing-image audit of Q6 C6 A2 reconstruction.
The source helper loads two16B-aligned chunks, selects words for four possible
dword offsets, then funnels a possible16-bit shift. Proposed A4 would use
five dword-aligned scalar loads and retain that funnel, seeking to remove
selectors without repacking weights. Q6 source already hoists weight decode
outside NCOLS, so that hoist is not a new lever. Root requires actual main
host Info/cache/guard/name/image binding, then comparison with the existing
S85 ORIGINAL computational body/metadata before choosing another decoder.
Prepared S102 audit source is not an executed result in this checkpoint.

Opus corrections: the word reconstruction has four offset choices, not a
five-way selection. Never dereference a misaligned int4. Base%4, allocation
bounds, exact bytes, compiled FP contraction and bounded strict runtime
checks remain distinct gates. Static ALU/send counts do not measure traffic
or dynamic cost, and arbitrary instruction-magnitude thresholds do not
justify a verdict. S98 is an IQ2 fixture: reuse actual S44/S85/S87 Q6 exports
and tools deliberately. No universal bitwise-by-construction assertion.

Fable recommends investigating one peer WG128 kernel combining epoch wait,
uncached inbox copy and resident plan. The aim is shorter peer readiness,
with math and downstream grouped/send operations unchanged. Root accepts
only a source/dependency lead. Whole-collection per-device wait medians do
not prove which branch controls the end-to-end critical path. The proposed
~8us prologue and attribution of1392vs1152 counts to cold/capture work are
unproved. Do not add copy/plan service or infer physical idle/PCIe traffic.

Fable corrections: preserve signed epoch difference, bounded timeout plus
error/continue semantics, helper and caller system acquires, all-thread
uncached copy and global/local visibility before the plan reads IDs. Pass
one actual copy-destination pointer with a proved ownership/lifetime ledger;
do not broadly remove restrict. A one-card primitive cannot qualify
cross-card freshness. If justified, reuse a bounded TWO-card IPC fixture
before expensive serving, with guards, exact plans, stale epochs/wraparound
and A-B-A graph replay. No blind readiness-wait removal is authorized.

Root's partial doorbell audit pins helper82b325b3... and unchanged build
ninja89cb173d...: no atomic-load/store or spin override is set in checked
compile flags. Default sys_load has system acquire plus L1/L3 uncached read;
sys_store has uncached write then release. ep_wait also has its own final
system acquire. Checked default B70 spin bound is20000. Include/macro scope,
actual emitted cache descriptors, whole-WG visibility and IPC freshness
remain unqualified; this is not approval to fuse.

VERDICT -> First bind and inspect current existing Q6 C6 code for the A4
hypothesis, reusing S85 ASM only if actual relevant body/metadata match.
Keep Fable's EP fusion separate until source visibility/ownership and
critical-dependency evidence justify a finite probe. Do not repeat rejected
GRF256, packed Q6, SWAR, MORE, SG16/DPAS, shared-streamOFF, headRPW2,
HC global fusion or commit-overlap experiments without a distinct mechanism.
Graph-boundary follow-up is tracked separately from this checkpoint;
no graph implementation or speed result is claimed here.
All serving targets and S30 no-recompute RAM recovery remain unchanged.
