S112 private top1 oracle preparation v2 (2026-10-16)

CONFIG -> SOURCE PREPARATION ONLY; no build, compiler, Docker, GPU or model
execution. No production source/archive edits or binary/kernel hashes now.
Root approved one private leader-acquire hardening after reviewing actual
SDK work-group barrier scope. Existing argmax is not currently selected on
B70; no observed runtime bug or production regression is claimed.

SOURCE -> private_argmax.hpp clones ONLY argmax helper/kernel body into
namespace s112_private with unique registered class GreedyTop1S112. The
sole semantic body delta is: leader captures relaxed fetch_add old value,
performs device-scope memory_order::acquire, THEN publishes last flag before
the WG barrier. Original pre-RMW release/acq_rel and post-barrier acq_rel
fences, FP comparisons, ties, layout, loops and reset remain exact.
source-projection.json and candidate-body.diff prove exact inverse source
reconstruction. Compile only oracle.cpp/new candidate kernel; old greedy,
row_top_prob_split and mtp_select come from the pinned S107 archive.

ORACLE -> Irregular4097 FIRST, then known vocabulary248320. Two owners, each with
separate reference/candidate logits, arg/top scratch, device IDs/probabilities,
residuals/token/selector row, guarded uncached mapped token/prob outputs.
All device/mapped allocations have128B guards. Scratch has4row capacity,
initialized ONCE; every completed replay checks all counters0 and unused
rows bytezero. No reset between replays, retries or counter-only recovery.
Same downstream probability/select functions on BOTH routes; exact int32
IDs also checked against an independent ascending host maximum/sentinel
reference. Full probability/residual/token/mapped bytes match, readonly
logits/residual input/selector row and all guards checked, unused output
slots must retain poison. Synthetic residual stride10240 uses source HC4*N2560;
no model weights or production activation equivalence is claimed.

Finite set: two shapes x eager/captured x four changing A/B/A triplets =
48cases, each checks two owners x two routes. Inputs include unique/tied
maxima crossing chunks, signedzero, all-negative finite background, finite
max amid NaN/-Inf, allNaN/-Inf fallback0, +Inf ties, minimum subnormals,
and ties on actual computed chunk boundaries. nb/per use the kernel's
formula for EACH shape; both owners' first A/B/A maxima change WG chunks.
The subnormal case places a lower-index+0 competitor against a higher-index
positive minimum subnormal: host-reference equality is meaningful.
Subnormal/NaN source behavior is not silently weakened: any old greedy vs
host mismatch is HOLD too, and would require explicit diagnosis. No claim
of exhaustive bit-pattern coverage. Inputs are uploaded OUTSIDE capture,
then the same captured operation graph reads fresh device contents A/B/A;
this is not a mapped-input-copy graph test. Mapped output is inspected after
queue completion; no early-publication/poll-timing proof is claimed.

Capture exceptions first balance recording using the pinned DPCT manager
ownership rule: begin/record failure retrieves/ends via DPCT; if end assigned
g before throwing, cleanup invokes g->end_recording directly. It then drains
and rethrows the ORIGINAL exception; cleanup failures are separately reported.
No recapture/replay/continue. All replay failures drain/report/stop. Route destruction drains before
freeing bound allocations; cleanup failures prevent PASS. No redraft,
automatic retry or alternate kernel selection on failure. Existing queues
are in-order; graph lifetime ends before buffers. No owner scratch overlap.

Optional --timing: only for n248320 AFTER ALL48 strict cases, two ABBA rounds,
one captured two-owner chain per observation, host wall + completion wait.
Staging/reset/readback are outside each measured chain; guards/counters
still checked after each. Includes greedy/prob/select work, not isolated
argmax GPU service, critical-path proof or model serving throughput. Root explicitly plans --timing in the same finite lease if strict gates pass.
The irregular shape is never timed.
No long soak or broad shape sweep is proposed.

CPU BUILD -> ROOT ONLY after full source/runner review:
  bash <this directory>/build.sh
Timeout0 bin/cpu-run, same e41/DPC++2026.1, O3 precise, SG32/per_kernel,
BMG-G31 AOT/jobs4, UID1000/cpus4, noDRI/networknone/diskTMP/cacheOFF.
Existing live build tree is mounted READ ONLY. Link the original kernels
archive normally, not --whole-archive; helper core/spec/CPU/ggml archives
are grouped to resolve existing dependencies. No configure, Ninja, source
compile of original kernels, ownership repair or automatic relink fallback.
Undefined-symbol/link/identity errors stop for root review. Build snapshots
extract/hash/cmp only original sampler/verify object members, write private
object/binary hashes and symbol/section inventory. Expected source/archive/
CMake identities are checked before/after. Successful link is NOT GPU GO.

Actual compiled registered name/Info/cache/guard/image bindings are PENDING
until a binary exists; see identity-plan.json. Root must audit selected
original and private candidate images before GPU. This preparation does not
invent those hashes or call a symbol listing an image-dispatch proof.

GPU RUNNER -> NOT AUTHORIZED/EXECUTED by this preparation. Root must first
supply actual binary identity/image-audit closure, then wrap a finite pinned
one-card container in bin/cpu-run --quiet OUTSIDE bin/gpu-run --card N,
with matching workload device pin and applicable per-card pre/post health.
No model payload, collective or full server is needed by this standalone
oracle. Retain owned full CID/name, native exit, normal teardown and health
before claims. Current S107 serving lease has priority.

VERDICT -> FROZEN SOURCE ONLY. Not a test result, speed claim or permission
to select the original unhardened argmax on B70. Root full-read and actual
post-build compiled-image audit remain required before any GPU execution.

VERSION -> Immutable v1 stays at greedy-top1-s112-oracle-prep. This v2 changes
fixture boundary arithmetic, meaningful subnormal competition, balanced
capture-error cleanup and timing gate/order only. Private argmax header/body,
original archive bindings, CPU build flags and failure-stop policy are exact
v1. from-v1.diff and v1-reference.json bind the recut; no build has run.
