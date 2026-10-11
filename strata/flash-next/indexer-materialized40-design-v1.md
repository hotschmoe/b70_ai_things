# Isolated materialized indexer input storage diagnostic

CONFIG: Existing source40 SDK d87740b5 and its fresh C1401403 baseline. Original
GGUF/quantization/reference/source40 functions remain unchanged. Input is the
four independently owned layer3 indexer.k_proj outputs and original k_norm
vector from producer-v4; resolved original epsilon/RopeScaling is checked
separately from literal CONFIG. No native capture feeds these computations.

COMMAND: ROOT only, after source/peer review and exclusive GPU coordination:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/prepare_indexer_materialized40_inputs_v1.py --own-producer-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-layer3-qsa-producer-v4-run-v2 --output <NEW_FIXTURE>

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/qualify_indexer_materialized40_v1.py --fixture <NEW_FIXTURE> --prepared /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/c140-v3-one-segmented-postboot-prepared-v3 --own-producer-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-layer3-qsa-producer-v4-run-v2 --output <NEW_RUN>

NUMEXPR remains absent, matching original host admission. The qualifier self
leases both cards before expensive semantics; the leaf pins card0. Compile
image39992 and runtimeC388 are explicit. Original frozen half lifecycle is
ported in a private function namespace with exact source SHA and declared
recipe/marker/prior adapters, never mutating original globals or files. The
runtime inside-bootstrap is standalone stdlib only, requiring no host modules.
Per-card strict +compiled P2P0 health, current source/ABI/library checks, exact
normal container/recipe/EOF/retirement, journals, full4 and page brackets remain
mandatory. Explicit pre-run health age<=300s, no model/pack mounts in the leaf.

A single complete original semantic admission runs per owned process; existing
complete ByteEpoch seals bracket that admission and reread every evidence/SDK/
pack/source byte on each later use, including before the helper starts. Model
stat and known pages are guarded; unchanged pre/post full4 remains mandatory.
This is not a persistent cached PASS. Original PID/roster/views and actual
boundaries are saved as producer-operation-witness.json and read-only rejoined.
No proof may transfer to another actor without its own semantic admission.

RESULT: Six modes: single_raw/single_materialized, steps_raw/steps_materialized,
batch_raw/batch_materialized. Every mode restores ownzero and runs direct +two
graph replays (18 frames). Half Store and separate Load use persistent owned
global half/expandedF32 scratch, outside graph allocation, on qualified in-order
queue. Production public append/append_steps/append_batch receives raw or
expanded; projected raw is never overwritten. Actual 16bit half bytes, widened
F32, raw/gamma/position echoes, complete before/after tail/dead/pooled/blockpos
are saved with exact rosters and stable consumed-byte bounds. Graph retirement
and queue waits precede all frees and post-mapped library capture.

The four production transient cast sites are append78-82,append_first244-247,
append_blocks304-310,append_tail358-362 in native_qsa_indexer.dp.cpp. Single and
steps execute site1; batch covers sites2-4. Materialized input makes their
existing conversions idempotent, without altering their arithmetic or source.

Independent integer IEEE F32->F16 ties-even and exact widening are storage
expectations. Actual half bytes/expanded and all materialized tail windows must
match bitwise; complete materialized states must match across all3 APIs.
Raw production modes are descriptive and may themselves round on another
compiler; no negative inference is made merely because raw!=widen. An explicit
WrongRawStore kernel writes original F32 to an owned shadow output: its actual
bytes must equal raw and fail the preregistered RNE storage expectation, on all
routes. Synthetic CPU controls cover half boundaries, wrong-store rejection,
changed/extra/alias/truncated artifacts, params, current-source routing, isolated
bootstrap and complete-byte re-use failure.

VERDICT: Source/CPU only until fresh compile +owned root GPU execution and final
public admission. A passing isolated storage contract is distinct from model
semantic repair, full model math, head improvement, compiler/driver cause or
speed. Existing target audit shows unrounded native tail for384 elements per
request with up to4079F32 ULP steps versus RNE; active selection remains all
visible cells, so this storage seam does not establish the prefix4 head cause.
All prior sources/captures/failures remain immutable; no canonical registry edit.
