# Prefix residual capture30: CPU source readiness

CONFIG -> Fresh29, native HC, serial fresh GEN, prefixes1/2/4/8, four armed requests.
COMMAND -> python3 strata/flash-next/test_prefix_residual30_cpu_v1.py
RESULT -> 23 test executions pass, including inherited coverage tests, whole T3
prefill matrices and pair0:32/32:48, missing rows/fields/stages, finite/layout,
PID/token/source/resume and actual producer-log roster negative controls.
VERDICT -> CPU contract/collector evidence only. SYCL TU compilation, off/on GPU
output equivalence, new matching390 and actual allocation/free gates unobserved.

The new0030 patch applies after immutable29. The separate combined30+31 plan
includes the frozen31 patch9b37b8cf. Both plans require a new SDK/build receipt;
old29/390/C1 results cannot qualify the changed source. The apply receipt records
fresh29 ->30 ->31 checks and mutations in a new private source-only staging tree.
The ledgers contain62 consumed source files,26 added-header payloads, six Python
runtime fingerprints, and the inherited eight SDK targets. The actual consumed
SYCL verify.hpp carries the new field; the shared shadowed type is unchanged.

STRATA_PREFIX30 is off by default. Enabling requires old SFD activations plus
nativeHC, existing absolute DIR, external64hex source binding, fresh resumed0,
serial supported old SFD mode, prefix<=8 and first4 requests. Unsupported enabled
verifier requests fail rather than claiming coverage. Off adds no allocation,
memcpy, wait, snapshot graph command or output. Old graph variants contain these
copies only while old fidelity_first is selected, including combined layer0 keys.
No inference math, reset, scheduler, cache or precision setting is changed.

Input is each layer's complete R matrix before PLE; attention and FFN are R after
the corresponding explicit HC write and existing cvec boundary. Prefill captures
all earlier rows, not only last-row slices. Verifier captures its last-token row.
Each field is LE_F32[actual_rows,4,2560]. Row position and GEN token IDs bind the
whole matrix, never a flattened last-row approximation. The completion nonce is
copied on the owning stage queue after all144 global phase/layer commands.

Data per prefix8 across48 layers is47,185,920B. Two separate Prefill/Verifier
owners can coexist:94,371,840B plus16 control bytes per owner per stage. A128MiB
aggregate diagnostic budget reserves16MiB for prior SFD/L0 observers. Actual
allocation chronology must still verify all buffers/contexts; no GPU lifecycle
claim follows from constants. Each owning queue stays alive until its snapshot
is released; graph retirement precedes verifier release including destructor,
explicit diagnostic release and mirror-graph release. Prefill has no retained
snapshot graph. Runtime graph-handle association remains unobserved until a
separate actual trace establishes it.

The collector CLI requires --producer-log, expected request token rosters,
expected exact stage ranges and external source binding. It cross-checks SFD
request/resume and P30 frame multisets with metadata PID/nonce/token/row scope.
Missing entire processes/requests/stages/earlier rows cannot pass. Python's direct
collect(...,producer_log=None) is explicitly metadata-only test mode, marked
producer_log_crossbinding_verified=false. All receipts retain fullmath=false.

Next actual gate: new build/390/identity/health, same source version off/on full
first-head and48 residual equivalence at1/2/4/8, then owned-original composition
comparison. These captures localize first differing residual; recurrent internal
state, all48 operator equivalence, coherence, concurrency and latency remain
separate requirements. No original model payload was read by these CPU tests.
