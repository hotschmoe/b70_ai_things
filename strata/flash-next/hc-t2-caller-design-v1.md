# Narrow T2 HC caller reuse

CONFIG -> Default-OFF STRATA_VERIFY_HC_T2_REUSE=1. ExactT2/G1, whole-model one
session, ordinary owned queue, device-plan stage mirror, no batch, remote,
ext-stream, split-stage or numerical observer route. All other cases retain the
original per-row caller. New kernel helper accepts exactlytwo rows. Public
HcNativeArgs/Verifier layouts are unchanged; fresh objects/link are required.

COMMAND -> Construct copied actual52 source plus the exact54 projection helper.
The target caller applies the original pending residual write, binds original
weights and sends a contiguous2-row descriptor. Down/up use source54 tile2
projection; norm, SiLU, F32 injection and mixed sums use the original composition.
Head remains per-row. PLE/GDN/QSA, expert dispatch, recurrence and commit source
are byte-identical. There is no checkpoint shortcut or new allocation/query/wait.

RESULT -> Dead-bank proof: parts_ allocation is max_t*K*N; with T2/K10/N2560 it
holds51200 floats. HC gate needs2*HC*N=20480 floats. Initial pre has no prior
parts producer. HC half1 precedes currentpost's expert writes. For laterlayers,
post consumes parts at combine1614-1627 after the existing shared-event join,
then applies residual1636 and invokes nextpre1651-1662 on the same orderedqueue.
The borrow is finished by HC mix before any newpost writes parts. G2/batch and
stage overlap are excluded; no shared array is inferred from recurrent pointers.
The old one-row gr.gated field is not enlarged or repurposed.

Fresh operator required: synthetic originalQ8_0/F32 weights, both pendingwrite
states, completeR/xn/lo/gate/rs/mixed/inject bitwise comparison against two
originalone-row calls, directT2 and graph replays, unusedparts-bank tail canary.
Fixture owns queue/graphs/buffers; failed drain retires the process before frees.
Root compiles the three changedcpp objects and fresh header overlay, retaining
originalSDK52 libraries only with duplicate originalobjects removed. Root lease,
health/kernel and actualcleanup are mandatory. No previous46/54 operator result
is transferred to this caller/bank qualification.

VERDICT -> Actual46 modelnegative used allT1 windows. The194HC T2 callcount is
source-derived, not measuredHC time. This candidate reduces attention/FFN calls
from192 to96 plus two unchangedhead reads, but promises no speed gain. Actualfull
HC operator and targetT2 output checks must precede model/performance claims.
