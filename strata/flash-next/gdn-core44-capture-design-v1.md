# Source44 output-only first GDN core/RMS capture

CONFIG -> Exact actual source43 kfiohqmn, parent17ff9e21. Existing NUM10
remains default OFF. NEW44 has NUM38 layout: unchanged first33 fields plus
scaled core6144 F32, raw RMS squared sum48, actual RMS argument48, actual
rsqrt scale48 and actual rsqrt128 scalar. Extra raw25156B/frame,100624B/four.

COMMAND -> prepare_gdn_core44_capture_v1.py produces a new patch/proposal
without writing the source43 SDK. Same output snapshot allocation, device/
context/in-order queue, four graph rosters, request/PID/nonce, final marker,
error word, owned release and64MiB raw cap. New observed T1 entry requires
complete pointers and actual one keep; unsupported keep sets existing error
and cannot publish. Old production host dispatch function remains byte-exact.
Old false template specialization keeps the original arithmetic expressions;
new true specialization stores actual already-used core/argument/scale values.
No additional query, barrier, wait, normal-model kernel or state update.

RESULT -> Only existing L0 NUM diagnostic uses the new entry. OFF routes the
original dispatch. Diagnostic ON adds output stores and five existing-style
raw dump readbacks/waits per frame; observer cost is explicitly unmeasured.
New metadata declares source_generation44/core observed. Old NUM33 readers
must refuse this layout. Fresh all8 compilation/public-header oracle, current
SDK/runtime/identity, new NUM38 actor/reader, OFF/ON full49 parity, all five
actual field witnesses and health/EOF/free/full4 are required before use.

VERDICT -> Source-only. No original math correction, reference substitution,
capture/runtime/latency qualification or old43 proof transfer. Existing exact
captured packet/state audits are conditional; independent original operands
remain the primary reference. A future own F32 core/reduction comparator can
consume new native fields as targets, with actual rsqrt128 rounding observed.
