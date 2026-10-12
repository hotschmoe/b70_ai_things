# Current52 normal-scheduler native2 latency probe

CONFIG -> Two sequential owned pair24/24 current52 processes; unchanged original
UD-Q4_K_XL, ctx4096/pref512, native2/group1/static no-borrow/cache3/fixedKV,
mirrorT1/copy/grouped/ESIMD OFF, no MTP, natural EOS and greedy requests.
Default STRATA_BATCH_DECODE_SHARE is absent: native source uses0.5. No SHARE0.
Exact authentic inputs are original P2048 operations A and P2048 code C.
Long A cap512, incoming C cap64, both fresh1/pin1536/ckpt1, expected reuse0.

Process1 records serial GEN A512 and C64 complete output/finish references.
Process2 starts BGEN A512 in slot0, waits for actual256th received A token while
A remains admitted/live, then sends BGEN C64 to slot1 and drains both. Arrival
binds the exact preceding256 emitted IDs and current slot generation. Long EOS
before the declared arrival fails. Both complete natural histories must match
serial references. At least one actual native rows2/mask3 window must carry
both current owners' BT tokens while both are live. No queued-only inference.

Existing STRATA_BATCH_FIDELITY_DIAG=1 emits compact completed batch_event lines.
Its ARM points to /results/NO_CAPTURE_ARM, which must never exist. Source
snapshot constructor returns before allocation when unarmed (headerline70);
begin/span/raw frame paths require ARM. Full-cache inventory is OFF. No raw
vectors, replay/row/span/allocation observations or captures may appear.
This retains actual-window proof without heavy array capture or a tool build.
H36/PTI/UR OFF; original1GiB/pipe/state/memory quotas remain unchanged.

Each original stdin send has before/postflush/after host monotonic ns bounds.
All token/admission/completion receipts use the protected pump's Python host
monotonic clock. No device clock is combined with these values. Perstream
native-client TTFT, completion, token-gap P50/P95/maximum, full token timestamps,
long arrival-straddling gap and gaps during incoming request are reported.
Percentiles interpolate sorted gaps at(n-1)*p. The timestamps describe local
native CLI pipe receipt, not HTTP/network latency or pure device duration.

Serial references and overlapping requests use identical source/config/roles,
but different stream scheduling and process/cache histories. Timing differences
are descriptive interference observations, not a balanced-temperature speed
promotion. One bounded incoming arrival cannot establish broad fairness or
stability. Actual generated EOS/counts remain visible; caps are not assumed
fully exhausted. Long output must reach256 tokens and remain live at arrival.
No original full-math, cache-family, API/shelf or full-goal claim.

COMMAND -> /tmp/b70-source52-native2-fairness-v1-launch-agent.json holds exact
root command/SDK/corpus/runtime/source pins. Construction appends
--construction-only and chooses a fresh external output path. Existing owned
boundary retains campaign complete4, per-phase source/config/stat/page/health/
kernel/memory, active guardian, raw quota, merged EOF/hash and normal retirement.
Failure closes the owner and post guards. Public reader after successful close:

  PYTHONPATH=strata/flash-next python3 strata/flash-next/validate_source52_native2_fairness_v1.py --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-native2-fairness-v1-run-v1 --output NEW_EXTERNAL_OUTPUT.json

RESULT -> Genuine actual52 pure construction passed2 phases,2 requests each,
P2048 caps512/64, mirror0/default share/no ARM/empty capture configuration.
Two tiny source-adapter controls passed: incoming follows actual256th live token;
early natural terminal fails before second send. No compiler, GPU, model payload
or native CPU model process executed by source author. No admission-chain or
capture-profile sweep added. Actual root runtime and public reading pending.

VERDICT -> Bounded normal-scheduler coherence/token-gap candidate addressing
incoming prompt interference. It cannot promote fairness, stability or speed.
