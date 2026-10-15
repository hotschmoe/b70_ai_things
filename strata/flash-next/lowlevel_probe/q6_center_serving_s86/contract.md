# S86 exact-shape Q6 serving proposal (2026-10-15)

CONFIG -> Same binary OFF/ON future candidate, STRATA_Q6_CENTER defaultOFF.
Only Q6 wide NCOLS6/RPW1/A2/SG32/PF0, K6144/M2560 or K2560/M10240. New CENTER
compile-time bool defaultsfalse; exactly one extra C6 instantiation/unique ID.
COMMAND -> Prepare original/candidate copies and unapplied patch; inverse
substitution reconstructs original bytes. No main apply/build/GPU/model calls.
RESULT -> Source-only candidate. VERDICT -> ROOT_UPSTREAM_REVIEW_BEFORE_APPLY.

Packed registry, supported positive PF2/3/4/6 earlyreturns, non-C6, other
dimensions, RPW!=1/A2OFF and SG16 fallbacks are unchanged. Unsupported PF>0
values fall through in original code; all nonzero PF values (also negative)
are explicitly excluded from the NEW center branch. Eligible PF0/flagOFF
follows original launch; PF0/flagON
uses same geometry/queue/buffers/capture API with new ID, then returns.
CENTERfalse retains original byte sub_sat expressions. CENTERtrue contains
exact S85 tested add/XOR expressions; masked byteu0..63,+96<=159 no carry,
XOR128=signed(u-32). Same loads/Q8/DP4A/FP source ledger/XOR tree. No new
memory planes, startup/perwindow logs, class layout or broader quant dispatch.
Flag is static on first eligible host call, before graph construction; capture
records selected kernel. Use fresh processes with fixed0/1env; no live toggle.

Root reports S85 primitivePASS146snapshots/10DOTs/S44+public/private strict
initial/changed/restore/readonly/guards, health/cleanup. Eight nested20-call
balanced rounds each: out63.172075->48.725638us medianpaired-22.7841%;
qkv73.817238->57.480712us -22.0951%, all16roundsnegative. Seeded-input
projection observations only, no serving/population/hardware cause claim.
Actual main build/ISA/runtime gate still pending; source/private PASS does
not prove new main-AOT dispatch. Preserve absent spill fields!=measuredzero.

Generic03bb executable/objects are staleS79candidate, not current source.
Later root main build must recompile restored verify.hpp/verify.cpp/generate.cpp
host dependencies plus native TU before same-new-binaryOFF/ON serve. Do not
serve/link generic as baseline without appropriate audited build. Root owns
fresh upstream/source reviews, harness/build and any finite GPU qualification.

V2 changes ONLY the new host condition by adding q6k_pf()==0. Removing that
guard reconstructs v1 candidate exactly; patch is against same81558e95 source.
V1 copies/patch/contract remain immutable. No test campaign or new math change.
