# S145: private GDN commit layer bundle

CONFIG -> Existing PUBLIC per-layer control against one private layer-batched
candidate. Geometry is36 GDN layers,two disjoint owners,three proposed rows,
MT8,state128,keyheads16,valueheads48,C10240. State-only modes are fixed:
STRATA_GDN_SPLIT0,STRATA_GDN_COMMIT_SPLIT1,STRATA_DBG_GDN0. Control uses the
unchanged503792B verify-kernel object, not a private clone. The candidate uses
source-renamed exact existing conv/state body spans with layer pointer offsets.
The inverse name projection restores the recorded original body bytes. Public
validation/output sentinel remains part of the finite contract, not omitted.

COMMAND -> Root fully read the original critical source packet and V2 delta.
V1 was unused. V2 removed ONLY the unused private use_root_sync property
declaration and its two launch arguments. Official root_group launch-property
specification itself constrains compatible max-work-group sync geometry even
without a root barrier; private bodies use work-group barriers only. Their
body spans, arithmetic, indexing/local shapes and work-group barriers remain
unchanged. Actual B70 max-work-group query and cooperative backend path remain
UNKNOWN. Public control retains its existing property. This source correction
preceded build, not a failed GPU attempt or alternate-run recovery.

One e41/noDRI/network-none/UID1000/four-CPU precise offline build compiled the
fixture and two private kernel classes, prelinked them with the existing public
object and linked the leaf. No main/archive rebuild or ISA certification.
Actual build execution23.552887910s, build/tee0. Components: fixture/private
compile9.710s,AOT2.699s,host link0.196s. New executable SHA:
`0c8e52871906ac3cad7bde8f5ddaa9e9c8157aa24a7c142ff9391a4723308a55`.
Input hashes were checked before/after; curation uses those completed identities
without binary/model/archive reads or rehashes.

One finite card0 CPU-quiet run then used exact physical PCI0000:0b:00.0 and
fresh pinned per-card pre/post health. Non-auto-removed owned container was
inspected, normally waited/removed, then full CID and NAME absence confirmed.
Unknown/live lifecycle would hold the same leases for root; no forced stop,
reset, retry or blind health. Root run execution68.240021042s exited0.

RESULT -> Nine strict A/B/A phases for accepted pairs(1,3),(2,2),(3,1) passed
full recurrent+convolution state checks across public/private eager/captured
routes. Each complete compared state is235339776B. All state floats were finite;
all spans/guards/read-only inputs and shared output sentinel passed. Accepted
state snapshots restored unchanged into both arms then continued with the next
seeded commit; continuation eager/captured results also matched. This exercises
byte-layout continuity, not a full-model200k RAM-cache restore qualification.

Eighteen compact FNV1a64 receipts follow actual complete memcmp checks. Saved
receipt diagnostics show A0=A2 and A0!=B1 for each accepted pair and both eager/
continuation kinds. FNV is NOT the numeric comparator or a collision-free byte
proof. No full snapshot files were written; no235MB payload export/read is
needed for this curation. Exact raw strict log, comparison and root summary
remain retained. Native/create/analyzer/wait/rm/pre-health/post-health each
recorded0; final non-OOM/noError and full CID+NAME absence passed.

| Completed HOST captured GDN-only chain | Public control | Private bundle |
| --- | ---: | ---: |
| Four retained samples, us | 1199.487,1186.632,1180.921,1195.790 | 861.262,885.659,879.627,882.462 |
| Median, us | 1191.2110 | 881.0445 |
| Source conv/state calls | 72+72 | 2+2 |

Change of medians:-26.0379143577%. All eight ABBA/BAAB samples remain in actual
interleaved order. Timing is completed whole captured GDN-only work; setup,
restores, readbacks, finiteness/guards and digest generation are outside the
timed interval. One process/four samples per arm are nested observations,
not independent population replication. No actual per-kernel submission-count
or DOT export proof is claimed from source call counts.

VERDICT -> Finite private complete-state/continuation/health contract passed.
The tested bundle combines layer batching, ordinary private launch property
and private compilation, so it does not isolate launch-count or any hardware
cause. No serving gain, whole-model latency budget, main image/ISA equivalence,
promotion or integration GO follows automatically. Group lengths/modes outside
the tested domain remain unqualified. Production owner spans/layout, original
norm-presence validation and PLE/cache/eviction completion remain separate
source obligations. S146 is preparation only. No new GPU/compiler/model CLI,
repo edits or runtime settings were executed by curator.
