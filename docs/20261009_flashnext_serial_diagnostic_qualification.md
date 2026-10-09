# Flash-Next serial diagnostic qualification

## One-card basic observer equivalence, 2026-10-09

CONFIG -> Exact pinned Unsloth UD-Q4_K_XL original four-shard model, corrected
Strata native-source HC/PLE engine through patches0001..0018. Build receipt:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/receipt.json.
Receipt SHA256: 00aa6fa32dbcee86dffc35e170cfd9ea58bdf10bdc3845fc10b06f4bef687593.
Frozen V5 controller46b15abba3fd60d432968420113a6f643a9e29a837f533ba5272d0481c91d253
and parent wrapper6f2a28ec6aadbce4989a10ed25048f57fd84b9694fd37cea4af1ca21a1c350f6.
Physical card0, all48 layers, context2048/prefill64/PLE cache65536, FP16 KV,
no MTP/adaptation/prefill borrowing. Parent holds both leases for health.
Native GEN transport with recorded primary name hotschmoe-dd and detailed
pilot geometry/topology alias; this is not a live HTTP identity query.

COMMAND ->

```bash
python3 strata/flash-next/qualify_serial_prefix.py \
  --plan /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-prefix-onecard-v5-prepared-v1/plan.json \
  --group basic \
  --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-basic-onecard-v1
```

RESULT -> Finalized parent and child PASS, exit0, 837 seconds including lifecycle
and fresh model hashing. Three separate native processes use identical useful
warmups followed by fresh A/B/C probes: arithmetic42, arithmetic42, capitalParis.
All probes end naturally at the pinned EOS. Diagnostics-off versus logits-only
processes have exact IDs and LP20 strings. Logits-only versus layers-enabled
processes have exact IDs/LP20 and bitwise identical full first-token248320 F32
logits for each probe (max_abs0, RMSE0).

The layers-enabled process supplies all48 finite first-window residual rows
for EACH probe, with mandatory stage/layer/token/position/head coverage checks.
Residual values are not compared to the activations-off process, which does not
observe them. This establishes availability of the diagnostic boundary, not an
independent residual reference. No prefill residuals are claimed for these short
prompts; the coverage receipt explicitly marks their absence unobserved.

All three native processes exit0 without OOM and their containers are removed.
Strict per-card health and compiled two-rank P2P0 collectives pass before/after;
parent kernel-fault and owned-container gates pass. After child terminal and
post-health, a fresh full buffered scan of all four original shards matches
publisher hashes with unchanged source stat records. The known page sentinel
remains intact. No cache repair, runtime replacement or driver change occurs.

Evidence directory:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-basic-onecard-v1.
Primary receipts: parent-qualification.json, child/report.json,
child/basic-equivalence.json, child/*/requests.json, post-health.json and
post-model-identity.json.

VERDICT -> PASS bounded one-card observer equivalence and lifecycle. This is
engine internal consistency, not independent whole-model mathematical fidelity,
full persistent-state equivalence, complete-prefix qualification, API sessions,
concurrency, production8192 capacity, clean latency or shelf promotion. V5 retains
complete_prefix_qualified=false. See strata/flash-next/serial-prefix-goal-gap-audit.md.

## Matching two-card basic suite started

CONFIG -> Same frozen engine/controller/wrapper and pilot geometry, physical
cards0/1 with actual32/16 layer split. Consumes the finalized one-card child
receipt above.
COMMAND -> Same wrapper, two-card V5 prepared plan, --group basic and
--one-card-receipt pointing to the finalized one-card child/report.json.
RESULT -> Parent acquired both leases and started pre-strict health. Output:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-basic-twocard-v1.
VERDICT -> RUNNING; no two-card numerical result or cross-topology equivalence
is claimed.

## Finalized two-card basic and cross-topology comparison

CONFIG -> Same frozen engine/V5/wrapper and exact original model, matched
context2048/prefill64/PLE65536 FP16KV pilot. Actual physical-card split is
card0 layers0-31, card1 layers32-47. Unlike the older8192-context C1 screen,
this pilot reserves fewer buffers: stage0 has8897 resident and7487 RAM-mirrored
experts,23,402,803,200 mirrored bytes; stage1 has all8192 experts resident and
no missing experts. Placement counts are configuration-specific.

COMMAND -> qualify_serial_prefix.py with two-card V5 prepared plan, --group basic
and finalized one-card receipt; then:

```bash
python3 strata/flash-next/serial_prefix_qualification_v5.py compare-topologies \
  --one-card /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-basic-onecard-v1/child/report.json \
  --two-card /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-basic-twocard-v1/child/report.json \
  --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-basic-cross-topology-v1.json
```

RESULT -> Two-card parent/child finalized PASS, exit0,498s including lifecycle
and fresh hashing. Three native processes exit0 and are removed. Observer
off/on IDs and LP20 match; activation0/1 full first logits match bitwise for
A/B/C. All48 finite first-window residuals per probe pass required32/16 stage
coverage. Pre/post per-card strict and compiled pair P2P0 health, kernel-fault
and owned-container gates pass. A NEW full buffered scan after child terminal
and post-health matches all four publisher shards with unchanged stat records.

CPU compare-topologies PASS across all9 process/probe pairs, enforcing matched
source/controller/tokens/geometry/math environments. Output IDs, LP20 and natural
finish match on every pair. Full248320 first logits are bitwise identical for
the6 observed pairs. The3 layers-enabled pairs have all48 residuals bitwise
identical:144 paired rows, max_abs0 and RMSE0. This is the first actual observed
whole-layer topology comparison; activations-off residuals remain unobserved.

Evidence: F06/serial-basic-twocard-v1 and
F06/serial-basic-cross-topology-v1.json under the campaign results root.
VERDICT -> PASS bounded two-card observer equivalence, healthy lifecycle and
matched one/two topology internal consistency. Independent model equations,
complete persistent-state proof, prefix-cache/concurrent/API/production capacity
and clean latency/shelf qualification remain open.

## Two-card shared-root prefix suite started

CONFIG -> Same frozen two-card pilot, finalized basic/one-card prerequisites.
COMMAND -> Parent wrapper --group root with both prerequisite child receipts.
RESULT -> Acquired both leases and began pre-health. Result directory:
F06/serial-root-twocard-v1.
VERDICT -> RUNNING; cached-versus-fresh and actual reuse are not yet qualified.

## Finalized two-card shared-root prefix reuse

CONFIG -> Same frozen0018 engine/V5/parent and two-card32/16 pilot, with
conversation parking disabled to isolate the intended root checkpoint. Exact
A/B input IDs share21 system-prefix tokens, then diverge in their user suffix.
COMMAND -> Parent wrapper --group root with finalized one-card and two-card
basic receipts; F06/serial-root-twocard-v1.
RESULT -> Final parent/child PASS, exit0,388s including health and fresh hashes.
Six armed requests form two matched triples: fresh B reference, fresh A
establishment, nonfresh B hit. Both hits select exactly21 reused tokens and
complete only28 newly evaluated prompt rows, versus49 in each fresh request.
Parking remains0 bytes/0 entries; the selected route is the targeted root
checkpoint rather than an unrelated full conversation image.

Both cached/fresh comparisons and fresh-reference repeat match output IDs, LP20,
natural EOS, full248320 first logits and all48 first-window residuals bitwise.
There are3 full comparisons/144 residual rows with max_abs0 and RMSE0. All6
noncancelled requests have mandatory source token/position/stage/head coverage.
No standalone prefill residual claim applies to these49-token prompts.

Native engine exits0 and its container is removed. Per-card strict and compiled
pair P2P0 pre/post health, kernel-fault and owned-container gates pass. NEW
post-terminal/post-health full buffered hashes match all four publisher shards
with unchanged stat records; no source repair or runtime change. Evidence:
F06/serial-root-twocard-v1/{parent-qualification.json,child/report.json,
child/root/requests.json,child/root/result.json,post-health.json,
post-model-identity.json}.
VERDICT -> PASS bounded two-card root reuse and cached/fresh observable output
equivalence. Persistent state is not independently dumped/compared; actual
per-stage row counts are still topology-derived in V5. Main checkpoint memory,
real continuation, exact eviction/cancellation/session isolation, concurrency,
independent mathematics and clean cache-hit latency remain unqualified.

## Two-card191-token pinned-prefix suite started

CONFIG -> Same two-card pilot,209-token exact inputs with191-token shared pin;
this crosses the64-token prefill chunk seam and exercises a longer state prefix.
COMMAND -> Parent wrapper --group pin with both finalized basic prerequisites.
RESULT -> Acquired both leases and started pre-health;
F06/serial-pin-twocard-v1.
VERDICT -> RUNNING; no pinned-prefix numerical or prefill-seam result yet.

## Finalized two-card191-token pinned-prefix reuse

CONFIG -> Frozen0018/V5/parent, same32/16 pilot,209-token exact A/B inputs
with191-token shared pin. Parking disabled; fresh references use the same pin
and cut geometry as hits. This exercises native64-token prefill before the pin.
COMMAND -> Parent wrapper --group pin; F06/serial-pin-twocard-v1.
RESULT -> Final parent/child PASS, exit0,439s including lifecycle and fresh
source hashing. Two matched triples repeat in one warmed process. Fresh
requests evaluate209 prompt tokens; both hits actually reuse191 and evaluate18.
Both hit/reference comparisons and the fresh-reference repeat have exact IDs,
LP20, natural EOS, full248320 first logits and all48 first-window residuals
(max_abs0/RMSE0 for144 paired residual rows).

Each fresh request also supplies96 bounded prefill-last residuals (two points
across all48 layers with required32/16 stage coverage); hit requests supply no
prefill residuals because their18-row suffix takes the verifier path. Additional
CPU comparison of reference/reference2 and establish/establish2, matching
(stage,layer,pos,token) rosters, finds192 paired prefill rows bitwise identical.
Receipt serial-pin-prefill-repeat-v1.json records this repeated-fresh comparison.
It does not compare unobserved cached prefill or prove independent equations.

Engine exits0/container removed. Strict per-card and compiled pair P2P0 pre/post
health, kernel/owned-container gates, and NEW post-terminal/post-health full
buffered publisher hashes of all4 shards pass with unchanged source stats.
VERDICT -> PASS bounded191-token pin reuse across the actual prefill seam.
Independent persistent-state/prefill math, production8192 and long contexts,
complete-cache admission/memory/session/concurrency and clean latency remain open.

## Two-card parked snapshot suite started

CONFIG -> Same frozen two-card pilot,512MiB parked budget/one slot, exact B/A/B
roster repeated twice. Requires actual ram_snapshot route.
COMMAND -> Parent wrapper --group parked; F06/serial-parked-twocard-v1.
RESULT -> Numerical/teardown child PASS: both restorations select ram_snapshot
and42 reused tokens. Parent post-health passes; NEW source hashes still running.
VERDICT -> NOT FINALIZED; no complete snapshot qualification until parent gates
finish. Exact victim/admission and total-memory coverage remain V6 work.
