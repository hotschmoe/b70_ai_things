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
