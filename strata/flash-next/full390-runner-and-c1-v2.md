# Full390 source gate and segmented C1 controller

CONFIG -> New controller generation preserves earlier C1 source/test/document
snapshots in c1-controller-v1-archive/. Existing narrow source-upload contracts,
actual failures and passing narrow lifecycle receipts remain unchanged. Model
SHA and source-cache sentinel identity gates remain mandatory.

COMMAND -> Prepare separate full390 GPU runner and add immutable --plan selection
to its builder. Upgrade C1 upload gate to the complete original387 HC plus3 PLE
scope, exact per-source SHA/type/shape/offset and native count/account/lifecycle.
Prepare explicit0011/0012 segmented profiles. CPU-only checks; no GPU launches.

RESULT -> New run_source_upload_oracle_full.py reads the frozen oracle snapshot
and requires its SHA match the selected immutable plan. It runs all five plan
cases:fullcard0,fullcard1,same-device24/24,two-device24/24 and actual-model static
bounds24/24. It uses12GiB host container limit and900-second owned deadline per
case. Parent pre/post strict card and compiledP2P0 health, fault audit, normal
exit/removal and empty logical-free trace ledgers remain required.

The runner accepts a completed independent full identity receipt:passedtrue,
matching lock/revision, four rows with actual/expected SHA equality, stable
before/after dev/inode/size/mtime/ctime arrays, and complete start/finish. It
rejects old C1 prepared metadata as a replacement for fresh post-run identity.
It rechecks current stat signatures and the known original4KB page before and
after each case, without performing another103GB scan.

Build and launch are separate parent actions:

```sh
python3 strata/flash-next/build_source_upload_oracle_full.py \
  --engine-receipt NEW_ENGINE/receipt.json \
  --plan strata/flash-next/native-source-upload-plan-full-v6.json
bin/gpu-run python3 strata/flash-next/run_source_upload_oracle_full.py \
  --oracle-receipt NEW_ORACLE/receipt.json --pack-receipt EXACT_PACK_RECEIPT \
  --oracle-plan strata/flash-next/native-source-upload-plan-full-v6.json \
  --model-identity NEW_POST_RUN_FOUR_SHARD_IDENTITY/receipt.json --output NEW_RESULT
```

C1 prepare now requires whole390 evidence across both cards, the real two-device
owners and actual static model-call bounds. Every layer must appear exactly once
per case, no source name may be duplicated, and all expected source images must
match the frozen roster. Ordinary native images are counted, byte-accounted and
retired through owning logical-free traces; their payload GPU readback remains
outside the source oracle scope and is explicitly unqualified.

The new profiles one-card-segmented and two-card-segmented require actual engine
patches0011 and0012. They set STRATA_STAGE_MIRRORS=1 and
STRATA_STAGE_MIRROR_SEGMENT_MIB=1024, --adapt-every0 --adapt-swaps0 and
--no-prefill-borrow, with complete VERIFY_NO_HOST/device-plan guards. The source
rejects unsupported MTP, adaptive async, peer/remote tiers, same-device serving
split and pipeline windows; the controller independently refuses them. It
retains C1, no MTP, FP16KV, prompt-cache0 and the original selected PLE table.
The old profiles remain available as distinct controls with the new full gate.

New research aliases append-seg1024-adapt0-borrow0 to existing2048/8192 IDs;
hotschmoe-dd stays first. Two-card segmented launch requires a completed matched
one-card-segmented qualification, not a prior legacy-profile screen. Source and
runtime/schema/manifest/library identities remain frozen and old prepared runs
cannot silently use the changed controller.

The known cached-view page sentinel is captured during preparation and checked
by the foreground launch loop, before/after every diagnostic request and after
teardown. A mismatch stops further model work and preserves evidence; it never
triggers redownload or cache mutation. This4KB watcher does not replace full
artifact hashing or establish a corruption origin.

VERDICT -> Python syntax/help checks, actual completed four-shard receipt/stat/
sentinel validation and five-case parsing pass without GPU. Synthetic full390
reports built from actual original source metadata pass the case gate, and14
negative controls reject narrow coverage, missing rows/layers, changed SHA/
offsets/accounting/counts/destructors and unqualified/incompatible segmented
profiles. Actual full-source and actual segmented serving remain parent-owned
experiments. A C1 text/token/lifecycle screen still does not establish complete
model logits/state/prefix/concurrency fidelity or a shelf promotion.
