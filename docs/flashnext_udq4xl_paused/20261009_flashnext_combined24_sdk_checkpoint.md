# Combined source24v2 SDK checkpoint

CONFIG -> Fresh pristine Strata fb58e0d plus reviewed24-patch chain
20/21/23/22v2/24v2, pinned39992 image and GGML3cf03257. All modified ABI callers
rebuild in a fresh private tree. No GPU devices or model weights exposed.

COMMAND -> build_native_hc_engine.py under pair GPU lease, immutable combined
plan8d708bff..., clean exact dependency copy, jobs4. Then rebuild/link the full
source-upload oracle with the actual new static libraries under a new lease.
Independently compare all51 final source hashes and all8 target binary hashes.

RESULT -> SDK PASS305s; oracle compile/link PASS44s. All51 actual final hashes
match the immutable plan, all8 target binaries exist and hash correctly. Source
and plan remain unchanged. Matching oracle binds new engine receipt and unchanged
new library hashes; no old ABI binary is reused. Root source identity check:
F09/combined-sdk-root-identity-check-v1.json.

Engine: /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T144515Z-k4gb2xtx
Engine receipt SHA2560c58553d71758bdf6563eb7dab02747808a2a34d8981fa782e9b5a0ccb0805f5.
Oracle: /mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-njugb9sn
Oracle receipt SHA256f55f12e95651f5e57326dc28d3b55e36c793642672fab630a5cfb1cf8e84d4f6.

VERDICT -> Compilation and linkage qualified only. Actual new source390 upload,
producer coverage, original reference math, owned lifetimes, source identity,
API1/2/4/6 coherence, full-state cache and matched clean latency remain required.
The prior pairV2 source failure remains failed even though source recovery passed.

CPU source prototypes C1combinedV3 and numericalV3 pass14/2 test commands,
respectively, but no genuine preparation or runtime follows. C1V3 accepts additive
new upload fields without requiring the new post-run full4 hash. Strict C1V4 and
numericalV4 will reject that legacy acceptance before any model experiment.
The frozen V3 prototypes stay preserved; they are not a production launch path.
