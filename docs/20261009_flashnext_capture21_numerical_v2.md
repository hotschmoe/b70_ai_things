# Capture21 numerical v2 one-card qualification

CONFIG -> Frozen20 reference and frozen21 candidate; same original UD-Q4_K_XL
artifact and matched geometry. Prefixes1/2/4/8, max_new1, native serial GEN,
one workload card with pair-leased pre/post health. QFUSE0, existing selfcommit1.
Collector v2 merges producer FD2 onto FD1 before engine startup. Failed v1
lifecycle evidence stays preserved and failed.

COMMAND -> python3 strata/flash-next/qualify_layer0_numerical_v2.py
with genuine layer0-numerical-onecard-prepared-v2 plan and NEW output at
f08-20261009/layer0-numerical-onecard-v2. After terminal exit0, run the tracked
verify_layer0_gdn_original_v1.py against candidate21_on/requests.json and the
new post-model-identity.json; output layer0-conditional-gdn-onecard-v2/receipt.json.

RESULT -> Parent terminal PASS (761s, diagnostic wall time only). All three
processes exit0 and are removed. Frozen20 versus21-off and same21 off/on each
have four bitwise full248320-F32-head comparisons and four complete48-layer
residual comparisons: eight heads and384 residual rows total. Bounded token IDs,
LP20 and finish agree. Four current26-field frames and12 native Q8_1 packets
pass. Logical ownership audit binds the entire7023104-byte target allocation,
owning context, owner marker and returned free in one canonical ordered log.
Strict per-card and compiled pair P2P0 pre/post health pass, kernel fault gate
passes and NEW complete four-shard buffered hashes match the publisher.

Independent conditional original-weight checks PASS28 (seven per prefix).
Original Q8_0 projections use verified actual Q8_1 packets; convolution uses
original F32 taps, supplied history/qkv, shift/SiLU and q/k sum normalization.
Recurrent update and gated output use supplied incoming state/normalized qkv/
decay-beta/z and original F32 gamma. Worst NMSE 2.1262469670281249e-14; worst normalized maximum 2.9579823077046734e-07.
Existing limits NMSE1e-6/max-normalized1e-4 are unchanged.

VERDICT -> The ordered collector fixes the observed v1 collection failure in a
NEW actual run. This qualifies the bounded one-card capture diagnostic, not
independent own-state/full-model math, graph retirement/physical memory release,
natural completion, API concurrency, production latency or shelf promotion.
Seven shared/expert fields remain unobserved in source21. Two-card numerical
qualification, complete first-layer/full-model reference and actual2/4/6 serving
remain required.

## Immutable external receipts

- `layer0-numerical-onecard-v2/parent-qualification.json` SHA256 `ffb3b00f91f7de069aac007da8edfbbacdaa3dcd04ca5e9ee341d31f785a7984`.
- `layer0-numerical-onecard-v2/child/report.json` SHA256 `c2718589f63f221164dca96cc0bf990d92591a00a98e2b870d69aecc20c13f39`.
- `layer0-numerical-onecard-v2/post-health.json` SHA256 `8edaca31aabe02cf94f2ba94c3f682d861e9fd204e89f8a898b457bb59504e88`.
- `layer0-numerical-onecard-v2/post-model-identity.json` SHA256 `4f21316b993994c8b09d2ba6fb1e495800a141a9e7f843f269be72ca317ce042`.
- `layer0-numerical-onecard-v2/layer0-logical-lifecycle.json` SHA256 `d518ad757ec0c6afb6e80e170b5661d80ea02b480702da30cba6c2595bc30e16`.
- `layer0-numerical-onecard-v2/packet-contract.json` SHA256 `fd85d69533b1ab78fb3be9edfe7e94c68172ff6f351e320d418bf79525d0a4c6`.
- `layer0-conditional-gdn-onecard-v2/receipt.json` SHA256 `a42b8b22fe8bc4e5b6f9014864c6b741bfcbd9b6e069de9560c64d42dcc600f2`.
