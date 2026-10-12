# Current mode0 fidelity oracle: source52/58/59

CONFIG -> Selected original UD-Q4_K_XL, unchanged original weights and own zero
state. This audit reads source and small saved reports only. Current59 receipt
SHA df52f1b799d013e84438dc50e77c4b81b7d5f655fa04ba14c0af05b6d0ea74d0;
SDK /mnt/vm_8tb/b70/build/strata-native59-incremental58-v1-20261012T215430Z-tmxlm889.
Actual52/58/59 relevant numerical kernels and existing NUM38/P30 headers are
byte-identical in the bounded source audit. This is source identity, not old
runtime proof transfer. Current58 genuine command metadata has materialized0.

COMMAND -> Read the source snapshots and frozen reports; run only four tiny
synthetic arithmetic tests. No model/provider reads, compile or GPU execution.
Source evidence: /tmp/b70-current-mode0-fidelity-source-audit-v1.json.
Recipe construction evidence from genuine58 saved metadata:
/tmp/b70-current-mode0-fidelity-genuine-command-construction-v1.json.
The construction is not runtime authority; its demonstration binding is zeros.

RESULT -> The stale checkpoint statement about an unresolved post-PLE layer2
error needs this qualification. Source35 original prefix1 V4 report
c4cc4ca6caae9e67ad57867adf9f65dc10f8cea03474c984e7fb78d082e2a244
had layer1 postFFN NMSE2.9245740445028e-14 and layer2 postFFN
1.593836659759951e-14 after the actual source35 PLE repair. That specific
layer2 gap was gone. Later FFN growth remained, final head NMSE
8.409171378729055e-5, maxnormalized0.002253121092107111. The later source35
prefix1 report dca1fb225e25422c6cc4b5ca89bc3cffefd42ff0f0b90bf94d20f60971a08301
saved all278 own arrays bitwise equal to that V4 lane. These historical records
do not qualify current52/58/59.

Latest completed independent full48 own-state comparison uses source43 mode1:

- Own report37dba63f1dcbe8d2ffb28280c5ebf184155fa762442bc1b710e5cdf9799367b4,
  F17/current-full48-own-state-bulk-mode1-v1. It used independent original
  weights/states and explicit bulkHC83c289, with fresh integrity closure.
- Actual native NUM10 actor d6713012 (F17/source43-num10-v3-run-v1) four fresh
  literal248045/MAX1 mode1 requests. Direct readonly comparison report
  df8295d8dea16f27ad57d35d78a36d50dc853ec1530007d5841bf035e0281619:
  inputexact, layer0attention3.960988499e-7, layer0FFN1.476465561e-6,
  head0.0001059580329, maxnormalized0.002389562478. Only postFFN all48/head
  and layer0input/attention were targets; other phases were unobserved.
- Later actual44 NUM38 actor23e5c2f9c5d09195ae859e99a103517e6b371b8821028dfff503323a59d75248,
  F17/source44-num38-v2-run-v1, captured five extra core fields. Original
  driver ec203ace16f0e4dc8f7b3601ef90eb4c17afaecb1a21c9914961dad39a03a4f5
  FAILED at serialization; it is not an admitted original-reference baseline.
- Independent firstGDN report77ee20cfda70b55e2aa5c69bda1d8c88fe549aa8850922cfa2a470796abc838e
  versus43 targets has initialconv/state exact, QKV NMSE3.393e-9, gated
  5.258e-9, block4.988e-7, postattention3.961e-7. Existing independent encoder
  applied to actual native gated reproduces native packet bitwise: a conditional
  storage check, not an independent primary operand. Own gated differs before
  packing. Neither this nor matching argmax assigns a tolerance or fullmathPASS.

The primary reference already models Q8_1 activation packing, BF16 alpha/beta
weights and FP16 KV/indexer storage. Do not repeat a missing-packer hypothesis.
Its remaining arithmetic differences are concrete:

- OriginalStorageProjector.project, full48_owned_composition_storage_v1.py44-69,
  encodes/decodes Q81, then decoded original-weight NumPy FP64 dot +F32 store.
  Native Q80 projection (native_mmvq.dp.cpp1894-1904,1927-1947,3628-3632)
  uses exact signed half16 integer dots, product d_weight*d_activation*intdot,
  F32 lane accumulation at kb=sub+16*k, then XOR16/8/4/2/1. Stored Q81 sum is
  unused for Q80; Q4_K/Q5_K/Q5_1 experts have different affine sum contracts.
- OwnedPrefillGdnStorage.row86 already requests BF16 weights for alpha/beta
  and F32 verifier input. Native verify_kernels.dp.cpp128 onward performs
  groups-of8 explicit FMA per lane then XOR32; reference NumPy dots differ.
- Host sqrt/exp/log1p, conv/norm/core reductions, router ties, expert FFN
  integer/affine accumulation, and final projection rounding remain unmodeled.
  The CPU HC candidate uses original rows/FMA/XOR and host rsqrt/exp candidates;
  its transport parity is not device-intrinsic parity.
- CurrentFull48OwnStateBulk explicitly accepts only43/47 and mode1. It refuses
  mode0 rather than pretending the inline half roundtrip is qualified.
  native_qsa_indexer.dp.cpp393 returns raw in mode0; mode1 performs explicit
  half-storage/widen before indexer append. Existing mode1 CPU reference must
  not be relabeled as current mode0. Actual current compiler lowering/storage
  behavior needs its own test at the first QSA layer3, then history/prefix4.

New explicit source-only arithmetic candidate:
original_q80_q81_wide32_reference_v1.py. project_owned uses independently encoded
own F32 input plus exact original packed Q80 rows. OwnFirstGdnArithmeticProjector
changes only blk0 QKV/Z/output and BF16 alpha/beta linear roles in a NEW OwnedGdn
instance; all other roles stay inherited. It consumes no native operands/state.
It is not installed into any executed full48 runner. Q80 source-expression
products/adds are separately rounded; current compiler contraction has not been
validated. BF16 prelinear uses explicit host fmaf/group8/XOR32, while device
FMA/intrinsics and nonlinearities remain unqualified. Four synthetic CPU tests
PASS; no actual original tensor calculation.

Smallest meaningful CURRENT capture is source59 mode0, four fresh literal248045
GEN/MAX1 requests, batch/cache/MTP0, paired24/24 and unchanged current geometry.
Use existing NUM38 +P30 input/attention/FFN48 +SFD49/full248320 head +input33;
QSA3/H36/PTI/HC-reuse/E2 OFF, UR2 for original ownership evidence. No new kernel
or engine build is needed: these observer bodies are identical52/58/59.
current_mode0_fidelity_capture_recipe_v1.recipe builds these exact native argv,
ENV and request descriptors from the current admitted command metadata. It
explicitly disables the concurrent scheduler for this first-row math diagnostic;
this is not a concurrent-serving result. Four fresh copies prove actual zero
before-state and repeatable current targets. PLE bytes must independently match
original hash/IQ4 input, not merely old source35 evidence.

Remaining executable seams are explicit: attach this recipe to the closed
current nativeCLI lifecycle/public reader (fresh currentSDK/all8/config/source,
per-card+compiledpairhealth/kernel/full4/knownpages/ownedEOF and raw hashes),
arm each ordinal via the original ARM protocol, rejoin current artifacts/UR
ownership and NUM38/P30 quotas, and expose actual engine generation separately
from unchanged observer schema. The recipe deliberately has launch_authorized
False until that actual port is complete; run_source44_num38_zero_v2 cannot run
this as-is because it pins44/mode1/intake/old registry gates. Do not waive them
or silently feed a59 binary through a44 alias.

For independent first-step root computation, instantiate existing own zero
OwnedGdn with a NEW OwnFirstGdnArithmeticProjector around the original projector
and independently computed ownHC mixed. Compare QKV/Z, gate/beta, normalized,
core/gated/packets and output against the admitted current capture. Use exact
original rows; one mandatory fresh full4/pages closing bracket after all reads,
including failures. A same-current-library bounded native operator on these OWN
operands can distinguish source-expression/contraction/intrinsics before broad
full48 reruns. No captured operand enters primary math. Then extend current
mode0 indexer/compiler storage and remaining arithmetic at the first divergent
phase, not with a fitted tolerance or blind global flag stack.

VERDICT -> Full-model fidelity remains open. Source/current capture recipe and
independent arithmetic pieces are ready for integration; complete current
capture/public port and mode0 own-state reference support are not implemented
by this audit. Full512 IDs equality is valuable greedy coherence, not a full
logit/intermediate or independent original/native math oracle.
