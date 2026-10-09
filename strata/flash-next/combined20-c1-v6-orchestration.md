# Genuine combined20 C1/V6 CPU orchestration

CONFIG -> Actual combined20 engine
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T112318Z-4uag6j5x,
receipt SHA1647fdb0941d6e960bdbded793849b998de9de963eb5a6e6c687f18ea5547b00,
source plan05041bd1. Root's new F07/source-upload-combined20-v1/receipt.json
passed all five whole390 cases and pre/post health. Actual corresponding oracle
is /mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-tkbp0s5d/receipt.json,
SHA1287b2f76a03cea857c91a25e083ad47b77074be970df020e614ac44fcfe3bf6.
No old-engine source upload is transferred to this generation.

COMMAND -> Actual CPU-only preparation commands below, independent complete
buffered four-shard hashing, V6 prepare and frozen parent prepared_chain_binding.
No GPU/model launch, compile, source/cache invalidation or repair by this agent.

```
combined20_engine=/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T112318Z-4uag6j5x
combined20_f07=/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f07-20261009
combined20_runtime=/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/c1-python-runtime-v1/receipt.json
combined20_oracle=/mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-tkbp0s5d/receipt.json

python3 strata/flash-next/c1_serve_controller.py prepare \
  --profile one-card-segmented --engine-root "$combined20_engine" \
  --runtime-receipt "$combined20_runtime" \
  --upload-lifecycle "$combined20_f07/source-upload-combined20-v1/receipt.json" \
  --oracle-receipt "$combined20_oracle" --verify-model-shards --port 18084 \
  --output "$combined20_f07/c1-onecard-combined20-prepared-v1"
python3 strata/flash-next/c1_serve_controller.py prepare \
  --profile two-card-segmented --engine-root "$combined20_engine" \
  --runtime-receipt "$combined20_runtime" \
  --upload-lifecycle "$combined20_f07/source-upload-combined20-v1/receipt.json" \
  --oracle-receipt "$combined20_oracle" --verify-model-shards --port 18085 \
  --output "$combined20_f07/c1-twocard-combined20-prepared-v1"
python3 strata/flash-next/hash_combined20_preparation_identity.py \
  --upload-lifecycle "$combined20_f07/source-upload-combined20-v1/receipt.json" \
  --output "$combined20_f07/model-full-identity-combined20-preparation-v1"
python3 strata/flash-next/serial_prefix_qualification_v6.py prepare \
  --prepared "$combined20_f07/c1-onecard-combined20-prepared-v1" \
  --engine-root "$combined20_engine" \
  --model-identity "$combined20_f07/model-full-identity-combined20-preparation-v1/receipt.json" \
  --output "$combined20_f07/serial-prefix-onecard-v6-prepared-v1"
python3 strata/flash-next/serial_prefix_qualification_v6.py prepare \
  --prepared "$combined20_f07/c1-twocard-combined20-prepared-v1" \
  --engine-root "$combined20_engine" \
  --model-identity "$combined20_f07/model-full-identity-combined20-preparation-v1/receipt.json" \
  --output "$combined20_f07/serial-prefix-twocard-v6-prepared-v1"
```

Those exact output directories now exist; reruns require new names, never
replacement of records. Every C1 prepare actually used --verify-model-shards.
The independent scan separately records start/finish and before/after stat
signatures after the completed upload qualification. It was not reconstructed
from C1 metadata or an old expected digest. A hash/sentinel failure must stop
dependent preparation and preserve evidence, without automatic cache changes.

RESULT -> Both genuine C1 preparations report launch_allowed=true. Independent
current four-shard hash/stat receipt passed. Actual V6 prepare and frozen
qualify_serial_prefix_v6.prepared_chain_binding passed on both new plans:

- one-card plan SHA
  d9c078895a8df6f575d19e22dd55c9146fa843e466b52f409f0fdfc4f2b85195.
- pair plan SHA
  2eadea269c987784e890a8719fe37f700e6d6e46759f0319c3a17ed00db0e9f7.

The local combined20-c1-v6-positive-prepare-receipt.json binds actual paths,
plans, engine/full390/prepared/binary chain and geometry. Frozen V6/controller,
0019/0020 headers, source and plans remain unchanged.

## Identity and geometry are separate scopes

C1's existing one-card-segmented API prerequisite is2048 context/64 prefill/
65536 PLE rows. Its existing pair API prerequisite is8192/128/1048576. Existing
C1 detailed method/quant aliases remain registered and unchanged, and generated
server-config.json keeps model_name=hotschmoe-dd first plus the secondary alias.
No API query or model launch took place. Preparation eligibility is not a screen.

V6 native GEN deliberately normalizes both to2048/64/65536 and emits distinct
new one/pair topology/lifecycle research aliases. Those exact aliases are in
combined20-v6-registry-proposal.json as PROPOSED native metadata only; no eval
registry was mutated and no fake HTTP served-ID claim is made. Do not alter C1
JSON/controller fingerprints to manufacture alias/geometry agreement. Current
V6 pilots do not qualify pair8192 API capacity or clean latency.

## Consumed combined20 warm/first-position audit

Actual generate.cpp invokes fidelity begin at9592 and PrefixDiagnostic/PCL begin
at9610-9611. Each requires the ARM marker before its ordinal advances; the three
fixed fresh max_new1 warmups precede creation of ARM. Thus first armed request
is ordinal1 and warmups consume no six-request raw capture quota. New diagnostic
buffers can exist when the diagnostic flag is on, but active first-window graph
selection remains false during unarmed warmups. No diagnostic copy nodes are
added to the normal warm graph.

Ordinary first generated prediction consumes the final prompt token at n-1 in
T1. V6 prompt-work counting clips completed real stage body spans to
[actual_reused,prompt_tokens), including that final verifier row exactly once;
body spans beyond the prompt are decode, not prefill. Actual prefill source at
2095-2096 counts completed chunk rows at4309 and emits final complete only when
count equals n at4562. Actual verifier body emits its own entry/return, not an
inferred topology multiplier. These are existing body-return observations,
not independent commit/partial-cancellation byte proofs.

real_live uses the actual seed reply and canonical committed_live IDs emitted
after existing commit/publication handling at11409-11413. The terminal predicted
EOS is emitted but not consumed as model input in that generation loop. The
continuation fixture validates exact seed input, output-ID/decoded-text equality,
natural finish and exact template render; committed IDs must be a strict prefix
of the new conversation before a live route can qualify. Thinking-tag removal,
BPE retokenization or another render mismatch fails explicitly rather than
rewriting tokens. This constraint is a real acceptance gate, not guaranteed by
CPU synthetic continuation or by readable replies. No new source bug was found
in this static audit; actual V6 execution still supplies the decisive evidence.

VERDICT -> Genuine new20 CPU/source prerequisite preparation passed. Parent
must qualify owner runtime, same-version V6 basic off/on, one/pair numerical
parity, new lifecycle/cache groups and API screens under leases with health/
full identity/teardown. No agent GPU inference, concurrency, shelf or latency
qualification is claimed.
