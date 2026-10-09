# QSA native count thresholds: additive audit v1

CONFIG -> Fresh29 consumed source and frozen qsa_owned_state_storage_v1. No GPU,
model payload or change to frozen full48/reference plans.
COMMAND -> Inspect qsa_step_fill, KV append/read page mapping and native layer call.
RESULT -> Attention KV cells are NOT compressed by indexer pooling or page granules.
VERDICT -> Total consumed context C, including current input token, gives pos=C-1,
n_kv=pos+1=C. Verifier window T=1 is distinct from C. The phrase token threshold
2052 means C2052, not a verifier window of2052 rows.

qsa.dp.cpp:1091-1094 sets step n_kv=pos+1, n_bid=n_kv/4 and width=min(n_kv,2051).
layer.cpp:1203 repeats that same calculation. qsa_decode_attn.dp.cpp:166-169 maps
every selected cell through page_table[cell/4] and cell%4, retaining one K/V row
per token. Indexer keys are pooled four-to-one; attention KV cells are not.
The Python owned model appends one key/value per advance and uses len(keys),
matching this native contract. Physical page count is ceil(C/4), not n_kv.

| Context C | Indexer completed rows floor(C/4) | Incomplete tail cells | KV pages | Selected width |
| --- | --- | --- | --- | --- |
|63|15|3|16|63|
|64|16|0|16|64|
|65|16|1|17|65|
|2047|511|3|512|2047|
|2048|512|0|512|2048|
|2051|512|3|513|2051|
|2052|513|0|513|2051|
|2053|513|1|514|2051|

C2052 is the first nonidentity selection: one KV cell is omitted. C2053 omits
two. The incomplete tail has finite1e9 bias; completed cells receive no tail
bias. Fresh first-generated capture at C2052 uses prefills2051 plus verifier1,
with verifier pos2051. For reused/cached sequences C must come from actual
logical position/step counts, not newly submitted suffix length. A context2048
fixture cannot reach this boundary. Physical resident/streaming tier changes
page residency, not logical n_kv, on this source profile. Other compressed KV
formats/modes require their own source audit and are not inferred here.
