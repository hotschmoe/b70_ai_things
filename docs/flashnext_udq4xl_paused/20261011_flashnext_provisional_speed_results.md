# Flash-Next provisional speed results

CONFIG -> Pinned Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL, source40 SDK,
C140 one-card-segmented profile on card0: context2048, prefill64,
PLE row-cache65536, prompt-cache0, fresh requests, temperature0/seed1,
heavy diagnostic observers OFF. Permanent API name hotschmoe-dd first.

COMMAND -> Reviewed run_provisional_speed_screen_v2.py using
f17/speed-screen-c140-one-postcache-prepared-v1. Self-leased session39500
completed terminal0/592s including admission, loading, untimed coherence,
four timed requests, normal stop, health and full source hash checks.

RESULT -> Six ordinary identity/coherence requests and both repeated timed
prompt pairs passed. All timed requests ended at the192-token length cap,
with cached_tokens0; these are bounded diagnostics, not natural completions.
Native times are engine-reported host request intervals, not device-only
execution. Client first-text time and usage arrival are independent receipt
clocks; SSE chunk gaps do not establish per-token gaps.

| Request | Prompt tokens | Output tokens | Native prefill tok/s | Native decode tok/s | Client first text s | Client usage arrival s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 54 | 192 | 6.0 | 5.0 | 9.060 | 47.457 |
| 1 | 524 | 192 | 28.4 | 6.9 | 18.541 | 46.333 |
| 2 | 524 | 192 | 33.2 | 8.8 | 15.866 | 37.723 |
| 3 | 54 | 192 | 12.6 | 8.7 | 4.371 | 26.420 |

Owned teardown was normal with API/native exit0 and container removal.
Post strict per-card and compiled pair health passed, kernel scan passed,
and allfour publisher hashes passed again. Parent receipt SHA256
032a928b86b76ad3af1ec2bd57bfda15f92f71cf434ba06b5c2614ae0915983f;
post full4 SHA256927e585a45af470dcfe6ea196519da23e82ce9246bde07d3ab44fa4b887d7216.
Root independently rehashed command logs, client/prepared/config/manifest
bindings and cross-bound parent/client/full4 receipts.

VERDICT -> Actual scoped one-card measurements, not original-model fidelity,
production stability, cache/concurrency qualification or shelf promotion.
The four samples vary substantially as the run warms; no stable-rate or
optimization speedup is assigned. Original fidelity still has the indexer
half-storage discrepancy and earlier gated/full-head uncertainty.

Dual-card speed session3027 is live using the genuine fresh pair preparation.
Its prerequisite CLI uses the actual one-card qualification.json, not
parent-qualification.json: frozen V2 recipe's example names the wrong file.
That documentation mistake was caught before launch and is not a code or
qualification waiver. Dual measurements remain pending. Its existing profile
uses context8192/prefill128/PLE row-cache1048576, so these initial profiles
cannot establish a matched hardware-only speedup. Identical prompts/seed/
cache policy and source identities will be retained and both configurations
reported. The complete original fidelity/cache/streams/placement/profiling/
latency/shelf campaign remains active.

## Completed dual-card diagnostic

CONFIG -> Same pinned model, source40, prompts, temperature0/seed1, fresh
cacheOFF policy and heavy observers OFF. Existing pair profile uses
context8192, prefill128 and PLE row-cache1048576; this differs from the
one-card profile. One stream at a time, not two concurrent clients.

COMMAND -> Session3027 completed terminal0/461s. The actual freshone
qualification.json prerequisite was used. Parent receipt SHA256
9eb8457de28db2902bae51d28ff32cea6b914ce88791f7a78e6a7586d2f4fa56.

RESULT -> Allfour requests generated192 tokens and ended at the length cap.
Both repeated prompt pairs passed exact text/reasoning/finish/usage equality;
zero prefix-cache reuse. Identity/source/normal owned teardown/strict and
compiled pair health/kernel/freshfull4 checks passed. Root rehashed command
logs and parent/client/full4 receipt joins.

| Request | Prompt tokens | Output tokens | Native prefill tok/s | Native decode tok/s | Client first text s | Client usage arrival s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 54 | 192 | 7.3 | 5.4 | 7.447 | 42.840 |
| 1 | 524 | 192 | 56.4 | 7.9 | 9.358 | 33.502 |
| 2 | 524 | 192 | 80.2 | 12.1 | 6.599 | 22.346 |
| 3 | 54 | 192 | 21.1 | 12.2 | 2.609 | 18.249 |

VERDICT -> Scoped measured dual-card diagnostics with substantial warming.
The later samples report12.1-12.2 decode tok/s and80.2 prefill tok/s for
524 prompt tokens. Profile differences and limited repetition preclude a
hardware-only or stable-rate speedup claim. These do not qualify original
fidelity, full state caching, concurrent clients, natural completions,
critical-path attribution, matched latency distributions or a serving shelf.

Cross-profile visible-output equality per corresponding request: [True, False, False, True].
This observation does not establish full-logit/intermediate equivalence.
The prepared deeper pair QSA9 diagnostic has now been launched; the complete
cacheV10 suite and remaining original campaign requirements are still open.
