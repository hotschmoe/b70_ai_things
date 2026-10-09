# Combined C1 preparation generation v3

CONFIG -> Separate source generation for combined20+21+23+22v2+24v2 engine plan
8d708bff8f59b1133f5156969e6a59cf8dee4a949637d75e850f58ede63d9d6a.
Existing C1 controllers, prepared results and numerical v1/v2 evidence remain
unchanged. No actual model payload reads, SDK build, GPU launch or positive
preparation were executed by this source task.

COMMAND ->

```
python3 strata/flash-next/test_c1_serve_controller_combined_v3.py
python3 -m py_compile strata/flash-next/c1_serve_controller_combined_v3.py
```

RESULT -> Fourteen CPU mock/source controls PASS. AST checks preserve the prior
upload, owned cleanup, lease, screen and lifecycle helper behavior. Mock fixtures
are not actual SDK/upload/identity evidence and are never published as real
prepared.json or qualification.json results.

VERDICT -> Source-ready preparation; actual qualification is pending.
The new controller rejects incomplete prerequisites before scanning any model
payload. It requires the exact reviewed SDK plan, all51 expected source hashes,
all8 compiled target hashes and six runtime Python source hashes. The old
five-source manifest and earlier engine plans are rejected. The new whole390
upload gate preserves oracle-to-engine receipt binding; an old21/20 upload cannot
admit the new combined engine. Genuine prepare always requires runtime receipt,
matching new upload/oracle receipts and --verify-model-shards. Every shard scan
checks its publisher hash and unchanged pre/post stat; unchanged stat alone is
not identity proof.

Six files bound by both artifact manifest and combined_generation gate:

- serve/server.py
- serve/frontend.py
- serve/artifact_identity.py
- serve/batch_request_identity.py
- tools/strata_tokenizer.py
- tools/gguf_reader.py

The actual API22 artifact identity checks these same six consumed sources.
Frontend and GGUF reader are unchanged pristine sources and are additionally
pinned; they are not falsely counted among the51 changed overlay files.
All existing stable-primary and exact tokenizer/template/API identity checks
remain. hotschmoe-dd is first in configuration and observed /v1/models order.
New detailed aliases retain Qwen3.8/Unsloth/UD-Q4_K_XL/native-HC-PLE/no-MTP/FP16KV,
context, mirror/adaptation method and explicit card/split topology. The separate
c1-combined-v3-model-registry-proposal.yaml is a proposal, not a served response.
The parent must admit its research aliases to evals/configs/models.yaml before
actual prepare; absent registry aliases fail closed.

The buffered known-page guard reads both shard3 pages before evaluating either:

- offset3857879040,4096B, SHA2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e
- offset39437303808,4096B, SHAd69f7ffbfaab20277926926a4e542ce906b7fe9f4a4791dc7c46e945f0b1bced

It checks pre/post stat and returns deterministic schema3 pages for repeated
prepared validation. It performs no cache invalidation, writes or repair.
It cannot diagnose filesystem/device health or replace current full4 hashing.
Any mismatch stops dependent work; the parent watchdog owns preservation of
both current views and independent cached/direct diagnostics.

After genuine prerequisites exist and the parent authorizes payload scans,
use new output directories; no missing path may be replaced with an older gate:

```
python3 strata/flash-next/c1_serve_controller_combined_v3.py prepare \
  --engine-root /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T144515Z-k4gb2xtx \
  --runtime-receipt <genuine-c1-python-runtime-receipt> \
  --upload-lifecycle <actual-new-combined-whole390-receipt> \
  --oracle-receipt <actual-new-combined-whole390-oracle-build-receipt> \
  --verify-model-shards --profile one-card-segmented --port 18082 \
  --output <new-onecard-combined-v3-prepared-dir>
python3 strata/flash-next/c1_serve_controller_combined_v3.py prepare \
  --engine-root /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T144515Z-k4gb2xtx \
  --runtime-receipt <genuine-c1-python-runtime-receipt> \
  --upload-lifecycle <actual-new-combined-whole390-receipt> \
  --oracle-receipt <actual-new-combined-whole390-oracle-build-receipt> \
  --verify-model-shards --profile two-card-segmented --port 18082 \
  --output <new-twocard-combined-v3-prepared-dir>
```

C1 one uses context2048/prefill64/card0; C1 pair uses context8192/prefill128,
split32/16/card0+1, as in the preserved API method. A numerical v3 driver may
separately normalize native context2048/prefill64 but cannot transfer that to
API latency/context qualification. Pair API launch requires actual matched
one-card API qualification first. Controller lifecycle operations require the
parent GPU leases and health gates; this source task invokes none of them.

The engine permits additional batch source routes, but this C1 profile remains
batch0/parallel1. No concurrency, full-model math, natural-language stability,
latency or shelf claim is made by source preparation.
