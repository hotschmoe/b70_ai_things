# C1v5 canonical source-stat correction

CONFIG -> Separate C1v5 and parent5 source after the actual C1v4 preparation
failed before payload scans/output. Frozen C1v4/parent4, numericalV5 prototype
and that failed attempt remain unchanged. The genuine uploadV2 and final full4
identity remain valid; their actual producer signature is a five-element list.
No actual model weights, GPU, runtime or SDK action occurs in this source task.

COMMAND ->

```
python3 strata/flash-next/test_c1_serve_controller_combined_v5.py
python3 strata/flash-next/test_qualify_c1_serving_combined_v5.py
python3 -m py_compile strata/flash-next/c1_serve_controller_combined_v5.py \
  strata/flash-next/qualify_c1_serving_combined_v5.py
```

RESULT -> C1v5 31 and parent5 20 CPU controls PASS. The real tiny-file chain runs
uploadV2.full_buffered_identity on four actual8KiB files, producing real SHA,
byte counts and list5 pre/post stat. strict_v2_upload_provenance consumes that
receipt using actual file reads, hashes and os.stat; stat is NOT mocked into
an assumed shared format. Separate parent5 consumption uses the same actual
producer/page/stat metadata. Physical chmod then restored-mtime changes ctime
and is rejected by both consumers. A legacy dictionary without ctime is rejected.
The actual prepare body also hashes the four tiny files and serializes canonical
model_shards/page metadata with explicitly mocked SDK/upload/runtime admission.
Its temporary mock output is removed and is not a genuine preparation.
GPU/SDK/API lifecycle fields and known-page offsets are explicitly synthetic in
these tests; no mock is a genuine prepared/upload/qualification result.

VERDICT -> Source-ready schema correction; actual preparation/serving pending.
The one C1 semantic correction makes stat_signature return exactly:

```
[st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns]
```

C1v4 returned a dictionary containing only device/inode/size/mtime. Its comparisons
against uploadV2/watchdog list5 could never succeed, despite identical real stat
values. C1v5 neither drops ctime nor converts list5 to a weaker dictionary. New
prepared model-shard stat, before/after hash scan checks, upload rows and both
page guards all use canonical list5. Prior prepared generations have different
controller hashes and are not silently admitted. Source stat remains admission
metadata, not independent content proof: full publisher hashes and both sampled
pages are still mandatory, and unchanged stat alone is insufficient.

Parent4 had the same mismatch when its strong source-proof helper compared the
list5 final identity to C1v4 stat_signature. Parent5 imports/pins C1v5, keeps the
same source/terminal/health/hash/proof gates and uses generation5 artifact names:
qualification-controller-v5.json, c1-post-model-identity-v5.json and
c1-source-identity-proof-v5.json. Final qualification.json includes
c1_parent_generation5; parent-qualification.json includes parent_generation5.
Public strong consumer interface remains:

```
from qualify_c1_serving_combined_v5 import validate_final_source_proof
proof = validate_final_source_proof(prepared_directory, prepared_metadata)
```

Default two-argument use requires completed parent+final/proof health/terminal/
source/epoch bindings and current list5 signatures. Internal candidate_final
validation remains solely a parent prepublication operation. NumericalV6 and
new concurrent consumers must pin parent5/helper5, rather than frozen numericalV5.

Audit of other boundaries: watchdog.signature, uploadV2.full_buffered_identity,
qualify_layer0_numerical_v5.stat_signature/full_buffered_identity and the inherited
run_source_upload_oracle_full.verify_model_identity all already use list5.
NumericalV5 candidate source metadata did not independently convert stat; it
called the broken C1v4/parent4 helpers. Its source is preserved, with zero genuine
positive prepares/executions. Old legacy C1 source retains its historical dict4
format in frozen evidence; no old producer/consumer is modified.

The actual reviewed combined51/all8 SDK, new390 uploader, six Python sources,
stable hotschmoe-dd first, exact four admitted method/context/topology aliases,
model math, API profiles, traces, owned cleanup and GPU leases remain the same.
No registry change is needed. Use the exact C1v4 preparation recipe with controller
filename c1_serve_controller_combined_v5.py and new output directories after parent
authorization. Onecard2048/64 and pair8192/128 remain; pair needs actual matched
onecard qualification. Run new parent with:

```
python3 strata/flash-next/qualify_c1_serving_combined_v5.py \
  --prepared <genuine-new-C1v5-directory>
```

Pair adds --one-card-receipt <genuine-onecard/qualification.json>. New source plans
pin both generations and preserve predecessor hashes. Neither this schema fix nor
CPU controls qualify full model math, own-state, concurrency, speed/stability or
shelf promotion. Both sampled page views and full4 hashes still cannot observe
source changes between polls; no cache/file repair is performed.
