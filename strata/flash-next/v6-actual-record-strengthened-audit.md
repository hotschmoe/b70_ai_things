# External strengthened audit for frozen V6 records

CONFIG -> Frozen combined20/V6 source, controller and live run remain unchanged.
A new CPU read-only auditor applies stronger acceptance outside the frozen run
directory. Stage0 is[0,48) for one card; pair owner bounds come from actual args.

COMMAND -> python3 strata/flash-next/test_audit_v6_actual_request_records_cpu.py
and:

```
python3 strata/flash-next/audit_v6_actual_request_records.py \
  --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f07-20261009/serial-basic-onecard-v6-v1/child \
  --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f07-20261009/serial-basic-onecard-v6-strengthened-readonly-v2/receipt.json
```

RESULT -> Actual nine completed records pass. Six diagnostic records have exact
body entry/return multiset equality, per-stage prompt coverage without gaps or
duplicates and consistent PREFIX selection/finish/span, PCL and SFD PID/request.
Source-token FNV/SHA/IDs, vector token positions and committed live IDs all match.
Three flag0 records explicitly have no raw/lifecycle scope. Parent final audit
is F07/serial-basic-onecard-v6-strengthened-final-v1/receipt.json and additionally
checks the completed expected nine-record roster with no pending processes.

VERDICT -> PASS for those bounded observed records only. The auditor's passed
boolean does not prove whole-run completeness: missing whole processes or an
unwritten request can be invisible to file enumeration. Require the declared
process/label roster, completed process result reports, numerical equivalence,
parent post-health/full model identity/owned teardown separately before advancing.
For basic that means exactly basic_off/basic_logits/basic_layers, each A/B/C.
Parent pair and cache groups must repeat this external check on actual records.

## Frozen V6 acceptance gaps and future V7

V6 checks returned entries with entered[key]>=returned[key], so an additional
unmatched enter can be ignored when completed prompt rows remain exact. Its PCL
records are self-bound to PCL begin but not cross-bound to PREFIX/SFD identity.
No such violation was found in the actual completed one-card records. This
stronger external audit requires exact entry/return equality for noncancelled
completed requests and binds each system to the same actual input/PID/request.
Cancelled unmatched entries are reported as partial/unobserved work, never zero;
returned-without-enter remains an error. Body completion is not raw state/commit
proof and source fields explicitly retain commit_proven=false.

The initial external v1 receipt reported nine 'pid' failures because the new
auditor incorrectly treated fresh_all_stage_reset as an accounting event with
identity fields. Frozen source does not give it PID/request, like pin_saved.
Preserve serial-basic-onecard-v6-strengthened-readonly-v1/receipt.json as an
AUDITOR FALSE FAILURE. Corrected v2 explicitly lists those auxiliary descriptive
events as unbound/unobserved; it does not supply missing ownership or silently
ignore missing identity on selection/finish/evaluated_span/PCL/SFD. Synthetic
negative controls reject extra/missing enters, coherent wrong PCL PID, wrong SFD
PID, wrong PREFIX ordinal, wrong FNV/SHA, device mismatch and committed suffix.

A later V7 collector should integrate these stronger checks while preserving
frozen V6 outcomes/scope, add exact expected process/request roster acceptance,
and cross-bind process/container identity because container-local PID1 can be
reused across distinct processes. It must retain finite full-vector, natural
finish/IDs/LP20 and lifecycle/identity gates. There is no current source patch or
math/reset change in this auditor.

## Meaningful basic equivalence limits

Flag0/on compares exact generated IDs, natural completion and existing LP20
strings, whose values are printed to six decimal places. True flag0 exports no
full raw vocabulary or residuals; matching LP20 does not prove full-distribution
bit equality. Logits-only versus activation arms compares all248320 raw first
logits bitwise plus outputs. Activation arms require all48 first-window residuals
but there is no activation-off residual vector to compare. These are bounded
instrumentation controls, not independent mathematical reference or full
persistent GDN/PLE/QSA/KV state proof. Diagnostic timing is not clean latency.

## Complete-hash helper provenance and association correction

The original actual combined20 complete hash scan was executed by
hash_combined20_preparation_identity_original_v1.py (preserved byte-for-byte
archive of the then-current helper). Its authentic receipt remains unchanged:
F07/model-full-identity-combined20-preparation-v1/receipt.json. Actual upload/new20
association was independently checked by parent and genuine C1/V6 preparation.
That original helper itself checked actual engine SHA, passed five-case upload
and post-health but did not enforce upload oracle-to-engine association.

The current hash_combined20_preparation_identity.py now requires --oracle-receipt
and checks upload-to-oracle SHA, oracle-to-exact-engine SHA/path and engine source
plan before claiming after-combined20. Tests validate the authentic new20 chain,
reject a matching old18 upload/oracle pair and wrong SHA/path, and use only four
tiny synthetic shards for reader controls. No unnecessary new full scan occurred.
Command: python3 strata/flash-next/test_hash_combined20_preparation_identity_cpu.py.
Receipt: combined20-hash-helper-cpu-receipt.json. Future scan command:

```
python3 strata/flash-next/hash_combined20_preparation_identity.py \
  --upload-lifecycle ACTUAL_NEW20_UPLOAD/receipt.json \
  --oracle-receipt ACTUAL_MATCHED_ORACLE/receipt.json --output NEW_HASH_DIRECTORY
```

The hash helper establishes actual file identity only, not GPU inference or
cache correctness. Source/CPU files and archived provenance are frozen together
in v6-strengthened-auditor-freeze.json; no frozen GPU-run files were altered.
