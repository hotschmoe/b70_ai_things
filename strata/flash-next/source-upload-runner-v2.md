# Whole390 source-upload runner v2

CONFIG -> Same original five whole390 HC/PLE source upload/probe cases, runtime
configuration, payload readback, owner lifecycle trace and negative controls as
frozen run_source_upload_oracle_full.py. New separate runner; no old evidence
rewritten. CPU source task only, no actual model payload/page reads or GPU work.

COMMAND ->

```
python3 strata/flash-next/test_run_source_upload_oracle_full_v2.py
python3 -m py_compile strata/flash-next/run_source_upload_oracle_full_v2.py
```

RESULT -> Ten CPU tiny-file/mock controls PASS. AST comparison preserves exact
full_cases, Docker workload args and command construction. Both failed page
views and inspection pre/post stats are preserved from the same failed read,
without re-reading and replacing the failure. Complete four tiny-file hashes
pass; wrong publisher bytes/expected SHA fail even with unchanged stat. Missing
terminal, post-health or fresh complete4 proof cannot finalize PASS.

VERDICT -> Source-ready, actual new combined-engine whole390 qualification
UNOBSERVED. Launch only via the parent's pair lease with genuine oracle/build,
pack and independent current all4 identity prerequisites:

```
python3 strata/flash-next/run_source_upload_oracle_full_v2.py \
  --oracle-receipt <actual-new-combined-oracle-build-receipt> \
  --pack-receipt <actual-native-pack-intake-receipt> \
  --oracle-plan strata/flash-next/native-source-upload-plan-full-v6.json \
  --model-identity <genuine-current-full4-source-identity-receipt> \
  --output <new-whole390-v2-run-dir>
```

The runner pins source_page_watchdog_v3.py. Both shard3 buffered views are checked
at admission, before/after each case, each existing two-second active poll, and
before/after final hashing: offset3857879040 SHA2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e;
offset39437303808 SHAd69f7ffbfaab20277926926a4e542ce906b7fe9f4a4791dc7c46e945f0b1bced.
Each page is4096B. Activity between polls is UNOBSERVED. Polling does not prove
continuous source identity, direct-I/O equivalence or disk health. No automatic
cache invalidation, file writes or repair occurs. Failure preserves both exact
inspection buffers/stats in the new run directory and prevents dependent cases.

The final PASS additionally requires all five cases complete, owned containers
terminal without forced-cleanup error, post-health, and a new independent complete
four-shard buffered publisher scan after both terminal and post-health boundaries.
post-model-identity.json includes each full byte count, hash, pre/post stat and
completion timestamps. The old admission receipt is not reused as final proof.
Final scan/guard failure remains a failed run; current source invalidation prevents
starting a dependent final hash, explicitly UNOBSERVED. Oracle-to-engine receipt
and actual consumed library hashes are verified before work and rechecked after
final hashing. Model math/ordinary payload readback/concurrency remain unqualified.
C1combined_v4 independently enforces this v2 provenance; frozen V3 remains a CPU
prototype and does not provide the strict new final-identity gate.
