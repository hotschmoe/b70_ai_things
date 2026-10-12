# Original paired MTP host timing result

CONFIG -> Exact original sidecar, target24/24 on two B70s, P2048/GEN512,
normal reference-derived prefix cuts. MTP ON, teacher head skip OFF, HC reuse
OFF, expert reuse OFF. Instrumentation adds no GPU waits, queries or copies.

COMMAND -> Actual root66379 completed Exit0. Root replayed full512 raw history
against faithful V6, source timing collector, complete artifact hash roster and
closed-report analysis. Both full5 scans passed; arm health/kernel/normal owned
retirement passed. Analysis requires the closed actual report digest.

RESULT -> Producer65af1c952e546054b57bb8b0890bb70219f39ae2c35839f6f222df21d8d95f21.
Results under f17-source37-20261010/original-mtp53-host-timing-v1; analyzed
artifact original-mtp53-host-timing-analysis-v1.json alongside that directory.
All512 tokens match reference,229 accepted/53 rejected windows,10227 spans.
Prompt40183.919488ms; decode30181.647776ms.

| Decode host phase | Calls | Total elapsed ms |
| --- | ---: | ---: |
| Target run (includes nested stage1) | 282 | 21391.833 |
| Draft record/wait | 511 | 8084.702 |
| Own-head read and argmax | 282 | 348.007 |
| Target commit | 282 | 239.735 |
| Actual stage submissions, both stages | 564 | 31.283 |

Stage0 inclusive host P50/P95=73.216/92.244ms, total21391.716ms.
Nested stage1 P50/P95=35.801/43.964ms, total10254.253ms.
Stage0 exclusive host elapsed (inclusive minus contained same-thread stage1)
totals11137.463ms. None of these durations establish physical device idle,
transfer bandwidth or kernel execution time. Target_run and stage0 are almost
the same enclosing interval; never add them or add stage1 again.

Teacher-prompt target-prefill part calls total38689.305ms and include draft
record waits totaling15218.771ms. These are also nonadditive. Prompt head
skipping is being repeated using the prepared four-run same-ELF ABBA; no clean
benefit follows from this instrumented single arm.

VERDICT -> Current faithful model diagnostic locates large host elapsed in
target calls and original-draft waits, with small measured submit-call spans.
Next optimization decisions should examine their device work and dependencies;
submit overhead alone does not explain the measured decode interval. Full
calibrated host/device/transfer/stream profile, matched clean repetitions,
cache/API concurrency and requested performance targets remain open.
