# Optimized HC kernels with Direct PLE control

CONFIG -> Exact Unsloth UD-Q4_K_XL, actual source56 SDK58aa4da5, pair24/24,
prefill512/context4096, mirror1/copy0, HC token tile2/coalescing1/prefill tile4,
PLEdirect/row cache65536/inflight256. Three P2048 useful roles plus fresh
repeats, GEN512, diagnostic tracing/cooperation/cap OFF. All outputs reach
length512; this does not demonstrate natural task completion.

COMMAND -> Producer92681 PASS605s; independent reader94090 PASS.
Producer fb0395163f26919c9288688f3fe64191eff09ad7075da17ef02c88934c451de7.
Public 95bfbffc669477ca16cf1a49408c9b2c78c05cbe2e352fcd7f01f02fce4487e9.
External root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source56-best-HC-PLEdirect-v1-run-v1.

RESULT -> All six complete histories and finishes match both closed source61
Direct arms; repeats, source/ABI/model identity, health and normal owned
teardown pass. Source61 lacks the compiled HC optimizations present in56.
The source/SDK difference is explicit: this is not a same-source flag-only
comparison. No compiler or model-file scan overlapped timed requests. The
reader does not qualify all possible host activity or cold cache state.

| Request | PP tok/s | Emitted/decode interval tok/s | Client TTFT ms | Token gap P50 ms | P95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose_operations_review 1 | 112.18 | 19.41 | 20696.8 | 46.29 | 53.75 | 60.81 |
| prose_operations_review 2 | 259.53 | 21.65 | 7937.3 | 45.51 | 53.06 | 60.15 |
| code_event_pipeline 3 | 256.06 | 21.41 | 8045.5 | 43.25 | 59.47 | 138.60 |
| code_event_pipeline 4 | 255.29 | 22.37 | 8069.4 | 42.27 | 57.25 | 63.96 |
| document_decision_memo 5 | 254.29 | 19.77 | 8096.5 | 48.88 | 60.82 | 171.37 |
| document_decision_memo 6 | 254.47 | 20.54 | 8094.8 | 47.93 | 58.47 | 71.47 |

VERDICT -> Warm repeats recover254-260tok/s prefill, with20.5-22.4tok/s
decode. This restores the known optimized configuration; it does not prove
a new Direct steady-state gain. Native PP timing and emitted/decode interval
are separately labeled; client intervals are stdout receive timings, not
HTTP service timings or calibrated device execution. Percentiles use linear
interpolation within each request. No2000/50, fullmath, concurrent service,
API or shelf promotion follows. Next profiling build must deliberately
retain these HC kernels and current source61 observation support.
