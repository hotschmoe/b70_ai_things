# S129 actual one-build timing attribution

CONFIG -> One completed required main build; five compiles, four archives,
one final SYCL AOT/main link. Current Ninja log snapshotted once; only the
last completed ten targets matched to the root-validated exact plan.

COMMAND -> Read retained ninja-log.snapshot.txt and exact ten-target plan;
start/end fields are Ninja elapsed milliseconds, not per-kernel GPU timings.

RESULT -> Wrapper217.870344533s; timed ninja205.843s. Logged ten-action
envelope0.019..205.836s =205.817s; active-interval union205.807s;
peak4 overlapping actions. Wrapper minus ninja12.027344533s includes container/
checks/backups/receipts, not isolated compiler time.

| Target | Start s | End s | Duration s |
| --- | ---: | ---: | ---: |
| CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o | 0.019 | 13.764 | 13.745 |
| ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o | 13.765 | 15.779 | 2.014 |
| ggml/src/libggml-base.a | 15.779 | 15.848 | 0.069 |
| CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o | 0.020 | 17.833 | 17.813 |
| CMakeFiles/strata.dir/src/program/generate.cpp.o | 0.020 | 42.855 | 42.835 |
| CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o | 0.019 | 58.631 | 58.612 |
| libstrata_kernels.a | 58.640 | 58.981 | 0.341 |
| libstrata_engine.a | 58.982 | 59.070 | 0.088 |
| libstrata_prefill.a | 59.070 | 59.126 | 0.056 |
| strata | 59.126 | 205.836 | 146.710 |

Native MMVQ TU compile58.612s (0.019..58.631); final whole-main link
146.710s (59.126..205.836). The latter includes driver/SYCL AOT/backend
and host link work; this log cannot isolate those subphases. Native TU was
the latest finishing compile; all remaining archives ended by59.126s.
Four initial compiles overlap; do not sum their durations as elapsed wall.

Cache was fresh/unknown before; actual after counters hits0/misses1608.
This one cold build provides no warm-main prediction or cache-hit validation.
No new configure/compiler/cache warm/decoder/GPU run was performed here.

VERDICT -> Final AOT/main link is the largest measured build phase; native
TU compilation is the largest individual compile. These are build-loop costs,
not serving latency or runtime model performance.
