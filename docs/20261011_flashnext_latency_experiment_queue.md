# FlashNext latency experiment queue

CONFIG -> User steering: prioritize token latency, bandwidth saturation and
gaps between operations. Synthetic substitutions and spoofed inputs are
authorized diagnostics. Native43 SDK:
`/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T120045Z-kfiohqmn`.
Source paths below are relative to its `source/sycl/src` directory.

COMMAND -> Finish the matched LPdisabled PTI OFF/ON capture first. Rank the
following controls by observed removable milliseconds per token. Keep GPU
timing free of competing model scans, compilation and CPU admission work.

| Order | Control | Source basis and measurement |
| --- | --- | --- |
| 1 | Real replay timeline and collector overhead | `core/verify.cpp:2174,2321`: submission versus blocking completion. Separate host submission/wait from observed per-device kernel/copy unions and uncovered intervals. Establish replay ownership and clock uncertainty before aligning clocks. |
| 2 | Shared SwiGLU/Q81 fused versus separate | `kernels/cuda/shared_expert.dp.cpp:305-354`: `STRATA_FUSED_SWIGLU_Q81` defaults ON. The multi-row fused branch removes one launch and nominal F32 intermediate write/read of `8*n_ff*n_tok` bytes. Confirm actual branch and operands; this byte count is not measured DRAM traffic. |
| 3 | Paired versus separate gate/up | `core/verify.cpp:77-79,843-845,1490` and `shared_expert.dp.cpp:325-330`: requires `STRATA_LFUSE`, enabled prerequisites and 2..8 rows; `STRATA_LFUSE_PAIR` alone is insufficient. Same-format supported pairing replaces two launches with one; retain fallback. |
| 4 | GEMV and multi-column shape sweep | `kernels/cuda/native_mmvq.dp.cpp:3685` and its `STRATA_MMVQ_*` dispatch controls. Capture actual dimensions, weight format, Q81 operands and outputs. Sweep subgroup, rows and supported wide paths separately for decode and prefill. |
| 5 | Resident versus host-mirror expert access | `core/verify.cpp:1520` onward. Preserve routing, operands, allocations and dependencies. Record actual transfer bytes and direction; allocation size does not establish transferred bytes or achieved bandwidth. |
| 6 | Shared-expert queue overlap | `core/verify.cpp:1372-1380`: `STRATA_SH_STREAM` and `STRATA_SH_FORK_LATE`. Require observed queue/dependency overlap and reduced token completion time. |

For isolated replay, captured real operands establish the workload; they are
not an independent mathematical reference. A dependency-preserving no-op can
bound scheduling cost. Resident-only substitution can isolate placement cost.
Label both synthetic and retain completion/dependency checks. Random operands
can explore kernel shape sensitivity but cannot reproduce routing or cache
behavior without evidence.

RESULT -> Source review identifies executable controls, not measured gains.
`STRATA_VERIFY_EAGER` explicitly adds stage waits (`verify.cpp:832-839`) and
must not qualify clean latency. Fixture replay coverage exists; normal-model
coverage, physical idle, bandwidth saturation and collector overhead remain
pending. The active PTI3 capture is still completing pre-run qualification.

VERDICT -> Report per-request TTFT and token-gap P50/P95, per-card observed
operation unions/gaps, launch counts and explicit byte models. Do not add
nested intervals or infer physical idle from host waits. Test promising
controls in repeated matched real-model runs with original-reference and
quantized-packet checks, normal completion, health and teardown. Use removable
critical-path time to decide between fusion, kernel tuning and placement.
