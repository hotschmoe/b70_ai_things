# Local Strata PR refreshes on v0.2.16

Base: 58ee3227720629ff2e41d640165ebe89c880dfef. All branches are local in the shared Strata git repository; none has been pushed or posted. They are preparation for review, not claims of upstream qualification. Full backend builds and GPU parity checks for each standalone branch were not run.

| PR | Local branch and HEAD | Status and limits |
| --- | --- | --- |
| #2802 | refresh/pr-2802-0216 ef2230c18 | Upstream #4185 covers much of the decode work; this branch carries remaining IQ2 gate/up and IQ4_NL/Q2 down row reuse. SYCL source-level compiler check passed. Existing parity coverage was adapted but not run. The measured DD binary uses upstream helpers and its EP row paths; it does not include this residual branch. |
| #2804 | refresh/pr-2804-0216 4fd981dc4 | KREG opt-in carry. Its source compiled in the assembled candidate; standalone branch parity and backend builds pending. |
| #2806 | refresh/pr-2806-0216 cefdfa288 | Residual score-transpose and top-k scan carry. Upstream separately implements batched attention rows and active merge; standalone parity pending. |
| #2808 | refresh/pr-2808-0216 fd927eb4d | Prompt attention union carry. Its source compiled and was exercised in the assembled candidate; standalone parity pending. |
| #2810 | refresh/pr-2810-0216 e84757694 | Q8 hyper-connection load/read carry, including namespace fix. Its source compiled and was exercised in the assembled candidate; standalone parity pending. |
| #2812 | refresh/pr-2812-0216 b6fa5ad90 | Fused BF16 HC read, stacked on refresh/pr-2810-0216. SYCL source-level compiler check passed; GPU parity and other backend builds pending. |
| #2526 | Moot on this base | Both engine/src/program/generate.cpp and engine/sycl/src/program/generate.cpp now park decoded slots before overwrite. The candidate also passed a four-slot forced-restore workload. No refreshed branch created. |
| #2511 | refresh/pr-2511-0216 48abb19ba | Draft-count carry adapted to current engine and server. Agent reported 14 targeted Python tests passing; CUDA/HIP/SYCL builds and live accounting checks remain unrun. |
| #3273 | refresh/pr-3273-0216 f5413b6a4 | EP base plus prompt reduction/PUSH stack on current upstream, and server placement guard. EP base and PUSH were validated in the assembled candidate; branch includes the EP base in the same commit and should be split into a reviewable stack before submission. |

The #2802 residual was discovered after the once-only candidate measurement. It is intentionally kept on its own local branch, not silently substituted into the measured binary. The candidate's upstream decode helpers are functionally equivalent, but the old IQ2 row-reuse optimization is not fully carried into that binary.
