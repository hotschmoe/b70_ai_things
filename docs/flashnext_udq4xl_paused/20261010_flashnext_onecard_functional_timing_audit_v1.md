# Source35 one-card functional timing audit V1

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Read-only audit of
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/functional-screen-v1-onecard-run.
One persistent source35/C113 engine, card0, ctx2048, prefill64/short-read64,
FP16 KV, PLE65536/mmap, device-plan/nohost, segmented host mirrors, no MTP,
adapt-every0/adapt-swaps0/no-prefill-borrow, fresh=1 for each GEN. Native HC F32,
input33/prefix30 diagnostics OFF, STRATA_VERIFY_EAGER absent,
SYCL_CACHE_PERSISTENT=0. Request order: function0, function1, arithmetic0,
arithmetic1; this is not four newly initialized processes.
Parent terminal qualification passed; raw child collection/teardown and all
four functional checks passed, while its broad passed/full-model/latency
qualification flags stay false. ROOT's current read-only recollection reports
all four outputs/IDs equal CPU V3; that is a tiny functional comparison, not
full48 arithmetic or performance qualification. This audit changes no model,
backend, registry, active controller, kernel or frozen reference.

COMMAND: Read actual current SDK source at
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source;
read saved case JSON/stdout/stderr and pinned latency methodology. CPU-only
recollection zips EVERY stdout line with its corresponding stdout_seconds
BEFORE filtering indices by line.startswith('T '). The actual key is stdout,
not stdout_lines; this protocol emits T, not a literal EMIT record. Assert
41/41 lines/times for each function and11/11 for each arithmetic, monotone
stamps, emitted IDs equal output_ids, and emitted count equals DONE count.
No GPU, Docker, profiler, build, model payload read or active-file edit.

RESULT: Host reader observations, milliseconds from its request start:

| Request | DONE prompt/decode ms | First T | Last T | DONE receipt | Consecutive T gaps min/median/max ms |
| --- | --- | --- | --- | --- | --- |
| function0,30 input/18 output | 17547.7 /4343.8 | 17789.784 | 21890.257 | 21892.471 | 123.753 /257.756 /299.458 |
| function1,30 input/18 output | 2346.5 /2153.5 | 2423.469 | 4498.318 | 4500.618 | 85.122 /124.173 /143.551 |
| arithmetic0,39 input/3 output | 6737.9 /562.1 | 6826.724 | 7298.444 | 7300.636 | 215.766 /235.860 /255.954 |
| arithmetic1,39 input/3 output | 3311.1 /299.3 | 3399.892 | 3608.502 | 3610.889 | 101.546 /104.305 /107.063 |

Function T line indices are4,6,...,38; arithmetic indices4,6,8, zero-based.
RESUME, PP, REUSED, LP and DONE retain their original timestamp positions.
Complete consecutive T gaps in emission order (ms):

- function0:218.662,258.492,239.756,257.756,250.391,198.581,123.753,
  139.277,266.388,270.713,299.458,249.252,292.783,277.057,261.144,
  271.498,225.512.
- function1:85.122,107.109,93.660,126.092,120.530,130.955,123.970,
  139.547,124.173,123.559,139.251,140.139,143.551,130.525,129.458,
  118.006,99.203.
- arithmetic0:215.766,255.954; arithmetic1:101.546,107.063.

All outputs include EOT248046. Arithmetic visible text12 consists of two
ordinary IDs plus EOT; do not treat all three as visible text tokens. The
collector records time.monotonic when its stdout-reader thread receives each
line, relative to a start BEFORE command construction/write/flush
(serial_prefix_qualification_v8.py:103-132). These are native-pipe receipt
latencies/gaps, not device timestamps, HTTP TTFT or frontend text delivery.
stdout is explicitly unbuffered (generate.cpp:1622). The T printf precedes LP
work, so first-T receipt is not delayed by an intentional T+LP flush. Reader
scheduling/pipe overhead remains. LP follows each T with observed median
receipt gap2.262/2.339/2.269/2.375ms respectively; this is not an isolated
measurement of LP CPU or device work.

Native counter boundaries (G=generate.cpp, V=core/verify.cpp):

- G:303 uses steady_clock. r0 at9771 follows request parsing/current-ID setup
  and starts before prefix selection, previous-commit wait and fresh state
  reset. Short prompt pieces run through verifier T2/T1 windows (10111-10117,
  10597-10610). prompt_ms ends at10661 after preceding n-1 prompt rows,
  checkpoint/refill work, before REUSED flush and final prompt-row window.
  DONE labels n=30/39, but prompt_ms is not a complete30/39-row prefill timer.
- d0 at10694 follows REUSED and small vector/history setup. Decode includes
  the final prompt-row T1 producing the first generated token, subsequent
  verify/commit/sample rounds, emitted EOT and LP diagnostics. It ends11590
  BEFORE final wait_commit at11592-11595 and later checkpoint/finish work;
  DONE is emitted11775-11784. prompt_ms+decode_ms is not complete request time.
- G:11524-11551 prints T, copies the complete vocab head to host, computes
  double-precision exp/log normalization and top20 partial_sort, then emits
  LP. Every T-to-next-T gap includes the preceding diagnostic work plus next
  forward/commit and host receipt overhead. These LP20 runs are not a clean
  streaming baseline; dividing output count by decode_ms is not qualified
  serving throughput.

Observed first-request work is concrete, its duration is not. stderr:468-482
places first T2 capture between request1 reset and span[0,2), and first T1
capture immediately before span[22,23). Both occur before first output and
neither repeats in the log. V:1791-1796 caches each executable;
V:2072-2074 invokes capture/capture_commit on run. Finalization and an existing queue wait are inside capture (V:1858-1887).
The printed upload-no-error field uses ue=0: the migrated CUDA upload is a
stub, not an observed SYCL upload. No elapsed capture timestamps exist.
READY precedes these lazy captures. The separate warm() helper V:2719-2725
and warm-on block G:12619-12632 belong to the later non-serve path; their
comments do not prove the persistent GEN path warmed before READY.

Startup stderr:401-464 records8092 resident expert slots,16484 missing experts
mirrored into49 host-USM segments/51660083200bytes, then session-up/token-graph
hit path. This placement/loading occurs before request1; it is not measured
by these prompt/decode counters. Fresh=1 resets semantic session/prefix reuse,
not process graphs or runtime/page residency. All selections report reused0;
that does not make repeats cold-process measurements. Both arithmetic repeats
also differ after T1/T2 capture already exists. No timestamped module/JIT,
page-fault/residency, dispatch or transfer trace attributes these differences.
SYCL persistent cache OFF does not prove absence of in-process compiled state.
Legacy DONE hits/lookups/offload zeros on device-plan/nohost do not establish
zero resident work or zero PCIe host-mirror reads.

VERDICT: The repeats show a large first-observation cost for each prompt, with
actual lazy graph capture present in the first function prompt. They do not
uniquely explain that cost or establish a warm-up optimization, stability or
qualified token rate. Follow the pinned x2 methodology: same-build balanced
prompt order/repeats, scoped trace and untraced controls, output equality,
separate startup/first-use/steady request records and distributions.

Next concrete diagnostic: retain this exact lane/IDs/LP setting initially and
add a default-off host timeline for request begin/reset, capture/finalize/
existing-wait entry-return (including T and executable-cache hit), existing SYCL
submit/wait, PLE gather, sampling, full-head LP readback/CPU top20, T print,
final commit wait and DONE. No added barriers or normal-graph EAGER mode.
A passive UR/Level Zero host API trace can test module-build/JIT/finalization
hypotheses; it does not by itself time device kernels. Retain existing replay
completion events and validate profiling support/clock units before any GPU
lane; the current zero GPU stamps cannot supply missing intervals. Host minor/
major faults and RSS are separate from unobserved GPU page residency/PCIe
traffic. Use the source35 trace-coverage audit for per-card calibration/event
ownership and mapped-host expert/handoff limits. Collect cold process first
use and balanced warm repeated prompts, then a separately labeled LP0 control
to isolate this diagnostic setting's sensitivity while preserving all IDs.
Do not move captures before READY and label it faster without measuring both
startup and matched request latency. No tracer/backend patch is prepared here.

Evidence SHA256 (R=run root above, S=SDK source above):

```
S/sycl/src/program/generate.cpp 27e52a23debf493b57654a7ca871859b2655c5d8bb11c51be9e473d5ba557a39
S/sycl/src/core/verify.cpp cdd837fd3dd31873094587e549623bae887765c48d8c4de7f542185f52c4d84f
R/input-plan.snapshot.json 8cf5cae200450d0d8f72390468a581dd3a409defba3eca7d20176952387ba645
R/parent-qualification.json c2100ece8cc8e7e96bbb3ca4557448bcac7cb06cf42de185a44d145a9aa79fe1
R/child/native/engine.stderr.log 431dceec455ad99d343653163f80eeddd23a99a524ed2e2476dc9e9617693afd
R/child/native/engine.stdout.log 485151e055485fbaf4a0e5bd0ae01a39100c20cf19e7b007206c0ccc0eb6b074
R/child/native/case0-repeat0.json 7cd3e9fb9ea932d0df0d3b00755cbda036d45863b6d3e879347c39b576abe970
R/child/native/case0-repeat1.json 4f62d1bd44c328c3dbec83d0f2f353949787210a38e962f201c74acbf26f323c
R/child/native/case1-repeat0.json 3025b74ab484830a1bdca84ec0d97bf34f622d437f7bbe4285b61c401b1584a2
R/child/native/case1-repeat1.json df09b23c84e6a7bda8032eb6e05089cce1b7a8105519d7782c4a981962d7714d
```

Pinned methodology: docs/20261009_flashnext_latency_methodology.md SHA
3104b8a07819775db35150a6def2c8b1a9537a857f49298f78e4532ebfcd649c;
source35 trace audit SHA31afbff0fdb3739df05908201b3484478e8794148a55e3be7be5afa9700f35de.
