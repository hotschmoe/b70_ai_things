# Source36 bounded native host trace H

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: New patch0036 on frozen source35, two edited native host source files
and one new SYCL-consumed header. No native kernels, math, tensors, queues,
graph properties, input/template/registry, source35 SDK or old plans changed.
Frozen proposal docs/20261010_flashnext_source35_critical_path_trace_plan_v1.md
remains unchanged. This implementation is the FIRST ONE-CARD NATIVE SERIAL
subset: serve/batch0/MTPempty/pipeline0/empty split/suffix0/lookup0. Paired,
worker/batch/cache4/6, frontend HTTP/SSE and passive/device tracing are deferred.
OFF preserves all original supported modes; ON rejects unsupported modes.

COMMAND: CPU reconstruction and collector controls:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s strata/flash-next \
  -p test_host_critical_path36_cpu_v1.py -v
```

Actual authorized host fixture used ordinary /usr/bin/g++13.3, -std=c++17 -O2
-pthread, pinned39992 image, env -i, networknone,2CPU/2GiB/no swap, only header/
fixture RO input and owned output mounts, no SDK/model/GPU devices or requests.
Owned container exited0/was removed. Durable V2 root is
/mnt/vm_8tb/b70/build/hosttrace36-cpu-actual-v2-20261010; its copy receipt proves
all original files copied by SHA, not reexecuted. CPU receipt V2 includes exact
source snapshot bytes and original parent/inside/copy SHA bindings, so /tmp
cleanup does not erase source association. V1 header/fixture/nine controls stay
historical: raw calloc Event lifetime was not defensible C++17; V2 uses bounded
nothrow new[]/delete[], with no header changes after actual V2 CPU execution.

RESULT: Nine actual C++ CPU cases passed: OFF no heap/file/event mutation;
normal ON; overflow; unexpected writer; reused pointer/newest generation;
invalid mode without allocation; actual RLIMIT_FSIZE write failure with closed
FD count; request limit; preexisting graph. Ten Python controls pass: canonical
host coverage and metadata, corruption, clock/order/open span, graph/stage/token/
printf/flush association negatives, final-close/terminal negatives, preexisting
status, patch application, unchanged native wait/graph-submit call inventory,
and read-only actual CPU/source closure. No native SYCL compilation, native
runtime/trace, event profiling, model input or GPU execution occurred.

Implementation and admission boundaries:

- STRATA_CRITICAL_PATH_TRACE unset/0 returns before trace allocation/path work.
  ON requires1, existing absolute nonsymlink DIR,64-lowercase-hex prerequisite
  BINDING_SHA256, native serial onecard mode and EAGER absent. Bind the directory
  and binding to actual fresh source/binaries/model/IDs/parent proof. The SHA
  argument alone is not a GPU qualification. Invalid flags fail startup.
- HeaderV2 allocates32768 typed Event objects once ON before readiness (about
  2.5 MiB), no per-token allocation. Max4 requests, input1..2048/new1..64;
  excess/overflow marks coverage invalid, not model routing. Single owning
  pthread; every mutating path rejects foreign writers before reading mutable
  fields. The violation flag is lock-free atomic and sticky. Only the owner
  exports/releases records. Unsupported pairing/worker modes are rejected.
- Per-request ring export follows RequestScope completion after DONE and finish
  work. Files are O_EXCL/O_NOFOLLOW, never overwritten. Unconditional fclose
  reports both write/close failures. Require CLOSED success and terminal
  SESSION flags. Records are copied before delete[] at owned process exit.
  Export can delay the next request: H is a bounded diagnostic, not a claim of
  transparent timing. Fatal _Exit/SIGKILL can lose a ring: missing coverage fails.
- Every scope has sequence, CLOCK_MONOTONIC ns/span, request/PID/rid/slotgen,
  absolute position/T, stage/device, opaque queue and graph generation. Native
  Request/Reset/PromptSpan, Verify/input/PLE, capture/cache-hit/finalize/existing
  capture wait, replay submit/existing wait, sampling/commit/final wait, LP
  readback/CPU, T instant/actual printf/actual stdout flush and DONE are hooked.
  No new SYCL/device wait, barrier, event query, USM allocation or graph node.
  Greedy sample/device residency/mapped host expert reads remain inside the
  existing graph; H does not time individual kernels or claim PCIe bytes.
- Graph recreation replaces the pointer mapping; cached hits keep the newest
  generation. A graph first seen already cached is explicitly PREEXISTING,
  with no creation/JIT timing claim. Whole Capture and separate finalize/wait
  scopes are distinct. The upload ue=0 migration stub is never a transfer.
  Queue pointers are run-local identities; physical card/context association
  needs the actual root-owned lease/health/build metadata. Device clock is
  explicitly not_requested. Header has no SYCL dependency or event ownership.
- Collector requires all prior/prompt/decode rows once for stage0/layers0..48,
  exact input/fresh/public IDs, complete output IDs/DONE and absolute positions.
  Each T is tied to the preceding completed T1 replay/graph/queue/position and
  immediate corresponding actual printf begin/end; one original flush per
  no-draft serial decode round. LP20 readback/CPU coverage must match token
  count. Dropped/duplicate/stale/out-of-order/zero-clock/unmatched spans,
  invalid footer/close/session/bounds/ownership fail. Overlapping scope sums
  are not additive costs. Collector success is HOST COVERAGE ONLY.

Native build is ROOT-only using NEW host-critical-path36-engine-build-plan-v1.json:
all36 ordered patches,64 source files/28 added-header payloads (29 consumed
SYCL .hpp paths versus28 before this patch), eight fresh ABI targets,
six unchanged Python sources and freshly rebuilt transitive archives in the
pinned compiler image. Do not reuse source35/C113 binaries or runtime evidence.
Preserve default native graph geometry/env and STRATA_VERIFY_EAGER absence.
New source390/uploadV2/logicalfree/full4/known-pages/strict per-card+compiled pair
health/identity/owned exit-removal gates remain required before model use.

Exact OFF/ON witnesses before interpreting trace latency:

1. Same fresh source36 SDK, image/libraries/card0, ctx2048/prefill64/FP16KV/
   PLE65536/segmented1024/device-plan/nohost/adapt0/no-borrow, identical accepted
   two functional prompt ID sequences and GEN64 fresh1 ckpt1 temperature0
   logprobs20, balanced same-build repeats. Require all emitted IDs/text/EOT,
   functional answers/repeats and actual work/ownership/source gates unchanged.
2. In a SEPARATE correctness-only capture lane, BOTH OFF/ON use identical
   existing SFD capture flags/binding: complete vocab248320 F32 first-output
   head bytes for each function/arithmetic repeat at actual last prompt
   position29/38, and matched prefix1/2/4/8 P30 corpus all4 full heads and
   every48x3xactualrow residual/attention/FFN row. Source36 fresh consumers and
   schema2 nonce/frame association are mandatory. Full heads/vectors must be
   bit-identical OFF versus ON; top20 text is insufficient. These diagnostics
   alter observation, so do not call their elapsed time normal serving latency.
3. Then native H OFF/ON normal-graph LP20 diagnostic runs with full H coverage,
   scoped first-use/warm distributions and trace perturbation measured. Never
   treat these counters as device clocks. LP0/passive UR/device envelope and
   frontend trace are separate future preregistered experiments.

VERDICT: Header CPU mechanics and source/parser controls qualify this proposal
for independent review and ROOT's fresh native build. They do not qualify GPU
trace coverage, speed/stability, full48 mathematical fidelity, broad/registered
quality or concurrent cache4/6. No captured native math inputs enter the frozen
original reference. All model/reference failures and historical evidence remain.
