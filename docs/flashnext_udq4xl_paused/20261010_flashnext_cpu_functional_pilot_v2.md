# Exclusive-RAM fresh CPU functional pilot V2

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: NEW V2 after actual V1 metadata-startup failure, BEFORE inference.
V1 runner44f4e54/sourceplan11632bc remain frozen. Its /harness/pilot.py mount
had only two parents, so module-level parents[2] raised IndexError before the
metadata helper executed. ROOT retained V1 source, command, failed container
receipt/normal removal, pre-full-four PASS and post-full-four evidence; no CPU
model case was launched. V2 changes both wrapper mounts to the same deep path
/harness/llamacpp/flash-next/pilot.py. Metadata, server, runtime-post-check and
server --version preflight all use that exact production path. A CPU test
computes parents[2] for the actual container path and deliberately rejects the
shallow V1 layout. No backend rebuild or arithmetic change is needed.

Historical CONFIG: V1 prelaunch revision after checkpoint6420fdf. Original runner201398f2
and sourceplan2a4e1abc are preserved in that commit; no model inference had
launched. Independent review identified that the original memory monitor only
logged failure while HTTP could remain blocked1200seconds. The revised monitor
initiates controlled owned-container stop immediately in its own thread,
unblocking the HTTP caller. Monitor and parent cleanup share a serialized,
idempotent stopper; foreign image/name/label is rejected before any stop.
Blocked-HTTP, foreign-ownership and repeated-stop CPU controls exercise this.
No original build/supplement/runtime receipt is modified. The runner saves exact
source-plan bytes plus SHA at start, uses that snapshot for both metadata
passes, and rechecks runner/plan/source hashes at completion. Exactly two
declared, correctly associated and context-bounded metadata fixtures are
required before model inference.

CONFIG: Narrow private diagnostic using exact frozen CPU build V3, compiler
image39992 for inference and c388 for original exported tokenizer/template
metadata. Fifteen synthetic/mock CPU checks plus one read-only actual completed
build metadata admission test pass (sixteen total). No model inference, Docker,
GPU touch or model payload read was performed by this source-preparation agent.

The original build receipt remains passed=false: compilation/configuration
and terminal exit succeeded, but its host ldd check lacked libgomp. Admission
requires its exact failure, SHA-bound host ldd file, unchanged original logs,
all-six build sentinels, successful independent pinned-container ELF/library
closure, host-loader-failure-binding-v1.json and actual CPU-runtime-smoke-v1.
Unknown build failures are rejected. All-six binaries are hashed before/after;
server dependencies are rechecked inside the intended container before/after
its request. V3 recipe design SHA80d9 remains unchanged; actual root build
history lives separately in cpu_reference_build_runtime_v3.md.

COMMAND (ROOT ONLY, after paired serving finishes and RAM is free):

```
PYTHONDONTWRITEBYTECODE=1 python3 llamacpp/flash-next/qualify_cpu_functional_pilot_v2.py \
  --build-root /mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010 \
  --source-receipt /mnt/vm_8tb/b70/build/flashnext-cpu-source-v2-20261010/source-receipt.json \
  --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/cpu-functional-v2-NEW
```

Wrapper preflight ONLY (ROOT; no lease, full weight scan or model mount):

```
PYTHONDONTWRITEBYTECODE=1 python3 llamacpp/flash-next/qualify_cpu_functional_pilot_v2.py \
  --wrapper-preflight \
  --build-root /mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010 \
  --source-receipt /mnt/vm_8tb/b70/build/flashnext-cpu-source-v2-20261010/source-receipt.json \
  --output /mnt/vm_8tb/b70/build/cpu-functional-v2-wrapper-preflight-NEW
```

This invokes actual c388 metadata fixtures and the actual production Python
server wrapper with --version in compiler39992, network none/no model mounts,
2 GiB/no swap cap. It records named owned terminal exit/removal and pre-exec
ELF/library identity. It must succeed before any pre-full-four or inference.
The actual model-pilot command repeats these fresh wrapper preflights itself;
a prior printed success alone is not transferred as runtime qualification.
Wrapper-preflight passed=true means ONLY metadata/startup passed, with zero
CPU cases/inference; functional/math/registered-quality claims stay false.

The actual inference runner acquires the ordinary bin/gpu-run pair lease ONLY to exclude GPU
serving/RAM-heavy campaign workloads. It validates inherited card0/1 descriptors;
CPU containers receive no GPU devices, device requests, privileged mode or
supplementary groups. No GPU health/runtime call is made. External CPU jobs
cannot be excluded by the GPU lease; ROOT schedules this exclusive RAM window.
Start requires112 GiB MemAvailable,116 GiB cgroup memory/no swap cap, and6 GiB
live host reserve. PID VmRSS/VmHWM/VmSwap, host MemAvailable/SwapFree and cgroup
memory current/peak/max/swap/events are sampled every second. Any swap growth,
pressure/max/OOM event or reserve loss initiates owned shutdown and fails.
Shutdown starts from the monitor without waiting for HTTP; Docker normal-stop
grace remains30seconds, after which an unclean forced exit still fails. These are pilot limits, not a
measured memory capacity claim; existing shared page-cache charges may differ
from process RSS. The full model remains111334654784bytes (~103.69 GiB).

Two declared prompts, each repeated in a fresh server container, maximum four
serial loads; first failure stops remaining model execution:

- Write only a Python function add(a, b) that returns their sum. No explanation.
- A box has 7 red balls and 5 blue balls. How many balls are there in total?
  Answer with only the number.

The code response must contain only the add(a,b) function with a single a+b
or b+a return, optionally in one Python code fence. Restricted AST proves the
three declared sum cases; generated Python is never executed. The numeric
response must strip to exactly12. Every response must reach actual EOS/EOT
below64 predicted tokens, temperature0, seed1234, no repeat penalty, temperature
sampler only, no truncation or thinking output. A cap/word stop is failure.
Fresh repeats must have identical generated IDs and text. This is scoped
functional evidence, not a benchmark of general quality or concurrency.

The c388 metadata container mounts ONLY five tokenizer assets and two frozen
source35 Python text helpers, no model tensors or SDK libraries. It records
Python/regex/Jinja source identity and renders both messages with thinking-off.
Inference /props must report exact original GGUF chat-template bytes, original
first-shard path,2048 context and one slot. /apply-template must equal independent
rendered bytes; /tokenize must equal independent original exported token IDs.
/completion receives those numeric IDs directly (server-common.cpp:1025-1037),
so no extra template/tokenization transform is allowed. It returns complete
output IDs; a second c388 metadata pass independently decodes successful cases
and verifies text equality. Both primary hotschmoe-dd and the CPU research alias
must appear in /v1/models with hotschmoe-dd as actual primary. Default-port28671
must be unused. The CPU server uses pinned compiler image, env -i, empty cwd,
local127.0.0.1 API via host network, original four files individually mounted ro,
mmap/lazy-on, no repack/MTP/projector, FP16 KV, ctx2048/prefill64, eight threads.
No outbound network action or implicit model fetch is prescribed.

All four original shard full hashes/stat identities are checked immediately
before model execution and after owned terminal teardown, with both known
buffered pages preserved before/after the scans and every server. The page
guard binds EXACT third shard00003-of-00004 (shards[2]), not the second shard;
CPU roster/reordering/missing-shard negatives enforce this association.
Cleanup verifies the named owned image/label, normal stop, exit0/no OOM and
removal. Forced cleanup fails. Unknown/live containers prevent the post-terminal
full scan; failures and partial output are preserved. No source/model repair,
cache eviction or original reference modification is performed.

RESULT: Source/CPU-only readiness; actual four-process functional pilot remains
unexecuted. Model identity, token/template equivalence, numerical heads, useful
responses, repeatability, memory and runtime teardown cannot be inferred from
successful CPU compilation or synthetic tests. Existing llama-debug head/ID
capture remains a separate subsequent diagnostic; this runner does not claim
complete logits or native arithmetic equivalence.

VERDICT: CPU HC Q8_0 activation operands, expert Q8_K operands and F32 GDN
association differ from Strata's current math. Preserve all frozen FP64 and
native numerical failures. This pilot can produce a useful independent
functional comparison; native bitwise authority, full48 math, broad/registered
quality, shelf/production, concurrency and speed qualification stay false.

Registry proposal for a LATER deliberate ROOT edit (NOT applied here): add a
CPU diagnostic entry under family qwen3.8-flash-next, weights
Unsloth-UD-Q4_K_XL-original-GGUF, method llama.cpp-fresh-CPU-Q8_0-Q8_K-F32-GDN,
served_model_id qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-llamacpp-cpu-v2,
and serve_cmd referencing this exact frozen pilot/build. Mark diagnostic only,
no shelf/math/speed/concurrency qualification. Keep actual primary
hotschmoe-dd. Before any registered eval, cross-check actual /v1/models against
that deliberate entry and regenerate affected future BATCH/cache plans with
the new registry SHA. V2 pilot freezes the existing registry SHA and leaves
models.yaml unchanged; registration requires a new explicit plan binding.
