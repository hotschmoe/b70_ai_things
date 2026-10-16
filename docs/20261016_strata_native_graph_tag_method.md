# Native graph profiling tags on B70 (S146, 2026-10-16)

CONFIG -> One card0 private GDN graph method test, reusing S145's exact
public/private device images and full numeric campaign. Native profiling-tag
support was checked inside the same process. The explicit in-order queue did
not enable queue profiling. Two native tags were submitted outside capture,
around the existing complete graph. No unitrace collector was injected.
Control records 72 convolution plus 72 state calls; candidate records two plus
two, using the same private layer wrappers already tested in S145. No new
device kernel, model, or serving configuration was introduced.

COMMAND -> Root launched the finite source/controller copies in this bundle
with EXE SHA `15ec15aa7df7c48ec2d53e41acf6402c016928c606956415038ce0a67e992652`.
The CPU-only host adapter build completed with build/tee exit0 in about 20s;
its actual commands and receipt are under `build/`. The executable, object,
and reused device images remain external, indexed by their existing receipts.
The runner used the immediately preceding same-boot S145 card0 post-health,
with root's explicit no-intervening-GPU attestation. It required S145 full
numeric PASS, clean teardown and post-health0 before starting. S146 still
performed its own card0 post-health after normal completion.

RESULT -> One leased process closed in about 28s, with native0, analyzer0,
post-health0, normal docker wait/remove, full CID and NAME absent, and no
force/reset/retry. Native aspect was true, so the unsupported branch was not
used. Every measured row checked all 235339776 consumed state bytes against
the eager reference, plus finite values, guards, read-only inputs and output
sentinels. FNV digests are descriptive receipts; complete byte comparisons
were the numeric check. Setup, state restoration and validation were outside
the recorded timing windows. Completed host chain includes tag submission
when enabled and the common queue completion wait. Host graph submit brackets
only graph replay, excluding the explicit tag calls. Event queries occur after
the completed-chain endpoint. All eight observations are retained below.

| Iteration | Arm | Tags | Completed host chain us | Host graph submit us | Device envelope us |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | control | OFF | 1186.893 | 89.771 | invalid |
| 1 | candidate | OFF | 1058.398 | 94.510 | invalid |
| 2 | candidate | ON | 2097.930 | 319.890 | 1039.792 |
| 3 | control | ON | 2099.813 | 321.483 | 1351.615 |
| 4 | control | ON | 1949.597 | 322.305 | 1355.989 |
| 5 | candidate | ON | 1774.242 | 321.193 | 1028.125 |
| 6 | candidate | OFF | 1027.739 | 94.239 | invalid |
| 7 | control | OFF | 1190.980 | 92.867 | invalid |

Completed-chain cell medians were control 1188.9365us OFF versus 2024.705us ON,
and candidate 1043.0685us OFF versus 1936.086us ON. Host graph-submission
medians were 91.319 versus 321.894us for control and 94.3745 versus 320.5415us
for candidate. These are two observations per cell in one nested process.
They establish that this tag method perturbed these measurements; they do
not qualify a repeated performance population or establish significance.
The S145 control/candidate speed result is not requalified by this method run.

VERDICT -> Native tags work on this pinned B70 process, and provide a
whole-graph device timestamp envelope, but their overhead is large enough
here to park routine integration. Retain as an occasional bounded diagnostic;
no follow-up GPU run is selected. The envelope is end-tag command_start
minus begin-tag command_end, queried after the existing queue completion.
It is not per-operation exclusive service, occupancy, physical idle,
readiness causality, a CPU/GPU clock join, or a serving gain. The returned
timestamps are in the device event time domain; they do not join S136 GPU
gaps to S142 CPU brackets.

The installed SDK API guard avoids the queue-profiling fallback entirely.
Its exact header/hash evidence is in `source/installed-api-evidence.json`.
The official extension describes the native tag aspect and timestamps:
[Intel profiling-tag specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_profiling_tag.asciidoc).
Graph recording and graph-profiling properties have separate rules:
[Intel graph specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).
No profiler, driver, adapter-selection knob or stack upgrade was changed.

The development loop reused existing device images and S145 correctness
coverage instead of repeating a full state campaign. The remaining timing
cost was one host adapter build and one finite leased process, with no large
trace, parser pass, decoder, new framework, or model reload. Exact copies,
original absolute paths, hashes, modes and external artifact receipts are
listed in `source-copy-index.json`. Actual root statistics are retained
unchanged in `run/root-summary.json`; all raw JSON method rows are retained
unchanged in `run/leaf.stdout`.
