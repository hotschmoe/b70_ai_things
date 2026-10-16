# 2026-10-16: S141/S142 actual reviews and next source decision

CONFIG -> Two independent actual Claude calls at the experiment decision point,
each resumed its own session and received the same frozen31,348-byte packet
(SHA3ebaac494891deb6a43b53db137eef116ba228c0130fd3646271e0330cfe6404).
Packet includes S141 private strict QSA result, S142 CPU diagnostic, retained
source profiles and the n3-only S143 production source proposal. No tools,
GPU/build or another model's answer were provided to either model. This draft
is outside the repository and experiment root; it adds no execution/review gate.

COMMAND -> Existing actual review.py invocations, no retries or new calls here.

| Actual responding model | Elapsed seconds | Actual result |
| --- | ---: | --- |
| claude-opus-5-5 | 53.78712227300275 | exit0; same resumed session; packet unchanged |
| claude-fable-5-1 | 54.71873091801535 | exit0; same resumed session; packet unchanged |

Full answers, exact prompts/commands/packet and filtered completion/session IDs
are in payload/. Raw CLI events and unfiltered completions remain external with
exact paths/bytes/SHA in source-copy-index.json. Cumulative usage/modelUsage,
API-duration and cost fields are deliberately not presented as these calls'
usage or elapsed time. Both answers were fully read by root.

RESULT -> Both accept n3-only source integration, pending runtime qualification.

Opus checks the exact first raw row, contiguous strideD, owner commitpositions
at+2/stride1, n_steps3/base0 and unchanged gamma/epsilon/buffers/capacity/RoPE/
queue. It flags analytic-RoPE coverage and validation exceptions as conditions.
Fable agrees that group length3 matches S141, the old loop remains for other
lengths, the flag is immutable before capture and the marker name is unique.
It treats production capacity/RoPE as integration coverage requirements and
warns that S141's private chain timing predicts no serving gain.

Root accepts SOURCE_GO_N3_ONLY, adds explicit STRATA_ROPE_TABLE=0 in BOTH future
S144 arms and requires the actual native environment gate. append_steps span/
overlap validation may throw into the existing capture exception path; no
fallback is added. Tailrestore/GDN/conv/KV/PLE history and error/lifetime order
stay unchanged. Main build was already running at this decision; this curation
is not an extra GO block. S144 same-new-binary control/ON, warm+3timed pairs,
strict output/work/functionals/normalclosure/fullhealth are still PENDING here.
No mainruntime qualification or serving gain is claimed by source approval.

Both models correctly notice S142's much smaller verifier/pair submission CPU
brackets and the larger later poll/PLE interval. Both overextend the result:
Opus infers LEFT/postpair trace artifacts and GPU-side cost; Fable extends it
to S95/S136 gaps, roughly one-fifth overhead and a general46ms real cycle.
Root REJECTS those extensions. S142 supports profiler-bundle inflation of CPU
brackets in ONE matched107x64/allflag0 diagnostic. It does not join CPU/GPU
clocks, prove physical causes, validate all traced gaps as artifacts, identify
one LevelZero event/API, or establish a general cycle budget. Native waits may
include other dependencies. The unprofiled1.585553ms commit median is a useful
diagnostic bracket, not an additive performance budget or serving forecast.
The unchanged full model answers retain these rejected claims for provenance.

NEXT -> Both propose batching GDN state/conv commit launches, so root accepts a
bounded SOURCE AUDIT, not implementation/build/GPU GO. Existing source already
has upstream gdn_commit_layers batching: two kinds per owner, FOUR total for
two owners, rather than inventing another descriptor framework as the first
step. Actual SYCL implementation is missing. The source audit must map that
existing alternative, parameter/storage ownership, selected state-only body,
FP recurrence order, complete state/conv outputs and lifetime before choosing
one finite strict private GPU oracle. It is not already a measured optimization.

Opus calls shared y_dummy output a required allocation fix. Root REJECTS that
for the actual selected state-only SYCL branch: it accepts only state/h/C/gate/
beta/hk/hv/keep and RETURNS before y/z/gamma/output kernels; upstream batched
commit has no y arguments. Do not allocate dummy output slices or infer a race
from an unused caller argument. The frozen forensic source audit is included at gdn-source-audit/: selected
SYCL verify_kernels.dp.cpp L232-303 andL1943-1966, fullSHA5faf60ccd6e8d0665281253374bd97d4d874deb86f0aad479ce94f8565b2673e.
L1834-1849 only validates the unused output pointers. Upstream public API/header
and clean CUDA implementation are pinned atL44-53 andL1019-1039 respectively.

The remaining GDN source constraints are real: two exact3row ownergroups need
disjoint state/input/keep spans; init_slots alone does not prove nonoverlap.
Reuse current SYCL recurrence order and verify full state/conv bytes in a finite
private oracle. Source-equivalent arithmetic is not universal compiled FP proof.

VERDICT -> Continue the already-authorized narrow QSA integration and pending
S144 test; investigate the existing batched GDN seam in parallel. Queueprofiling,
Q6 pre-expansion and broad tracing remain parked. Preserve all actual answers,
source-only status and rejected inferences. No new model call, workload, build,
GPU access, rawtrace read or repository modification was performed for this draft.
