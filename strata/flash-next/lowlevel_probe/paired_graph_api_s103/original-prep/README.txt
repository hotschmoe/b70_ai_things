S103 SOURCE ONLY: one-card graph API, ordering and mapped freshness probe

No build, probe, GPU, profiler or numerical process has been executed by the
preparation author. No repo/current docs or S102 preparation were changed.
Root must full-read frozen sources and authorize the build/run separately.

CONFIG -> One small TU, six explicit kernel classes Step0..Step5, one
in-order Level Zero queue. Mode0 captures six one-operation component
graphs; mode1 records those identical six operation calls into one graph.
No nested-subgraph API or kernel fusion, model, weight, common archive,
FP/MTP math or accepted-prefix commit. No root_sync: each work-item owns
one output element and queue/graph edges order six separate kernels.

Both modes run in separate fresh processes within ONE owned container and
ONE quiet-CPU/card0 lease, with per-card health once before and once after.
Each mode runs three A/B/A replays, no warmup/eager execution/timing loop.
All kernels read the current mapped host input through the exact pinned
strata::sys_load_mapped uint32 L1/L3 uncached helper, plus the preceding
step's device output (Step0 starts from its index). Every step stores its
own257-word plane. Both modes use identical logical args/shape/values;
raw allocation addresses naturally differ between fresh processes.

Input uses actual zeMemAllocHost with BIAS_UNCACHED through the pinned
host_malloc_polled helper. The probe rejects its ordinary-host fallback.
Host mutation follows a drained previous replay/readback and a host fence,
before graph submission. The device does not spin and the host does not
mutate input while a graph is running. This bounded sequencing is not a
proof of concurrent host/device writes or cross-card visibility.

Unsigned32 multiply/XOR/add has defined modulo2^32 arithmetic. The CPU
ledger computes all six active planes independently from the current input.
Exactly one7704B full-output readback per replay checks every intermediate
step, all guards, mapped input/guards readonly and A/B/A restore equality.
N257 with512 work-items also checks the partial group's bounds. Captures
and replay graphs live until after final queue drain; graphs die before
storage and queue. No fallback/retry follows a partial failure.

COMMAND -> ROOT after source review, other-agent quiet-window release and
fresh lease/boot checks:
  bash R/paired-graph-s103-prep/build.sh
This uses bin/cpu-run, pinned e41, noDRI/network, UID1000, four CPU/OMP/MKL,
disk TMP and one direct icpx compile+AOT link. Exact source/header hashes
are checked pre/post. No native/common archives are linked. Build records
actual command, compiler identity, source/binary/profiler hashes. Inspect
the SAME live handle to terminal; no duplicate build after a timeout.

After root reviews the successful build, announces physical card0 and an
approximately60s finite probe window, and checks other holders:
  bash R/paired-graph-s103-prep/ROOT_RUN.sh
Expand R to this existing results root. Root must verify the preparation
manifest before either command. Scripts refuse existing build/run dirs.
Unitrace executable00fc84e5... is the existing S39 profiler; its tool library
is hashed during the CPU build, then both hashes are checked before the
two GPU processes. No profiler build or whole-model input is needed.

RESULT GATES -> Existing unitrace -h -d -v emits small text tables only;
no Chrome trace or raw270MB reparse. Analyze mode0 before starting mode1.
Expected per-process actual kernel counts: Step0..Step5 exactly3 each,
18 total. Expected zeCommandListImmediateAppendCommandListsExp counts:
18(mode0) versus3(mode1), hence6 versus1 per replay. No unclassified
graph submissions are permitted. Unexpected backend/API counts produce
HOLD for inspection, not an automatic rerun or an assumed saved budget.

Common app setup/readback is explicitly reconciled: one7704B device fill
and three7704B full-output copies per process, no other device commands
beyond the18 named kernel executions. All host API counts are retained,
including graph construction and allocation/cleanup, but are not assumed
equal or removable. Capturing six graphs versus one changes construction;
only actual graph replay submission counts and equal device work are the
selected mechanism. Raw instrumented time columns are deliberately not
used for a speed verdict. DOT files retain actual capture structures.

Normal container process exits are inspected by FULL created CID/image/name,
with Runningfalse/OOMfalse/ExitCode0 required. Owned CID is then removed and
full CID+NAME absence is confirmed before post-health. The GPU attempt has
a180s outer bound. On failure there is no new workload; only owned-CID
cleanup is allowed. As in the retained S98 runner, an unconfirmed teardown
holds leases until absence is confirmed, rather than releasing a live GPU
container or overlapping health. Never kill a foreign/shared name.

VERDICT -> Only PASS_GRAPH_API_ONLY can authorize discussion of one
default-off depth2 main integration. It is not MTP numeric/state isolation,
partial-main-failure recovery, graph speed, production stability or serving
qualification. Actual MtpPair state/pointer lifetimes, mutable inputs,
host poll/PLE dependencies, no-redraft failure rules and acceptance-gated
commit remain separate source and matched serving gates. Do not include
accepted-prefix commit before the host acceptance decision.

Development-loop choice: ordinary re-recording avoids an unsupported nested
API investigation; the same18 tiny operations plus A/B/A checks and two
small profiler summaries test the required API/freshness/count hypothesis
before any expensive main AOT build or model startup. S102 separately
reuses existing S85 ISA and is not touched here.
