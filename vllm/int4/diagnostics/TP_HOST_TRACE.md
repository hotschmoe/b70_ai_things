# R276 startup TP collective boundaries

CONFIG -> Opt-in source-pinned host instrumentation in
`vllm/fp8/kv_hooks/b70_tp_host_trace.py`. The reviewed phase image is
`sha256:0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077`.
Runner, XPU communicator, inherited base communicator and distributed route
source bytes must match their exact gates. No native code is changed.

COMMAND -> CPU verification:

```sh
python3 -m unittest discover -s vllm/int4/diagnostics -p test_tp_host_trace.py -v
```

RESULT -> Five CPU tests cover nested profile/capture scopes, paired profile
records, actual tensor metadata, other-group exclusion, custom-op route guard,
normal/error/logging-failure scope restoration, bounded output and source-gate
rejection. Fake tensors forbid value reads/copies. No GPU smoke was performed
for this hook; native completion and compilation behavior remain unqualified.

VERDICT -> Ready for a separately identified, leased diagnostic startup. It
provides observed profile shapes and per-rank Python collective entry/return
counts, not an inference from requested batch limits. Instrumented timing must
not be presented as uninstrumented performance.

## Next-arm integration

`kv_campaign_server.py` now accepts `--tp-host-trace phase` or
`--tp-host-trace phase-mrv1`, plus `--tp-host-trace-max-events` (default100000).
Tracing requires TP2 and explicit `--p2p 0`; TP1, P2P1, omitted P2P, packaged
hooks and nonpositive event caps are rejected before lease or subprocess work.
Select the immutable image matching the explicit source profile below.

The server copies the complete hook directory into each new readonly source
snapshot. Tracing forces the existing entrypoint/PYTHONPATH path even with
`--hook none`, preserving the installed-package guard and `/tmp` workdir.
It explicitly forwards the output directory `/kv-campaign/tp-host-trace`,
profile selector and event cap into Docker. A host shell export alone is not
an activation mechanism. `sitecustomize.py` adds its target only when that
forwarded directory is present; existing calibration/offload targets coexist.
The existing HookLoader invokes `install()` after normal runner import and
fails closed on installation or source mismatch. Spawned workers inherit this
Python startup behavior. No eager runner import or model/Dynamo wrapper is added.

Before the next TP2 run, freeze a new plan and output directory with the chosen
trace flags, image and source hashes. Never modify active snapshots. Retain the
existing both-card lease, P2P0, strict per-card/compiled-pair health, verified
teardown and post-health. No lifecycle bypass or extra GPU waits are added.

Four CPU integration tests check both selectors and forwarded argv, actual
opt-in import-target registration, pre-lease refusals, hook coexistence and
byte-identical default Docker argv against pre-change fixtures (`none`/`load`).
Run these and the existing campaign lifecycle checks with:

```sh
python3 -m unittest discover -s vllm/fp8 -p test_kv_campaign_tp_host_trace.py -v
python3 -m unittest discover -s vllm/fp8 -p test_kv_campaign_preflight.py -v
```

## Evidence interpretation

Each process writes `host-PID.jsonl`, mode0600, under the supplied private
directory. `dummy_enter`/`dummy_return` label profile, capture, or warmup/replay;
they include bound `_dummy_run` arguments and output metadata. `shape_counts`
records actual communicator tensor shapes, strides and dtypes; do not substitute
the requested `num_tokens` for observed shapes.

The TP communicator is checked by its actual unique group name. Profile-only
`collective_enter`/`collective_return` pairs include actual local/global rank,
group, world size and call ID. Every dummy scope aggregates operation entry,
return and exception counts. Other communicator groups are skipped and counted
separately. Nested Python method calls may still represent the same underlying
operation, so report by operation rather than blindly summing all boundaries.
Match ranks by phase/arguments and operation order, not PID or process-local ID.

The actual all-reduce route clones the input, submits c10d all_reduce with
async_op=True, executes its existing work.wait(), and returns. The hook adds no
GPU reads, transfers, synchronization, event queries or waits. Host returns are
not independently proven device completion. The original final profile sync
also remains unchanged; this hook does not trace its completion.

Captured graph replay normally bypasses Python communicator callbacks. Zero
callbacks in a replay scope do not mean zero device collectives. Capture counts
describe observed Python invocations, not graph-node execution counts. The
hook requires the opaque custom-op route to keep its communicator wrapper out
of Dynamo's model trace. GPU compilation/capture still needs actual qualification.

The event cap emits `truncated` and suppresses later records; capped evidence is
incomplete. Logging and metadata handling perturb host timing. Exceptions retain
partial evidence, with scope restoration, but cannot guarantee log durability
after process/host failure. This is startup instrumentation only; it does not
trace live accepted-token row permutations or cache publication/reuse.

The default `B70_TP_HOST_TRACE_PROFILE=phase` accepts only the phase runner.
An explicit `B70_TP_HOST_TRACE_PROFILE=phase-mrv1` instead requires runner SHA
`7299b4cfabc447b15b66a5fcaf5bb858a0783fc46340d561f292f5f96e99e789`, associated
with image `sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1`.
This separate pin was added only after its CPU image identity passed: exactly
one runner changed among2227 Python files, all282 packages and2106 native files
matched phase, and the GDN phase source stayed unchanged. Evidence is
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/mrv1-counts-phase-image/manifest.json`.
All other route gates are shared and unchanged. Unknown profiles fail closed;
selecting MRV1 does not also allow phase runner bytes. Forward this selector
explicitly with the next plan, and externally verify the exact image: the hook
checks source files, not Docker image identity. No GPU qualification is implied.
