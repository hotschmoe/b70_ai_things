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

## Next-arm integration, not applied by this change

The server already copies the complete `vllm/fp8/kv_hooks` directory into its
new readonly source snapshot. In that directory's `sitecustomize.py`, add this
opt-in target after `TARGETS = {}`:

```python
if os.environ.get('B70_TP_HOST_TRACE_DIR'):
    TARGETS['vllm.v1.worker.gpu_model_runner'] = ['b70_tp_host_trace']
```

The existing HookLoader calls `install()` only after the runner's normal module
import completes, and fails closed if installation/source verification fails.
It propagates to spawned worker interpreters through their Python startup.
Do not import the runner eagerly from an unrelated entrypoint or wrap model
forward/Dynamo tracing functions.

Add a deliberate server/plan flag for this diagnostic. Its integration must:

- Force the existing `use_entry` path and prepend `/kv-source/kv_hooks` to
  PYTHONPATH, including when `--hook none` is selected. Keep the installed
  package identity guard and `/tmp` workdir.
- Forward `B70_TP_HOST_TRACE_DIR=/kv-campaign/tp-host-trace` and optionally
  `B70_TP_HOST_TRACE_MAX_EVENTS=100000` into the container. Merely exporting
  variables in the host shell does not forward them through the current
  explicit Docker environment list.
- Reject combining this copied-source hook with packaged-hooks mode unless
  that mode has separately reviewed integration.
- Record the helper, sitecustomize, server, exact image, plan and runtime source
  hashes before launch; create a new output/snapshot. Never edit active inputs.
- Retain the existing both-card lease, P2P0, strict per-card/compiled-pair health,
  verified teardown and post-health. This change adds no lifecycle bypass.

No lifecycle or sitecustomize edits are included here. A raw example patch and
independent review live in
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/collective-host-trace/`.

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
