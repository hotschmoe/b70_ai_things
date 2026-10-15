# S38: guarded peer gather alias in serving (2026-10-15)

CONFIG -> The opt-in alias path passes a bounded real-IPC serving comparison,
but remains OFF. Its small observed prefill benefit and unchanged concurrent
decode do not justify another full-context qualification campaign. The
qualified S28 engine `152055a6`, head pairing ON and batch MTP depth2 remain
the recoverable winner. Targets of 2000+ prefill and 150+ decode per stream
remain unmet. S30's true RAM recovery qualification remains valid for that
winner; it is not transferred to this new executable.

Both arms use frozen `strata-peer-gather-s38`, SHA256
`9d30ad8bbcde267e73ed650707002168b001a0aaacf59f9b7234e849d999a9aa`.
Only `STRATA_EP_PREFILL_GATHER_ALIAS=0/1` and log paths differ. Other controls:
EP/PUSH, 149 primary plus 363 peer experts per layer, all 24576 resident,
head pairing ON, batch MTP depth2, spec4, C12288, scheduling Q0, FIRST0,
no borrowed prompt buffers, two 200000-token slots, INT8 KV, 32 GiB parking.
Native trace and GPU phase-timer environment keys are absent, including
presence-tested keys that would remain enabled if assigned string0.

The clean optional patch records existing IPC output-allocation capacity,
guards the fixed Flash-Next geometry and full backed owner tables, validates
the per-layer contiguous routing ranges and same in-order GEMM queue, then
gathers all peer inputs into the FP16 view of existing FP32 output storage.
Experts run in reverse offset order. Earlier unread inputs lie below the
current output; each expert's GU read completes before its down write on
that queue. Dequantization, GEMM, SiLU, offsets and reduction remain the same.
There is no extra GPU allocation, wait or event. A failed guard preserves
the original forward per-expert gathers before any alias write.

COMMAND -> Actual Claude Opus5.5 and Fable5.1 review the exact combined
patch and the later one-line diagnostic quota change. Root checks actual
source facts, exact before/after hashes and one linked TLS instance. One
CPU-only offline build takes183s; only CMake link parallelism changes8->4.
The upstream main remains296b59d2; existing PUSH, publication-scope and
accounting contributions remain open. Both live native process executable
hashes match the frozen binary, in addition to config/model identity checks.

Each fresh process first serves one distinct4049-token/cap32 diagnostic.
Buffered CPU steady_ns markers record all48 layers with complete matching
header, records and trailer; dropped0/failed0. The first-admission quota is
per TLS owner thread, not a global atomic process quota. With this server's
single admission thread, exactly one frame per process is observed. Later
requests retain compiled inactive marker checks but take no timestamps,
records or dumps. Diagnostic timing is excluded from timed samples.

Existing clients then run three fresh requests each at actual4059 and32728
input tokens, cap256, and three fresh C2 coding pairs with107 input and1500
output tokens per stream. The same first request warms both processes.
This is fixed OFF-then-ON order, not a cold-start or interleaved replication.
CPU-quiet windows enclose clients; each server retains the pair GPU lease.
Concurrent323/667, separate reasoning391 and a real tool-call-ID round trip
pass in each arm. No daily-server restoration, new GPU trace, knob sweep,
soak, mixed-arrival repeat or full200k/cache rerun is added.

RESULT -> All13 exact request/output hashes match across arms, including
the diagnostic, six prefill requests and six coding responses. Native
prompt/cache counts and API prompt/completion/cached counts match exactly;
all are fresh cache0. Four available retained S16 coding hashes also match.
Both diagnostics have identical all48-layer routing/expert counts and shapes;
OFF falls back48/48, ON activates48/48 with one allgather each. This tests
actual IPC serving, not a byte readback of every internal peer output.

| Metric, median of3 | Alias OFF | Alias ON | Observation |
| --- | ---: | ---: | --- |
| 4K native prefill tok/s | 1303.0 | 1326.2 | +1.78% |
| 4K client TTFT, s | 3.1469 | 3.0923 | -0.0546s |
| 32K native prefill tok/s | 1546.1 | 1556.1 | +0.65% |
| 32K client TTFT, s | 21.2337 | 21.0955 | -0.1382s |
| C2 aggregate output tok/s | 104.516 | 104.017 | -0.48% |

All samples, observed ranges, per-stream TTFT/completion/throughput, SSE
content-event arrivals and p50/p95/maximum gaps remain in the raw comparison
and clients. SSE events can contain multiple tokens; these are not individual
token timestamps. Three observations do not establish a stability bound or
statistical significance. No matched run reaches either absolute speed goal.

The first diagnostic's peer-loop CPU span has median15.826->14.830ms per
layer, sum859.107->815.047ms across48 layers. Primary gather-return to first
primary dequant-entry has median15.947->14.963ms, sum867.196->823.293ms.
ON allgather call spans have median0.00966ms, sum0.856ms. These include cold
runtime behavior and submission calls that may block; they are not GPU
service times, physical idle, bandwidth or an additive/removable budget.
They replace neither S31's operation census nor S34's historical trace.

Both owned containers stop normally; lease-wrapper exit0 and removal are
recorded. Both-card health plus compiled world2/shape4x5120/10iterations/
P2P0 health pass after each arm, no reset. The immediately preceding external
E2a full normal-exit/health closure is reused before arm0; no redundant
pre-health run. Final CPU/GPU leases are released to the other session.

VERDICT -> Keep alias OFF and retain the reviewed optional patch as research
evidence. Do not add another ABBA or full-cache run to resolve a small prefill
difference while decode is the priority. Prepare the existing GRF resource
knob's full-byte/guard/graph oracle against unchanged production kernels.
This is a source-only next step, not a measured GRF result or promotion.

Raw: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/peer-gather-serving-s38/`
with `s38-report.json`, `serving-comparison.json`, `host-spans.{json,csv}`,
`diagnostic-cross-arm-gate.json`, per-arm raw clients/frames/identities,
normal-stop/post-health closures, exact reviews and build/source audits.
Frozen raw manifest:201 files/15,090,535 bytes, SHA256
`f2cff4c669d28c54a58a7e27d350fc89166dd333f03bf9ec31f856ab2011a672`.
