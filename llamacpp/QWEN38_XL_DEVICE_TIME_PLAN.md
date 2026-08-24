# Qwen3.8 XL per-quant device-time plan

## Decision

Status: all full-service timing mechanisms in this document are closed NO-GO.
Keep the runners and parsers as negative evidence; do not rerun them unchanged.

- Main-queue `enable_profiling` caused `DEVICE_LOST` during model load before
  any timing barrier or event read could run.
- Launching under paused VTune Pin did not become healthy within 20 minutes.
- Attach-after-load captured a coherent 512-token request, but stop killed the
  PID-1 server with exit 255 and offline finalization could not load the raw
  collector data. It also measured 1.298x reference TTFT, beyond the 1.25x
  perturbation ceiling.

Use the exact counts/packed-byte census to select candidates, then measure the
candidate end to end. If isolated shape timing is still necessary, use the
normal-queue one-card microreplay in the second-stage section and label the
result as a service-demand estimate, not TP=2 critical-path time.

The candidate image contains VTune 2025.10. It does not contain `unitrace`.
The compiled `libggml-sycl.so` has attributable kernel symbols for every quant
family observed in the XL counts-only census:

- `q3_K`, `q4_K`, `q5_K`, `q6_K`, and `q8_0`
- `iq3_s`, `iq4_nl`, and `iq4_xs`

Some symbols name the family directly. Reordered MMVQ templates may encode only
the stable `ggml_type` integer. `parse_vtune_quant_tasks.py` handles both forms,
does not allocate unknown time, and fails when either adapter or a required
family is missing.

## Rejected default-off collection contract

Use only the profiling image and profiling entrypoint. Do not change a shelf
entry or a production image tag. The runner must require the caller to hold the
two-card `gpu-run` lease and must leave the endpoint down on every exit.

The launch-under runner used the following contract. It is retained only to
explain the negative evidence:

```text
vtune -collect gpu-offload \
  -knob collect-programming-api=true \
  -knob enable-tasks-stack-collection=false \
  -knob enable-stack-collection=false \
  -knob enable-characterization-insights=false \
  -knob dump-compute-task-binaries=false \
  -knob target-gpu=0:11:0.0,0:68:0.0 \
  -start-paused \
  -result-dir /profile/result -- llama-server <the exact XL arguments>
```

Set `GGML_SYCL_QUANT_CENSUS=1` for route evidence and explicitly set all of the
following to zero:

```text
GGML_SYCL_QUANT_TIMING_SAMPLE=0
GGML_SYCL_PROFILE=0
GGML_SYCL_DEBUG=0
PROFILE_VERBOSE=0
PROFILE_STATS=0
```

After health and a 32-token warmup, resume the paused result from inside the
same container. Submit one temperature-zero, seed-1234, ignore-EOS request for
exactly 512 completion tokens. Pause immediately after the response. Save the
response before stopping the server. Stop and finalize VTune gracefully, wait
for the result database, archive container inspect and logs, then remove the
container. Do not start another TP=2 process if stop/finalization reports a
Level Zero failure.

Generate both reports after finalization:

```text
vtune -report summary -r /profile/result -format csv \
  -csv-delimiter comma -report-output /profile/summary.csv
vtune -report hotspots -r /profile/result \
  -group-by gpu-adapter,computing-task-offload \
  -format csv -csv-delimiter comma \
  -report-output /profile/tasks.csv
```

The first live trial must validate the exact `vtune -command stop` behavior in
the container. The runner is not production-safe until it proves that stop
terminates the child server, finalizes the database, returns success, and leaves
the endpoint down. If `stop` leaves the target alive, pause, send SIGTERM to the
child server, and allow the parent VTune process to finalize; do not use SIGKILL
as the normal path.

## Gate

The profile is accepted only when all checks pass:

1. Image ID, model size/SHA256, entrypoint SHA256, parser SHA256, and full
   container environment are archived. The model is the Unsloth UD-Q4_K_XL
   artifact, TP=2, MTP off, lab doors off, and P2P access off.
2. Pre- and post-run `xpu-health` pass under the caller's lease. No
   `DEVICE_LOST`, `OUT_OF_RESOURCES`, `UR_RESULT_ERROR`, `nan`, fatal exception,
   VTune data-limit warning, incomplete finalization, or stopped-early request
   occurs.
3. Inspection proves queue profiling and all verbose source instrumentation are
   off. No `[QUANT-TIMING] enabled` marker is allowed.
4. An unprofiled reference and the collected request each return exactly 512
   completion tokens and identical deterministic content. The collected
   post-first-token rate is at least 85 percent of reference and TTFT is no more
   than 1.25 times reference. Larger perturbation makes the timing diagnostic
   only and blocks a performance conclusion.
5. The VTune result has positive task time on two distinct GPU adapters. The
   parser finds all eight census families listed above. Each of the four
   dominant families (`q5_K`, `q8_0`, `iq4_xs`, and `q4_K`) must occur on both
   adapters. Unknown tasks are retained verbatim; they are never assigned to a
   quant family by proportion.
6. This first-live script is a mechanism gate with one reference and one traced
   arm. A pass proves collection safety and attribution, not repeatability. The
   following qualification must run the same collected arm twice more from
   clean server starts. For each dominant family and adapter, total task time
   must be within 15 percent of the three-run median. Otherwise report every
   sample and do not rank families by cost.

After the repeatability qualification, the headline output is total GPU task-duration seconds by
`adapter x quant-family x route`, plus the fraction of classified quant time.
It is not `time per layer` or critical-path latency.

## Second-stage exact-shape replay

VTune kernel names generally do not carry `width`, `K`, `N`, and `rows`. Do not
spread a family total across census shapes. If a family is important enough to
need shape attribution, add a separate one-card microreplay executable using
the exact candidate source and normal non-profiled queue:

1. Replay one census key at a time through the real standard `MUL_MAT` dispatch.
2. Allocate real quant bytes and input/output buffers once, warm the dispatch,
   synchronize, launch `R` identical operations, synchronize once, and measure
   host monotonic elapsed time.
3. Use `R=32,64,128`, five repetitions each. Fit `elapsed = intercept + R*slope`;
   require positive slope, fit `R^2 >= 0.99`, and coefficient of variation at
   `R=128 <= 5 percent` on each card.
4. Measure an empty synchronization control and reject a key when its fitted
   slope is not at least five times the control uncertainty.
5. Run cards separately under `gpu-run --card N`. No TP, oneCCL, queue profiling,
   callbacks, or server is involved.

Multiplying isolated slope by the real census call count is a service-demand
estimate only. It excludes TP overlap, contention, collectives, cache residency,
and graph scheduling, so it must not be presented as measured end-to-end decode
time.

## Rejected alternatives

- Source callback timing with a queue wait before and after every sample avoids
  `enable_profiling`, but serializes the queue and destroys the overlap being
  measured. It is suitable only as a sparse cross-check, never a speed result.
- `gpu-hotspots` hardware-metric collection is more perturbing than
  `gpu-offload`. Use it only after this gate identifies one or two dominant
  kernels, with `computing-tasks-of-interest` restricted to those symbols.
- Programming a Level Zero tracer or adding event timestamps in llama.cpp adds
  a new runtime mechanism when VTune already supplies external API correlation.
  Keep that as a fallback if VTune cannot resolve task names on this driver.
