# Strata data readiness probes

These standalone probes complement the [data map](../data-map/README.md).
They measure transfer preparation and dependencies; they do not claim a
serving optimization. Actual Claude Fable 5.1 authored the Level Zero
capability query; Claude 5.5 authored the readiness probe, reviewed and
corrected before GPU execution.

Build offline, without a GPU:

    bash build.sh

Every GPU run uses the lease and matching device pin. From this directory:

    ZET_ENABLE_METRICS=1 ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' ../../../bin/gpu-run --card 0 build/ze_metric_enum --json build/card0-metrics.json
    bash run_readiness.sh card 0 2 full
    bash run_readiness.sh card 1 2 full

The launcher pins CPU 2, checks the expected PCI BDF and binary hash, and
runs the AOT binary inside strata-ab-build:ocloc. Card selection is physical;
the masked device is logical 0. K_USED=10 is confirmed from served GGUF
qwen4exp.expert_used_count; rows=6, N=2560, hc=4. Exact payload sizes:
24-byte tokens, 17280-byte Q8_1 activations, 245760-byte residual,
17520-byte ids+activation and 614400-byte expert return. EP sizes are
single-card transfer stand-ins here, not P2P transport measurements.

Each timed arm includes fresh salted bytes, destination poison/guards and
exact verification. The finite producer has an observable reduction checksum
verified on the host; an earlier removable busy loop was caught in review.
Host allocation/touch/synthetic pack/enqueue/wait/consume times are separate.
Device start/end timestamps measure service and producer-copy-consumer gaps.
Host and device clocks are never subtracted. command_submit versus start
was inconsistent (including negative queued differences); preserve those raw
fields as diagnostic only and exclude ALL queued-time values from conclusions.
Synthetic pack/consume/checksum costs are not Strata's quant/pack costs.

S24: one full run per card, 680 samples each, all byte/guard/checksum checks
passed; both-card post-health passed. Sample 0 is excluded from summaries.
Fresh pageable versus preallocated pinned D2H at 614400 bytes had median
producer-end to copy-start gaps 121.20 versus 1.41 us (GPU0), 116.93 versus
1.41 us (GPU1), 11 warm samples per arm. This warrants serving-path readiness
instrumentation, not a claimed model speedup or a bandwidth limit.
Both host metric-group queries returned ERROR_DEPENDENCY_UNAVAILABLE despite
timestamp support; no driver, kernel, sysctl or package was changed.

Raw provenance, exact commands, frozen binaries, hashes and summaries:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/lowlevel-s24/

No server restoration, duplicated smoke matrix or new whole-model build was
needed. The GR no-root-sync experiment passed 32 bitwise cases but was neutral;
its rejected source patch is retained separately. No serving defaults changed.

The retained metric enumerator fixes a display-only review finding: timer
units depend on properties.stype, not the driver API version. The installed
ze_api.h documents DEVICE_PROPERTIES as ns, DEVICE_PROPERTIES_1_2 as Hz.
Raw original values are retained; no performance conclusion used that label.


## Real vocabulary head replay (S25)

    bash build_head.sh

This separate CPU-only AOT build links the recorded runtime archives and
prints source/archive/object/binary hashes. It writes under build/head-cols.
Run inside the matching build container with bin/gpu-run --card N,
ZE_AFFINITY_MASK=N and ONEAPI_DEVICE_SELECTOR=level_zero:*, and mount the
GGUF read-only. Arguments:

    head_cols_probe --iters 30 --expect-bdf 0000:0b:00.0 --weights FILE 11024256 --json OUT

FILE is the first served IQ3_XXS GGUF; output.weight has Q5_K type13,
shape2560x248320,437043200bytes, SHA256
93f1c7cb7a7cf8134ce9f8cf80c5416df968f7f1ddcc16f2159a9521080826a5.
The offset is specific to this model file. Re-derive it from GGUF metadata
before using another artifact. C=1/2/3/4/6/8 use deterministic normal
activations, every column compared bitwise with its independent C1 result,
plus argmax, finiteness, inactive rows and tail guards. Actual read weight
FNV-1a64 must be da8746c3dcc2469e for this tensor.

Both cards passed all six cases. Card0/card1 minimum-of-three mean spans
in us: C1 1006.0/1006.6; C2 1322.0/1307.9; C3 1721.7/1676.5;
C4 2043.2/1978.0; C6 2700.5/2620.7; C8 3376.6/3302.6. Quantization is
timed separately, about4.2-4.6us. These event-barrier spans include queue
and launch gaps. Tiny unitrace per-kernel medians on card0: C1 1016.666us,
C2 1309.010us, C6 2681.146us, grid62080x1x1, workgroup32x4x1, SIMD32.
This is a standalone replay; cache/frequency/context differ from serving.

Only two slots at the SAME draft depth may be paired. Four serial draft
heads cannot become one C4 call because successive depths are autoregressive.
Ideal head-only saving at depth2 is about1.38ms/window before extra graph
launches and coordination; no serving gain is claimed.

## Existing production PLE trace (S25)

STRATA_PLE_TRACE=1 on the frozen S21 binary changes timer/logging only;
keep STRATA_VERIFY_PROFILE unset (it changes shared-stream scheduling).

    python3 -I parse_ple_trace.py ENGINE_LOG --window START:END --json SUMMARY

The existing trace omits all-cache-hit tickets (jobs0) and tickets with
200+ I/O jobs. It reports same-clock differences without absolute anchors:
issue->submit, submit->last, last->return and collect wait. Large
last->return can mean useful prefetch slack, not a stall. The collect wait
includes caller locking/scheduling. It cannot be joined exactly to GPU
events and submit->last is not device-only NVMe latency.
S25 had21 miss-ticket records between batch capture and the24-window summary:
collect median667.9us, p90 817.4us, sum13.2942ms; jobs8-64.
Two107-input/64-output replies match S16 prefixes, active slots offer46
and accept40 drafts each; arithmetic canary and both-card/10 compiled
collective post-health pass. No clean throughput claim from this diagnostic.
Raw commands, frozen hashes and trace summaries: cachefix_20261014/head-profile-s25/.
