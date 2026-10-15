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
