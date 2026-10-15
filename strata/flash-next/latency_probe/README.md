# Snapshot transfer latency probes

Standalone source prepared by Claude Opus 5.5 and reviewed/run by root.
These probes measure components, not serving speed or cache correctness.
Every GPU execution must use `bin/gpu-run --card N` and the matching
`ZE_AFFINITY_MASK=N`. The probe requires exactly one visible GPU and checks
its PCI address with `--expect-pci`. Keep host conditions quiet when comparing
latency; record CPU affinity, governor, image/binary identity and topology.

Build offline in `strata-ab-build:ocloc`, without `/dev/dri`:

```sh
source /opt/intel/oneapi/setvars.sh --force
icpx -O2 -std=c++20 -fsycl -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen \
  "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" \
  -fsycl-device-code-split=per_kernel xfer_probe.cpp -o xfer_probe
```

Run the resulting binary in the same image with `/dev/dri` exposed, through
the lease wrapper. For example, inside that leased container on card0:

```sh
taskset -c 2 ./xfer_probe --expect-pci 0000:0b:00.0 \
  --label card0_cpu2 --sizes 64M:10,256M:8 --arms 0xaff \
  --json /results/card0_cpu2.json
```

`--oracle-only` performs 42 exact-byte/guard cases and exits before timings.
Timed runs also perform that oracle and verify every sample using a fresh
pattern and poisoned destination. No separate small/full matrix is needed.
The first sample is separated from subsequent samples. New segmented
allocations can still fault on every iteration; "warm" means later samples,
not necessarily reused pages. Destination correctness checks are untimed.

Arms cover warm pageable/pinned host copies, device copies, host-to-host
queue copies, and 16MiB segmented save/restore pairs. Segmented save arms
compare fresh zeroed vectors, fresh uninitialized storage, and an amortized
persistent host-USM pool. Include `alloc_us + total_us` when comparing saves;
`free_us` is charged to the paired restore. Persistent-pool creation, poison,
and verification are outside its timed interval. The probe does not establish
that uninitialized or pinned storage is safe or profitable in production.

`total_us` is host enqueue plus wait; `service_us` is device-event duration.
Neither alone measures physical PCIe traffic. Host-to-host arms must not be
reported as VRAM transfer bandwidth. Save/restore pairs include separate
event timing per chunk, matching the synchronous transfer call shape.

`host_cpu_probe.cpp` is a CPU-only first-touch/allocation diagnostic. Its
source documents arguments. Compile with `g++ -O2 -std=c++20`; record load
and do not run it during clean GPU latency windows. Its warm memcpy arm
repeatedly copies the same 16MiB source/destination. That cache footprint
does not establish bandwidth for a multi-GiB snapshot; do not extrapolate
its result as a measured full-snapshot copy time.

S18 results, commands, raw samples, topology, identity and health live under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/agents/opus`.
Do not use the agent's unqualified native launcher as the runtime recipe;
root used the build image to keep SYCL and driver libraries matched.
