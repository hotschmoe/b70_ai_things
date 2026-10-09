# FlashNext USM free-query controls, 2026-10-09

CONFIG -> Installed runtime39992d70 and original SYCL guarded allocator header;
no model weights, forward pass or serving. Frozen control source0a1eb04c.
Default allocator settings remain unset. Parent holds both GPU leases for the
whole per-card+compiled-pair lifecycle, with each separate control process pinned
by ZE_AFFINITY_MASK and local device0.

COMMAND -> run_usm_free_control.py default and separate --trace process on both
cards. Seven sizes8192/40960/163840/3481600/6963200/27852800/33554432 bytes,
three allocation paths (ordinary SYCL, Strata guarded SYCL, raw Level Zero),
two rounds:42 cases per card. Same owning queue waits/free, then only metadata
queries on the freed address; no freed-pointer dereference/copy/fill/kernel.
Fresh same-size live allocations receive bounded first/last-byte probes.
UR tracing arm enables only tracing layers/logging, with allocator knobs unset.

RESULT -> Both default and tracing controls complete42/42 operations per card.
All matching frees return, fresh probes pass and addresses are reused42/42.
Every post-free SYCL type query still says device. For ordinary/guarded paths,
Level Zero address ranges retain backend slabs (e.g.8KiB request in64KiB slab).
For raw zeMemFree, the range query fails2013265953 with null base/zero bytes,
while type/properties still report the old device/allocation ID. Therefore
neither SYCL pointer type nor allocation-property type proves logical liveness
on this installed runtime. Pool retention alone does not explain every path.

Actual tracing logs contain60 successful urUSMDeviceAlloc and60 successful
urUSMFree returns per card. An independent chronological ledger keyed by
context+pointer rejects allocating an already-live pointer or freeing a missing
one; all120 events retire with0 live logical allocations. Removing one free or
duplicating a free is rejected. These negative controls validate this parser
against the actual recorded trace format, not against hypothetical log strings.
The raw Level Zero path bypasses UR allocation/free, as declared; tracing covers
the SYCL paths and their guarded allocator auxiliary allocations.

Both arms exit/remove normally, pass strict finite per-card and ten compiled
P2P0 collective pre/post-health. Traced kernel-journal fault audit is clean;
parent separately inspected the default run interval with no selected faults.
Initial source compile failed missing explicit fcntl/unistd includes; failed
snapshot/log retained and corrected source compilation passes without devices.

VERDICT -> The first source-upload oracle's post-free type==unknown gate is
invalid for this runtime, demonstrated independently of NativeDense. Its failed
receipt remains failed; it is not retrospectively promoted. A new oracle contract
must preserve source SHA/bytes, live USM/device/context, stage ownership and
allocation-accounting checks, then correlate actual native-owner pointers and
scratch with successful matched logical-free trace events, fresh probes and
normal process teardown/health. Control success alone does not qualify model
owners, full source upload, serving, prefix caching, concurrent coherence or speed.

Raw controls: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f05-20261009/.
Independent trace ledger: usm-free-trace-v1/independent-trace-audit.json.
No driver, kernel, source weights or serving allocator configuration changed.
