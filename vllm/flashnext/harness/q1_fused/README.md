# Q1 fused pack and serial threshold compaction

This preserves the measured exact-but-slower serial candidate. Kernel
namespace b70_q1_packcompact_r1 and explicit PackKernel/CompactKernel.
The measured source hashes/profile are recorded in
measured-source-sha256.json; kernels/key/header/Torch control unchanged.
Build output is disk scratch q1/fused-r1; no binaries in campaign tree.

    B70_AGENT=Q1 MAX_JOBS=4 bin/cpu-run bash vllm/flashnext/harness/q1_fused/build.sh

The shared key.h is compiled into the GPU library and a CPU key helper.
CPU4166key pairs/252threshold cases and GPU189adversarial comparisons plus
all6populated shapes/canaries/changed-input graph/exact gated parity PASS.
Native pre/posthealth and source/lifecycle PASS. Serial compaction is
slow: M1kernel0.416/1.447/3.091ms at4K/16K/32K. Do not select this candidate
for serving on performance grounds. Torch-only remains measured control.

Interface: PackedTopk(scores_n,cap,device).__call__(scores,ids,step).
The first call lazily allocates exact-shape int64 keys; warm before capture.
SO default: /mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r1/libq1_fused.so
B70_Q1_FUSED_SO overrides it. Loader FakeTensor support is not yet added;
component capture is distinct from model Dynamo integration.

A future replay requires explicit coordinated GPU grant and new identity:

    Q1_PROFILE_TAG=profile-new bash vllm/flashnext/harness/q1_fused/window.sh

Physicalcard1/logical0,240s body, CPUquiet outsidelease, mandatory native
pre/posthealth, source/binary hashes, nooverwrite and actual teardown.
Measured native/Torch-only/fused arms share147456scorecols and seeded
real-geometry pages, capacity4/activeM1,M2,contexts4K/16K/32K. No model
or pressure/fidelity/liveness qualification. Memory counters are raw
order-dependent component allocations, not model KV capacity evidence.
