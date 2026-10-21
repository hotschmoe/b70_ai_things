# Q1 fused r2 parallel selected-ID compaction

Measured source/profile:
/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r2/profile-r1
The candidate_sha256 field in original result.json hashes selector.py,
NOT the compiled library. identity.json and measured-source-sha256.json
provide explicit source-to-port hashes and actual library identity:

23e456afdc5ab7bff44b40a45fe51744773d19fe2191eb6f3963f57871d69325
/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r2/libq1_fused_r2.so

Kernel/key/Torch/v1control source bytes are unchanged. Port changes only
source/build/CPUhelper/SO paths. Unique namespaceb70_q1_packcompact_r2;
PackKernelR2 and CompactBitonicKernelR2 can co-load with measured v1.

One full-width grid packs native exact uint64 keys then sign-bit XOR/
bit_cast to int64. Upstream Torch topk2051 sortedTrue is unchanged. IDs
are inverted from low32bits of the first devicewidth selected values,
padded with INT_MAX to4096 and sorted in16KiB SLM via256threads with
78bitonic stages plus initialfill barrier. Cooperative write touches
firstwidth only. Actual-n serial scanning is removed. All native clamps,
NaN payload ordering, ties, inactive rows and canary tails preserved.

CPU4166compiled keypairs/252bitonic cases PASS. GPU252adversarial variant
comparisons and all6real147456-column shapes/canaries/changed topk+chain
capture/exact gated parity PASS. Nativepre/posthealth, source/binary and
actual container closure PASS. Raw72chains validate3replays/24phases:
11native/136Torch/63v1/63v2ops, unique compact symbols distinguish arms.

M1devicechain native/r2 at4K1.186/1.342ms,16K2.606/1.360,
32K4.653/1.352. Shortcontext still regresses versus native; r2 improves
versus Torch-only at every tested shape. Bitonic~0.123ms, pack~0.006ms,
first-sort kernel sum~0.515ms. No model serving/speed/fidelity/pressure/
liveness or200K qualification follows from these component checks.

Interface: PackedTopk(scores_n,cap,device).__call__(scores,ids,step).
B70_Q1_FUSED_SO explicitly pins the library; default is the measured path
above. Warm first call to allocate exact-shape keys before graph capture.
Keys4.72MB at4x147456; warmed transient peak increment19.04MB vs42.64MB
Torch-only. Raw reserved counters depend on arm/order/graph pools; do not
infer model KV capacity. Compilation0/full XPUGraph baseline needs no
Dynamo FakeTensor dispatch; other compilation modes are unqualified.

Build (CPU only):

    B70_AGENT=Q1 MAX_JOBS=4 bin/cpu-run bash vllm/flashnext/harness/q1_fused_r2/build.sh

A future replay requires explicit grant and new no-overwrite identity:

    Q1_PROFILE_TAG=profile-new bash vllm/flashnext/harness/q1_fused_r2/window.sh

Physicalcard1/logical0,240s body, CPUquiet outsidelease, native pre/post
health, source/SO hashes and actual teardown. Parent owns the separately
default-off serving API/shape cache/workspace review and full GPU grant.
Measured v1 remains in q1_fused; Torch-only remains in q1_torch.
