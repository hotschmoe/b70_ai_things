# Q1 top-k reduction component experiment

The first candidate changes only the per-bit 256-thread count reduction
from the original manual local-memory tree to sycl::reduce_over_group.
It preserves all 64 radix scans, unsigned 64-bit packed score/index keys,
tie rules, prefix selection, serial ascending-ID output, and clamps.
Native QSA and both isolated variants load together; all copied kernels
have the unique b70_q1_topk_r1 namespace and explicit template names.

Original qsa_batch.cpp SHA256: 028208d94aa263dece9ff659bc77808a789f761462c59ee910e5fb2285ab4428

CPU build (MAX_JOBS4, disk TMPDIR, no GPU):

    B70_AGENT=Q1 MAX_JOBS=4 bin/cpu-run bash vllm/flashnext/harness/q1_topk/build.sh
    B70_AGENT=Q1 bin/cpu-run env PYTHONPATH=/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages /home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python vllm/flashnext/harness/q1_topk/oracle.py

Only after an explicit coordinated GPU grant:

    Q1_PROFILE_TAG=profile-r1 bash vllm/flashnext/harness/q1_topk/window.sh

Physical card1 with matching device pin; native pre/post health, CPU quiet
outside GPU lease, 240s process cap, normal container teardown, source/SO
hash checks and no-overwrite outputs. Expected total 2-4min.
Q1_TOPK_ROOT may select a separate build/output directory; build first.

probe.py checks 63 adversarial unsigned-key oracle cases against native,
copied tree and candidate. Cases include ties, NaN payloads, infinities,
signed zero, extremes, width/n clamps, zero-width and inactive canaries.
Then it measures top-k and the entire captured one-layer decode chain
(write step, quantized KV append, indexer append, score, top-k, gather,
attention, gate and durable length). Geometry is24/2/256, indexer4x128,
compression4, budget2048, cap2051, block2752; serving capacity4 with active
M1/M2 and full populated paged KV at4K/16K/32K. Inputs are seeded component
fixtures, not actual captured model activations. Quant scales are positive
fp16, codes signed int8 and pooled state finite f32. Whole-chain exact
gated-output parity, exact selected IDs and changed-input graph replay
are required. Inactive selected-ID rows retain canaries.

Native unitrace logs three uninstrumented captured-chain replays per arm
and shape. Separate event timing reports top-k and total chain averages;
auxiliary uncaptured per-op event boundaries have no host waits between
operations and include prior gaps/length commit. Event instrumentation
can perturb scheduling; use the uninstrumented trace to rank kernels and
readiness gaps before attributing critical-path cost. Sum across12layers
only as an explicit projection, not a measured model result.

This experiment does not establish full-model fidelity, speed, pressure
liveness, or fault clearance. Any full-model change needs an independently
coordinated serve with activation, matched fixtures, health and teardown.
