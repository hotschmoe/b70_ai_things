# Actual FP8 cache geometry oracles, prepared only

CONFIG -> Immutable phase+MRV1 image7b107d. Reuse the unchanged tracked
vllm/fp8/kv_fp8_scale_oracle.py with length23240, block-size1600, permuted block
table, interleaved hybrid K/V layout and exact-write enabled. Separate q1/q4
jobs use12 query heads,2 KV heads and dimension256, matching the target TP2
full-attention shard geometry on one physical device. These are independent
synthetic attention tests, not a TP2 model launch.

COMMAND -> plans.json contains the two exact owned launch commands, both on
card0 so they can be scheduled separately from future SG card1 work. Each
requires the selected inherited lease and the image-matched7b pair-preflight
marker, verifies immutable image/source/helper hashes, runs strict selected
pre-health, one bounded oracle and strict selected post-health. Per-job limits
are8GiB,4CPU and420s, with separate fresh cache/output and a read-only source
snapshot. No pair reset occurs under one card lease. Shared reviewed ownership
helpers terminate launcher groups and verify labeled Docker cleanup.

Geometry ->15 logical blocks map to physical blocks16 through2. Two extra
physical blocks and the final partial page remain untouched in the exact-byte
write check. Joint storage is [17,2,1600,2,256]; K/V view strides are
[1638400,512,256,1]. These strided views and the23240-key reduction differ
materially from the earlier67-token/block64 fixture.

RESULT -> Actual argparse AST parsing confirms both command lines. CPU
lifecycle fixtures pass q1/q4 geometry checks, partial JSON retention after
numerical exit1, selected post-health and post-health failure blocking success.
Both unleased commands fail before GPU work. Source identities and resource
pins validate. The actual oracle imports/executes GPU code only inside the
future owned container; no GPU action occurred during this preparation.

Original gates remain unchanged: relative K/V write error<0.04, attention
relative L2<0.015, correct doubled-V behavior and nontrivial wrong-K sensitivity,
plus exact representable write bytes and untouched slots. A success requires
the final completed oracle receipt, expected block table/stride/query geometry,
logical device0 under the physical pin, and process exit0. Partial rows and the
full raw log survive failure; health success cannot mask numerical failure.

VERDICT -> Ready for separately scheduled geometry diagnostics. No numerical
pass, kernel defect, model quality, native GDN/Mamba copy, MTP, TP2 collective
or throughput conclusion is established by these CPU checks. Pair-preflight
marker provenance is supplied by the frozen plan's exact image-matched path;
it is not a generic substitute for image identity.

Raw plans: /mnt/vm_8tb/b70/results/bang_isolation_20260910/vllm-fp8-geometry-plans
