# Exact FP8 cache-write extension

CONFIG -> Existing installed-XPU FP8 oracle, with optional --exact-write mode.
No GPU execution was performed while preparing this change.

COMMAND -> Run test_kv_fp8_scale_oracle.py in a CPU-only R276 container with no
network/device mounts. Its extracted validator executes actual CPU Torch casts;
manual IEEE E4M3 bytes independently encode the expected finite extrema, zero,
minimum subnormal and normal values. Source syntax and git diff checks also run.

RESULT -> Five checks pass: exact manually specified bytes at permuted locations;
reject corrupted active K; reject corrupted active V; reject modified unused
slot; reject swapped K/V scales. Actual XPU execution remains outstanding.

VERDICT -> The added mode strengthens the existing oracle without replacing its
installed reshape_and_cache_flash write, attention-versus-CPU reference, V-scale
linearity, K-scale sensitivity, grouped heads, causal query offsets, permuted
blocks or hybrid cache-stride checks. Exact writes use distinct power-of-two
K/V scales (0.125 and0.25) and representable FP8 values; the cache is then cleared
and all existing random write/read tests still run with scales0.08 and0.06.
This does not define an out-of-range conversion policy or qualify model-level
prefix copying, MTP state rollback, graph replay or SGLang Triton attention.

## Prepared bounded GPU qualification

The parent owns a separate raw runner at:

`/mnt/vm_8tb/b70/results/bang_isolation_20260910/fresh-calibration-plan/scale-oracle/run-leased-oracle.py`

It self-leases both cards through tracked bin/gpu-run, reuses the reviewed
preflight owned-container/process-group cleanup helpers, and requires strict
per-card plus compiled P2P0 collective pre/post-health. It executes phase-only
R276 image0328900c with length67, block64, permuted hybrid layout and query
lengths1 and4. Every run records source/helper/image identity; cleanup is verified
before reset. The runner is prepared but has not been executed.

The two inside-container commands are:

    python3 /oracle.py --length 67 --block-size 64 --permute --hybrid-layout --query-length 1 --exact-write
    python3 /oracle.py --length 67 --block-size 64 --permute --hybrid-layout --query-length 4 --exact-write

Do not invoke those directly outside the leased lifecycle. CPU evidence and
review detail live under the raw scale-oracle directory above.
