# Qwen3.8 AutoRound INT4 neural.download replica campaign

User authorization: overnight replication, quality comparison against current
FP8 weights, conditional RAM offload with FP16 KV, and final hotschmoe-dd
restoration. Commit/push milestones; provide a local systemctl activation
command. No FP8 KV work or host kernel/driver changes.

## Frozen acquisition

CONFIG -> Recipe https://neural.download/models/qwen38-27b-autoround-int4-b70.html
retrieved 2026-09-08. Steve Seguin's source repository
https://github.com/steveseguin/b70-optimization-lab at
54aefaf067b422d507f51f1e362efeecc58668ba, cloned separately under
/mnt/vm_8tb/b70/steve-repro/qwen38-int4-neural-20260908.

COMMAND -> Fetch exact model revision bce40cacab0a4535b92fb3d57615c2bea9adf3d1
from devan-carlin/Qwen3.8-27B-int4-AutoRound into models/files/qwen3.8-27b/;
pull published R276 image from GHCR by digest:
521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad.

RESULT -> Acquisition in progress. Raw root:
/mnt/vm_8tb/b70/results/int4_replica_20260908.

VERDICT -> The live recipe supersedes the older AutoRound route discussed
in docs/20260905_qwen38_fp8_vs_int4_quality_review.md. Do not transfer either
its older rejection or the author's newer speed claim into a local verdict.

## Sequence and decision gates

1. Finish FP16 KV RAM-offload qualification on official FP8 weights. Keep
   the preserved serving configuration available as the fallback. Concurrent
   artifact downloads/builds make overlapping timings provisional; repeat
   promotion comparisons on a quiet host.
2. Verify all publisher model hashes/byte counts and reproduce the exact
   GPTQ relabel with the pinned script. The tensors stay AutoRound INT4;
   GPTQ here selects the fixed-K oneDNN kernel, not a new quantization.
3. Record native library hashes, torch/vLLM/UMD/Level Zero/oneCCL and image
   identities before GPU execution. Keep every native extension within the
   published image's matching stack. Preserve source patches/build scripts;
   no quarantine binaries or host ABI replacement.
4. Run per-card and compiled TP2 collective health, then the scoped exact
   R276 recipe. Its direct-P2P setting is confined to this documented replay
   with pre/post health and reset on failure, not a general serving default.
5. Reproduce the author's 12-prompt, six-class, 512-token strict suite on
   two fresh MTP0 and MTP4 lifecycles. Bind MTP verification to the same
   image's MTP0 output. Report the author's 99-interval token1-to-token100
   decode metric separately from total wall throughput and natural output.
6. Qualify C1/C4, long generations, tool histories, cancellation and actual
   long contexts on the intended daily-driver shape. The exact strict
   1024-context/cache-off recipe is not a 200K/cache-on qualification.
7. Compare all 164 sandboxed HumanEval+ problems with official FP8 weights,
   same harness, seed, thinking-off and 2048 output cap. Review new failures
   individually. More than two additional failures (1.22 percentage points)
   or any new critical arithmetic/tool/state corruption rejects promotion;
   also require default-thinking task checks. This is bounded local evidence,
   not proof of parity on every agent workload.
8. Only after the INT4 recipe passes, port the guarded RAM offload repair
   deliberately and rerun eviction/reload/coherence/performance/lifecycle
   gates. More GPU KV capacity requires more distinct histories to force
   eviction; equal input size alone does not create matched memory pressure.
9. Restore hotschmoe-dd on the best fully qualified recipe: INT4 if it
   meets speed and quality gates, otherwise original FP8 weights with FP16
   KV. Offload is independently conditional. Prepare the system service
   update and exact user command; do not leave the endpoint down at handoff.

Published reference, not a local measurement: R276 MTP4 strict pair
112.90/113.00 tok/s; R256 earlier pair about 112.3. The recipe credits fixed-K
W4A16 GEMM (R220/R221), FP16 row chunks (R224), grouped GDN (R228/R276),
size-independent Inductor reductions, graph capture and draft-only INT4
lm_head. Source details and all immutable inputs are in its pinned
publication-manifest.json. No local speed or quality result yet.

## Acquisition complete

CONFIG -> Frozen R276 GHCR digest and bce40cac AutoRound checkpoint.
COMMAND -> Download the model; verify every listed file with both O_DIRECT
and ordinary cached reads; run the author's R212 hard-link/config relabel
and verify its separate manifest. Fetch eight release assets and four
revision-pinned source archives; verify published hashes and byte counts.
RESULT -> Original and relabelled model identity checks passed. R212 has
99 dynamic exclusions and unchanged tensor payloads. All 12 release/source
artifacts downloaded (153604055 bytes). R276 installed _xpu_C is
271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f;
_xpu_ops.py is 6ee6b8db18759873246aca28e85ca6d2ba177eb08bfd3b9b0f0feea168cee9b3;
layernorm.py is 50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8.
All three match the recipe. The three core PyTorch native library hashes
match the current FP8 image. Runtime: torch 2.13.0+xpu, vLLM
0.27.2rc1.dev77+gac7509e2b.xpu, kernels 1e90ffa67, oneCCL 2022.0,
image UMD 26.27.39122.11-0 and Level Zero loader 1.32.0. The first inventory
attempt used an absent distribution name; the complete corrected inventory
is runtime-r276-v2.txt. No runtime was modified by inventory collection.
VERDICT -> Exact published artifacts are available locally. No local GPU
speed/quality claim yet. Acquisition identities are tracked in
neural_replica_lock.json. prepare_replica.py extracts the published
standalone invocation without executing shell text; strict config has no
model/runtime parameter overrides beyond local paths, port and served ID.
run_strict_replica.py owns the four fresh leased lifecycles and stops on
health, workload, canary or token-parity failure.

Lowest-priority follow-up: the user permits one bounded FP8 KV retest after
all FP16 offload, INT4 and serving-restoration work. The earlier package-path
mismatch makes a corrected retest useful, but it must not delay those tasks.
