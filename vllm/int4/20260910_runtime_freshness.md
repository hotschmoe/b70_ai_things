# Runtime freshness audit, 2026-09-10

CONFIG -> CPU/network-only audit; no GPU access, serving, host package change, or active-plan mutation.
COMMAND -> Fetch Steve origin/main, query neural.download, GHCR tags and OCI manifests, GitHub release/PR APIs, and version-pinned upstream source. Pull the official XPU image by immutable platform digest.
RESULT -> Steve origin/main is ac2f723ec0eb150976c744f54cddb7c4e0d99c82. His public INT4 GHCR repository lists R228, R256, R276 only. R276 is exactly our existing registry/local image identity sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad. His public FP8 repository lists only R187, sha256:173660ec18c6e98a14b9a4f573922abe9d3414999056f07ab5c3c14b55d6ceb0. No newer published replacement for our Steve image was found.
VERDICT -> We are current on Steve's published INT4 image, but not upstream vLLM. Steve's September 9 GDN phase-guard result is explicitly a local, unpromoted candidate, not a newer public image. Its matched bounded screens do not establish the original client's FP8 KV/prefix-cache soak.

CONFIG -> Official vLLM v0.29.0 XPU, released 2026-09-09, image built 2026-09-09T01:58:28Z.
COMMAND -> Resolve docker.io/vllm/vllm-openai-xpu:v0.29.0 through Docker registry manifests.
RESULT -> Index digest sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901; Linux amd64 digest sha256:1db27a8b75ae1d6b3cbf16ebbb88310c2b5548221c8df1335082b2a54dba209a. Compressed layers total 4.165 GB. Source requirements pin torch 2.13.0, Triton 3.7.2+xpu, vllm_xpu_kernels 0.1.14.1, auto_round_lib 0.14.2. Image build history installs Compute Runtime 26.27.39122.11, IGC 2.38.2+22051, Level Zero loader 1.32.0, GMM 22.10.0. Actual CPU-only dpkg and package-metadata inspection of our R276 confirms the same UMD 26.27.39122.11, Level Zero 1.32.0, IGC 2.38.2, GMM 22.10.0, torch 2.13.0+xpu, and triton-xpu 3.7.2 generation. The project historical 26.14 finding is not this R276 image. No host driver change is needed to stage the new image.
VERDICT -> A useful separate current-upstream viability arm, not a single-variable replacement. Do not transplant Steve's ABI-specific native libraries or blindly carry his tuning environment. Establish fresh per-card health, model identity, coherence, teardown, and post-health before concurrency or promotion.

CONFIG -> Upstream v0.29.0 source and current main 8c34a372323b3becc92c183257d195562882ef1a.
COMMAND -> Read version-pinned GDN metadata builder, KV manager, model runner, Dockerfile, requirements; query PR status.
RESULT -> Both release and main still call split_decodes_and_prefills(m, decode_threshold=1) without the phase guard at the relevant GDN non-speculative branch. PR 48375 (Mamba EAGLE block drop) and PR 53919 (accepted-count ordering) remain open/unmerged at audit time. Release notes include Mamba state-copy race 50729, XPU Mamba pointer overflow 48109, and dense hybrid EAGLE/MTP retention 55760/55861. Model Runner V2 becomes default, so backend update also changes runner behavior.
VERDICT -> Updating alone does not include the specific GDN phase guard or those two open correctness fixes. Compare the exact-source phase candidate on the preserved baseline separately from the upstream-update arm.

Raw evidence: /mnt/vm_8tb/b70/results/bang_isolation_20260910/runtime-freshness/

Sources:
- https://neural.download/
- https://github.com/steveseguin/b70-optimization-lab/tree/ac2f723ec0eb150976c744f54cddb7c4e0d99c82/packages/qwen38-27b-int4-fixed-k-tp2-b70
- https://github.com/steveseguin/b70-optimization-lab/tree/ac2f723ec0eb150976c744f54cddb7c4e0d99c82/community/dominick253-qwen38-27b-fp8-uniform-decode-alias/validation/priority-20260909
- https://github.com/vllm-project/vllm/releases/tag/v0.29.0
- https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/gdn_attn.py
- https://github.com/vllm-project/vllm/pull/48375
- https://github.com/vllm-project/vllm/pull/53919

CONFIG -> Fresh upstream single-card viability recipe, prepared but not GPU-executed.
COMMAND -> Inspect version-pinned registry and XPU mixed-precision source, artifact config, and lifecycle argument transformations.
RESULT -> Qwen3_5ForConditionalGeneration and Qwen3_5MTP are registered. GPTQ routes through AutoGPTQConfig, which retains dynamic exclusions. XPUwNa16LinearKernel accepts FP16/BF16, uint4/uint4b8 and group sizes divisible by 32; the artifact uses g128. It dispatches int4_gemm_w4a16. Fresh Config.json/Mounts.json are in raw evidence upstream-v029-config: 8K context, one sequence, FP16 KV, no speculation, eager, prefix off, no hooks or Steve tuning. The saved speculative-config placeholder is removed by the required lifecycle --mtp 0 option.
VERDICT -> Source-compatible minimal viability candidate. This is not proof of successful weight load or generation. Model Runner V2/default selection and actual runtime dispatch must be captured in the GPU arm. Run fresh-image per-card AND compiled two-rank health under both leases before serving; use a new collective cache directory to avoid reusing old compiled artifacts. Then allocate the single-card arm under bin/gpu-run --card N with matching ZE_AFFINITY_MASK. Any existing-native source port needs rebuilding for this image before qualification.

CONFIG -> Installed official image CPU-only audit, no device mounts.
COMMAND -> docker run --rm --network none --cpus 2 --memory 2g --entrypoint /opt/venv/bin/python -v <raw-evidence>:/audit <immutable-image> /audit/inspect-installed.py.
RESULT -> Pull completed. Local image ID equals Linux amd64 digest 1db27a8b75ae1d6b3cbf16ebbb88310c2b5548221c8df1335082b2a54dba209a. Installed vllm is 0.29.0+xpu; torch 2.13.0+xpu; triton-xpu 3.7.2; vllm-xpu-kernels 0.1.14.1. Actual GMM is 22.10.1-1~24.04~ppa1 (upgraded after the initial Dockerfile install); R276 has 22.10.0. UMD, IGC and Level Zero match the versions above. The health script's torchrun, libccl.so.1.0 and CCL kernels paths all exist under /opt/venv. Native SHA256 ledger is upstream-v029-installed.txt. Installed GDN source SHA256 c65552d9aad86472544033d44ad8a872221a83e6b60ba9918cc049ab0c580c7c exactly matches fetched release source and retains the unguarded split call.
VERDICT -> Staged and inspected without GPU touch. Parent paused further Docker work while current-arm teardown/health completes because concurrent image extraction delayed Docker lifecycle calls. Future heavy image pulls should be serialized with teardown despite parallel CPU source analysis.

CONFIG -> Separate official0.29 phase-guard candidate, not built or served.
COMMAND -> Apply the same phase-guard source edit at the sole matching split call; compile Python source for syntax validation.
RESULT -> Tracked patches/v029-gdn-prefill-phase.patch and raw upstream-v029-gdn-phase-candidate/Dockerfile prepared; source input hash matches the actual installed image. Candidate changes one Python file only. Base and candidate must both run the same tiny-prefill screen after the required fresh-stack pair health.
VERDICT -> Ready for build after parent Docker clearance; not yet a verified repair.

CONFIG -> Official0.29 actual-source CPU regression.
COMMAND -> Run the existing Steve AST-extracted builder/helper test inside the official image, with no devices/network and CPU/memory limits.
RESULT -> Eight stock cases pass their expected-bug assertions. An initial unmodified CLI help attempt fails to infer a device because the CPU-only container deliberately has no XPU devices; this is not a backend viability failure. A separate parser-only check sets the platform metadata to XPU without creating devices; engine configuration is not instantiated.
VERDICT -> The installed current release preserves the same CPU-proven one-token misclassification. GPU tiny-prefill baseline/candidate tests remain required.

CONFIG -> Fresh CLI parser and standalone candidate build.
COMMAND -> Parse transformed fresh lifecycle engine arguments using official installed AsyncEngineArgs, with XPU platform metadata in a container that has no devices. Build the separate phase-guard Dockerfile.
RESULT -> ENGINE_PARSE_OK /model 1 gptq True False: model path, TP1, GPTQ, enforce-eager, prefix-cache-disabled arguments accepted. Candidate immutable image sha256:29f480e90dc6b88e7898392500b89a22a49e13209f7b6e3e3c7b47088748a0e1 built.
VERDICT -> CPU CLI/schema gate passes; no model config construction, weight load, or serving claimed. Baseline and candidate remain independent exact image IDs.

CONFIG -> Exact official0.29 base/candidate native equality and GDN CPU test.
COMMAND -> Hash installed native/source files inside both images with no GPU devices; run eight actual-source AST cases on installed candidate.
RESULT -> All 13 inspected native files are byte-identical; the only inspected source difference is gdn_attn.py. Eight candidate cases pass with fixed=True, versus eight stock cases with fixed=False. Manifest and native-equality.json preserve identities and outcomes.
VERDICT -> One-Python-file candidate built and CPU-verified on the current upstream image. No GPU-serving, coherence, speed, or long-session repair claim.

CONFIG -> Official vLLM0.29.0 XPU stock versus phase-guard image on card0, same INT4 AutoRound W4A16 g128 artifact, TP1/P2P0, MTP0, FP16 computation and KV, eager, prefix caching off, context8192, max sequences1, batch8192, GPU memory0.90. Primary alias hotschmoe-dd; research aliases record v029 stock/gdnphase and this configuration. The full-feature production recipe is not this isolated arm.
COMMAND -> Review parent-run tiny-prefill screens and compare complete returned token-ID arrays by (phase,prompt_id), not hashes alone, across official0.29 stock/candidate and R276 phase-fixed card0/card1. Raw comparison: runtime-freshness/v029-gdn-full-array-comparison.json. Parent GPU work used the leased lifecycle and new-image health gates.
RESULT -> Official0.29 stock fails all six one-token prompt attempts: the first repeats token17 (character2) for64 tokens; the other five repeat token1023 (duct) for64 tokens. All18 longer-prompt controls pass. The phase-guard image passes24/24; all four sequential repeat comparisons and all16 mixed-versus-sequential comparisons are exact. Every complete candidate token array matches both R276 patched-card0 and patched-card1 output (24/24 against each). Stock and candidate match18/24 full arrays, differing at the first token on all six singleton cases. Both official0.29 lifecycle exit codes are0; card0 post-health is HEALTHY in both logs. Candidate completed2026-09-10T04:12:10Z. Endpoint identities and immutable images are preserved in each manifest/models.json.
VERDICT -> The same isolated initialization defect survives upgrading to official0.29, and the one-Python-file phase guard removes its observed singleton degeneration in this matched screen. The complete output agreement across the two patched runtime families strengthens this bounded result. These are24-request screens at max-num-seqs1, so client mixed issuance does not establish concurrent batch execution, long-session coherence, production cache/MTP/graph parity, or resolution of the exact Pi incidents. No speed claim or shelf promotion follows from these data.
