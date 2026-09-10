# SGLang XPU feature parity review, 2026-09-10

CONFIG -> CPU/network-only source and image review. No GPU execution by this
review. The original endpoint uses Qwen3.8-27B AutoRound GPTQ INT4 W4A16 g128,
MTP3, prefix caching, decode graphs, calibrated FP8 KV, and TP2. A replacement
must qualify that combination; a successful eager MTP0 start is only viability.

COMMAND -> Inspect official GitHub source, issues, release metadata and Docker
Hub Intel image manifests. Save downloaded sources and API records under
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-freshness/`.

RESULT -> Official latest release is v0.5.19, published 2026-09-05. Main at
2026-09-10 03:10:38 UTC is `2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed`.
The newest published `intel/sglang-dev` tag returned by Docker Hub is
`nightly-dev-xpu-bmg-20260903-f5bed25`, also tagged `latest`, with amd64 manifest
`sha256:1a13d3d2b0adc63422a643eb5c59524842577d99ca3935b4b8c2c432e739e127`.
The short tag suffix is not accepted as an upstream SGLang commit by the API;
do not treat the tag text as verified installed source identity.

CPU-only installed inspection resolves the image to SGLang
`fd70325c108c17b635e7caaa57fc7ab4e6e63ad9`, package
`0.5.19.dev931+gfd70325c1`, torch 2.13.0+xpu, triton-xpu 3.7.2 and sgl-kernel
0.11.0. Its container UMD is 26.18.38308.1 and loader 1.32.0; this does not
change the host. Main is 307 commits ahead. `/opt/venv/bin/torchrun` and
`/opt/venv/lib/libccl.so.1.0` exist, so the shared health tools can use their
default `/opt/venv` CCL root. The native GPTQ XPU kernel is present.

The current local image is
`sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`:
SGLang `bede6bc37c5d9638099ebb948d93b9e2a7799f10`, kernel
`2d10888c069350ff20a192338d568dec945c9594`, torch 2.13.0+xpu, UMD
26.22.38646.4. No old ABI libraries are to be copied into a newer image.

VERDICT -> The installed backend is not current main. The newest published
Intel image is a useful bounded baseline, but main contains relevant fixes
after that image. An updated source build or deliberate source ports are
needed before claiming current feature parity.

## Confirmed corrections to earlier issue references

- [Issue 34720](https://github.com/sgl-project/sglang/issues/34720), unexpected
  GDN speculative conv arguments, is closed. PR 35775 merged as
  `5c4622341ffdeb73109af0521fe7db0d79c42498` on Aug 24.
- [Issue 35144](https://github.com/sgl-project/sglang/issues/35144), XPU MTP
  verify broadcasts, is closed. PR 35195 merged as
  `9401db3f290cb1bdffbd29c99114d3d68e2e0c0b` on Aug 18.
- Both fix commits are ancestors of our Aug 26 source, verified using
  `git merge-base --is-ancestor`; raw exit statuses are saved. These issue
  titles are not evidence of unfixed current failures.
- Our measured SGLang large all-reduce failure also occurs without MTP and
  is not explained by the old verify-broadcast issue.

## Material recent fixes

[PR 35051](https://github.com/sgl-project/sglang/pull/35051), merged Sep 10 as
`fd596a474c0f`, packs XPU device-pointer tables through uint64, then views them
as int64. XPU addresses may set the high bit; constructing int64 directly
can overflow. The change also replaces hardcoded CUDA allocations in Mamba
speculative state buffers with the actual device. This is directly relevant
to a new XPU MTP experiment, although it is not evidence about vLLM bangs.

[PR 34820](https://github.com/sgl-project/sglang/pull/34820), merged Sep 9 as
`13469c16d3e9f857c53c3f33da8d33268fd4f570`, preserves configured SSM precision
in KDA tracked prefix-cache checkpoints. Previously an intermediate BF16 state
could lose precision even with a FP32 or FP16 state pool. Its CPU test checks
the snapshot copy and a FP16 double-rounding counterexample. This matters for
prefix-on investigation, but latest Qwen GDN does not supply the new
`h_track_buf` argument to the shared tracker. Do not claim the KDA fix
qualifies Qwen GDN precision. It spans metadata, checkpoint tracking and kernels;
do not copy one function and assume the port is complete.

The full source-only precision patch was checked against the exact installed
commit. Ten source files applied, but five shared-metadata hunks and two KDA
extend hunks rejected because intermediate changes altered the interfaces.
The incomplete port is retained only in raw `precision-port/` and must not be
mounted or built. Its rejection logs are evidence of a port dependency, not
a runtime result.

The pointer/device correction has a deliberate four-file source port in
`sglang-sep3-xpu-pointer-device.patch` and raw `pointer-port/`. The manifest
records exact installed before hashes. No native library is copied by its
Dockerfile. Building this candidate and checking unchanged native hashes is
separate from preparing its source; GPU qualification remains outstanding.

Actual-source CPU checks in the exact image passed: all three installed before
hashes match, all four candidate modules compile, signed pointer construction
raises ValueError for high-bit addresses, and the candidate round-trips the
full uint64 bit patterns. Baseline hashes for torch XPU and 79 SGLang native
libraries are recorded in `pointer-port-cpu-checks.json`; candidate-image
comparison remains pending until the image is built.

For a full current-source rebuild, pin SGLang main above and sgl-kernel-xpu
`3802d7d87829073535fe5f84d3aef4ed76e7ef90`. Relative to
Sep 3, `pyproject_xpu.toml` adds setuptools-rust and torch to build requirements
and renames the native dependency from `sgl-kernel` to `sglang-kernel-xpu`.
The current Dockerfile also source-builds torch_memory_saver at
`a5c99f11b18ebb8e9fda71a68812e476ae49e417`. Hold torch and UMD explicitly and
rebuild the native pieces from these pinned sources; copying Python main over
the old binary tree would skip the required ABI/dependency qualification.

## Feature gates on current main

| Feature | Source finding | Qualification needed |
| --- | --- | --- |
| INT4 GPTQ | Native XPU dense GPTQ kernel uses PyTorch INT4 packing; explicit group-size and act-order checks | Verify the published image includes this route, then load our exact artifact |
| FP8 KV | `intel_xpu` prefill still explicitly leaves descales unset, with unsupported comment; Triton has scale plumbing | Use Triton candidate and verify cache write, prefill, decode and MTP verify against FP16 KV with actual scales |
| Prefix cache | Unified cache supports hybrid state tracking; Sep 9 precision fix is relevant | Warm/cold identity, branch histories, cancellation and concurrent reuse |
| Graph | XPU graph runner exists; decode graph is opt-in, with full backend supported | Explicit `--cuda-graph-backend-decode full`, capture/replay and teardown evidence |
| MTP | NEXTN path exists; latest XPU greedy verification is a sampling semantic difference | Three draft steps, topk1, acceptance/state correctness, user temperature behavior |
| GDN | Triton baseline; optional fused XPU route falls back for verify and tracked states | Tiny fresh prefill, resumed prefix and state reuse tests; no assumed equivalence to vLLM |
| TP2 | Existing local large all-reduce control fails | Both-rank matched entry/return and health before model qualification |

Candidate parity launch settings, not a tested recipe:

```text
--served-model-name hotschmoe-dd
--device xpu --dtype float16 --quantization gptq
--attention-backend triton --linear-attn-backend triton
--kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32
--cuda-graph-backend-decode full
--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Leave radix caching enabled. Supply a verified SGLang-compatible scale
artifact; the vLLM scale JSON is not assumed drop-in. Match context, chunk
shape, concurrency, model weights and sampling before comparing results.
SGLang steps/draft-token counts must be checked against actual acceptance
metrics rather than equating flag names across backends.

## Single-card viability plan

`tp1_viability.py` is a dedicated bounded lifecycle with `bin/gpu-run --card N`,
matching physical device pin, private localhost port, and per-card pre/post
health. It never runs a pair collective or resets both devices while holding
one lease. Health containers have a unique ownership label; launcher groups are
retired and owned containers are verified removed before post-health. Cleanup
uncertainty retains the lease with a marker until resolved. The inherited
selected lease descriptor and owned localhost container binding are checked.
A failure writes `recovery-required.txt` for the owner to resolve
after releasing the single lease. An optional JSON job is executed while the
server owns its lease; no tools in generated model output are executed.

Before the first serve in the new image, the coordinator must separately hold
both leases and complete per-card plus compiled two-rank collective health.
The single-card lifecycle does not substitute for this new-stack gate.

The initial command uses INT4, FP16 KV, MTP0, eager, prefix off, context 8192,
chunk 512 and concurrency cap 4. This isolates single-card loading and basic
backend execution from the known TP2 collective problem. Follow with short
deterministic coherence and tool parsing, then prefix, graph, MTP and FP8 one
variable at a time. A second device may run an independent TP1 vLLM test once
pair-wide preflight has completed and the coordinator assigns both leases.

CPU validation: syntax compile and command checks for both card pins, TP1,
stable alias, baseline feature flags, and absence of pair reset/collective
calls passed. These are lifecycle checks, not GPU or model qualification.

Primary sources: [current SGLang source](https://github.com/sgl-project/sglang/tree/2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed),
[official releases](https://github.com/sgl-project/sglang/releases),
[Intel Docker tags](https://hub.docker.com/r/intel/sglang-dev/tags).

## Pointer/device candidate image result

CONFIG -> Exact Sep 3 official image plus only the four reviewed source files.

COMMAND -> `docker build --pull=false` from raw `pointer-port/Dockerfile`;
run the CPU verification with no GPU devices and no network.

RESULT -> Immutable Docker image ID
`sha256:1660b87ca7ab262e205132a7c23b0fd0c75aeaff91bef4ead31ddf4ad1c4f672`.
Actual installed candidate source hashes match the manifest. Candidate
uint64 pointer roundtrip passes, all four modules compile, and all 81 native
library hashes exactly equal the unpatched baseline. Raw checks are
`pointer-port-candidate-cpu-checks.json`.

VERDICT -> Source-only candidate ready for coordinator-owned new-stack health
and TP1 viability. Not a full-main refresh or a GPU/model qualification. Use
`--image` with the immutable ID above; `tp1-plan.json` now contains that override.
The unpatched official image remains the baseline.

## Calibrated FP8 scale loading gate on pinned main

CONFIG -> Source `2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed`; exact Qwen3.8
artifact uses the Qwen3.5 architecture implementation.

COMMAND -> Inspect `model_runner_components/load_model_utils.py`,
`models/qwen3_5.py`, `models/qwen3_5_mtp.py`, and its Qwen3VL parent.

RESULT -> FP8 plus `--quantization-param-path` requires a model
`load_kv_cache_scales` method and otherwise raises. The Qwen3.5 causal,
conditional, MTP and Qwen3VL parent classes do not define that loader.
Qwen3.5 weight loading does have attention scale-name mappings, which is a
possible deliberate data-transformation route; it is not qualification.

VERDICT -> The earlier candidate parity flags are insufficient for our
calibrated KV artifact. Implement and test a model scale-loader or a reviewed
weight transformation, then verify all attention and MTP scale values in the
live process. Do not silently fall back to unit scales. FP16-KV viability is
unaffected.

## Measured TP1 viability on the pointer/device candidate

CONFIG -> Image `sha256:1660b87ca7ab262e205132a7c23b0fd0c75aeaff91bef4ead31ddf4ad1c4f672`,
card 1, exact AutoRound GPTQ INT4 artifact, FP16 compute/KV, FP32 SSM, MTP0,
eager, prefix off, context 8192, chunk 512, concurrency cap 4. Primary alias
`hotschmoe-dd`. Compiler cache copied from the same-image first attempt;
HTTP request bound 300 seconds. This arm retains the multimodal architecture;
`--language-model-only` was removed after an unsupported architecture failure.

COMMAND -> Coordinator ran the prepared lifecycle and viability job. Raw root:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-tp1-nightly-card1-warm-retry-v4/`.

RESULT -> All 26 probe checks passed with no bangs: two identity/arithmetic
canaries, eight serial requests across one-token/two-token/arithmetic/operations
prompts, and sixteen requests in four mixed concurrent batches. `/v1/models`
identity and owned container binding were captured. Job and lifecycle exited
0, teardown completed, and the strict experimental card-1 post-health probe
passed. The immutable image's four source hashes and all 81 unchanged native
hashes were verified before this GPU arm.

VERDICT -> Measured single-card text/structure viability for this bounded
configuration. This is not exact token-output equivalence, broad semantic
quality, TP2 qualification, MTP, graphs, prefix reuse or calibrated FP8 KV
qualification. The early 90-second cold-compilation timeout is not a semantic
failure: the server eventually returned HTTP 200 and exited gracefully.
Warm-cache changes and concurrent CPU compilation preclude a matched speed
claim. A separate prefix/tool-history arm is now running; no result is claimed
here until its teardown and post-health finish.

## Interpreting `#new-token: 128` for one-token requests

CONFIG -> Installed source `fd70325c108c17b635e7caaa57fc7ab4e6e63ad9`, page
size 128, request `prompt: [17]`, no multimodal inputs.

COMMAND -> Download exact installed scheduler/batch/GDN metadata source and
execute the actual accounting, input slicing/count and metadata assignments
with CPU fakes. Raw `sglang-freshness/prefill-accounting/` contains source
hashes, the test and `result.json`.

RESULT -> `PrefillAdder._update_prefill_budget` rounds the budget to page size
before updating its logging counter, so it reports 128. Actual batch slicing
retains `[17]` and a logical token count of 1; non-speculative GDN metadata
constructs `query_start_loc=[0,1]`. The live probe responses also report
`usage.prompt_tokens=1` for all six one-token requests. Model-specific
multimodal padding is gated on non-None multimodal inputs. GDN uses EXTEND and
`has_initial_states = extend_prefix_lens > 0` for these fresh requests.

VERDICT -> The 128 log count is page-rounded accounting, not evidence that
127 real tokens were added or that the tiny-prefill test was bypassed. These
CPU/source checks do not instrument physical GPU tensor shapes; internal
kernel tiling/padding remains distinct from logical query length. Do not
claim an identical vLLM kernel path, only the same one-token client input.

## Prefix-cache configuration failure and corrected retry

CONFIG -> Same pointer/device image and TP1 viability settings, now prefix
cache enabled. First prefix arm left Mamba cache strategy at automatic.

COMMAND -> Coordinator launched `sglang-tp1-nightly-card1-prefix-tools`.

RESULT -> Weights and pools loaded, then scheduler initialization refused
`no_buffer` with page size 128. No inference qualification resulted. The
coordinator completed cleanup and strict post-health before a new arm.
Exact installed-source CPU checks explain the mismatch: with overlap disabled
and page size unresolved, automatic Mamba strategy selects `no_buffer`.
Intel attention later forces page size 128. Its supported non-MLA page sizes
are 64/128, so forcing page size 1 alone would be changed back to 128.

VERDICT -> Use the minimal explicit companion option
`--mamba-radix-cache-strategy extra_buffer` when enabling prefix cache on this
Intel-attention/Triton-GDN configuration. Exact source marks this Qwen
architecture and Triton linear backend supported, and the extra-buffer
validator permits XPU. CPU checks exercise the actual resolver, support
predicate and validator with page128/track256/chunk512; they do not substitute
for the running cache-coherence test. Raw checks:
`sglang-freshness/prefix-config/{manifest.json,check.py,result.json}`.

The coordinator launched the corrected, separately named
`sglang-tp1-nightly-card1-prefix-tools-extra-buffer` arm on port 18208. Its
result is pending here; neither prefix correctness nor stability is claimed.

## Completed bounded prefix/tool-history qualification

CONFIG -> Same immutable pointer/device image, card 1, FP16 KV, eager, MTP0,
INT4 artifact and context/chunk/concurrency limits as above. Enable prefix
cache with explicit `extra_buffer`; retain native Intel attention at page128
and Triton GDN. The source/ABI image is unchanged.

COMMAND -> Coordinator ran four concurrent histories, four turns each, with
tool-call and tool-answer checks, under the owned lifecycle. Raw root:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-tp1-nightly-card1-prefix-tools-extra-buffer/`.

RESULT -> 32/32 checks passed in 32 attempts, with zero bang attempts and no
retries. Server logs reached four running requests. Final metrics report
65536 device-cached tokens and a cache hit-rate gauge of 0.8947368421052632.
Identity was captured; job/lifecycle exited 0, owned teardown completed, and
strict card-1 post-health passed.

VERDICT -> Bounded concurrent prefix-cache/tool-history qualification passed
for this FP16-KV/eager/MTP0 configuration. Cache reuse is measured, not inferred
from a flag. This is a diagnostic harness rather than actual Pi/OMP extension
execution. It does not qualify TP2, MTP, graphs, FP8 KV, long-run stability or
matched throughput. The current-main native build remains a separate ongoing
CPU experiment; these results belong to the Sept3 source-port image only.
