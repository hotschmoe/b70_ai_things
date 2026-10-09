# Full native-HC Strata build, intake and launch recipe

CONFIG -> CPU-only source/intake audit, Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf,
patches0001..0004, selected four Unsloth UD-Q4_K_XL shards. Compiler/runtime image
39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7.
No full build, pack conversion, GPU execution or server launch performed here.

COMMAND -> Inspect the actual SYCL CMake, iq_pack.py FORM/conversion rules,
NativeDense eligibility and model call sites, PLE scalar/batch paths, native-pack
CLI checks and Python server config. Read actual backbone F32 conversion
candidates in bounded CPU buffers, excluding MTP/vision artifacts.

RESULT -> Prepared executable build/pack-audit scripts and pinned JSON plans.
The actual backbone has194 Q8 HC down/up matrices,264 F32 BF16-form tensors whose
values are all exactly representable in BF16, one native Q8 PLE key and one Q8
PLE value. Of40960 original F32 convolution values,101 change upon F16 narrowing;
maximum absolute change2.9802322387695312e-08. Raw census:
/mnt/vm_8tb/b70/results/flashnext-reset-diag-source-20261009/strata-pack-conversion-census.json.

VERDICT -> Full native-HC source can be built now. Patches0001..0004 alone do not
qualify the whole exact-source model: PLE value/convolution (and old key input
rounding) need the source-preserving PLE increment, and actual per-stage HC
ownership needs model-call validation. The parent is preparing these separately.
No inexact active compatibility copy is silently accepted as a native input.

## Build executable source

Run only the builder; it does not chain a device test or serve:

```sh
python3 strata/flash-next/build_native_hc_engine.py --jobs 4
```

It takes the parent compiler lease, clones clean Strata into a new independent
source tree, applies the four pinned patches, and obtains a complete independent
llama.cpp dependency at3cf03257f219afbe7334045ff7c6a06ac68c627d. This is the exact
SYCL CMake dependency. Existing22 header files cannot provide ggml-cpu/base
implementations or a full CMake project. Optional --ggml-source accepts a complete
clean checkout at that pin and still clones it into the independent build.
It never restores archive binaries or modifies existing checkouts.

The build container receives no DRM devices and has network disabled. Dependency
fetching is a host CPU/network step before compilation. Configure:

```sh
cmake -S /src/sycl -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=/ggml \
  -DSTRATA_SYCL_Q8_HC_PRIMITIVES=ON -DSTRATA_SYCL_PARITY=ON \
  -DSTRATA_SYCL_AOT=
cmake --build /build --target strata native_expert_parity conversation_snapshot_test -j 4
```

CMake itself forces the dependency's GGML_SYCL/CUDA/Omp off and staticCPU on;
Strata's own kernels compile as SYCL. Leave JIT/AOT and subgroup/FP flags at the
reviewed defaults. The script snapshots exact plan bytes before work and records
commands, image, patched source, dependency commit/tree, binary and CMake hashes.
Source link/executable compilation is still pending. Later PLE/mirror patches
use --plan NEW_IMMUTABLE_PLAN; do not edit a plan after a build has started.

## Explicit compatibility pack, with native override accounting

The runtime needs index.txt, dense.bin, native_experts.txt, conversions.json,
compat-bf16.json and tokenizer exports (vocab, merges, token_type, tokenizer.json,
chat_template.jinja). It does not need experts.bin, an MTP pack or a replacement
PLE artifact. Quantized experts, embedding/head and eligible dense projections
come directly from the selected GGUF. native_experts.txt v4 records per-role
shards when a layer straddles shard boundaries; keep those paths and shapes.

Upstream --native / --native-dense-gguf does not remove the packer's requirement
for --compat-bf16. NativeDense's ordinary eligible() excludes HC and PLE value;
the new HC owner attaches originals after the canonical table loads. Therefore
HC compatibility copies still exist but native routes must never consume them.
PLE value is rounded to BF16 until its separate override exists. F32 convolution
is stored as original bytes in dense.bin indexkind3 but narrowed to F16 by the
loader. No source-preserving claim follows from storing its original bytes.

Prepare only after the independent build/dependency exist:

```sh
python3 strata/flash-next/prepare_native_hc_pack.py \
  --source ISOLATED_STRATA_SOURCE --ggml-source ISOLATED_GGML_SOURCE \
  --pack NEW_PACK --prepare --receipt NEW_PACK_RECEIPT
```

This explicit CPU conversion runs the pinned packer with --compat-bf16, without
--base/--experts-bin or setup.py. It hashes every produced file, identifies all
inexact copies requiring native overrides, rejects unexpected inexact imports,
and always records full_source_fidelity_qualified=false. A successful metadata
receipt is not launch permission. The backing selected GGUF bytes are preserved.

Use an isolated Python environment with the pinned source requirements for the
packer/server; do not upgrade the host Python stack as an incidental step.
STRATA_GGUF_PY points to the exact independent dependency's gguf-py. Exported chat
schema/template and tokenizer behavior need comparison with selected GGUF token
IDs and rendering; output file hashes alone do not prove tokenizer equivalence.
Source/template receipt paths and the model lock must remain in the experiment.

## Model routes, placement and proposed command arguments

native-hc-launch-plan.json has exact common arguments, env and three placement
profiles. They are unexecuted and launch_allowed=false until the listed gates.
Use the immutable full build and pack as separate read-only container mounts.

Important native-pack CLI behavior: --spec0 is rejected. Native packs require
--spec2 or larger and --prefillCHUNK. Use --spec2 --suffix-draft0 --lookup-chain0,
omit --mtp, and verify startup reports no drafter. This is verifier capacity:
with both draft routes disabled and no MTP, it executes plain target decode.
Do not import the stock setup configuration, which installs and enables MTP.

The initial profile uses fixed prefill128,8K context,FP16KV, one active stream,
no prefix reuse/adaptive refill and no extra batch slots. Keep source graph
defaults: do not add --no-token-graph or graph-only controls. Set persistent SYCL
cache0 for the already documented image workaround. Set strict native flag1;
legacy HC_Q8/HC_Q8_INJECT/QFUSE/GR_V3 flags0 to prevent unintended fallback.
--native automatically resolves all four shards and enables native dense/head
and PLE key/postops; explicitly name the original shard2 PLE table.

One card uses physical card lease plus matching ZE_AFFINITY_MASK. The two-stage
same-device control uses --layer-split32 --split-device0; real two-card handoff
uses --layer-split32 --split-device1 and both-card lease/affinity.32 is a sizing
hypothesis, not a fit result. Account canonical weights still loaded, original
HC/PLE uploads, padding, state, head, prefill workspace and expert cache bytes.

A64GiB global mirror cap plus at least16GiB actual free host headroom is an
initial bound, not a promise that mirrors fit. Original PLE stays mmap-backed
with1048576 cached rows (at most94371840 raw row bytes, plus metadata/workspace).
Do not select full PLE RAM residency or relocate to NVMe without measurements.
VRAM reserve2GiB per stage and smaller initial prefill protect cache/state
headroom; assert measured allocation receipts rather than configured numbers.

Stock patches0001..0004 bind mirrors only for stage0. A two-stage configuration
requires every stage1 expert resident or a qualified stage-owned mirror patch.
The same-device split also needs this proof; sharing a card does not fill the
second cache or fix a single global address table. With VERIFY_NO_HOST, fail on
any unmirrored expert. Do not lower the guard to get a readable response.

The model calls currently set NativeDense::set_layer_range globals but pass no
explicit layer_lo/hi to load(). New HC owner outside()/owns_head checks use those
arguments. Wire effective stage ownership deliberately, then assert per-stage
HC tensor names/counts and only one final-mixer owner. Source-descriptor fixtures
alone cannot prove real model uploads honor that carve.

The Python service config uses model_name=hotschmoe-dd first and the detailed
quant/backend/placement/native-mode alias second. Launch through the parent
controller, validate /v1/models and eval registry, and retain normal stop/health
receipts. The config's env is supported by engine_env(); the Python service
supplies --serve itself. No service launch is chained to this recipe.

## Required device progression

Before whole-model generation: read back actual uploaded HC387 tensors and the
original PLE key/value/convolution bytes under their actual owner/queue; compare
against GGUF SHA256 receipts and descriptor bounds. Current native owner FNV
source logging is a host-input fingerprint, not a device-byte readback.

Require source-preserving scalar/prefill/verifier/write primitives first, then
one-card whole-model teacher-forced/logit/state/fresh-start/history cases. Test
same-device layer split before real two-card handoff. Cover every stage's expert
hit/miss and original PLE state, then prefix/cancellation/concurrency. No model
launch merely because the executable links or compatibility pack is complete.
