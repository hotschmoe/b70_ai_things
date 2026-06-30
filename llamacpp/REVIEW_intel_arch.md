# llama.cpp on Intel Arc Pro B70 (dual Battlemage/Xe2) -- deployment review

Review of a fresh llama.cpp checkout at `/mnt/vm_8tb/b70/llama.cpp`
(HEAD `86b94708`, "Revert sched ... (#25138)"). Read-only investigation; no GPU
touched, nothing built. All citations are `file:line` from that checkout unless a
URL is given.

Target models (HF safetensors on disk):
- `/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/bf16` -- dense VLM,
  `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`.
- `/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-35b-a3b/bf16` -- MoE VLM,
  `architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: "qwen3_5_moe"`.

Hardware: 2x Intel Arc Pro B70 (32 GB each, Battlemage / Xe2). oneAPI 2025.3 +
icpx + Level-Zero available inside docker images, not on host.

---

## TL;DR verdict

- **Qwen3.6-27B is FULLY supported in THIS checkout** (text + vision + MTP), not
  text-only and not blocked. This is a bleeding-edge tree that carries a dedicated
  `qwen35`/`qwen35moe` architecture, a SYCL gated-delta-net (GDN) kernel, and a
  modular `conversion/` package that registers `Qwen3_5ForConditionalGeneration`
  for both the text model and the mmproj vision tower. Upstream master also merged
  Qwen3.5/3.6 + MTP (web-confirmed, June 2026).
- **35B-A3B MoE is also recognized** (`qwen35moe`, `conversion/qwen.py:625`) and
  convertible; same vision/MTP path. It is bigger and less-trodden -- treat as
  "convertible, runtime un-benchmarked here."
- **"W8A8-like" maps to Q8_0 weights only** -- llama.cpp is weight-only quant with
  fp16/fp32 activations; there is no int8-activation path. The SYCL int8 matmul
  kernels use `dpct::dp4a` (INT8 SIMD dot product), so int8 *weight* math is XMX/
  DP4A-accelerated, but "A8" does not exist.
- **TP=2 maps to `--split-mode tensor` (NOT `row`)** on SYCL. `row` is explicitly
  unsupported on SYCL; a real tensor-parallel meta-backend (`tensor`, enum value 3)
  was added with a dual-GPU ring all-reduce. **Caveat:** `tensor` split is
  arch-gated and, while it is *not refused* for `qwen35`, every other recurrent/
  hybrid arch is explicitly excluded -- so TP across the GDN recurrent state is an
  unverified risk (see section 2/7).
- **DP=2 (two single-GPU `llama-server` + nginx round-robin) is the low-risk path**
  and matches the existing daily-driver pattern. Use Q4_K_M per card.
- **B70-specific landmines:** (a) weight corruption / garbage output with
  `GGML_SYCL_F16=ON` + AOT `bmg_g21` unless `GGML_SYCL_DISABLE_OPT=1`
  (issue #21893); (b) Q8_0 historically ~4x slower than Q4_K_M on B70
  (issue #21517, partially fixed by PR #21527, ~3.1x); (c) compute-runtime 26.x
  multi-GPU known issue (#21747), called out in the Intel Dockerfile.

---

## 1. SYCL backend: build, Battlemage support, dtypes

**Build flags** (`docs/backend/SYCL.md:313-332`, `:771-800`):

```sh
source /opt/intel/oneapi/setvars.sh
# FP16 (recommended)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build --config Release -j -v
```

Build-time env/flags (`SYCL.md:771-784`, `ggml/CMakeLists.txt:256`,
`ggml/src/ggml-sycl/CMakeLists.txt:167,195-205`):

| Flag | Default | Notes |
|------|---------|-------|
| `GGML_SYCL` | (must be ON) | enables the SYCL code path |
| `GGML_SYCL_TARGET` | `INTEL` | target device type |
| `GGML_SYCL_DEVICE_ARCH` | empty (JIT) | AOT via `spir64_gen -device <arch>`. **For B70 = `bmg_g21`** (Battlemage). Optional: default build is JIT (slow first-run, then fine). |
| `GGML_SYCL_F16` | OFF | FP16 build; rebuild required to flip |
| `GGML_SYCL_GRAPH` | ON | SYCL Graph extension (runtime-disabled by default, `GGML_SYCL_DISABLE_GRAPH=1`) |
| `GGML_SYCL_DNN` | ON | use oneDNN for GEMM (else oneMKL) |
| `GGML_SYCL_SUPPORT_LEVEL_ZERO_API` | ON | L0 alloc path, reduces host RAM in multi-GPU; >4GiB alloc support |

Compilers are `icx`/`icpx` (Linux). The compiled image we already have
(`sglang-xpu:mtp`) ships oneAPI 2025.3 + icpx + L0, which is exactly what
`SYCL.md:282-288` lists as a verified oneAPI release (2025.3.3 / 2025.2.1 / 2025.1).

**Battlemage / Xe2:** Verified-devices table lists "Intel Arc B-Series ... Support |
Arc B580" (`SYCL.md:136`); B580 and B70 are the same Battlemage/Xe2 family. B70 is
not named explicitly, but B70 is in active community use (issues #21517, #21893,
#22413; a Medium write-up runs Qwen3.6-27B on B70). `SYCL.md:800` also calls out
"Intel Xe2+ GPU such as BMG or newer" for USM-system buffers. So Battlemage is a
supported, if rougher, target.

**Dtypes:** FP16 via `GGML_SYCL_F16` (perf knob; affects accuracy/speed, must be
benchmarked). KV cache default f16. BF16 weights are accepted at convert time
(`--outtype bf16`); runtime compute is fp16/fp32. INT8/INT4 are weight-only
(see section 4). `SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt`
is suggested for precision (`SYCL.md:334-336`, also set in the Intel Dockerfile).

**Reference build (`.devops/intel.Dockerfile`):** base
`intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04`, builds with
`-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DGGML_SYCL_F16=ON` (line 47), and installs
level-zero 1.28.2, IGC v2.34.4, compute-runtime 26.18.38308.1. **Note the comment
block (lines 77-91):** the older 25.40 compute-runtime is preserved in comments
because "26.x has known issue" for multiple GPUs (links #21747 and
intel/compute-runtime#921). Relevant to our dual-B70 plan.

---

## 2. Multi-GPU on SYCL: split modes, tensor parallel, peer copy

**Split-mode parsing** (`common/arg.cpp:2498-2520`) accepts `{none,layer,row,tensor}`.
Enum (`include/llama.h:194-198`):

```
LLAMA_SPLIT_MODE_NONE   = 0  // single GPU
LLAMA_SPLIT_MODE_LAYER  = 1  // split layers and KV across GPUs (pipeline-ish)
LLAMA_SPLIT_MODE_ROW    = 2  // "split layers and KV across GPUs, use tensor parallelism if supported"
LLAMA_SPLIT_MODE_TENSOR = 3  // explicit tensor parallel meta-backend
```

`--main-gpu` (`arg.cpp:2550`): the GPU for the whole model with `none`, or for
intermediate results + KV with `row`.

**On SYCL, `row` is NOT supported** -- `SYCL.md:838` Known Issues:
"`Split-mode:[row]` is not supported." The working tensor-parallel mode is the new
`--split-mode tensor` (mode 3), documented at `SYCL.md:416-424` and `:727-735`:

> `--split-mode tensor` (tensor parallelism) shards each layer across the selected
> GPUs. It requires flash attention, which is auto-enabled when `--flash-attn` is
> left at its default `auto` ... Passing `--flash-attn off` together with
> `--split-mode tensor` is rejected. The default `f16` KV cache is recommended.
> Tensor parallelism is currently optimized for 2 GPUs; other counts fall back to a
> generic all-reduce.

This is confirmed in code:
- Flash-attn auto-enable for tensor split: `src/llama-context.cpp:3517-3519`
  ("enabling flash_attn since it is required for SPLIT_MODE_TENSOR").
- Backend sampling is disabled with tensor split (falls back to CPU sampling):
  `src/llama-context.cpp:1195-1198`.
- The SYCL tensor-parallel comm backend lives at
  `ggml/src/ggml-sycl/ggml-sycl.cpp:5862-5949`:
  `ggml_backend_sycl_comm_init` / `..._allreduce_tensor` / `..._free`. For N=2 it is
  a degenerate ring all-reduce: small tensors (<32K elems) take an FP32
  memcpy+ADD path; large tensors are BF16-compressed before the cross-device copy
  (half the PCIe bytes) then decompressed+added (`:5870-5894`). Anything that is not
  N=2 / F32-or-F16 / contiguous returns false and the meta-backend uses a generic
  butterfly all-reduce (`:5959-5977`).

**Peer / P2P copy:**
- Cross-device copies go through `dev2dev_memcpy`, which checks
  `ext_oneapi_can_access_peer(... peer_access::access_supported)`
  (`ggml-sycl.cpp:639`).
- The legacy CUDA-style `ggml_sycl_set_peer_access(...)` is effectively a **no-op**
  in this tree: it is wrapped in `#ifdef NDEBUG` and its enable/disable peer calls
  are commented out (`ggml-sycl.cpp:2767-2806`). There is no `GGML_SYCL_PEER_COPY`
  env knob; the relevant compile guard is `GGML_SYCL_NO_PEER_COPY`
  (`ggml-sycl.cpp:5355,5778`) which only disables peer paths in the IPC/event code.
  So unlike vLLM there is no `CCL_TOPO_P2P_ACCESS`-style wedge surface here; the
  tensor-parallel all-reduce is bandwidth-managed in user code.
- `--split-mode layer` benefits from `ZES_ENABLE_SYSMAN=1` to query free memory
  (`SYCL.md:798`).

**Arch gate on tensor split (IMPORTANT for Qwen3.6):**
`src/llama-model.cpp:317-318` throws *"LLAMA_SPLIT_MODE_TENSOR not implemented for
architecture '...'"* when `!llm_arch_supports_sm_tensor(arch)`. That predicate
(`src/llama-arch.cpp:976`) returns `false` for an exclude-list and `true` by
default. The exclude-list is **all the recurrent/hybrid SSM arches**: `MAMBA`,
`MAMBA2`, `JAMBA`, `FALCON_H1`, `GRANITE_HYBRID`, `LFM2`, `LFM2MOE`, `NEMOTRON_H`,
`NEMOTRON_H_MOE`, `PLAMO2`, `GEMMA3N`, `KIMI_LINEAR`, plus a few dense ones
(`GROK`, `MPT`, `DEEPSEEK2/32/4`, `GLM_DSA`, `BITNET`, `T5`, `MINIMAX_M2`,
`MISTRAL4`, ...). **`QWEN35` and `QWEN35MOE` are NOT in the exclude-list**, so they
return `true` and `--split-mode tensor` is *not refused*. BUT qwen35 is itself a
hybrid recurrent GDN model (section 3); the fact that every other recurrent arch is
excluded makes qwen35's inclusion either deliberate-and-tested or an untested gap.
**This is the single biggest unknown for the W8A8/TP=2 config and must be verified
empirically before trusting it** (see section 7).

---

## 3. Qwen3.6 architecture support (the decisive question)

### What the configs say
`qwen3.6-27b/bf16/config.json`: `architectures:["Qwen3_5ForConditionalGeneration"]`,
`model_type:"qwen3_5"`; `text_config.model_type:"qwen3_5_text"` with a 64-layer
hybrid stack (`layer_types` = `linear_attention` x3 then `full_attention`, repeating;
`full_attention_interval:4`), GDN params (`linear_key_head_dim`,
`linear_num_value_heads`, `linear_conv_kernel_dim`, `mamba_ssm_dtype`),
`mtp_num_hidden_layers:1`, and a `vision_config` (`out_hidden_size:5120`,
`patch_size:16`, `spatial_merge_size:2`). The 35B
(`Qwen3_5MoeForConditionalGeneration`, `qwen3_5_moe`) adds `num_experts:256`,
`num_experts_per_tok:8`, `moe_intermediate_size:512`.

### Converter support (text)
`conversion/__init__.py` maps the architecture to the modular `conversion/` package:
- `:208 "Qwen3_5ForConditionalGeneration": "qwen"` (text)
- `:210 "Qwen3_5MoeForConditionalGeneration": "qwen"` (text MoE)

`conversion/qwen.py`:
- `:620 @ModelBase.register("Qwen3_5ForConditionalGeneration","Qwen3_5ForCausalLM")`
  -> `class Qwen3_5TextModel(_Qwen35MtpMixin, _Qwen35MRopeMixin, _LinearAttentionVReorderBase)`
- `:625 @ModelBase.register("Qwen3_5MoeForConditionalGeneration","Qwen3_5MoeForCausalLM")`
  -> `class Qwen3_5MoeTextModel(...)`
- linear-attention V-reorder, interleaved MRoPE, and MTP wiring are handled by the
  mixins (`:353,:522,:538`). MTP via `_Qwen35MtpMixin`.

### Converter support (vision / mmproj)
`conversion/__init__.py` MMPROJ map:
- `:299 "Qwen3_5ForConditionalGeneration": "qwen3vl"`
- `:300 "Qwen3_5MoeForConditionalGeneration": "qwen3vl"`

and `conversion/qwen3vl.py:16` registers
`Qwen3_5ForConditionalGeneration`/`Qwen3_5MoeForConditionalGeneration` for the
vision tower. So **vision converts via a separate `--mmproj` run** (section 6),
producing a `mmproj-*.gguf`.

### Runtime support
- Arch enums: `src/llama-arch.h:46-47` `LLM_ARCH_QWEN35`, `LLM_ARCH_QWEN35MOE`;
  names "qwen35"/"qwen35moe" (`src/llama-arch.cpp:41-42`).
- Model classes: `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`; dispatched at
  `src/llama-model.cpp:289-292`.
- `qwen35.cpp:4-35` `load_arch_hparams`: loads GDN/SSM params
  (`LLM_KV_SSM_CONV_KERNEL`, `..._INNER_SIZE`, `..._STATE_SIZE`, `..._TIME_STEP_RANK`,
  `..._GROUP_COUNT`), NextN/MTP (`LLM_KV_NEXTN_PREDICT_LAYERS`), marks recurrent
  (linear-attention) layers vs full-attention every 4th, and maps `n_layer==64` ->
  `LLM_TYPE_27B`.
- Recurrent classification: `llm_arch_is_recurrent` includes `QWEN35`/`QWEN35MOE`
  (`src/llama-arch.cpp:920-947`); runtime uses `llama_memory_recurrent`
  (`src/llama-model.cpp:2060-2107`) for the GDN state.
- **GDN has a real SYCL kernel:** `ggml/src/ggml-sycl/gated_delta_net.cpp` (plus
  `ssm_scan.cpp`, `gla.cpp`, `wkv.cpp`, `cumsum.cpp`). So the hybrid linear-attention
  path is implemented for the SYCL backend, not just CUDA.

### Web corroboration (June 2026)
- Upstream llama.cpp merged Qwen3.5/3.6 conversion + runtime, and MTP was merged to
  master (the serve-time MTP flag is `--mtp` / `--spec-type draft-mtp`).
- A community write-up runs **Qwen3.6-27B on an Arc Pro B70** end-to-end with
  llama.cpp SYCL using a prebuilt **Q4_K_M** GGUF + auto-loaded **mmproj vision**,
  `--jinja`, single GPU (`--device SYCL0 -ngl 999`). Vision reported working.

**Verdict: 27B dense VLM = FULL support (text + vision + MTP) in this checkout.**
**35B MoE VLM = recognized + convertible (text + vision + MTP); runtime feasible but
heavier and unverified here.** Neither is "text-only" or "blocked."

---

## 4. Quantization mapping (W8A8 / W4A16 analogs)

Available types (`tools/quantize/quantize.cpp`): `Q8_0` (`:68`, ~8 bpw), `Q4_0`
(`:36`), `Q4_K_S`/`Q4_K_M` (`:62-63`), `IQ4_XS` (`:60`, 4.25 bpw). `Q4_K` is an alias
for `Q4_K_M` (`:61`).

| Our scheme | llama.cpp analog | Honest mapping |
|------------|------------------|----------------|
| W8A8 (int8 weights + int8 activations) | **Q8_0** | Q8_0 is ~8-bit *weights only*. **Activations stay fp16/fp32** -- llama.cpp has no int8-activation path. So "A8" does not transfer; this is W8A16-ish, not W8A8. |
| W4A16 (int4 weights, fp16 act) | **Q4_K_M** (or `Q4_0`/`IQ4_XS`) | Good analog: ~4-bit weights, fp16 compute. Q4_K_M is the community default and the B70-validated choice. |

**SYCL int8/int4 math:** the quantized matmul kernels do use INT8 DP4A --
`ggml/src/ggml-sycl/mmq.cpp` calls `dpct::dp4a(...)` ("SIMD dot product") for the
Q8_0/Q4_x mul-mat tiles (`mmq.cpp:578,733,876,1021,...`), and `mmvq.cpp` holds the
vec-dot quantized path. `SYCL.md` News (2026.04-05) adds a "mul_mat by reorder"
optimization specifically for `Q4_K, Q5_K, Q6_K, Q8_0`, plus fused MoE. So int8
*weight* dequant/matmul is XMX/DP4A-accelerated; there is just no int8 *activation*
quantization to leverage B70's INT8 XMX the way our sglang W8A8 kernels do.

**B70 perf caveat:** issue #21517 reports Q8_0 ~4x slower than Q4_K_M on B70
(Xe2/Battlemage), a kernel-efficiency problem; PR #21527 reportedly gave a ~3.1x
Q8_0 token-gen speedup. This checkout is newer than both, so it likely contains the
fix -- but Q8_0-vs-Q4_K_M on B70 must be benchmarked, not assumed.

---

## 5. Server (`llama-server`) -- mimicking the daily-driver

Everything we need exists:
- OpenAI-compatible routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
  `/models`, `/health`, `/v1/health` (`tools/server/server.cpp:205-211`).
- **Prometheus metrics:** `/metrics` route (`server.cpp:207`); enabled by `--metrics`
  (`common/arg.cpp:3169-3174`, env `LLAMA_ARG_ENDPOINT_METRICS`).
- **API key:** `--api-key KEY` (comma-separated multi-key) and `--api-key-file`
  (`arg.cpp:3073-3099`, env `LLAMA_API_KEY`). `/health` and `/models` are public; the
  rest are key-gated.
- **Concurrency:** `--parallel N` (`arg.cpp:2279`), `-cb/--cont-batching` and
  `-nocb` (`arg.cpp:2305-2311`, default-on env `LLAMA_ARG_CONT_BATCHING`).
- **Tool-calling / chat templates:** `--jinja` enables jinja chat templates; tool
  calls are parsed/emitted in `tools/server/server-chat.cpp:167-201`
  (`function_call`, `tool_calls`, `function_call_output`).
- **Vision at serve time:** the server links mtmd
  (`tools/server/server-context.cpp:17-18,700-755`, `process_mtmd_chunk`,
  `mtmd_helper_decode_image_chunk`) and takes `--mmproj FILE` (`arg.cpp:2313-2319`).
- **MTP / speculative at serve time:** `--mtp` pushes
  `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` (`arg.cpp:2789-2792`); generic form
  `--spec-type draft-mtp` (`arg.cpp:3769`). The server auto-downloads the MTP draft
  when that spec type is set (`arg.cpp:359-374`).

Daily-driver parity command sketch (single card / DP member):
```sh
ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR="level_zero:0" \
./build/bin/llama-server \
  -m /models/qwen3.6-27b-Q4_K_M.gguf \
  --mmproj /models/mmproj-qwen3.6-27b-f16.gguf \
  -ngl 999 --ctx-size 131072 \
  --split-mode none --main-gpu 0 \
  --parallel 4 --cont-batching \
  --jinja --metrics \
  --api-key "$B70_API_KEY" \
  --host 0.0.0.0 --port 18080
```

---

## 6. GGUF conversion workflow

The converter is the thin `convert_hf_to_gguf.py` dispatching into the modular
`conversion/` package (`convert_hf_to_gguf.py:236-270`).

**Text model (with bundled MTP by default):**
```sh
python convert_hf_to_gguf.py \
  /mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/bf16 \
  --outtype bf16 \
  --outfile /models/qwen3.6-27b-bf16.gguf
```
- `--mtp` exports *only* the MTP head as a separate `mtp-*.gguf` draft;
  `--no-mtp` excludes it (`convert_hf_to_gguf.py:121-126`). The mixin gate restricts
  `--mtp/--no-mtp` to Qwen3.5/3.6 + Step3.5 text variants (`:257-265`). Default
  (neither flag) bundles MTP into the main file.

**Vision tower (separate run, required for VLM):**
```sh
python convert_hf_to_gguf.py \
  /mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/bf16 \
  --mmproj \
  --outfile /models/mmproj-qwen3.6-27b-f16.gguf
```
- `--mmproj` exports the multimodal projector with an `mmproj-` prefix
  (`convert_hf_to_gguf.py:117-118`). This is a distinct artifact from the text GGUF
  and is passed to `llama-server --mmproj`.

**Quantize to the serve target:**
```sh
# W4A16-like (recommended, B70-validated)
./build/bin/llama-quantize /models/qwen3.6-27b-bf16.gguf /models/qwen3.6-27b-Q4_K_M.gguf Q4_K_M
# W8A8-like (weight-only Q8_0; benchmark vs Q4_K_M on B70)
./build/bin/llama-quantize /models/qwen3.6-27b-bf16.gguf /models/qwen3.6-27b-Q8_0.gguf  Q8_0
```
(`tools/quantize/quantize.cpp` also supports per-tensor overrides via
`--tensor-type`, e.g. `attn_q=q8_0`, `:148`.) The mmproj file is normally kept at
f16 and not quantized.

---

## 7. Stock-vs-patch: what works out of the box, what is a risk

### Works stock (no source patch expected)
- SYCL build with `icx`/`icpx` + `GGML_SYCL_F16=ON` (per Intel Dockerfile, which is
  the reference recipe).
- Qwen3.6-27B convert (text + MTP + mmproj) and Qwen3.6-35B-A3B convert -- arch and
  conversion are present in-tree.
- Single-GPU serve of Q4_K_M (+ mmproj) with OpenAI API, `--api-key`, `--metrics`,
  `--parallel`, `--cont-batching`, `--jinja` -- all flags exist.
- **DP=2** (two single-GPU servers, each `--split-mode none --main-gpu N` /
  `ONEAPI_DEVICE_SELECTOR=level_zero:N`, nginx round-robin) -- the existing
  daily-driver pattern, lowest risk.

### Needs care / flags (not patches)
- **B70 corruption bug (#21893):** `GGML_SYCL_F16=ON` + AOT `GGML_SYCL_DEVICE_ARCH=
  bmg_g21` produced "hallucinated fragments / repetitive tokens" on B70 until
  `GGML_SYCL_DISABLE_OPT=1` was set (root cause suspected in the Xe2 reorder/DPAS
  optimized kernels). The corruption fix is reorder-related and may interact with the
  new "mul_mat by reorder" Q8_0/Q4_K optimization. **Mitigation:** start with a JIT
  build (no `GGML_SYCL_DEVICE_ARCH`), validate output, and if you go AOT for B70 keep
  `GGML_SYCL_DISABLE_OPT=1` in your back pocket. Build is out of scope for this
  review -- flag for the build session.
- **Q8_0 throughput on B70 (#21517 / PR #21527):** historically ~4x slower than
  Q4_K_M; partially fixed. The "W8A8-like" Q8_0 config may be slower than the
  "W4A16-like" Q4_K_M config -- the opposite of our sglang result where W8A8 fused
  kernels win. Benchmark Q8_0 vs Q4_K_M on B70 before choosing.
- **compute-runtime 26.x multi-GPU (#21747):** the Intel Dockerfile keeps 25.40 in
  comments because 26.x has a known multi-GPU issue. If TP=2 misbehaves, try pinning
  the older compute-runtime.

### Real risks / unknowns to verify
1. **TP=2 (`--split-mode tensor`) correctness on the GDN recurrent state.** qwen35 is
   *not* refused by the arch gate (`llama-arch.cpp:976`), yet every other
   recurrent/hybrid arch (Mamba, Jamba, Falcon-H1, Granite-Hybrid, LFM2, Nemotron-H,
   Kimi-Linear, ...) *is* excluded. Whether the SYCL dual-GPU all-reduce + recurrent
   memory produce coherent output for qwen35 is **unverified** and is the make-or-break
   for the W8A8-like config. Test for "!!!!"-style garbage under concurrent load,
   exactly like the sglang W8A8 warmup-poisoning we already hit.
2. **Backend sampling is forced to CPU with tensor split** (`llama-context.cpp:1195-
   1198`) -- a possible throughput cost at TP=2.
3. **`--split-mode row` is unsupported on SYCL** (`SYCL.md:838`). Do not use it; the
   TP analog is `tensor`. `layer` is the safe pipeline split if `tensor` proves
   incoherent on the GDN model.
4. **MTP draft + tensor split + recurrent** is a triple stack none of which is
   individually battle-tested together on Xe2 here. Bring up text-only, single-GPU,
   no-MTP first; add vision, then MTP, then multi-GPU, one axis at a time.
5. **35B-A3B MoE** adds fused-MoE SYCL kernels (new, 2026.04-05) on top of the GDN +
   vision + MTP stack -- treat as a second-phase target after 27B is solid.

### Recommended bring-up order for Qwen3.6-27B
1. JIT SYCL build (Intel Dockerfile recipe), `GGML_SYCL_F16=ON`.
2. Convert: text bf16 GGUF (MTP bundled) + `--mmproj` f16; quantize Q4_K_M.
3. Single-GPU serve (`--split-mode none --main-gpu 0`), text-only, validate coherence.
4. Add `--mmproj` (vision), then `--mtp` (speculative), re-validate.
5. **DP=2** = two such servers + nginx (matches daily-driver; low risk) -- this is the
   "W4A16-like TP=1, DP=2" target and should be the production default.
6. Only then attempt **TP=2** via `--split-mode tensor` for the "W8A8-like" target;
   gate on a concurrent-load coherence sweep before trusting it. Benchmark Q8_0 vs
   Q4_K_M; expect Q4_K_M to likely win on B70.

---

## Key file:line index

- SYCL build flags / env: `docs/backend/SYCL.md:313-336,771-800`
- Battlemage in verified table: `docs/backend/SYCL.md:136`; USM/Xe2: `:800`
- `row` unsupported on SYCL: `docs/backend/SYCL.md:838`
- `--split-mode tensor` semantics: `docs/backend/SYCL.md:416-424,727-735`
- Intel Dockerfile build + 26.x multi-GPU note: `.devops/intel.Dockerfile:41-48,77-91`
- split-mode parse / enum: `common/arg.cpp:2498-2520`; `include/llama.h:194-198`
- SYCL tensor-parallel all-reduce: `ggml/src/ggml-sycl/ggml-sycl.cpp:5862-5949`
- peer-access no-op / dev2dev: `ggml-sycl.cpp:2767-2806,639`
- tensor-split flash-attn auto / CPU-sampling: `src/llama-context.cpp:3517-3519,1195-1198`
- tensor-split arch gate + allow-list: `src/llama-model.cpp:317-318`; `src/llama-arch.cpp:976`
- qwen35 arch enums / dispatch / hparams: `src/llama-arch.h:46-47`, `src/llama-arch.cpp:41-42`, `src/llama-model.cpp:289-292`, `src/models/qwen35.cpp:4-35`
- GDN SYCL kernel: `ggml/src/ggml-sycl/gated_delta_net.cpp` (+ `ssm_scan.cpp`,`gla.cpp`,`wkv.cpp`)
- INT8 DP4A matmul: `ggml/src/ggml-sycl/mmq.cpp:578,733,876,1021` (`dpct::dp4a`)
- converter text/mmproj registration: `conversion/__init__.py:208,210,299,300`; `conversion/qwen.py:620,625`; `conversion/qwen3vl.py:16`
- convert flags `--mmproj/--mtp/--no-mtp`: `convert_hf_to_gguf.py:117-126,257-270`
- quant types: `tools/quantize/quantize.cpp:36,60-63,68`
- server routes / metrics / api-key / parallel / mmproj / mtp: `tools/server/server.cpp:205-211`; `common/arg.cpp:3169-3174,3073-3099,2279,2305-2311,2313-2319,2789-2792,3769`

## External references
- B70 weight corruption / `GGML_SYCL_DISABLE_OPT=1`: github.com/ggml-org/llama.cpp/issues/21893
- Q8_0 4x slower than Q4_K_M on B70 (+PR #21527): github.com/ggml-org/llama.cpp/issues/21517
- compute-runtime 26.x multi-GPU: github.com/ggml-org/llama.cpp/issues/21747
- B70 SYCL perf: github.com/ggml-org/llama.cpp/issues/22413; Discussion #12570
- Qwen3.6-27B on B70 community recipe (Q4_K_M + mmproj + --jinja): bibek-poudel.medium.com (How to Run Qwen3.6-27B Locally on Intel Arc Pro B70)
