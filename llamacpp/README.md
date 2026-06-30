# llamacpp/ -- llama.cpp (SYCL) backend for the dual B70

Third serving backend, added 2026-06-30. llama.cpp's SYCL/GGML path runs on Intel Arc Pro B70
(Battlemage/Xe2) via oneAPI + Level-Zero. Unlike vLLM/sglang (torch + compressed-tensors + our oneDNN
int8 kernels), llama.cpp is C++/GGML with **weight-only GGUF quant and fp16/fp32 compute** -- there is
no int8-activation path, so it does not consume any of our W8A8/W4A8/W4A16 compressed-tensors artifacts
or our custom kernels. It has its own server, quant formats, and multi-GPU model.

**Headline: this checkout has FULL dedicated Qwen3.6 support** (text + vision + MTP). The fresh HEAD
carries a `qwen35`/`qwen35moe` architecture with a real SYCL gated-delta-net (GDN) kernel, an mmproj
vision tower (`qwen3vl`), and bundled MTP. So llama.cpp can genuinely serve our headline 27B VLM --
a real daily-driver-class alternative. Full evidence + file:line citations in `REVIEW_intel_arch.md`.

## Layout

- `REVIEW_intel_arch.md` -- the deep Intel-Arc/SYCL + Qwen3.6 arch-support review (read this first).
- `build_sycl.sh`        -- build llama.cpp SYCL inside the oneAPI image (no host toolchain). GPU-free.
- `convert_gguf.sh`      -- HF bf16 -> f16 GGUF (+ mmproj vision) -> quantize Q8_0 + Q4_K_M. GPU-free.
- `serve_dp2_q4km.sh`    -- "W4A16-like, TP=1 DP=2": Q4_K_M, one server per card + nginx. LOW RISK / prod default.
- `serve_tp2_q8.sh`      -- "W8A8-like, TP=2 DP=1": Q8_0 via `--split-mode tensor`. HIGHER RISK (GDN+TP).

Upstream source clone (git-ignored runtime, NOT repo content): `/mnt/vm_8tb/b70/llama.cpp` (HEAD 86b94708).
GGUF artifacts (git-ignored, `*.gguf`): `/mnt/vm_8tb/b70/llamacpp/gguf/`.

## Quant -> config mapping (honest)

| Our scheme        | llama.cpp     | Multi-GPU                          | Notes |
|-------------------|---------------|------------------------------------|-------|
| W8A8 (tp=2, dp=1) | **Q8_0**      | `--split-mode tensor` (both cards) | Q8_0 = 8-bit WEIGHTS only, fp16 act (~W8A16, not W8A8). TP across GDN recurrent state is UNVERIFIED -- coherence-gate. Q8_0 historically ~4x slower than Q4_K_M on B70 (#21517). |
| W4A16 (tp=1, dp=2)| **Q4_K_M**    | 2x single-card server + nginx      | 4-bit weights, fp16 compute. Community B70-validated; expected production default. |

## Quick start (GPU-free prep is already done by the night run)

```sh
# 1. build (GPU-free; inside sglang-xpu:mtp). Artifact: /mnt/vm_8tb/b70/llama.cpp/build/bin
bash llamacpp/build_sycl.sh

# 2. convert + quantize (GPU-free). Artifacts: /mnt/vm_8tb/b70/llamacpp/gguf/*.gguf
bash llamacpp/convert_gguf.sh

# 3. serve (NEEDS the GPU lease -- both cards). Run when the GPUs are idle.
./bin/gpu-run bash llamacpp/serve_dp2_q4km.sh start   # low-risk DP=2 Q4_K_M
#   or
./bin/gpu-run bash llamacpp/serve_tp2_q8.sh start     # TP=2 Q8_0 (coherence-gated; may fall back to DP=2)
bash llamacpp/serve_dp2_q4km.sh stop                  # release
```

## Build environment

No native oneAPI on the host. We build + serve INSIDE `sglang-xpu:mtp`, which ships oneAPI 2025.3 +
icx/icpx + Level-Zero + oneMKL (`libmkl_sycl`) + oneDNN + cmake + ninja (verified 2026-06-30). The
binaries are ABI-locked to that image's oneAPI libs, so we serve from the same image (LD_LIBRARY_PATH
prepends the compiler libs + `build/bin`). Build is JIT/SPIR-V (portable, no AOT arch guess) -- see
`build_sycl.sh` for the F16 / AOT (`bmg_g21`) perf knobs and the #21893 corruption caveat.

## Bring-up order (from REVIEW sec 7) and status

1. SYCL build (JIT). -- DONE 2026-06-30 (build_sycl.sh, rc=0; llama-{server,quantize,mtmd-cli} built).
   NOTE: the cli was renamed `llama-cli` -> `llama-completion` in this HEAD; the server is `llama-server`.
2. Convert text+MTP+mmproj, quantize Q4_K_M + Q8_0. -- DONE (convert_gguf.sh). GGUF verified: arch=qwen35,
   65 blocks (64 + MTP head), GDN ssm tensors, 27-layer clip vision; CPU coherence probe = "...is Paris."
3. DP=2 serve Q4_K_M, validate coherence. -- DONE + PASS 2026-06-30 (GPU): replica0 ~95s cold, COHERENCE
   OK, both replicas serve the alias, clean teardown, box HEALTHY. (Needed the --served-model-name->--alias fix.)
4. TP=2 `--split-mode tensor` Q8_0. -- DONE + PASS 2026-06-30 (GPU): /health ~160s, vision mmproj loaded,
   COHERENCE OK, decode ~14 t/s, clean teardown, NO wedge. The GDN+tensor-split coherence unknown resolved
   POSITIVELY. (Q8_0 ~14 t/s < sglang W8A8 fused+MTP ~25 -- llama.cpp is weight-only, no fused int8 act/MTP.)

Server parity with the daily driver is complete: OpenAI `/v1/*`, `--api-key`, Prometheus `--metrics`,
`--parallel`/`--cont-batching`, `--jinja` tool-calling, `--mmproj` vision, `--mtp` speculative.

## Promotion to the shelf

Per the repo shelf rules, these bring-up scripts stay under `llamacpp/` until a config is MEASURED
faster-or-equal AND coherent under concurrent load on the GPU. Only then does a verified config get
promoted to `rdy_to_serve/llamacpp/<model-quant>/serve.sh`.
