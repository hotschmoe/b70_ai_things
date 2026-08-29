# Neural.Download Qwen3.8 FP8 vLLM port ledger

Date: 2026-08-29

Status: F01 source and identity preflight passed. The recipe is reproducible
from tracked source on this host, but it is not safe to run verbatim. Weight
fetch and deterministic image builds remain pending.

## Requested source

Package page:

`https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html`

Primary reproduction source:

`https://github.com/steveseguin/b70-optimization-lab/tree/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70`

The page labels the package `candidate-portable-repro`, not an install guide.
It explicitly lists clean-host Intel/Docker setup, clean-host endpoint replay,
and beginner recovery as missing. Its qualified strict headline is 51.918757
tok/s for a 12-prompt natural suite, not a long-agent or clean-host result.

## Exact identities

- Reproduction repository commit:
  `0948f7c2c2e21f0e8fcc444e319e5e8f5b83d0e7`.
- Sparse source checkout:
  `/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260829/source/`.
- vLLM source commit: `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`.
- Base image:
  `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- Base image is present locally with the exact image ID above.
- Published runtime: vLLM `0.27.2rc1.dev77+gac7509e2b`, PyTorch
  `2.13.0+xpu`.
- Model: `Qwen/Qwen3.8-27B-FP8` at revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Expected weights: 66 Safetensors files, 30,866,866,928 bytes, aggregate
  basename-sorted manifest
  `82fb8f84fa117c81c3e8639c4675709dfb667d70ddaa2fd097d35fc37d95453a`.

The official FP8 revision is now tracked in `models/manifest.yaml` as
`qwen3.8-27b/fp8-official`. Both the reproduction repository and Hugging Face
remote HEAD values matched the published pins on 2026-08-29.

## Source patch ledger

The deterministic MTP0 base applies three patches to the pinned vLLM source:

| Patch | Scope | SHA256 |
| --- | --- | --- |
| `vllm-qwen38-fp8-block-w8a16-20260826.patch` | Default-off block-FP8 weight plus FP16 activation dispatch in `scaled_mm/xpu.py` | `5db7f1af1156f3490ca91d0d74a07aa2d0909e175eeb1ae23f2074c55c44ff8a` |
| `vllm-qwen38-xpu-deterministic-gdn-ba-state-20260828.patch` | Deterministic GDN B/A reduction and native recurrent-state behavior | `cda7dd1e42a1e0fed2dd34f3936303cb038852a46d8d00786a1c2ebae326f8eb` |
| `vllm-qwen38-xpu-compiled-gdn-state-ccl-wait-20260828.patch` | Compiler-visible GDN state plus explicit oneCCL asynchronous work and `Work.wait()` | `8f8febcd0abc59bc9b69830827cd7607c00870414b17bd02cf32e2d879858ac8` |

The strict MTP1 overlay then applies:

| Patch | Scope | SHA256 |
| --- | --- | --- |
| `vllm-qwen38-xpu-gemma-rmsnorm-mtp1-serial-exact-r30-20260828.patch` | Replay the exact two packed MTP1 Gemma RMSNorm rows through the one-row native operator | `ff5b4f33f5596efbad75112bdbbca2bbf81b6c84688476bfa1c9ec9e546c78c4` |

The deterministic base build copies only the patched W8A16 integration,
`_xpu_ops.py`, Qwen GDN, and XPU communicator files over the pinned base image.
The MTP1 layer copies only `layernorm.py` over that qualified base. No archived
or ABI-specific local binary is an input.

## Why the published launch is not run directly

The target-only `run-server.sh` defaults to P2P off, but the published
qualified MTP0 command overrides `CCL_P2P_ACCESS=1`. The strict MTP1 launcher
hardcodes direct P2P. Current project rules prohibit an arbitrary full-model
vLLM TP2 direct-P2P serve because the queue-handoff failure remains open.

Both upstream launchers also hardcode `--memory 9g --memory-swap 12g`, allowing
up to 3 GiB of container swap. The strict MTP1 route uses 0.96 GPU memory
utilization. Those settings are inconsistent with the host-stall safeguards
established after the 2026-08-29 incident.

The local first-live port therefore changes only lifecycle and safety policy:

- run through `bin/gpu-run` with a whole-box lease;
- require at least 96 GiB MemAvailable and at most 1 GiB used swap;
- use a no-swap container ceiling and persistent host memory/PSI sampling;
- run card and compiled P2P-off collective health before and after;
- begin with graph-off W8A16 MTP0, FP16 target/KV, one request, and P2P off;
- preserve exact image, model, source, patch, compiler, GDN, and completion
  settings;
- require two-fresh-server target arrays before testing MTP1;
- keep direct P2P quarantined to a bounded loaded-context oracle.

These changes create a new local control. They do not reproduce the published
51.9 tok/s configuration and must not be compared as though P2P, MTP, graph,
and safety settings were matched.

## F01 verdict

F01 passes. Exact source, model, image, patch, and expected weight identities
are available and the base image is already installed. Proceed with the pinned
31 GB weight fetch, verify the 66-file manifest, build the deterministic MTP0
image from the dedicated source root, then implement and test the leased
P2P-off launcher before touching the GPUs.
