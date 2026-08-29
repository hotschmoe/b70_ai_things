# Neural.Download Qwen3.8 FP8 vLLM port ledger

Date: 2026-08-29

Status: F01 source, checkpoint, and image preflight passed. The F02 leased,
P2P-off, no-swap qualification harness is ready for its first live run. The
recipe is reproducible from tracked source on this host, but it is not safe to
run verbatim.

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

The checkpoint was downloaded to
`models/files/qwen3.8-27b/fp8-official`. Direct and ordinary verification
matched all 66 publisher weight identities, 30,866,866,928 bytes, and the
published aggregate manifest SHA256.

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

## Local deterministic MTP0 build

The pinned build completed from the dedicated root
`/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260829/build-mtp0`.
The local image is
`neural-download/vllm-openai-xpu:qwen38-fp8-collective-work-wait-r15` with ID
`sha256:dce80db0a1ad861145e88b1c565f29172641912dc75a3b50d08e370f7d58e291`.
This differs from the publisher's validated image ID because Docker rebuild
metadata is not reproducible. The publisher notes that rebuild IDs can vary
and publishes the installed `xpu.py` hash as a content check. The local gate
extends that check to all four copied files.

All four installed runtime files match the patched source checkout:

| Installed file | SHA256 |
| --- | --- |
| `vllm/_xpu_ops.py` | `f3273ccfb41be44c3c02080c26df10e8b200060366b900d940803f4221224c59` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | `5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d` |
| `vllm/model_executor/kernels/linear/scaled_mm/xpu.py` | `7c36e4a8dab4bfc06b1d5be2d8466e8cdc94099dd5409424fecc6dd8ffc2c208` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `7afb4de8b87d7f180d696f7cadad8b9d48d9ab7b706ae19616425c4f9456fb19` |

An interactive `python -` from the image working directory imports the source
tree under `/workspace/vllm`; that is not the launch path. Import tracing of
both the direct image entrypoint and the recipe's `bash -lc 'exec vllm ...'`
path showed the real console launcher importing
`/opt/venv/lib/python3.12/site-packages/vllm`, where the verified overlay files
are installed.

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

The tracked local launcher is
`vllm/fp8/serve_qwen38_fp8_neural_f02.sh`. It fails closed on the local image
ID and all four installed-file hashes, serves an unambiguous model ID, fixes
P2P and XPU Graph off, retains deterministic Inductor and the recipe's
`Work.wait()` path, and uses equal 32 GiB memory and memory-swap ceilings so
the container cannot allocate swap. The qualification wrapper
`vllm/fp8/qualify_qwen38_fp8_neural_f02.sh` adds the whole-box lease, model
direct-read verification, 96 GiB host admission gate, continuous memory/PSI
sampling, two fresh compile caches, the complete publisher 12-prompt suite,
independent canaries, raw token-array equality, graceful teardown, and card
plus compiled P2P-off collective health.

## F01 verdict

F01 passes completely. Exact source, model, image, patch, and weight identities
are present; the deterministic MTP0 overlay has been rebuilt and its effective
runtime files verified. Proceed with F02 using the tracked leased P2P-off
launcher. F02 is a local safety-port qualification, not a direct reproduction
of the publisher's P2P-on headline.
