# Storage policy — ALL heavy data lives on the 8TB VM SSD

**Ground truth for where bytes go on the Threadripper/B70 box.** The README and HANDOFF both point
here. Rule of thumb: **nothing large ever touches the Unraid array HDDs or the docker.img** — models,
HF caches, pip caches, build caches, quant outputs, and logs all live on the single fast SSD.

## The disk

- **Device:** `/dev/sdd1`, mounted at **`/mnt/vm_8tb`** (the "8TB VM SSD", ~7.3 TB usable).
- As of 2026-06-19: **706 G used / 6.6 T available (10%)** — lots of headroom.
- This is a single SSD bind-mounted into every container, *never* the Unraid parity array (slow HDDs)
  and *never* baked into image layers.

## Canonical paths (everything under `/mnt/vm_8tb/b70/`)

| Path | Purpose |
|---|---|
| `/mnt/vm_8tb/b70/models/` | All model checkpoints (BF16 sources + our quantized outputs). |
| `/mnt/vm_8tb/b70/hf_cache/` | HuggingFace cache. Containers get `-e HF_HOME=/hf_cache` mapped here. |
| `/mnt/vm_8tb/b70/results/` | Benchmark output, quant logs (`quant27b_w8a8_*.log`), metrics. |
| `/mnt/vm_8tb/b70/pip_cache/` | pip wheel cache (`PIP_CACHE_DIR`) so re-installs in containers are fast. |
| `/mnt/vm_8tb/b70/vllm_cache/` | `XDG_CACHE_HOME` for vLLM/torch.compile caches. |
| `/mnt/vm_8tb/b70/vllm-xpu-kernels/` | Forked kernels repo (our int8 GEMM + fused quant `.so`). |
| `/mnt/vm_8tb/b70/ccache/` | `CCACHE_DIR` for fast kernel rebuilds. |
| `/mnt/vm_8tb/specula-build/models/` | A few extra source weights (e.g. `Qwen3-14B` BF16). |

## How containers are wired to it (the pattern every `scripts/*.sh` follows)

```bash
ROOT=/mnt/vm_8tb/b70
docker run --rm \
  -v "$ROOT:$ROOT" \                 # bind the SSD straight through, same path inside & out
  -e HF_HOME=/hf_cache \             # (mapped under $ROOT)
  -e XDG_CACHE_HOME="$ROOT/vllm_cache" \
  -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  --device /dev/dri -e ZE_AFFINITY_MASK=0 \   # B70 passthrough
  "$IMG" ...
```

Key points:
- Bind the SSD at the **same absolute path inside the container** (`-v $ROOT:$ROOT`) — this also keeps
  CMake/ccache happy, which pin absolute build paths (see HANDOFF "CMakeCache is path-pinned").
- Quant **outputs** are written to `$ROOT/models/<OUTNAME>` and **logs** to `$ROOT/results/` — both on
  the SSD, so they survive container teardown (`--rm`) and SSH drops.
- The big BF16 sources (e.g. `models/Qwen_Qwen3.6-27B`, 72 G) and outputs (W8A8 ≈ 33 G) all stay here;
  do **not** copy them to the dev machine.

## Don'ts

- **Don't** write to the Unraid array (`/mnt/user/...`, `/mnt/disk*`): slow HDDs, and it pollutes shares.
- **Don't** let HF/pip default to `~/.cache` inside a container (ephemeral, and can overflow `docker.img`).
- **Don't** bake weights/caches into image layers — Unraid's `docker.img` is small (we grew it to 200 G
  for the *images themselves*; data still belongs on the SSD).
