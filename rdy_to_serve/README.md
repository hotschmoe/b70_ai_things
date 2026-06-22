# rdy_to_serve -- self-contained, ready-to-serve B70 models

One directory per model. Each is **self-contained**: a `serve.sh` plus any `patches/` it needs to bring
the model up on the Intel Arc Pro B70 box. No hunting through JOURNAL/scripts -- `cd` into a model dir and
run `serve.sh`.

## [!!!] ALWAYS START WITH vLLM 0.23 -- image `vllm-xpu-env:v0230`. NEVER llm-scaler 0.14.x.
`vllm-xpu-env:v0230` = **vLLM 0.23.0+xpu** is our newest, most-capable B70 stack (Triton fused-MoE on XPU,
current Qwen3.6 / `Qwen3_5Moe` + Quark int8/int4 dispatch, graph capture). The old
`intel/llm-scaler-vllm:0.14.x` image is an **ANCIENT 0.14 vLLM fork** with no `_moe_C` MoE op suite --
int8 MoE hard-fails on it, and it has burned multiple agent-days as a dead end (docs/kernel/20 sec 6-9).
**Newest-first preference: v0230 (0.23.0) > `:tf` (0.20.2rc1) > 0.14.x.** If a vLLM-XPU image NEWER than
0.23.0 exists, prefer it and update this line (and CLAUDE.md). Every `serve.sh` here defaults to v0230.

## How to use
These run ON THE GPU HOST (Unraid @ 192.168.10.5), where the models, the `vllm-xpu-env:*` images, `gpu-run`
and `35_sweep_bench.sh` live (under `/mnt/vm_8tb/b70`). Sync this dir to the host, then:

```bash
ssh root@192.168.10.5
cd /mnt/vm_8tb/b70/rdy_to_serve/<model>          # (or wherever you synced it)
/mnt/vm_8tb/b70/gpu-run bash serve.sh            # acquire GPU lease, start, wait healthy, gen-probe
bash serve.sh stop                                # release the GPU
```
Common knobs (env): `TP`, `PORT`, `MAXLEN`, `MAXSEQS`, `UTIL`, `GRAPH=1` (graph-capture decode lever).
Every GPU touch must go through `gpu-run` (one B70, possibly several agents) -- see CLAUDE.md.

## Models
| dir | model | quant | cards | image | status |
|---|---|---|---|---|---|
| `qwen36-35b-a3b-quark-w8a8-int8/` | Qwen3.6-35B-A3B (MoE) | Quark **W8A8 INT8** | 2x B70 TP=2 | v0230 | WORKING |

(Add new models as sibling dirs. Keep each self-contained: serve.sh + patches/ + a short README.)
