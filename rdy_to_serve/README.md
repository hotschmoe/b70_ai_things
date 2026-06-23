# rdy_to_serve -- self-contained, ready-to-serve B70 models (the GOLDEN PATH)

One directory per **verified, current-best** serve config. `cd` into a model dir and run `serve.sh` --
no hunting through JOURNAL/scripts. The golden shelf is CURATED: a model lands here only when it is the
current best AND has been verified to serve. Everything else on the host is accounted for in the status
table below with a reason -- we do NOT put broken / dead-end / image-blocked recipes on the shelf.

See `../ORGANIZATION.md` for the layout + mutability contract.

## How each model dir is built
```
  <model>/
    serve.sh    model-specific knobs + local patch mounts; sources ../_common/lib.sh ; b70_dispatch
    patches/    pure-Python patches THIS model bind-mounts at runtime (copied in, LOCAL)
    README.md   recipe + verified perf + the verified: manifest line
  _common/
    lib.sh      SHARED model-agnostic engine (docker-run builder, graph flags, health wait, probes)
```
`serve.sh` keeps everything model-specific LOCAL (image, TP, graph flags, patches); only the boring
model-agnostic plumbing is shared in `_common/`. See the SWEEP GATE below.

## [!!!] ALWAYS START WITH vLLM 0.23 -- images `vllm-xpu-env:v0230*`. NEVER llm-scaler 0.14.x.
`vllm-xpu-env:v0230` = vLLM 0.23.0+xpu, our newest/most-capable B70 stack (Triton fused-MoE on XPU,
Qwen3.6 / `Qwen3_5Moe` + Quark int8/int4 dispatch, graph capture). `:v0230moe` = `:v0230` + the baked
MoE-routing patch (own leaf tag so it cannot affect dense models). The old `intel/llm-scaler-vllm:0.14.x`
is an ANCIENT 0.14 fork with no `_moe_C` -- int8 MoE hard-fails; a multi-agent-day dead end. If a
vLLM-XPU image NEWER than 0.23.0 exists, prefer it and update this line + CLAUDE.md.

## How to use (on the GPU host: local Ubuntu box b70s4dayz, since the 2026-06-23 migration)
```bash
cd /mnt/vm_8tb/b70/rdy_to_serve/<model>
/mnt/vm_8tb/b70/gpu-run bash serve.sh            # acquire GPU lease, start, wait healthy, gen-probe
bash serve.sh stop                                # release the GPU
```
Sub-commands: `start` (default) | `stop` | `logs` | `bench` | `run` (serve+bench+stop) | `smoke`.
Common env knobs: `GRAPH` (1=capture), `TP`, `PORT`, `DEVICE` (card 0|1 for single-card), `MAXLEN`,
`MAXSEQS`, `UTIL`, `KVDTYPE`. Every GPU touch goes through `gpu-run` (one B70, several agents -- CLAUDE.md).

Two single-card models can run AT ONCE (one per card) under one lease:
```bash
/mnt/vm_8tb/b70/gpu-run bash -c '
  DEVICE=0 PORT=8001 NAME=t0 bash qwen36-27b-int4/serve.sh start
  DEVICE=1 PORT=8002 NAME=t1 bash qwen36-35b-a3b-int4/serve.sh start'
```

## THE SHELF (verified, current-best)
| dir | model | quant | cards | image | notes |
|---|---|---|---|---|---|
| `qwen36-27b-int4/` | Qwen3.6-27B | int4 AutoRound (W4A16) | 1 | `:v0230` | ~30.8 t/s -- PRIMARY quality, daily driver |
| `qwen36-35b-a3b-int4/` | Qwen3.6-35B-A3B MoE | int4 AutoRound | 1 | `:v0230moe` | ~56.8 / ~65 (fp8KV) t/s -- FASTEST |
| `qwen36-35b-a3b-quark-w8a8-int8/` | Qwen3.6-35B-A3B MoE | Quark **W8A8 INT8** | 2 (TP=2) | `:v0230` | eager 4.8; GRAPH agg ~45.7 -- true-int8 MoE |
| `qwen3-14b-w8a8/` | Qwen3-14B | **W8A8 INT8** (compressed-tensors) | 1 | `:int8g` | the int8-kernel BASELINE (XPUInt8ScaledMM) |
| `qwen3-14b-w4a8/` | Qwen3-14B | **W4A8** int4w/int8a, GPTQ, prepacked | 1 | `:int8g` | int8-activation 14B (~9.3 GiB packed) |
| `qwen36-27b-w4a8/` | Qwen3.6-27B | **W4A8** int4w/int8a, SQ+GPTQ, prepacked | 1 | `:int8g` | int8-activation 27B (prepack + GDN; SECONDARY to w4a16) |
| `qwen36-27b-w4a16/` | Qwen3.6-27B | **W4A16** compressed-tensors int4 | 1 | `:v0230` | the COMPRESSED-TENSORS 27B (parity / W4A16 research); text-only-hybrid load shim |

## NOT ON THE SHELF (every other host model, with the reason)
| host model dir | status | reason / next step |
|---|---|---|
| (`Qwen3.6-27B-W4A8` / `Qwen3-14B-W4A8` / `Qwen3-14B-W8A8`) | SHELVED | UNBLOCKED 2026-06-23 by rebuilding `:int8g` (images/int8g/) -> now on THE SHELF above. |
| `Qwen3-14B-W4A16-gptq` | UNTESTED | 14B int4-gptq; recipe not re-verified this pass. Likely `:v0230`. Promote after a smoke. |
| `Qwen_Qwen3.6-27B-FP8` | UNTESTED | B70/Xe2 has NO FP8 ALU -> needs dequant path; serve-correctness unverified. Smoke before shelving. |
| `Qwen3.6-27B-W8A8-sqgptq` | DEAD-END | dense true-int8 W8A8 serves but ~1.7 t/s (~13x slower than the `:int8` path). Kept for reference; not a serve target. |
| (`Qwen3.6-27B-W4A16`) | SHELVED | FIXED 2026-06-23 -> `qwen36-27b-w4a16/` above. (Old note "won't serve, 4304 dim" was a red herring -- the real bug was a text-only-checkpoint name-prefix mismatch; see kernel/22.) |
| `Qwen_Qwen3.6-27B` | RESEARCH | full BF16 27B (72G) -- too big for one card; TP=2/PP=2 capacity studies only. |
| `Qwen_Qwen3.6-35B-A3B` | RESEARCH | full BF16 35B MoE (67G) -- TP=2 only; reference/baseline. |
| `Qwen_Qwen3-0.6B` | DRAFT | tiny; speculative-decode draft / smoke target, not a standalone serve. |
| `google_gemma-4-12B-it` | OLD | early-bringup experiment (scripts 24-33); superseded, not a current pick. |

## [!!!] SWEEP GATE
Any change to `_common/` or `bin/` (shared infra) requires `bin/serve-sweep --smoke` GREEN across all
shelf models before commit (and `--bench` if it could move perf). Each model README carries a `verified:`
line recording the last green sweep. A break in `_common/` breaks every model -- hence the gate.
