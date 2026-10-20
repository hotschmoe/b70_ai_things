# 1C3 -- b70_lowbit wired into vLLM for qwen3_5_moe

Plugin: `vllm/flashnext/plugins/b70_lowbit_xpu`. Quant name `b70_lowbit`.
Entry group `vllm.general_plugins`, key `b70_lowbit_xpu` =
`b70_lowbit_xpu:register`. No vLLM core edit. Serve script is
`plugins/b70_lowbit_xpu/serve_lowbit.sh`. It execs
`harness/xpu-run.sh` and does not wrap `bin/gpu-run` a second time.

Binary used for the launch microbench and the int3 serve:

```
/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so
1442856 bytes, mtime 2026-10-20 04:05 UTC
```

AOT `intel_gpu_bmg_g31`. The int4 Ornith serve earlier in the day used
the previous 773928-byte binary (4 device submits only). Those int4
numbers are not from this binary.

## What is wired

- `B70LowbitConfig` selects `b70_lowbit` only when the user passes
  `--quantization b70_lowbit` or the checkpoint quant method is already
  `b70_lowbit`. A3's INC / auto_gptq path is unchanged.
- GPTQ and AutoRound configs repack in `process_weights_after_loading`.
  Signed codes are `unsigned - maxq`, group 128, symmetric.
- Linears: QKV, GDN `in_proj_qkvz` / `in_proj_ba`, row-parallel `o_proj`
  and `out_proj`, shared-expert gate/up/down. `lm_head`, router, norms,
  conv, embeddings, vision, and MTP stay bf16. TP>1 raises on a packed-K
  row split.
- Routed experts keep `RoutedExperts.weight_loader` (`is_transposed`).
  Decode `T<=8` is `moe_gemv` with no host sync. `T>8` is one expert
  histogram per layer, then per-expert `prefill_wna16`.
- `B70_LOWBIT_MOE_LAUNCHES`: unset = 4 submits, `2` = GateUpSilu +
  DownReduce, `1` = one work-group per token. The serve used `2`.
- GEMV SLM holds `8*2560` bf16 elements. `o_proj` / `out_proj` are
  `K=4096`, so `M=6..8` takes `prefill_wna16`. `M=5` fits and was the
  coherence prompt.

## CONFIG

Int4 control (already journaled 04:02, old .so, 4 submits):

- Card 0. Ornith GPTQ int4 mixedcal-v2. Routed experts only. Trunk bf16.
- TP=1, dtype bfloat16, eager, language-model-only, max-model-len 33792,
  gpu-memory-utilization 0.90, max-num-seqs 4, port 18083.
- Names: `hotschmoe-dev` then `ornith-1.5-35b-a3b-b70-lowbit-int4`.

Launch microbench (this .so, card 1, cpu-run --quiet, 32g container
`b70-lowbit-launch`):

- Ornith decode shape T=1, top-8, E=256, H=2048, I=512, int4 random
  planes. Device event time. Match gate max abs 0.05 vs 4 submits.
- Then `tests/bench_lowbit.py` with the launch env unset.

Int3 serve (this .so, card 0, `B70_LOWBIT_MOE_LAUNCHES=2`):

- `/mnt/cache/b70/devmodel/quant/qwen36-w3a16-g128-experts-w4a16-trunk`
- Experts int3, trunk linears int4, skips bf16. Same serve flags as the
  int4 control. Alias `qwen36-w3a16-b70-lowbit`.
- vLLM `0.32.1rc1.dev94+g5fd62688f.xpu`, torch 2.14+xpu, image
  `vllm-flashnext-xpu:py312`.

A3 reference (JOURNAL 00:55): Ornith GPTQ int4, card 1, dtype float16,
XPUExpertsWNA16, eager, same context and util. Trunk bf16.

## COMMAND

```
B70_AGENT=C3 B70_GPU_LOCK_TIMEOUT=30 B70_CPU_LOCK_TIMEOUT=20 \
  bin/cpu-run --quiet harness/xpu-run.sh --card 1 --memory 32g \
  --name b70-lowbit-launch -- \
  bash plugins/b70_lowbit_xpu/tests/run_gpu_window.sh

B70_AGENT=C3 B70_GPU_LOCK_TIMEOUT=20 B70_LOWBIT_MOE_LAUNCHES=2 \
  MODEL=/mnt/cache/b70/devmodel/quant/qwen36-w3a16-g128-experts-w4a16-trunk \
  ALIAS=qwen36-w3a16-b70-lowbit CARD=0 PORT=18083 \
  plugins/b70_lowbit_xpu/serve_lowbit.sh

B70_AGENT=C3 B70_CPU_LOCK_TIMEOUT=30 \
  bin/cpu-run --quiet python3 harness/bench_latency.py \
  --base-url http://127.0.0.1:18083 --model hotschmoe-dev \
  --alias qwen36-w3a16-b70-lowbit --tag b70-lowbit-int3-eager \
  --decode-tokens 64 \
  --out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c3-int3-bench.json
```

Logs:

- `/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/launch_bench_c3.log`
- `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c3-int3-serve.log`
- `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c3-int3-bench.json`

## RESULT

Launch bench, GPU_WINDOW_RC 0, wall 122 s. `bench_lowbit.py` PASS on
pack, gemv, moe, prefill, quant config, linear apply, and MoE
decode/prefill for int2, int3, and int4.

| Submits | Ornith-shape time | Max abs vs 4 |
| --- | --- | --- |
| 4 | 143.7 us | reference |
| 2 | 128.9 us | 0 |
| 1 | 3099.6 us | 0 |

The 4-submit repeat inside the 1-submit process was 207.4 us. The
1-submit kernel is correct and about 15x slower. Serve path is 2.
About 15 us per MoE call, about 0.6 ms per token across 40 layers.

Int4 control (04:02, experts-only, 4 submits, old .so). Identity ok.
Coherence ok (Paris, identical). C1 ITL p50/p90/p99 79.55/82.15/116.59
ms, 12.98 tok/s, n=64, prompt 24, TTFT 2868 ms. C2 ITL p50 77.36 and
77.06 ms, 13.39 and 13.62 tok/s. Prefill 1017/8185/32762 tok TTFT
8279/49502/196439 ms (123/165/167 tok/s).

Int3 serve. `gpu-run` acquired card 0. `/v1/models` lists
`hotschmoe-dev` first, then `qwen36-w3a16-b70-lowbit`. Load 14.38 GiB
in 867 s. KV 478,423 tokens, 14.16x at 33792. init engine 15.07 s.
40/40 expert layers repacked at bits=3:

- w13 qweight `(256, 192, 1024)` to plane2 `(256, 1024, 512)`
- w2 qweight `(256, 48, 2048)` to plane2 `(256, 2048, 128)`

Coherence, temperature 0, prompt "The capital of France is" (5 tokens):
both completions ` Paris, a city renowned for its rich history, culture, and iconic landmarks.`
Bench verdicts OK/OK, same=True, mentions Paris. A 5-token prefill
hits GEMV on `K=4096` (`5*4096` equals the SLM cap) and returned in
about 1 s, so the trunk int4 linear path ran.

Bench, quiet CPU, BENCH_RC 0, wall 385 s:

| Case | ITL p50/p90/p99 ms | tok/s | n | TTFT ms |
| --- | --- | --- | --- | --- |
| C1 | 51.73 / 52.55 / 53.82 | 19.67 | 57 | 2412 |
| C2 stream 0 | 52.19 / 53.88 / 62.55 | 19.37 | 46 | 5268 |
| C2 stream 1 | 52.18 / 53.86 / 62.46 | 19.38 | 47 | 5320 |

Prompt tokens were 24 (C1) and 29 (C2). n is below 64 because the
completion stopped at EOS. Prefill TTFT: 1017 tok 10275 ms (99 tok/s),
8185 tok 72614 ms (113 tok/s), 32762 tok 282846 ms (116 tok/s). MTP off.

A3 eager experts-only int4, float16: C1 74.78/75.36/76.40 ms, 13.57
tok/s, n=64. C2 about 73.9 ms, 13.52 and 13.74 tok/s. Prefill
169/1038/5121 ms (6001/7888/6398 tok/s).

Teardown: `docker stop vllm-fn-lowbit-c0` exited 0. Container gone.
Port 18083 closed. The serve wrapper exited. Card 0 was then held by
`bin/gpu-run --card 0 vllm/flashnext/plugins/b70_gguf_xpu/scripts/prefill_gpu.sh`
(pid 2910396), which started `bin/xpu-health --card 0` itself. This
run did not issue a post-health probe.

## VERDICT

The plugin serves qwen3_5_moe. The int4 experts-only control is
coherent and within about 6 percent of A3's eager ITL on the same
Ornith checkpoint. The int3 checkpoint is coherent. Its decode ITL is
about 52 ms (19.7 tok/s), faster than that eager int4 baseline,
because the trunk is int4 and the experts are int3, with a 2-submit
MoE. It is not a claim that `moe_gemv` beat `XPUExpertsWNA16` on
matched weights. The matched comparison remains the int4 control.

Prefill does not. At 32k the int3 serve is about 116 tok/s against
A3's 6398 tok/s. `T>8`, including the 24-token decode prompt (TTFT
2.4 s), still launches one WNA16 GEMM per active expert, and the
trunk GEMMs are quantized as well. The histogram removed a host sync
per expert. It did not remove the launches. The 2-submit decode
fusion is about 0.6 ms/token and is not the TTFT gap.

`bench_lowbit.py` is green on this .so. The 1-submit MoE kernel stays
out of the serve path.

## Next

1. Grouped prefill: one device submit per projection for every expert
   in the layer, the way `XPUExpertsWNA16` does, then re-measure TTFT
   on this checkpoint with `B70_LOWBIT_MOE_LAUNCHES=2`.
2. When `fuser /mnt/vm_8tb/b70/gpu.lock.0` is empty and no container
   has `XPU_RUN_CARD=0`, run `bin/xpu-health --card 0`. Do not health
   a card another agent holds. Do not trust `flock -n` or
   `gpu-run --status` alone; both have reported a card free while
   fuser and a live container said otherwise.
