# D2 in-graph decode time and Shisa MTP

Agent D2. Card 1 only, shared with E1 through the lease. Card 0 was
left alone. Stack matches D1: vLLM `0.32.1rc1.dev94+g5fd62688f` in
`/mnt/vm_8tb/b70/build/vllm-flashnext/venv`, torch 2.14.0+xpu, image
`vllm-flashnext-xpu:py312`. Model Ornith-1.5-35B-A3B GPTQ int4
mixedcal-v2, TP=1, dtype float16, language-model-only. Served name
`hotschmoe-dev`, alias `ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2`.
Roofline used below is 608 GB/s. Latency harness is
`vllm/flashnext/harness/bench_latency.py` (decode 64, temperature 0,
ITL is the SSE inter-chunk gap). Logs and JSON are under
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/`. ASCII only.

The live worker is the V2 runner,
`vllm.v1.worker.gpu.model_runner.GPUModelRunner` (log line
"Using V2 Model Runner"). Overlays that wrap `execute_model` or
`load_model` have to patch that class. The old
`vllm.v1.worker.gpu_model_runner.GPUModelRunner` is also patched so a
future runner switch still hits the hook.

Claude consult: `claude --bare -p --model fable-5.1` and
`--model claude-5.5` (CLI 2.1.304) both exited `unrecognized_model`
and "Not logged in". No second-model review.

## 1. Answer

The 11.4 ms graph is already the sum of the kernels. At T=1 the
non-overlapping device time is 11.3 ms:

| class | layers | one layer, ms | scaled, ms | notes |
| --- | --- | --- | --- | --- |
| GDN module | 30 | 0.160 | 4.79 | includes qkvz, ba, out, and the core |
| MoE runner | 40 | 0.123 | 4.91 | includes routed int4, shared expert, topk |
| full attention | 10 | 0.143 | 1.43 | qkv 0.069 + core 0.038 + o 0.035, short KV |
| norms | 80 | | 0.18 | one graph of all 80, from the GEMV floor |
| sum | | | 11.3 | D1 full-graph replay was 11.4 ms |

The fp16 trunk is not the miss. A stack of the distinct projection
weights plus lm_head is 3.611 GiB and replays in 7.153 ms at 542 GB/s
(89% of 608). The 1.5-2 GB / ~3 ms figure is the int4 active-weight
estimate. It leaves out the fp16 GDN and attention projections and the
fp16 lm_head.

The class far from the roof is routed MoE. Active bytes are 13.3 MB
per layer (stored expert weights times 8/256), 0.53 GB across 40
layers. At 608 GB/s that is 0.9 ms. The graph measures 4.91 ms, 108
GB/s, 18% of the roof. `b70_lowbit` moe_gemv on this shape was about
0.129 ms per layer (report 1C3) against this graph's 0.123 ms, so
swapping it does not win. int4 GEMV on the fp16 trunk is slower than
oneDNN fp16 on the dominant GDN projection (0.169 ms vs 0.093 ms) and
was not served.

Shisa MTP does lower effective per-token latency. Native spec=2 stays
the D1 row: C1 ITL 18.81 ms, 93.88 tok/s, acceptance 0.383, 10.65
ms/token. The same graph with the Shisa head at spec=2 is C1 ITL 18.72
ms, 121.73 tok/s, acceptance 0.665, 8.22 ms/token. The no-MTP baseline
is 12.81 ms/token. Spec=1 accepts more often (0.775) and is slower
(111.90 tok/s, 8.94 ms/token) because it has only one draft slot.
Spec=3's first serve timed out after 600 s while E1 held card 1. A
rerun with an 1800 s lease wait is in flight; this file will be
updated when that bench lands.

## 2. GEMV floor

CONFIG: card 1, graph replay, no server. Script
`harness/d2_gemv_floor.py`. Log `d2-gemv-floor.log`. int4 is
`b70_lowbit` gemv, bits=4, RTN, group 128. Each int4 replay matched
its own dequant (max_abs 0), including the graph replay. E1's int2
capture failure (max_abs about 45) did not reproduce on these int4
shapes.

COMMAND: `bin/gpu-run --card 1` around that script. Container
`vllm-fn-d2-gemv`, removed after exit 0.

M=1 fp16 oneDNN, per call:

| weight | ms | GB/s | roof % |
| --- | --- | --- | --- |
| gdn qkvz [12288, 2048] | 0.0931 | 540 | 89 |
| gdn ba | 0.0055 | 48 | 8 |
| gdn out [2048, 4096] | 0.0363 | 463 | 76 |
| attn qkv [9216, 2048] | 0.0708 | 533 | 88 |
| attn o | 0.0367 | 457 | 75 |
| shared gate+up | 0.0070 | 602 | 99 |
| shared down | 0.0049 | 431 | 71 |
| router | 0.0049 | 216 | 36 |
| lm_head [248320, 2048] | 1.707 | 596 | 98 |

Distinct-weight stack, 231 calls, 3.611 GiB: M=1 7.153 ms at 542
GB/s; M=4 7.194 ms at 539 GB/s. Eighty Gemma norms in one graph were
0.183 ms. Isolated tiny graphs (router, ba, a single norm) sit near
0.03 ms and are launch-bound; scaling those by 40 or 80 overstates
them. The stack number is the one to use.

int4 graph replay, M=1:

| weight | ms | GB/s | fp16 ms |
| --- | --- | --- | --- |
| gdn qkvz | 0.1689 | 77 | 0.0931 |
| gdn out | 0.0655 | 66 | 0.0363 |
| attn qkv | 0.0544 | 179 | 0.0708 |
| lm_head | 1.1909 | 220 | 1.707 |

Triton fp16 qkvz best was 0.180 ms at 280 GB/s, max_abs 1.6e-2.
Triton lm_head best was 1.760 ms. Both lose to oneDNN and are
numerically worse.

VERDICT: `B70_D2_INT4_LINEAR` stays off. The overlay exists at
`harness/patches/d2_int4_linear.py` and is not enabled. An int4 lm_head
would save about 0.5 ms and is the only trunk GEMV where int4 won;
it is not worth the coherence risk against the MoE gap.

## 3. Eager op census and region replay

CONFIG: `GRAPH=0 ATTN=FLASH_ATTN MTP=0 MAX_LEN=4096 PORT=18081`,
`B70_D2_PROFILE=1`. Four short streams, first sight of T=1,2,3,4.
Prompt is one sentence, so full-attention KV is tens of tokens. This
is the short-context decode the 11.4 ms sample is, not a 200k
context. Log `d2-profile.log`. The T=1/T=2 capture from the first
hook (context not restored) is `d2-profile-t12.log`; those context
ops failed and are not in the table below.

Replay calls the module again under the `ForwardContext` saved during
the real step (`override_forward_context`). MoE, GDN, and attention
raise "Forward context is not set" if that restore is skipped. Each
region is an XPU graph replay, same method as D1's 11.4 ms sample.
`compute_logits` did not run on the profiled worker step (V2 logits
are not `ParallelLMHead.forward`), so lm_head is the floor number,
1.707 ms, and is outside this sum. The 11.3 ms sum matching 11.4 ms
says the D1 replay is the model forward without that GEMM.

Op counts are identical at T=1 and T=4 (1005 calls):

| class | calls | op |
| --- | --- | --- |
| linear | 110 | aten.linear (30*3 GDN projs + 10*2 attn projs) |
| norm | 81 | 80 xpu_fused_add_gemma_rms_norm + 1 xpu_gemma_rms_norm |
| moe | 40 | vllm.moe_forward_shared |
| gdn core | 30 | vllm.gdn_attention_core_xpu |
| full attn | 10 | vllm.unified_attention_with_output |
| kv update | 10 | vllm.unified_kv_cache_update |
| embed | 1 | aten.embedding |
| glue | 492 | view 130, reshape 90, empty_like 70, empty 64, add 43, slice 42 |
| other | 231 | detach 95, zeros 32, flatten 30, plus a long tail |

One-layer graph times, ms:

| region | T=1 | T=2 | T=3 | T=4 |
| --- | --- | --- | --- | --- |
| gdn module | 0.160 | 0.171 | 0.183 | 0.194 |
| gdn qkvz alone | 0.100 | 0.099 | 0.098 | 0.098 |
| moe runner | 0.123 | 0.136 | 0.171 | 0.186 |
| attn qkv | 0.069 | 0.069 | 0.068 | 0.068 |
| attn core | 0.038 | 0.035 | 0.035 | 0.036 |
| attn o | 0.035 | 0.037 | 0.035 | 0.034 |

Scaled the way section 1 scales T=1 (GDN module, not the three
projections on top of it; MoE runner, not shared expert and router
again):

| T | step ms | ms/token |
| --- | --- | --- |
| 1 | 11.3 | 11.3 |
| 2 | 12.2 | 6.1 |
| 3 | 13.9 | 4.6 |
| 4 | 14.8 | 3.7 |

Qkv stays on the roof as T grows (0.069 to 0.068 ms, about 90% of
608). MoE grows 0.123 to 0.186 ms from T=1 to T=4, which is 1.5x time
for 4x tokens. Per-token MoE cost falls. Attention core stays about
0.036 ms because the prompt is short; a long KV will move this row
and was not measured here.

Eager event brackets around the fused ops are not device time. The
first `aten.linear` bracket is 0.42 ms against a 0.100 ms graph of
the same projection. `moe_forward_shared` brackets average 1.07 to
1.38 ms against the 0.123 ms graph. That Python gap is what the
cudagraph removes. Forty of those brackets are about 55 ms, which is
most of the eager C1 ITL of 74.8 ms from D1. Glue fusion inside the
graph cannot move the 0.18 ms of norms, and the 492 glue ops are
views and allocations the graph already folds.

## 4. What was not shipped

No overlay is on for the daily path except the Shisa one, and that
one is opt-in via `B70_D2_SHISA_MTP=1`.

- int4 trunk GEMV: slower on GDN qkvz, the largest projection. Off.
- lowbit MoE swap: 0.129 ms vs the stock XPU WNA16 graph at 0.123 ms.
  Off.
- Norm fusion: 0.18 ms for 80 norms. Not worth a new kernel.

Kernel request for C4, not a fork and not a patch against vLLM
source: an int4 grouped MoE for E=256, top-8, H=2048, I=512, T=1,
under about 0.04 ms (today 0.123 ms, 108 GB/s on 13.3 MB active).
The bytes want 0.022 ms at 608 GB/s; the rest is launch and the two
small grouped GEMMs. A kernel that only reaches 200 GB/s still saves
roughly half of the 4.9 ms. The fp16 projections do not need a new
kernel.

`harness/patches/d2_int4_linear.py` is the rejected trunk swap.
`harness/patches/d2_profile.py` is the region profiler.
`harness/patches/d2_mtp_shisa.py` is the MTP overlay.
`harness/patches/sitecustomize.py` and `serve_ornith_gptq.sh` pass
the env vars through. None of these edit the vLLM fork.

## 5. MTP

CPU check, before any Shisa serve. The GPTQ file's routed experts are
int4. Its MTP subtree is 785 tensors, all BF16, not qweight. Trunk
layernorms match the Shisa BF16 trunk numerically (final norm
max_abs 0; the GPTQ norms are stored F16 and the Shisa trunk norms
are BF16). The MTP tensors do not match the Shisa 19-tensor head
(q_proj, fc, router, expert row 0, cosine about 0). GPTQ `mtp.norm`
is nearly constant (mean 0.023, std 0.0002). Shisa `mtp.norm` has
mean 1.92 and std 0.30. Acceptance 0.38 is that native head. The
drafter was not running int4, and it was not a failed load of the
Shisa weights.

Shisa file:
`models/files/ornith-1.5-35b-a3b/bf16-mtp-shisa/model-mtp.safetensors`
(19 BF16 tensors). The overlay copies onto
`runner.speculator.model` (`Qwen3_5MoeMTP`) after `load_model`.
Embed and lm_head are skipped so they stay tied to the target.
Experts needed a (1, 2) transpose: runtime w13 is [256, 2048, 1024]
and w2 is [256, 512, 2048]. Every serve logged
`d2_shisa_where speculator type=Qwen3_5MoeMTP`, `copied=16`,
`skipped=2`, `unused=[]`. Cosine before the copy was about 0 on fc,
qkv, o, and the experts.

CONFIG for the benches: `GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY
ATTN=FLASH_ATTN`, default SYCL GDN, async, max-model-len 33792,
`B70_XPU_CG_RECLAIM=500`, `B70_D2_SHISA_MTP=1`, MTP method `mtp`.
Bench `--skip-prefill`. Identity and coherence passed on spec=1 and
spec=2 (both greedy calls were the same Paris text).

| config | C1 ITL p50/p90/p99 | C1 tok/s | TTFT ms | C2 ITL p50 | acceptance | ms/token |
| --- | --- | --- | --- | --- | --- | --- |
| no MTP, D1 | 12.81/12.99/13.12 | 78.23 | 88.6 | 13.75 | | 12.8 |
| native spec=2, D1 | 18.81/19.17/19.37 | 93.88 | | 20.9 | 0.383 | 10.7 |
| Shisa spec=1 | 15.74/16.22/16.43 | 111.90 | 92.3 | 17.43 / 17.47 | 0.775 (86/111) | 8.94 |
| Shisa spec=2 | 18.72/18.97/19.26 | 121.73 | 115.1 | 20.93 / 20.99 | 0.665 (113/170) | 8.22 |
| Shisa spec=3 | | | | | | rerun in flight |

ms/token is 1000/C1 tok/s. Chunk ITL did not fall below the no-MTP
row. The extra accepted tokens did. Spec=2 is the faster of the two
measured Shisa settings. C2 tok/s was 102 and 108 at spec=1, and 111
and 123 at spec=2.

JSON: `d2-bench-shisa-spec1.json`, `d2-bench-shisa-spec2.json`.
Shisa copy log: `d2-shisa.log` (rotated per serve; the spec=2 copy
is in the journal entry and the spec=1 chain log).

Strata, `docs/20261016_night_lab.md`, is a different model
(Flash-Next IQ3, two cards), so the rates are not a matched pair.
N4 solo MTP accepted/offered 343-348/492-504 (0.69-0.71) against
355/471 (0.75). N6 MtpPair was 308/404 (0.76) against 317/388 (0.82).
Night-final C1 is about 141 tok/s. Ornith native 0.38 is below that
band. Ornith Shisa spec=2 at 0.665 sits next to Strata's lower N4
rate and still trails the N6 pair. The tok/s figures are not
comparable across the two stacks.

## 6. Spec=3 status

COMMAND: the first spec=3 serve used `B70_GPU_LOCK_TIMEOUT=600`.
`gpu-run` printed `timed out after 600s on card 1`. E1
(`vllm-fn-lowbit-e1-mix`) held the lease from 07:36. No spec=3
tokens were generated. The rerun uses an 1800 s wait, same flags,
`MTP_TOKENS=3`, port 18081. Capture sizes are 1,2,3,4,8, which cover
a decode query of length 4.

VERDICT so far: ship the Shisa overlay for a latency run, at
`num_speculative_tokens=2`, if the goal is effective ms/token.
Leave the int4 trunk overlay off. The MoE kernel request in section
4 is the way through the remaining device time.
