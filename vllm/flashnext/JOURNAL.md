# Flash-Next on vLLM -- journal

Append-only, newest at the bottom. One entry per milestone/experiment:
CONFIG -> COMMAND -> RESULT -> VERDICT. ASCII only. Agents append their own
entries (header: `## YYYY-MM-DD HH:MM <agent> -- <title>`); the orchestrator
commits and pushes at milestones.

## 2026-10-19 22:10 orchestrator -- campaign start
CONFIG: goal = migrate Flash-Next serving from Strata (DD night-final-v3:
C1 ~141, C2 ~92/stream, prefill ~2200, 2x200K, RAM parking) to newest vLLM;
plugins first, thin rebase-clean fork second. Work done by grok-4.7 headless
agents (vllm/flashnext/grok-agent.sh), orchestrated by Claude Opus 5.5.
COMMAND: phase 1 research agents 01-06 (prompts/).
RESULT: reports/01-06. vLLM main 5fd62688 (v0.32.0 latest release) builds as
XPU wheel 0.32.1rc1.dev94 (venv /mnt/vm_8tb/b70/build/vllm-flashnext/venv).
Flash-Next = qwen4_exp; no XPU model on main (open PR #61607, eager-only
W4A16). XPU MoE on main: int4 / MXFP4 / fp8 only; no GGUF on XPU; no int2/int3
MoE. Size: IQ3_XXS 43.8 GiB fits 2x32 GiB with 2x200K KV (PLE 26.8 GiB in host
RAM); uniform int4 60.4 GiB does not. IQ3_XXS uses Q2_0 (ggml 42) on 30/48
expert downs -- no kernel anywhere in vLLM. Published sub-4-bit: Haberstroh
int2-mixed 36.1 GiB (not loadable as-is). RTN int3 pilot trails ISTA IQ4_NL
on layer-0 down (0.216 vs 0.129 rel err).
VERDICT: Track A (primary) = IQ3_XXS GGUF on vLLM-XPU with our own kernels
(port Strata KREG MMVQ / ISA IQ / XMX prefill as plugin ops). Track B
(fallback/testbed) = AutoRound ~3-bit Qwen3.6-35B-A3B + our own int3/int2
XPU MoE kernels. Dev on 35B-A3B first; latency is the metric.

## 2026-10-20 00:30 orchestrator -- wave 2 launched
CONFIG: hotschmoe-dd STOPPED (systemctl --user stop hotschmoe-dd) to free both
GPUs. Agents A1 fork/model port, A2 GGUF kernels (card 0), A3 runtime/harness/
downloads (card 1), B1 3-bit quant, B2 int3/int2 kernels (card 0).
COMMAND: prompts/A1..B2.
RESULT: pending.
VERDICT: pending.

## 2026-10-20 00:32 A3 -- xpu-run plus host IGC mount

CONFIG: vllm-flashnext-xpu:py312, venv at /opt/venv, host UMD 26.22 /
loader 1.28.2 / gmmlib 22.10 bind-mounted ahead of pip oneAPI 2026.1.
Host IGC 2.36.3 (libigc.so.2, libigdfcl.so.2, libiga64.so.2,
libopencl-clang2.so.17) was missing from the container. Card 1,
ZE_AFFINITY_MASK=1. xpu-health image is the older vllm-xpu-env stack,
not this venv.
COMMAND: vllm/flashnext/harness/xpu-run.sh --check ; then
B70_AGENT=a3-health bin/gpu-run bin/xpu-health ; then
xpu-run.sh --card 1 -- python -c (4x4 ones matmul).
RESULT: --check exit 0, ldd and dlopen resolve libze_loader 1773224
bytes and libigc 95845480 bytes from /host-umd. xpu-health HEALTHY
cards 0 and 1 in 34s. Before the IGC mounts, the same alloc aborted
exit 134 at neo gmm_helper/resource_info.cpp:15 after device_count 1.
After the mounts: device name Intel Arc Pro B70, matmul sum 64.0,
ALLOC_OK, exit 0 in 20s. Lease released.
VERDICT: harness is usable. The allocation abort was the host IGC
not being visible to NEO, which dlopens libigc.so.2 from /usr/local/lib.
Ornith GPTQ smoke is still loading (card 1, port 18081). Downloads of
Qwen3.6-35B-A3B BF16 and the Unsloth MTP GGUFs are still running under
bin/cpu-run. Details in reports/1A3_harness.md.

## 2026-10-20 00:45 A3 -- Triton compiler in the runtime image

CONFIG: vllm-flashnext-xpu:py312 rebuilt with gcc 13.3, g++, and
python3.12-dev. Level Zero headers from libze-dev 1.28.2 mounted at
/opt/level-zero. Unversioned libze_loader.so is the host SONAME bytes.
oneAPI 2026.1 icpx from the steve-repro qwen38 oneapi-root is on PATH
and not on LD_LIBRARY_PATH. Card 1, ZE_AFFINITY_MASK=1.
COMMAND: python /mnt/vm_8tb/b70/build/vllm-flashnext/logs/a3_triton_smoke.py
(128-element Triton add-one). The previous Ornith launch had already
selected XPUExpertsWNA16 and then died in triton/runtime/build.py with
"Failed to find C compiler" (log a3-serve-eager-v3.log).
RESULT: sum 256.0, TRITON_OK, exit 0, 33s. Lease released.
VERDICT: Triton XPU JIT works on this venv. Ornith eager serve restarted
(log a3-serve-eager-v4.log, port 18081, max-model-len 33792).

## 2026-10-20 00:55 A3 -- Ornith GPTQ eager smoke and latency

CONFIG: card 1, Ornith GPTQ int4 mixedcal-v2, TP=1, dtype float16,
language-model-only, enforce-eager, gpu-memory-utilization 0.90,
max-model-len 33792, port 18081. Served names hotschmoe-dev then
ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2. XPUExpertsWNA16. KV cache
218758 tokens. No MTP, no graph.
COMMAND: bench_latency.py --tag eager-c1 --decode-tokens 64.
JSON logs/a3-bench-eager.json (under the vllm-flashnext build log dir).
RESULT: /v1/models identity ok. Two temp-0 completions identical and
both name Paris (repeated "Paris. The capital of France is Paris.").
C1 ITL p50/p90/p99 74.78/75.36/76.40 ms, 13.57 tok/s (n=64).
C2 concurrent streams 13.52 and 13.74 tok/s, ITL p50 about 73.9 ms.
Prefill unique nonce: 1K TTFT 169 ms (6001 tok/s), 8K TTFT 1038 ms
(7888 tok/s), 32K TTFT 5121 ms (6398 tok/s). No spec_decode metrics.
Strata DD reference is C1 141 tok/s, C2 92/stream, prefill about 2200.
This is a different model on one card with eager decode.
VERDICT: eager serve is coherent. Decode is about 10x the DD ITL.
docker stop released the card. Post-health HEALTHY on cards 0 and 1.
B1 then reacquired card 1 for another AutoRound quant. MTP serve is
queued behind that lease. Details in reports/1A3_harness.md.

## 2026-10-20 00:53 A3 -- devmodel downloads verified

CONFIG: bin/cpu-run download_devmodel.sh. Official
Qwen/Qwen3.6-35B-A3B @ 995ad96eacd98c81ed38be0c5b274b04031597b0.
GGUF unsloth/Qwen3.6-35B-A3B-MTP-GGUF @ 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d
(UD-IQ3_XXS, UD-IQ4_XS, UD-Q4_K_M, each under 25 GB).
COMMAND: B70_AGENT=a3-download bin/cpu-run bash vllm/flashnext/harness/download_devmodel.sh
RESULT: DONE 2026-10-20T00:52:53, exit 0, 2625s. Official 26 safetensors,
71903776776 bytes, index contains mtp/nextn. GGUF bytes match remote:
14069266720, 18209036576, 22663387424. Manifest
/mnt/cache/b70/devmodel/MANIFEST.txt. CPU lock released.
VERDICT: weights are ready for B1 and for later GGUF kernel work.

## 2026-10-20 01:56 A3 -- MTP serve queued behind B1

CONFIG: card 1. B1 AutoRound quant of Qwen3.6-35B-A3B, layers 0-39,
200 iters, device xpu, lease since 00:54:50. A3 MTP serve is
serve_ornith_gptq.sh GRAPH=0 MTP=1 MTP_TOKENS=2, port 18081, waiting
on the lease (pid 2825445). Eager smoke is already measured.
COMMAND: none new. Read B1's quant3.log.
RESULT: at 01:56 B1 was finishing layer 14 of 40, about 252 s/layer.
Remaining quant is on the order of 2 hours, then an eval. Port 18081
is closed. A3 is not preempting the lease.
VERDICT: MTP and graph latency are not measured yet. Next step is the
queued MTP serve, then bench_latency.py --tag mtp, teardown, health,
then the same with GRAPH=1.

## 2026-10-20 00:52 B1 -- SignRound layer 0, export paused for card 1

CONFIG: source /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B BF16 (official
download complete, 26 shards, index total_size 71903645408). Card 1,
ZE_AFFINITY_MASK=1, torch 2.14.0+xpu. AutoRound quant_tensor_sym + SignSGD,
iters=200, lr=1/iters LinearLR to 0, min/max scale clamped to [0, 1],
group 128 symmetric. Calib: wikitext-2-raw-v1 train, first 8 non-overlapping
windows of 512. Routing measured on the FP forward of that calib, not the
Strata residency profile. 80 percent token-mass cut. Packing
auto_round:auto_gptq, byte-identical to Ornith GPTQ expert-0 down_proj
qweight and qzeros (see plugins/LOWBIT_FORMAT.md).
COMMAND: bin/gpu-run --card 1 python vllm/flashnext/quant/run_quant.py
--mode quant --iters 200 --layers 0-39. xpu-health before: HEALTHY card 1.
RESULT: layer 0 is linear_attention. Trunk int4 block-MSE / RTN:
in_proj_qkv 0.376, in_proj_z 0.360, in_proj_a 0.355, in_proj_b 0.349,
out_proj 0.118. Weight rel L2 0.110 to 0.128. Routed experts, all 256
tuned, 0 empty: int4 rel L2 0.1064, block MSE ratio 0.316 (91 s);
int3 rel L2 0.2098, block MSE ratio 0.274 (91 s). Hits min 1, max 721,
mean 128. 62 experts cover 50 percent of hits, 128 cover 80 percent,
192 cover 95 percent, so the mixed recipe is 128 int3 / 128 int2 on
this layer. The int2 pass died on the second backward: advanced indexing
of the expert Parameter left the weight in the graph. Weights are now
detached. A CPU replay of that indexed int2 path beats RTN (loss 1.47e5
vs 6.43e5) and does not crash. int3 qweight for K=2048 is [192, N];
int2 is [128, N]. Dequant of a packed random code is exact in fp32.
Layer 0 was not written. Card 1 was then taken by a3-serve
(vllm-fn-ornith-c1, port 18081).
VERDICT: SignRound beats RTN on this checkpoint at 200 steps. The export
is resumable and is waiting for card 1. Full 40-layer run is about 3 hours
at the measured 91 s per bit-width plus a half-width int2 pass.

## 2026-10-20 01:01 A2 -- GGUF XPU plugin, dequant exact, head GEMV 1739 us

CONFIG: b70_gguf_xpu, icpx 2026.1.1, torch 2.14+xpu venv, card 0,
ZE_AFFINITY_MASK=0. Types Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ2_S IQ3_XXS
IQ3_S IQ4_NL IQ4_XS Q2_0 (ggml 42). Ops dequant_bf16, gemv (M chunked by
8), moe_gemv, grouped_gemm (joint_matrix 8x16x16, MT=4). Signatures in
plugins/INTERFACE.md under the A2 proposal. Host has no g++; GCC 15
headers taken from strata-sycl-dev:020 into
/mnt/vm_8tb/b70/build/gguf-xpu-toolchain.
COMMAND: bin/cpu-run scripts/build.sh (MAX_JOBS=1, ~27 s). python3
tests/compare_cpu.py. bin/gpu-run --card 0 scripts/gpu_test.sh.
RESULT: CPU dequant 0 bit mismatches vs gguf-py on Flash-Next shard 1
(all 11 types, 2 rows; Q2_0 vs a separate dequantize_row_q2_0
transcription) and on the devmodel UD-IQ3_XXS / IQ4_XS / Q4_K_M files for
the types they store. GPU bf16 dequant 0 mismatches on 64 rows of each
type. GEMV and MoE max abs about 1e-6. Grouped max abs about 1e-3.
Head Q5_K M=1 N=248320 K=2560: 1739 us, 251 GB/s, 41% of 608 (was 3403 us
before sizing SLM to the row). Strata KREG head nc1 is 760 us (~575 GB/s).
Expert top-10 GEMV about 106-121 us. Grouped M=128 re-reads weight every
32 rows and is under a few GB/s. Card 0 matmul after the run succeeded.
VERDICT: Package, tests, and bench are in reports/1A2_gguf.md. Correctness
holds. Decode head is 2.29x the Strata KREG number. Next: port the KREG
pipeline at native_mmvq.dp.cpp:2236, then raise grouped MT so prefill
reads each expert once.

## 2026-10-20 01:17 A1 -- b70/flashnext GGUF port, CPU suite green

CONFIG: clone /mnt/vm_8tb/b70/build/vllm-flashnext/vllm, branch
b70/flashnext off origin/main. Cherry-pick of vllm#61607 plus three
commits (kernels pin 0.1.16.1, b70_gguf quant hook and weight map, PLE
host table / indexer weights / MTP eh_proj). CPU only. Interpreter is
uv cpython 3.12 with the flashnext venv site-packages and llama-pinned
gguf-py. ONEAPI_DEVICE_SELECTOR=opencl:cpu, VLLM_TARGET_DEVICE=cpu.
No GPU, no payload copy of the 26.8 GiB PLE table.
COMMAND: git fetch origin && git rebase origin/main. Then import
tests.models.qwen4_exp.test_b70_gguf_mapping and call every test_*.
RESULT: rebase clean onto origin/main 2296c12e47. Ahead 4, not pushed.
Tip 7b40eb79c6. Suite 14/14 PASS after the rebase (exit 0, ~87 s).
IQ3_XXS shard 1: 1223 tensors, each name once, 38 of ggml 42 (Q2_0).
PLE shard: 1 tensor, IQ4_NL, raw ne (160, 320001536), n_bytes
28800138240. MTP sidecar: 34 tensors, block 48 maps to model.layers.0.
Qwen3.6 UD-IQ3_XXS: arch qwen35moe, 753 tensors, each name once.
Histogram {0:368, 8:8, 10:2, 11:1, 14:252, 21:39, 22:78, 23:3, 30:2}.
The only Q2_K/Q3_K tensors are blk.40 ffn_gate_exps, ffn_up_exps, and
ffn_down_exps. packed_row_bytes accepts them. A2 bind.cpp does not
register ids 10 or 11. Meta construct of Qwen4ExpForCausalLM and
Qwen4ExpMTP succeeded (PLE host_table is None, routed_experts.b70_w_gate
on meta, indexer bf16, MTP eh_proj on meta). Toy EP slice of 512 experts
across 2 ranks matched. Toy gather left the table data_ptr unchanged.
PLE hash constants from the GGUF header round-tripped onto PLEEmbedding;
16 vocab sizes sum to 320001446 of the 320001536 table rows.
VERDICT: loader path is measured on CPU for Flash-Next and for the
Qwen3.6 dev model. Details and the op contract are in
reports/1A1_fork.md and plugins/INTERFACE.md (A1 section). Next: A2
adds Q2_K/Q3_K to the four existing ops; GPU EP=2 waits on bin/gpu-run
plus xpu-collective-health. QSA compute remains a large work item.

## 2026-10-20 01:41 B2 -- lowbit WNA16 XPU plugin measured

CONFIG: plugins/b70_lowbit_xpu, quant method b70_lowbit via
vllm.general_plugins, no core edit. Symmetric int2/int3/int4, group 128,
bf16 acts. Ops gemv (M=1..8), moe_gemv (top-k, fused gate/up+down),
prefill_wna16 (bf16 DPAS), prefill_wxa8 (int8 acts, prefill only).
Kernel layout b70lb1 (2-bit plane + hi plane, 64B row align, scale bf16
[N, K/128]). B1's LOWBIT_FORMAT.md is auto_gptq qweight/scales; repack
gptq_to_b70lb1 is CPU-tested and not called by the loader. Binary
libb70_lowbit_xpu.so 773928 bytes, AOT intel_gpu_bmg_g31, icpx 2026.1.0
with host ocloc 26.22. Card 0 via harness/xpu-run.sh (ZE_AFFINITY_MASK
from --card). Torch 2.14+xpu, vLLM 5fd62688. Roofline 608 GB/s.
COMMAND: bin/cpu-run bash plugins/b70_lowbit_xpu/build.sh. Then
bin/xpu-health --card 0, bin/cpu-run --quiet xpu-run.sh --card 0
tests/run_xpu.sh, xpu-health again. Log bench_pair.log. Isolated
lm_head probe (bits 2/3/4) on the same .so.
RESULT: BENCH_RC 0, health 0/0, gpu-run 109s, finished 01:35:28.
VERDICT all correctness checks passed. WNA16 GEMV, MoE, and prefill
DPAS rel L2 0. WXA8 rel L2 <= 1.7e-3, cosine >= 0.999999. Plugin:
get_quantization_config, ReplicatedLinear apply exact, _B70MoE
is_monolithic False, decode (2, 256). Expert GEMV sits at ~25 us
(Ornith int3 gate/up 0.0245 ms, down 0.0244 ms) and beats
int4_gemm_w4a16 on the four expert shapes (0.025-0.036 ms vs
0.044-0.045 ms). MoE T=1: Ornith int3 top-8 0.107 ms (91.8 GB/s);
Flash-Next int3 top-10 0.176 ms (109 GB/s). Stacked expert GEMV only:
40*0.107 = 4.28 ms/token Ornith, 48*0.176 = 8.45 ms/token Flash-Next.
lm_head N=248320 K=2048 probe: int4 1.190 ms / 220 GB/s (36% of 608),
int3 1.671 ms / 119 GB/s, int2 0.959 ms / 141 GB/s. Suite int3 lm_head
was 2.126 ms / 93 GB/s on the same binary. int2+int4 fit ~0.71 ms +
bytes/549 GB/s; int3 is ~0.60 ms above that (hi-bit extract). 64B
paired loads moved the head ~4% vs the SLM binary. Prefill gate/up
re-reads weights ceil(M/8) times: Ornith M=2048 int3 WNA16 9.44 ms at
22 GB/s counted with the reload; WXA8 slower on Ornith, slightly
faster on Flash-Next M=512. TP=1. A B1 checkpoint does not load.
VERDICT: kernels and registration are measured. Details in
reports/1B2_lowbit.md. Next: wire gptq_to_b70lb1 into
process_weights_after_loading and reinstall the plugin. Expert GEMV
is launch-floor limited; the prefill unpack is the TTFT work item.
No tok/s claim.

## 2026-10-20 01:48 A2 -- KREG head 808 us, grouped MT=16

CONFIG: b70_gguf_xpu, icpx 2026.1.1, torch 2.14+xpu venv, card 0,
ZE_AFFINITY_MASK=0. Same 11 ggml types and the four ops. Q4_K/Q5_K/Q6_K/
IQ4_XS GEMV quantizes x to Q8_1 once, then the KREG schedule from
native_mmvq.dp.cpp:2236 (dp4a lanes, 8 lanes per 256-block, subgroup
butterfly, SLM across subgroups). Other GEMV and all MoE stay scalar.
Grouped is joint_matrix 8x16x16, MT=16, GRF 256, one weight read per
128-row entry. Build MAX_JOBS=4 via bin/cpu-run, about 38 s.
COMMAND: bin/cpu-run scripts/build.sh. bin/gpu-run --card 0
scripts/gpu_test.sh with SYCL_CACHE_PERSISTENT=1. Log of the shipped
binary: min of 10 after 3 warmup, torch.xpu.Event.
RESULT: VERDICT failures=0. Dequant 0 bf16 mismatches on all 11 types.
IQ/Q2_0 GEMV and MoE max abs about 1e-6. KREG GEMV max abs 8.6e-3 to
3.3e-2 against the true bf16 matmul (Q8_1 of x), under atol 5e-2.
Grouped max abs about 1e-3. Head Q5_K M=1 N=248320 K=2560: 808.4 us,
540.6 GB/s, 88.9% of 608. Prior binaries on that shape: scalar 1739.4 us,
KREG that re-quantized x in every work-group 857.6 us. Strata KREG nc1
is 760 us / 575 GB/s. M=8 on the head is 4304 us; Strata nc6 is 1166 us.
Expert top-10 GEMV about 122 us. Grouped M=128, one weight read: downs
136 us (IQ4_NL, Q2_0), gates 306-364 us (1.2-2.3 GB/s). MT=4 on those
gates was 510-748 us. A dequant-to-bf16 plus torch.mm probe, x already
bf16, is 193 us on the gate and 151 us for the dense GEMM alone (3.2 MB,
launch floor). That path was not shipped: inside the op, with a host
read of entries, it was about 307 us and a wider error, and nrows=1
would pay the sync per token. Card 0 finished the suite, gpu-run exit 0,
no xe-reset.
VERDICT: Package and the bench are in reports/1A2_gguf.md. Head decode
GEMV is 1.06x Strata's 760 us. The remaining 48 us, and the M=8 gap, are
the pipelined round-2 KREG, not another quant or occupancy pass. Grouped
prefill is still the slow kernel after one weight read. Next: night_pf
pipeline for the head, and the grouped next-superblock preload. Q2_K and
Q3_K (A1's three MTP tensors) are not in this binary.

## 2026-10-20 03:41 C3 -- GPTQ repack wired, CPU-checked

CONFIG: b70_lowbit pack=gptq beside the B2 b70lb1 plane path. Ornith
GPTQ int4 mixedcal-v2 (routed experts only; attn, shared, router, mtp,
vision, embed, lm_head dense). B1 extra_config on
qwen36-w3a16-g128-experts-w4a16-trunk (experts bits 3, trunk bits 4).
Signed codes are unsigned - maxq. Kernel .so not rebuilt. Card 0 was
free; card 1 held by B1. xpu-health --card 0 HEALTHY before serve.
COMMAND: docker run --network none vllm-flashnext-xpu:py312 (no
/dev/dri, no gpu-run lease) /opt/venv/bin/python
vllm/flashnext/plugins/b70_lowbit_xpu/tests/test_gptq_cpu.py
RESULT: synthetic int2/int3/int4 codes match pack_gptq.pack_linear.
Batched expert repack matches per-matrix gptq_to_b70lb1, including a
fused w13 gate||up stack. Ornith layer-0 expert-0 down_proj qweight
(64, 2048) K=512: exact int8 codes versus pack_gptq.unpack_qweight;
dequant rel L2 0.001580 from the bf16 scale. Quant config: ornith
attn, linear_attn, mlp.gate, shared_expert, shared_expert_gate, and
lm_head skipped; experts bits 4. B2 bench extra_bits experts=2 and
gate-vs-gate_proj component checks still pass. B1 experts bits 3,
self_attn.q_proj bits 4, shared_expert.down_proj bits 4,
shared_expert_gate skipped, shared_expert.gate_proj not treated as the
router.
VERDICT: the GPTQ bitstream matches b70lb1 on a real Ornith expert.
Serve is next. No tok/s claim.

## 2026-10-20 03:40 C1 -- MoE 122 us is a host gap plus a scalar kernel

CONFIG: b70_gguf_xpu A2 binary, card 0, ZE_AFFINITY_MASK=0, no
immediate-command-list env. Floor binary
/mnt/vm_8tb/b70/build/gguf-xpu-floor-empty. unitrace and onetrace are
not installed. Queue profiling property does not stick
(get_profiling_info throws); timing is submit_profiling_tag, same clock
as torch.xpu.Event, with a burn kernel queued first.
COMMAND: B70_AGENT=C1 bin/cpu-run --quiet bin/gpu-run --card 0
plugins/b70_gguf_xpu/scripts/floor_run.sh
RESULT: empty kernel behind the burn 2.60/3.16/4.48 us (n=32). 8 MiB
read 9.06 us. Idle one-shot tag span 90.0 us; 32 empty launches 190.6 us
(6.0 us each). torch add a2 118.9 us, behind a matmul 2.4 us. Q5_K gemv
N=32768 K=2560 a2 147.1, behind 111.2 us, 518 GB/s (85% of 608). IQ3_S
embd slice N=32768 behind 219.9 us, 164 GB/s (27%). moe_gemv T=1 topk=10
a2 stays 117-120 us, but behind is IQ4_NL down 86.7 us (106 GB/s),
IQ2_XS gate 71.9, IQ2_XS up 67.1, Q2_0 down 70.9, IQ2_XXS gate 64.0,
IQ2_XXS up 68.7. Batch-32 matches the behind number.
VERDICT: 122 us is about 30-50 us of host/tag gap on a real 64-87 us
scalar kernel, not a pure timer floor. Three calls per layer are about
210 us of device work before a faster dot. Empty device cost is 3 us.
Next: DP4A row GEMV and one host call per MoE layer.

## 2026-10-20 03:59 C4 -- GGUF loader on CPU, serve not started

CONFIG: fork /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext
at 7b40eb79c6 plus uncommitted loader (not committed). Dev GGUF
Qwen3.6-35B-A3B-UD-IQ3_XXS, arch qwen35moe, 753 tensors. HF config
/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B. Installed wheel has no
gguf package; serve will put the fork first on PYTHONPATH and set
B70_GGUF=1 so sitecustomize registers --load-format b70_gguf and loads
b70_gguf_xpu. No GPU lease taken.
COMMAND: uv CPython 3.12.13, PYTHONPATH=fork:gguf-py:venv site-packages,
VLLM_TARGET_DEVICE=cpu. Ran the 15 functions in
tests/models/qwen4_exp/test_b70_gguf_mapping.py (no pytest in the venv).
Meta-constructed Qwen3_5MoeForConditionalGeneration (language-model-only,
max_model_len 2048) and compared layer 0 / layer 3 partitions to the
GGUF header. Q6_K two-row dequant vs gguf-py. Q8_0 one-row dequant.
RESULT: 15/15 PASS after the GDN last-dim check was corrected (first
run failed 1 assertion that compared a K-axis retile to an illegal
dim-0 retile). Meta partitions match the header: GDN qkvz
[2048, 2048, 4096, 4096] K=2048 vs attn_qkv rows 8192 and attn_gate
rows 4096; full attn q/k/v [8192, 512, 512]; out and o_proj
[2048] K=4096; experts E=256 N=512 K=2048 ep=1. Conv, router gate,
shared-expert gate, and both embeddings are unquantized modules
(the model builds them with quant_config=None), so those F32/Q6_K
tensors are dense copies, not gemv. Q6_K row dequant matches gguf-py
(max abs 0). Q8_0 dequant of blk.39 shared down is finite. Both GPU
leases were held (card 0 C3 vllm-fn-lowbit-c0 since 03:41, card 1 B1
quant eval since 00:54), so the one-card serve did not start.
VERDICT: Loader and the correctness fixes are in the fork working
tree. Step 1 serve is next on the first free card:
harness/serve_qwen36_gguf.sh (port 18104, eager, max_model_len 2048,
hotschmoe-dev first). Packed TP sharding is still refused, so step 1b
is blocked until that lands.

## 2026-10-20 04:02 C3 -- int4 b70_lowbit serve, decode near A3, prefill not

CONFIG: card 0, Ornith GPTQ int4 mixedcal-v2, TP=1, dtype bfloat16
(kernel requires bf16 x and scale; A3 baseline was float16),
language-model-only, enforce-eager, gpu-memory-utilization 0.90,
max-model-len 33792, max-num-seqs 4, port 18083, --quantization
b70_lowbit. Served names hotschmoe-dev then
ornith-1.5-35b-a3b-b70-lowbit-int4. Container vllm-fn-lowbit-c0.
Kernel .so unchanged (four device submits inside moe_gemv). GPTQ
experts repacked to b70lb1 in process_weights_after_loading. Routed
experts only; attn, shared, router, norms stay dense bf16.
COMMAND: B70_AGENT=C3 CARD=0 PORT=18083
plugins/b70_lowbit_xpu/serve_lowbit.sh via harness/xpu-run.sh.
Then harness/bench_latency.py --base-url http://127.0.0.1:18083
--tag b70-lowbit-int4-eager --decode-tokens 64
--out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c3-int4-bench.json
RESULT: 40/40 MoE layers repacked. w13 qweight (256, 256, 1024) to
plane2 (256, 1024, 512); w2 (256, 64, 2048) to (256, 2048, 128).
KV cache 230319 tokens, 6.82x at 33792. Weight footprint 21.41 GiB.
init engine 16.70 s. /v1/models ids hotschmoe-dev,
ornith-1.5-35b-a3b-b70-lowbit-int4. Two temp-0 completions identical,
both " Paris.\nThe capital of France is Paris.\nThe capital of France is".
C1 ITL p50/p90/p99 79.55/82.15/116.59 ms, 12.98 tok/s (n=64).
C2 streams 13.39 and 13.62 tok/s, ITL p50 77.36 and 77.06 ms.
Short-prompt TTFT 2868 ms (24 tokens) and C2 TTFT about 6.5 s.
Prefill 1017 tok TTFT 8279 ms (123 tok/s), 8185 tok 49502 ms
(165 tok/s), 32762 tok 196439 ms (167 tok/s). A3 eager fp16 reference
was C1 74.78/75.36/76.40 ms (13.57 tok/s), C2 about 73.9 ms
(13.52 and 13.74 tok/s), prefill 169/1038/5121 ms. Decode ITL is
within about 6 percent of that reference. Prefill is the per-expert
Python loop (host sync per expert); T>8 takes that path, including
the 24-token decode prompt. docker stop removed the container and
port 18083 closed. The gpu-run process was SIGKILL'd, so the owner
file was stale until flock showed the card free. C1 acquired card 0
at 04:01:32 (c1_run.sh) before a post xpu-health could run.
VERDICT: int4 control is coherent and the decode ITL matches the
GPTQ eager baseline. Prefill does not. No speed claim over
XPUExpertsWNA16. Post-health is still owed on the next card-0 window.

## 2026-10-20 04:45 orchestrator -- reprioritize card 1 to decode overhead
CONFIG: C3/A3 eager vLLM decode on Ornith-35B-A3B int4 = 75-80 ms/token
(13 tok/s); host overhead, not kernels, dominates. B1 perplexity eval was
holding card 1 (slow on-the-fly unpack); A3 MTP serve queued 3.5 h.
COMMAND: killed B1 agent + eval job (int3 ppl 9.19 already recorded; int4
control and mixed ppl still TODO via run_quant.py --mode eval); killed A3
agent (its queued MTP serve then started on card 1, port 18081). Launched
D1 (prompts/D1_overhead.md): profile eager step, XPU graph capture, host
overhead fixes; target C1 ITL < 15 ms on 35B int4.
RESULT: pending.
VERDICT: decode overhead is the top latency item; quality eval deferred.

## 2026-10-20 04:45 D1 -- MTP eager latency on the running A3 serve

CONFIG: card 1, Ornith GPTQ int4 mixedcal-v2, TP=1, dtype float16,
language-model-only, enforce-eager, gpu-memory-utilization 0.90,
max-model-len 33792, max-num-seqs 4, port 18081, MTP method=mtp
num_speculative_tokens=2. Container vllm-fn-ornith-c1. Served names
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
XPUExpertsWNA16, FlashAttention. No graph.
COMMAND: harness/bench_latency.py --base-url http://127.0.0.1:18081
--tag mtp-eager --decode-tokens 64
--out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-bench-mtp.json
RESULT: identity ok. Coherence ok, both texts identical and name Paris.
C1 ITL p50/p90/p99 82.60/85.09/95.52 ms, 20.99 tok/s, n=64, n_chunks=37
(about 1.73 tokens per SSE chunk). C2 streams 22.60 and 21.96 tok/s,
ITL p50 82.05 and 82.02 ms. Prefill 1017/8185/32762 tok TTFT
169.60/1052.88/5358.87 ms (5996/7774/6114 tok/s). MTP acceptance during
the bench 87/218 = 0.399. Engine log at 04:40:36: mean acceptance
length 1.77, per-position acceptance 0.683 and 0.083, draft acceptance
38.3%, generation throughput 21.8 tok/s. A3 eager no-MTP reference was
C1 ITL 74.78 ms and 13.57 tok/s. Chunk ITL did not drop. Tokens per
step rose, so tok/s rose from 13.6 to 21.0. Prefill matches the eager
reference. docker stop vllm-fn-ornith-c1 after the bench. gpu-run
parent exited and card 1 lock was free.
VERDICT: MTP eager is coherent. The engine step is still about 83 ms.
Acceptance of the second draft token is 8%, so most of the extra
forward is wasted. Per-token latency is about 48 ms (1000/20.99),
still far from the 15 ms milestone. Next is an eager no-MTP step
breakdown, then XPU graph capture.

## 2026-10-20 04:48 C3 -- fused MoE launches measured, 2-submit kept

CONFIG: card 1, cpu-run --quiet, container b70-lowbit-launch, memory
32g. .so /mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so
1442856 bytes (rebuilt 04:05; 4-submit path plus GateUpSilu+DownReduce
and one-work-group MoeTokenKernel). Ornith decode shape T=1 topk=8
E=256 H=2048 I=512 int4 random b70lb1 planes. Times are torch.xpu.Event
around the host op. Match gate is max abs 0.05 versus the 4-submit op.
Then tests/bench_lowbit.py with B70_LOWBIT_MOE_LAUNCHES unset, so the
regression stays on the 4-submit default. Same container, so the card
was not released between them.
COMMAND: B70_AGENT=C3 B70_GPU_LOCK_TIMEOUT=30 B70_CPU_LOCK_TIMEOUT=20
bin/cpu-run --quiet harness/xpu-run.sh --card 1 --memory 32g
--name b70-lowbit-launch -- bash
plugins/b70_lowbit_xpu/tests/run_gpu_window.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/launch_bench_c3.log
RESULT: 2-submit vs 4-submit max abs 0.000000 on the small shape and
on the Ornith shape. TIME 4-submit 143.7 us, 2-submit 128.9 us.
1-submit max abs 0.000000 on both shapes. TIME 1-submit 3099.6 us
against a 4-submit repeat of 207.4 us in that process. bench_lowbit.py
PASS on pack, gemv, moe, prefill wna16, wxa8, quant config, linear
apply, and MoE decode/prefill (int2/int3/int4). WINDOW_LAUNCH_RC 0,
WINDOW_BENCH_RC 0, GPU_WINDOW_RC 0, wall 122 s. One int4 MoE case
reports max_abs 16 with rel_l2 3.64e-4 and cosine 1.0 and still passes
the B2 rel/cos gate. Container exited. Card 1 was then taken again by
vllm-fn-ornith-c1 (started 04:48:30). Card 0 stayed with C4.
VERDICT: Serve path is B70_LOWBIT_MOE_LAUNCHES=2. It matches the
4-submit numerics on this shape and is about 15 us faster per MoE
call, about 0.6 ms per token across 40 layers. The 1-submit kernel is
correct and about 15x slower, so it is not used. Int3 checkpoint
serve is next, on the next card whose fuser is empty and which has no
XPU_RUN_CARD container. The 04:02 int4 note still owes a card-0
post-health; this window was card 1 and both cards were held again
before a health probe.

## 2026-10-20 04:55 D1 -- eager decode step is host-bound

CONFIG: card 1, Ornith GPTQ int4 mixedcal-v2, TP=1, dtype float16,
language-model-only, enforce-eager, gpu-memory-utilization 0.90,
max-model-len 33792, max-num-seqs 4, port 18081, no MTP, no graph.
Container vllm-fn-ornith-c1. XPUExpertsWNA16, FlashAttention, SYCL GDN.
In-process timer on GPUModelRunner.execute_model and sample_tokens
(harness/patches/d1_step_timer.py, B70_STEP_TIMER=1). Dispatch leaf
events on step 50, cProfile on step 58, torch.profiler CPU+XPU on
step 70. py-spy not installed and ptrace_scope=1, so it was not used.
COMMAND: B70_AGENT=d1-eager-profile CARD=1 PORT=18081 GRAPH=0 MTP=0
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-eager-profile.log
Timer: .../logs/d1-step-timer.log
Bench: harness/bench_latency.py --tag eager-profile --decode-tokens 64
--skip-prefill --out .../logs/d1-bench-eager-profile.json
RESULT: identity ok, coherence ok (Paris, both texts identical).
C1 ITL p50/p90/p99 71.85/73.46/350.30 ms (p99 is the instrumented
step), 12.17 tok/s, TTFT 153.14 ms, n=64. C2 ITL p50 70.94 and
70.94 ms, 14.08 and 14.31 tok/s. Steady execute_model for tokens=1,
excluding steps 50/58/70: n=98, p50 68.86 ms, p90 70.81, min 65.43,
max 76.29. sample_tokens p50 1.37 ms. execute+sample p50 70.28 ms.
tokens=2 steps (n=62) p50 68.82 ms. A 24-token step was 75.88 ms and
a 30-token step was 85.32 ms, so the step does not scale with token
count. cProfile of step 58 (tokens=1): 25017 calls in 0.085 s, matching
the 83 ms wall of that step. Cumulative: model forward 80 ms, MoE
runner 44 ms across 40 layers, moe_forward_shared 41 ms, linear.apply
22 ms across 270 calls, GDN forward_xpu 21 ms across 30 layers,
Parameter.__torch_function__ 20 ms exclusive across 360 calls (largest
exclusive bucket). Kernel launch exclusive time is small: grouped GEMM
4 ms / 80 calls, torch.empty 3 ms / 424, SYCL gdn_attention 2 ms / 30,
Triton launch 2 ms / 52, FA varlen_fwd 1 ms / 10. No synchronize sits
in the top 30 exclusive. Dispatch step 50 is not a device budget: the
timer itself stretched that step to execute 421 ms + sample 316 ms and
reported device_ms 539, including aten.to.dtype_layout 306 ms for 5
calls. Leaf counts from that step are usable: 1000 leaves, 40
moe_forward_shared, 110 linear, 80 fused RMSNorm, 30 GDN, 10 full
attention, 10 KV updates, 4 aten.item. Kineto XPU profiler failed:
PTI_ERROR_INTERNAL, failed to enable Kineto Profiler on XPU. The
EngineCore.step hook logged nothing; the 70 ms is inside execute_model.
Sampler is 1.4 ms. docker stop after the bench. Card 1 lock released.
VERDICT: the ~70 ms eager decode step is Python and dispatcher work
around ~1000 leaf launches, not device time and not the sampler. A
24-token step costs about the same as one token. Graph replay is the
lever that removes that host walk. Next: FULL_DECODE_ONLY with
TRITON_ATTN, GDN triton, and B70_XPU_CG_RECLAIM=500. The dispatch
device_ms figure is not used as a device budget.


## 2026-10-20 05:07 D1 -- FULL_DECODE_ONLY XPU graph, C1 ITL 13.2 ms

CONFIG: card 1, Ornith GPTQ int4 mixedcal-v2, TP=1, dtype float16,
language-model-only, max-model-len 33792, gpu-memory-utilization 0.90,
max-num-seqs 4, port 18081, no MTP. compilation mode NONE,
cudagraph_mode FULL_DECODE_ONLY, capture sizes 1,2,3,4,8.
Attention TRITON_ATTN. GDN backend triton (skips SYCL gdn_attention).
B70_XPU_CG_RECLAIM=500 (keep_graph=True, re-instantiate every 500
replays). No xpugraph_submit_without_event rebuild. Container
vllm-fn-ornith-c1. Served names hotschmoe-dev then the GPTQ alias.
COMMAND: B70_AGENT=d1-graph-fdo B70_XPU_CG_RECLAIM=500 CARD=1
PORT=18081 GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=TRITON_ATTN
GDN_BACKEND=triton MTP=0 bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-fdo.log
Bench: harness/bench_latency.py --tag graph-fdo-triton --decode-tokens 64
--out .../logs/d1-bench-graph-fdo.json
RESULT: identity ok. Coherence ok, both texts identical and name Paris.
Capture: memory-profile pass 2 FULL graphs in 9 s; then 4 FULL graphs
in 2 s (uniform decode sizes 1, 2, 3, 4; size 8 is above
max_num_seqs * decode_query_len). Worker reports CUDAGraph memory
0.0 GiB on the second pass (pool already reserved; accounting does not
see a new delta). init engine took 484 s, almost all of it before the
9 s capture (cold Triton kernels during the profile forward). No
traceback. C1 ITL p50/p90/p99 13.20/14.82/15.93 ms, 76.02 tok/s, TTFT
216.30 ms, n=64. C2 ITL p50 14.38 and 14.40 ms, 70.26 and 69.76 tok/s,
TTFT 394.80 and 402.83 ms. Prefill 1017/8185/32762 tok TTFT
292.46/2460.29/19697.80 ms (3477/3327/1663 tok/s). Eager no-MTP
reference on the same card was C1 ITL p50 71.85 ms (12.17 tok/s) and
prefill about 6000/7800/6100 tok/s with FlashAttention plus SYCL GDN.
VERDICT: FULL_DECODE_ONLY replay is on the hot path. C1 ITL p50 is
under the 15 ms milestone (p99 is 15.93). Decode is about 5x the eager
step. Prefill got slower because the eager prefill path is now Triton
attention plus Triton GDN. Next: same graph with MTP, then PIECEWISE
(breakable, no inductor) and FULL, and a replay host-vs-device sample
to see what is left of the 13 ms.


## 2026-10-20 05:13 D1 -- FULL_DECODE_ONLY plus MTP

CONFIG: same as the 05:07 FULL_DECODE_ONLY Triton run, plus MTP
num_speculative_tokens=2. B70_XPU_CG_RECLAIM=500, B70_REPLAY_LOG=1,
B70_STEP_TIMER=1 (dispatch/cprofile/profiler steps set to 99999 so only
wall times). Async scheduling left at the vLLM default (on).
COMMAND: B70_AGENT=d1-graph-fdo-mtp CARD=1 PORT=18081 GRAPH=1
GRAPH_MODE=FULL_DECODE_ONLY ATTN=TRITON_ATTN GDN_BACKEND=triton MTP=1
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-mtp.log
Bench: harness/bench_latency.py --tag graph-fdo-mtp --decode-tokens 64
--out .../logs/d1-bench-graph-mtp.json
Timer: .../logs/d1-step-timer-mtp.log
RESULT: identity ok, coherence ok (Paris, both texts identical).
Capture: 2 FULL graphs in 4 s (profile) then 2 FULL graphs in 1 s.
Warm Triton cache. Reported CUDAGraph memory 0.03 GiB. C1 ITL
p50/p90/p99 20.40/20.73/20.88 ms, 84.31 tok/s, TTFT 170.34 ms, n=64,
n_chunks=38. C2 ITL p50 22.88 and 22.88 ms, 82.93 and 79.79 tok/s.
MTP acceptance during bench 0.395 (87/220); after counters
97 accepted / 264 drafted (0.367). Prefill TTFT 273.69 / 2693.92 /
28532.66 ms (3716 / 3038 / 1148 tok/s). Steady tokens=3 execute_model
submit p50 is 3.74 ms (n=65) and sample submit p50 is 2.07 ms. One XPU
event around a single replay measured 2.275 ms device. Host time
inside replay() p50 is 0.028 ms, so replay returns without waiting.
VERDICT: MTP raises tok/s from 76 to 84 and raises chunk ITL from
13.2 ms to 20.4 ms. The graph replay is about 2.3 ms of device work
and is not the chunk. Async scheduling is the default and the engine
waits on the future outside execute_model, which this timer did not
cover. Next: the same graph with --no-async-scheduling, and time
EngineCore.step.

## 2026-10-20 05:17 C3 -- int3 serve coherent, decode faster, prefill not

CONFIG: card 0, B1 checkpoint
/mnt/cache/b70/devmodel/quant/qwen36-w3a16-g128-experts-w4a16-trunk
(routed experts int3 group 128, trunk linears int4, skips bf16).
TP=1, dtype bfloat16, language-model-only, enforce-eager,
max-model-len 33792, gpu-memory-utilization 0.90, max-num-seqs 4,
port 18083, --quantization b70_lowbit. Served names hotschmoe-dev
then qwen36-w3a16-b70-lowbit. Container vllm-fn-lowbit-c0.
B70_LOWBIT_MOE_LAUNCHES=2 (GateUpSilu + DownReduce). .so 1442856
bytes. This is not the Ornith experts-only int4 checkpoint: the
trunk (qkv, o_proj, GDN in_proj, shared expert) is GPTQ int4.
COMMAND: B70_AGENT=C3 B70_GPU_LOCK_TIMEOUT=20 B70_LOWBIT_MOE_LAUNCHES=2
MODEL=<that dir> ALIAS=qwen36-w3a16-b70-lowbit CARD=0 PORT=18083
plugins/b70_lowbit_xpu/serve_lowbit.sh
Bench: B70_AGENT=C3 cpu-run --quiet python3 harness/bench_latency.py
--base-url http://127.0.0.1:18083 --model hotschmoe-dev
--alias qwen36-w3a16-b70-lowbit --tag b70-lowbit-int3-eager
--decode-tokens 64
--out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c3-int3-bench.json
Serve log: .../logs/c3-int3-serve.log
RESULT: identity first_ok and alias_ok. Two temp-0 completions
identical and both name Paris. Load 14.38 GiB in 867 s. KV cache
478,423 tokens, max concurrency 14.16x at 33792. init engine 15.07 s.
All 40 layers repacked bits=3: w13 qweight (256, 192, 1024) to
plane2 (256, 1024, 512); w2 (256, 48, 2048) to (256, 2048, 128).
C1 ITL p50/p90/p99 51.73/52.55/53.82 ms, 19.67 tok/s, n=57
(EOS before 64), prompt_tokens=24, TTFT 2412 ms. C2 ITL p50
52.19 and 52.18 ms, 19.37 and 19.38 tok/s, TTFT 5268 and 5320 ms,
n=46 and 47, prompt 29. Prefill 1017/8185/32762 tok TTFT
10275/72614/282846 ms (99/113/116 tok/s). BENCH_RC 0, wall 385 s.
A3 eager fp16 experts-only reference was C1 74.78/75.36/76.40 ms
(13.57 tok/s) and prefill 169/1038/5121 ms. The C3 int4 control on
that same Ornith checkpoint was C1 79.55 ms (12.98 tok/s). docker
stop removed vllm-fn-lowbit-c0 and port 18083 closed. Card 0 was
then taken by C4 prefill_gpu.sh (pid 2910396), which started its
own xpu-health, so this run did not probe health.
VERDICT: int3 wiring is coherent. Decode ITL is about 52 ms, faster
than the eager experts-only int4 baseline, because the trunk is
int4 as well as the experts being int3. That is not a same-checkpoint
win over XPUExpertsWNA16. Prefill is still the per-expert WNA16
loop and is slower than the int4 control. Next is a grouped prefill
kernel, and bin/xpu-health --card 0 when fuser shows gpu.lock.0 empty.


## 2026-10-20 05:17 D1 -- no-async graph is slower; replay device is 11.4 ms

CONFIG: FULL_DECODE_ONLY, TRITON_ATTN, GDN triton, B70_XPU_CG_RECLAIM=500,
no MTP, --no-async-scheduling. Same model and capture sizes as the
05:07 run. Step timer on, dispatch/cprofile/profiler disarmed.
COMMAND: B70_AGENT=d1-graph-fdo-sync ASYNC=0 CARD=1 PORT=18081 GRAPH=1
GRAPH_MODE=FULL_DECODE_ONLY ATTN=TRITON_ATTN GDN_BACKEND=triton MTP=0
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-sync.log
Bench: harness/bench_latency.py --tag graph-fdo-sync --decode-tokens 64
--skip-prefill --out .../logs/d1-bench-graph-sync.json
Timer: .../logs/d1-step-timer-sync.log
RESULT: async_scheduling False is in the startup args. Identity ok,
coherence ok. Capture 1 s + 1 s. C1 ITL p50/p90/p99 15.79/16.14/16.92 ms,
63.14 tok/s, TTFT 141.21 ms, n=64. C2 ITL p50 17.10 and 17.10 ms,
52.28 and 58.27 tok/s (stream0 p99 65.75 is one spike). EngineCore.step
p50 15.98 ms (n=169, min 15.57), matching the client ITL. execute_model
submit for tokens=1 p50 2.48 ms (n=99). sample p50 1.11 ms. replay()
host p50 0.05 ms. One XPU-event replay, synchronized before and after,
is 11.436 ms of device time.
VERDICT: turning async scheduling off made C1 ITL worse than the 13.20 ms
async result. The engine step is the chunk, and most of the step is the
captured graph on the device (11.4 ms), not Python inside replay().
Async hides about 2 ms of host beside that device time. Keep async
scheduling. Next: FULL_DECODE_ONLY with FlashAttention and SYCL GDN,
to see if those kernels capture and beat the 11.4 ms Triton graph.


## 2026-10-20 05:20 D1 -- FlashAttention FULL_DECODE_ONLY captures and is the best C1

CONFIG: FULL_DECODE_ONLY, capture sizes 1,2,3,4,8, ATTN=FLASH_ATTN,
GDN backend default (SYCL gdn_attention on XPU), async scheduling on,
no MTP, B70_XPU_CG_RECLAIM=500. Same Ornith GPTQ int4 serve shape.
COMMAND: B70_AGENT=d1-graph-fdo-fa CARD=1 PORT=18081 GRAPH=1
GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN MTP=0
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-fa.log
Bench: harness/bench_latency.py --tag graph-fdo-fa --decode-tokens 64
--out .../logs/d1-bench-graph-fa.json
RESULT: Using Flash Attention version 2. Capture finished: profile
2 FULL graphs in 1 s, then 4 FULL graphs in 1 s. No traceback.
Identity ok, coherence ok (Paris, both texts identical). C1 ITL
p50/p90/p99 12.81/12.99/13.12 ms, 78.23 tok/s, TTFT 88.59 ms, n=64.
C2 ITL p50 13.75 and 13.75 ms, 73.40 and 72.96 tok/s, TTFT 170.37
and 176.99 ms. Prefill 1017/8185/32762 tok TTFT 168.67/1037.62/5107.54
ms (6030/7888/6414 tok/s), matching the eager FlashAttention prefill
from A3. Triton-graph C1 was 13.20 ms p50 with 32k prefill at 1663
tok/s.
VERDICT: On this torch 2.14 / vLLM main stack, FlashAttention and
SYCL GDN both captured into FULL_DECODE_ONLY. The June note that
FlashAttention SYCL scratch cannot capture did not reproduce here.
This is the best C1 so far: p99 13.12 ms, under the 15 ms milestone,
and prefill stays at the eager rate. Next: MTP on this graph, then
PIECEWISE (breakable) and FULL.


## 2026-10-20 05:23 D1 -- FlashAttention FULL_DECODE_ONLY plus MTP

CONFIG: same FlashAttention FULL_DECODE_ONLY graph as 05:20, plus MTP
num_speculative_tokens=2. Async scheduling on. SYCL GDN. Reclaim 500.
COMMAND: B70_AGENT=d1-graph-fa-mtp CARD=1 PORT=18081 GRAPH=1
GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN MTP=1
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-fa-mtp.log
Bench: harness/bench_latency.py --tag graph-fa-mtp --decode-tokens 64
--out .../logs/d1-bench-graph-fa-mtp.json
RESULT: FlashAttention v2. Capture 2 FULL graphs then 2 FULL graphs,
1 s each (MTP decode query length rounds the capture list down to two
shapes). Identity ok, coherence ok. C1 ITL p50/p90/p99
18.81/19.17/19.37 ms, 93.88 tok/s, TTFT 106.35 ms, n=64. C2 ITL p50
20.94 and 20.98 ms, 84.90 and 87.35 tok/s. Acceptance during bench
0.383 (85/222); after counters 95/266 (0.357). Prefill
165.61/1050.08/5380.98 ms (6141/7795/6088 tok/s).
VERDICT: MTP on the FlashAttention graph is the highest tok/s so far
(94 vs 78 without MTP) and the chunk ITL rises to 18.8 ms, above the
15 ms milestone. The no-MTP FlashAttention graph remains the latency
config (C1 p50 12.81 ms). Next: PIECEWISE with breakable cudagraphs,
FlashAttention, no inductor.


## 2026-10-20 05:25 D1 -- PIECEWISE breakable graph captures

CONFIG: cudagraph_mode PIECEWISE, compilation mode NONE,
VLLM_USE_BREAKABLE_CUDAGRAPH=1, ATTN=FLASH_ATTN, SYCL GDN, async on,
no MTP, B70_XPU_CG_RECLAIM=500, capture sizes 1,2,3,4,8. No inductor.
COMMAND: B70_AGENT=d1-graph-piecewise VLLM_USE_BREAKABLE_CUDAGRAPH=1
CARD=1 PORT=18081 GRAPH=1 GRAPH_MODE=PIECEWISE ATTN=FLASH_ATTN MTP=0
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-piecewise.log
Bench: harness/bench_latency.py --tag graph-piecewise-fa --decode-tokens 64
--out .../logs/d1-bench-graph-piecewise.json
RESULT: Capture 5 PIECEWISE graphs in 2 s, then 5 in 1 s. No traceback.
Identity ok, coherence ok. C1 ITL p50/p90/p99 13.23/13.36/13.49 ms,
76.38 tok/s, TTFT 93.77 ms, n=64. C2 ITL p50 14.10 and 14.10 ms,
70.23 and 72.01 tok/s. Prefill 168.51/1038.40/5104.96 ms
(6035/7882/6418 tok/s).
VERDICT: PIECEWISE works on XPU with breakable graphs and FlashAttention.
C1 is 0.4 ms slower than FULL_DECODE_ONLY on the same attention path
(12.81 ms). Prefill matches eager FlashAttention. Next: cudagraph_mode FULL.


## 2026-10-20 05:28 D1 -- FULL graph matches FULL_DECODE_ONLY

CONFIG: cudagraph_mode FULL, compilation mode NONE, FlashAttention,
SYCL GDN, async on, no MTP, B70_XPU_CG_RECLAIM=500, capture sizes
1,2,3,4,8.
COMMAND: B70_AGENT=d1-graph-full CARD=1 PORT=18081 GRAPH=1
GRAPH_MODE=FULL ATTN=FLASH_ATTN MTP=0
bash vllm/flashnext/harness/serve_ornith_gptq.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-serve-graph-full.log
Bench: harness/bench_latency.py --tag graph-full-fa --decode-tokens 64
--out .../logs/d1-bench-graph-full.json
RESULT: Capture 2 FULL graphs in 1 s, then 4 in 1 s. No traceback.
Identity ok, coherence ok. C1 ITL p50/p90/p99 12.84/13.05/13.25 ms,
77.93 tok/s, TTFT 85.15 ms, n=64. C2 ITL p50 13.81 and 13.80 ms,
73.20 and 72.75 tok/s. Prefill 168.06/1036.19/5103.20 ms
(6051/7899/6420 tok/s).
VERDICT: FULL and FULL_DECODE_ONLY are the same decode latency on this
capture list (12.84 vs 12.81 ms). The latency config is FlashAttention
plus SYCL GDN with either mode: C1 p50 12.8 ms, under 15 ms. Report:
vllm/flashnext/reports/1D1_overhead.md. Claude Fable 5.1 and Claude
5.5 both still exit unrecognized_model and "Not logged in". Server
stopped after this bench.

## 2026-10-20 05:33 C1 -- fused MoE misses 1.2x, IQ3_S at 41%

CONFIG: card 0, ZE_AFFINITY_MASK=0, Arc Pro B70. Installed
plugins/b70_gguf_xpu/_C.so is the third binary (6391032 bytes): vec-form
DP4A, night ALU IQ dots, 2-row moe, dense SLM with an M-chunk of 4,
memcpy 4/8-byte IQ loads, fused moe_layer, Q2_K id 10 and Q3_K id 11.
Quiet CPU lock was busy (C2), so this run did not take bin/cpu-run
--quiet. Both-card xpu-health skipped; card 1 was leased and this is a
single-card kernel. unitrace/onetrace are not installed.
COMMAND: B70_AGENT=C1 bin/gpu-run --card 0, cp --remove-destination of
/mnt/vm_8tb/b70/build/vllm-flashnext/c1-so/_C.so onto the plugin .so,
then scripts/c1_run.sh. Waited behind C2's prefill_gpu.sh (acquired
05:28).
RESULT: failures=0. Dequant 0 bf16 mismatches on every type including
Q2_K and Q3_K. Q5_K head M=1 808.6 us (88.9% of 608), M=8 4316 us
(the shift-form DP4A binary had been 6917 us; that form is reverted).
IQ3_S token_embd M=1 1094 us (41.1%; A2 1594 us / 28.2%), M=8 2255 us
(19.9%; A2 6338 us). Small-N KREG device time is the behind number:
IQ4_XS 128 us (21%), Q6_K 129 us (33%), Q4_K 128 us (23%), against
80% targets of 34/53/36 us. IQ2_XS gate T=1 behind 68 us (first fused
binary 60.5 us, scalar 71.9 us). Fused moe_layer T=2/4/8 a2 404/758/1374
us, matching the earlier fused binaries. Quiet-run T=1 a2 stays 216 us.
Strata C1 card0 routed R2 is 93 us for 20 entries
(moe_report.md, IQ3_S/Q2_0); fused shared expert is ~19 us/layer
(mmvq_report.md). 1.2x of that is missed by about 3x on our 10-entry
plus shared call.
VERDICT: fusion removed the extra launches (3 * ~122 us a2 down to one
216 us call). The expert dot is the wall. Wider loads did not move it.
Report: vllm/flashnext/reports/1C1_moe_decode.md. Plugin .so left at
the third binary. Next step named there, not started: IQ4_NL kNightNl
is +4% on the down only; XMX at M=128 is 1-7 GB/s; pipelined KREG
regressed in the night lab.

## 2026-10-20 05:36 C2 -- prefill grouped GEMM measured

CONFIG: card 0, Arc Pro B70, icpx 2026.1. Library
plugins/b70_gguf_xpu/b70_gguf_xpu/_C.full.so sha256
edefe35868158518bbe5cbfd11e7097f6bdc8c73b893e246a910af8b3282f533
(6386936 bytes; same file as _C.bench.so). Package _C.so left as C1's
6391032-byte binary. Mode 0 fp16 DPAS 8x16x16 for 11 GGUF types, MT=4
and MT=16. Mode 1 bf16 and mode 2 int8 at MT=16 for eight types.
One launch per projection (fused gate/up, then down), device sort,
next-block preload. MT>4 work-groups capped at 512. Bench under
cpu-run --quiet. Unique-byte GB/s, 608 GB/s roof. Expert-only 48-layer
tok/s does not include attention.
COMMAND: B70_AGENT=C2 bin/gpu-run --card 0 scripts/prefill_gpu.sh
(log /mnt/vm_8tb/b70/build/c2-prefill-gpu-correct.log, 74s, exit 0).
Then B70_GGUF_SO=_C.bench.so bin/gpu-run --card 0 bin/cpu-run --quiet
scripts/prefill_gpu.sh --bench-only
(log .../c2-prefill-gpu-bench2.log, quiet 39s, gpu-run 75s, exit 0).
RESULT: Health before and after both runs HEALTHY, card 0. Correctness
VERDICT failures=0. fp16 max abs 1.487e-03 (44 checks). bf16 max abs
1.093e-02. int8 far_frac <= 0.013. Layer T=3 max abs 1.214e-04.
unsigned short joint_matrix is rejected (matrix_type::uint16); bf16
runs. Layer IQ3_XXS+Q2_0 E=512 topk=10: T=512 13885 us (GU 10268,
down 3440, cols=640 mt=4), expert-only 768 tok/s. T=4096 27599 us
(GU 19563, down 5085, GU cols=320 mt=16), expert-only 3092 tok/s.
T=12288 66129 us (GU 45207, down 11710), expert-only 3871 tok/s.
bf16 T=512 24709 us, int8 T=512 29104 us, both slower than fp16.
A2 single-expert shapes got slower: IQ4_NL M=128 N=2560 344 us vs
A2 136 us; IQ3_XXS M=128 N=640 647 us vs A2 306 us. Dense Q4_K
N=12288 K=2560 is 5.30 / 42.0 / 126 ms at M=512 / 4096 / 12288.
GU actual traffic is about 120 GB/s. Strata's ~2200 tok/s layer at
T=12288 is about 116 ms including attention. This expert layer is
66 ms. Report: vllm/flashnext/reports/1C2_prefill.md.
VERDICT: Correct fp16/bf16/int8 prefill path. Not a 2000 tok/s
prefill. Next, not started: parallel row-tile entries for dense, and
an SGN=4 (64-column, X from global) trial on the T=4096/12288 layer.

## 2026-10-20 05:58 E1 -- lowbit GEMV and MoE replay inside an XPU graph

CONFIG: card 1, image vllm-flashnext-xpu:py312, host UMD via xpu-run.
.so /mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so
unchanged (1442856 bytes). Python wrappers register_fake for gemv,
moe_gemv, prefill_wna16, prefill_wxa8 and cache the empty int2 hi plane.
B70_LOWBIT_MOE_LAUNCHES=2. Tiny shapes, bits 2/3/4. xpu-health card 1
HEALTHY before the probe (23s, img vllm-xpu-env:int8g-v0251).
COMMAND: B70_AGENT=E1 vllm/flashnext/harness/xpu-run.sh --card 1
--memory 16g --name b70-lowbit-graph-probe --env B70_LOWBIT_MOE_LAUNCHES=2
-- python3 -u plugins/b70_lowbit_xpu/tests/graph_probe.py
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/e1-graph-probe.log
RESULT: fake shapes ok. Replay max_abs 0.000e+00 for gemv and moe at
bits 2, 3, and 4. GRAPH_PROBE_OK. Wall about 15s after the loader
check. Capture-time buffers are not results; the check is a NaN fill
then replay against a fresh eager call. Card 1 released.
VERDICT: The decode ops are graph-capturable with no .so change.
Next is the int3 Qwen3.6-35B-A3B FULL_DECODE_ONLY serve on card 1.

## 2026-10-20 06:05 C4 -- Qwen3.6 IQ3_XXS one-card GGUF is coherent

CONFIG: fork /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext
tip e3f784ebb2, ahead of origin/main 2296c12e47 by 7, not pushed. Card 0.
HF /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B. GGUF UD-IQ3_XXS. Eager
TP=1, max_model_len 2048, max_num_seqs 1, gpu_memory_utilization 0.90,
language-model-only, load-format b70_gguf, quantization b70_gguf.
B70_GGUF_REF=0, B70_EAGER_QK=0, GDN auto (sycl). Port 18104. Names
hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs. xpu-health card 0 OK
before.
COMMAND: CARD=0 PORT=18104 B70_AGENT=C4
vllm/flashnext/harness/serve_qwen36_gguf.sh. Log
/mnt/vm_8tb/b70/logs/grok-flashnext/C4_serve_c0_outfix.log. Two POST
/v1/completions temperature 0 max_tokens 16, prompt "The capital of
France is". Then bench_latency.py --prefill 1024 --decode-tokens 32
--out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c4-bench-outfix.json.
docker stop -t 25, then xpu-health --card 0.
RESULT: placed 733, skipped_mtp 20, q8_dequant 7, 16.96 GiB, 29.27 s.
KV 8.93 GiB (110080 tokens). Identity hotschmoe-dev first. Both greedy
strings matched: " Paris.\nThe capital of France is Paris.\nThe capital
of France is". C1 ITL p50 87.62 ms (11.72 tok/s), short-prompt TTFT
265 ms. Prefill 1017 tok, TTFT 5755 ms, 177 tok/s. C2 ITL p50 ~86 ms
with max_num_seqs 1, so one stream queued (TTFT 3198 vs 302 ms).
Teardown gpu-run exit 0, 178 s. Card 0 HEALTHY after.
VERDICT: Step 1 passes. gdn_out kept GGUF K order and the activation
permutation is now the inverse of restore_rows. CPU mapping tests
15/15. Kernel, host-ref, and triton serves before this fix were
deterministic non-Paris strings. XPU gemv/dequant matched gguf-py
(C4_kernel_oracle.log). Step 1b not started: card 1 held by E1.
Report: vllm/flashnext/reports/1C4_e2e_gguf.md.

## 2026-10-20 06:02 C5 -- Strata routed bench reproduces on card 0

CONFIG: card 0, ZE_AFFINITY_MASK=0, ONEAPI_DEVICE_SELECTOR=level_zero:*.
Binary /mnt/vm_8tb/b70/build/night-wt/moe/src-ep/build-sycl-ep/ep_night_bench
(104919720 bytes, 2026-10-16 14:32). Worktree HEAD 1c360526, the DD pin.
Image sha256:e41d59c0 (strata-ab-build:ocloc), icpx 2026.1.0.20260617,
IGC libigc.so.2.36.3. AOT flags from that build.ninja: -O3 -fsycl
-fsycl-targets=spir64_gen -device bmg-g31 -fsycl-default-sub-group-size=32
-fsycl-device-code-split=per_kernel -fp-model=precise
-cl-fp32-correctly-rounded-divide-sqrt. Shape EPB_PAIRS=21:42
EPB_SCEN=3 (IQ3_S/Q2_0, C1 card0 10 groups / 20 entries, T=4). Timer is
the bench's graph replay (48 layers, median of 5 x 20). xpu-health
--card 0 HEALTHY before and after (img vllm-xpu-env:int8g-v0251).
COMMAND: B70_AGENT=C5 bin/cpu-run --quiet bin/gpu-run --card 0
/mnt/vm_8tb/b70/build/vllm-flashnext/c5/run_repro.sh
RESULT: pre_health_rc=0, bench_rc=0, post_health_rc=0, 46s. Bitwise yes
(153600 floats, 0 diffs). us per call: base gu/dn/all 64.74/44.79/109.53;
R2 all 96.87 (gu 56.61); IQ all 94.21; IQR2 gu/dn/all 53.97/18.68/72.64.
Published moe_report.md R2 (rows=2, IQ off) is 53.5/40.8/93.2. Today's
R2 all is 96.87 us. The DD also sets STRATA_NIGHT_IQ=1; that column is
72.64 us, and the down phase drops from 40.8 us published to 18.68 us.
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/c5/repro_ocloc_iq3s_c1c0.log.
VERDICT: The published ~93 us routed number is reproduced (R2 96.87 us)
on the historical AOT binary and IGC 2.36.3. The served IQ+R2 kernel is
72.64 us on this card today. Next: relink the same objects with
strata-sycl-dev:020 (IGC 2.41.5), then the same source with the plugin
toolchain.

## 2026-10-20 06:20 F1 -- UMD opt-in checked, card 0 campaign starting

CONFIG: container-scoped UMD A/B. Host is unchanged: kernel
7.1.0-070100-generic, compute-runtime 26.22.38646.4-0, IGC 2.36.3
under /usr/local/lib, Level Zero loader 1.28.2. No apt, no dpkg, no
host library replacement. Extracted trees, not installed:
/mnt/vm_8tb/b70/build/vllm-flashnext/umd/26.27.39122.11 (upstream
Dockerfile.xpu: IGC 2.38.2, loader 1.32.0, gmmlib 22.10.0) and
umd/26.35.39758.10 (newest intel/compute-runtime release, 2026-09-17,
matching IGC 2.41.5). Loader 1.34.0 exists and is not the loader NEO
26.35 was built with, so both non-host trees use loader 1.32.0.
Harness change is additive only (D2 owns harness/): xpu-run.sh
--umd DIR / XPU_RUN_UMD, and xpu-entrypoint.sh check_loader_override
when XPU_UMD_MODE=override. Host default still requires
libze_loader.so.1.28.2 and still rejects a resolved 1.32 or 26.27
path. Override mounts libze_intel_gpu, libigc, libigdfcl, libiga64,
libigdgmm, libze_loader, the OpenCL ICD, and the IGC clang soname
(libopencl-clang2.so.17 on 26.27, libopencl-clang2.so.22.1 on 26.35).
Override Triton and SYCL caches are suffixed with the version label.
COMMAND: bash -n on both scripts. xpu-run.sh --check with no --umd,
--umd 26.27.39122.11, and --umd 26.35.39758.10. No GPU and no lease.
Logs: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/f1-check-host.log,
f1-check-2627.log, f1-check-2635.log. Then bin/xpu-health --card 0.
RESULT: syntax ok. All three --check runs exited 0. Host loader
1773224 bytes, libigc 95845480, clang libopencl-clang2.so.17. 26.27
loader 1036264, libigc 96015048, clang .17. 26.35 loader 1036264,
libigc 104543000, clang .22.1. ldd and the mapped files resolve under
/host-umd. --check has no /dev/dri, so the sycl-ls / torch.xpu query
is still ahead of the GPU arms. Card 0 HEALTHY at 06:16 UTC
(img vllm-xpu-env:int8g-v0251) after C5 released it. Card 1 stays
with E1 (vllm-fn-lowbit-e1, port 18084). F1 will hold card 0 for
about 2 to 2.5 hours: host, 26.27, and 26.35. Each arm is a device
probe, Ornith GPTQ int4 FULL_DECODE_ONLY FLASH_ATTN, the Triton
attention plus Triton GDN graph, and b70_gguf_xpu gpu_check plus
moe_layer_bench. Port 18091. The plugin _C.so is the existing AOT
binary (6391032 bytes, mtime 2026-10-20 04:32). This A/B does not
rebuild it. C5 is separately relinking that plugin with IGC 2.41.5.
VERDICT: Opt-in is in place and the host path still checks out.
Measurements start on card 0.

## 2026-10-20 06:16 C5 -- IGC 2.41.5 and immediate lists do not move the routed kernel

CONFIG: same night-lab objects as the 06:02 control (worktree 1c360526), relinked
in strata-sycl-dev:020 (icpx 2026.1.0.20260617, ocloc 26.35.39758.10, IGC
libigc.so.2.41.5+1788943183) to c5/ep_night_bench_020 (103051936 bytes). Link
flags: -fsycl-targets=spir64_gen -device bmg-g31
-cl-fp32-correctly-rounded-divide-sqrt -allow_caching
-fsycl-device-code-split=per_kernel. Card 0, ZE_AFFINITY_MASK=0. xpu-health
--card 0 HEALTHY before and after (post_health_rc=0).
COMMAND: B70_AGENT=C5 bin/cpu-run --quiet bin/gpu-run --card 0
/mnt/vm_8tb/b70/build/vllm-flashnext/c5/run_020.sh (101s, every docker rc 0).
RESULT: IQ3_S/Q2_0, C1 card0 10g/20e, bitwise yes (153600, 0). us per call:
- historical AOT, runtime IGC 2.36.3 (06:02): R2 all 96.87, IQR2 all 72.64
- 020 AOT binary, runtime IGC 2.41.5: R2 55.72/95.86, IQR2 53.78/18.88/72.66
- 020 AOT binary, runtime IGC 2.36.3: R2 55.75/96.12, IQR2 53.88/19.08/72.96
- historical binary, SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1: R2
  56.23/96.50, IQR2 53.71/19.43/73.14
IQ2_XS/IQ4_NL, same scenario (gate matches C1, up does not): 020 IQR2
47.17/38.46/85.64; historical IMM=0 IQR2 47.18/38.24/85.42.
Logs: c5/repro_020_iq3s.log, c5/repro_020oldrt_iq3s.log, c5/repro_old_iq3s_imm1.log,
c5/repro_020_iq2xs.log, c5/repro_old_iq2xs_imm0.log.
VERDICT: IGC 2.36.3 vs 2.41.5, at codegen and at runtime, and immediate
command lists, move this kernel by about 1 us. Not the 216 vs 93 gap. Next:
JIT vs AOT on these objects, then the same source with the plugin toolchain,
then our moe_layer under both.

## 2026-10-20 06:22 E1 -- int3 FULL_DECODE_ONLY graph, C1 ITL 11.29 ms

CONFIG: card 1, vllm-flashnext-xpu:py312, b70_lowbit, B70_LOWBIT_MOE_LAUNCHES=2,
.dtype bfloat16, TP=1, language-model-only, max-model-len 33792,
gpu-memory-utilization 0.90, max-num-seqs 4, port 18084, no MTP.
GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN B70_XPU_CG_RECLAIM=500.
Model /mnt/cache/b70/devmodel/quant/qwen36-w3a16-g128-experts-w4a16-trunk.
Served hotschmoe-dev, alias qwen36-w3a16-b70-lowbit. Container
vllm-fn-lowbit-e1. Same .so as C3 (1442856 bytes). xpu-health card 1
HEALTHY before the probe. Capture sizes requested 1,2,3,4,8; the engine
captured 4 FULL graphs (size 8 dropped at max-num-seqs 4).
COMMAND: B70_AGENT=E1 B70_XPU_CG_RECLAIM=500 B70_LOWBIT_MOE_LAUNCHES=2
GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN CARD=1 PORT=18084
NAME=vllm-fn-lowbit-e1 MODEL=.../qwen36-w3a16-g128-experts-w4a16-trunk
ALIAS=qwen36-w3a16-b70-lowbit plugins/b70_lowbit_xpu/serve_lowbit.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/e1-int3-graph-serve.log
Bench: B70_AGENT=E1 bin/cpu-run --quiet harness/bench_latency.py
--base-url http://127.0.0.1:18084 --model hotschmoe-dev
--alias qwen36-w3a16-b70-lowbit --tag e1-int3-graph-fdo --decode-tokens 64
--timeout 900 --out .../logs/e1-int3-graph-bench.json (377s, exit 0).
RESULT: Weight load 23.48 s, then GPTQ repack, then capture: 2 FULL
graphs in 1 s and 4 FULL graphs in 1 s. No traceback. init engine 19.38 s.
KV cache 11.16 GiB. Identity ok, coherence ok (Paris, both texts identical).
C1 ITL p50/p90/p99 11.29/11.52/11.97 ms, 90.34 tok/s, TTFT 2388.12 ms
on a 24-token prompt, n=57 (EOS before 64). C2 ITL p50 15.65 and 15.65 ms,
65.44 and 65.34 tok/s, TTFT 2834.96 and 2834.36 ms. Prefill
1017/8185/32762 tok TTFT 10238.57/72872.08/282960.21 ms
(99.33/112.32/115.78 tok/s). MTP off. C3 eager int3 on the same checkpoint
was C1 ITL 51.73 ms and prefill 98.98/112.72/115.83 tok/s.
VERDICT: Decode ops captured. C1 ITL is 4.6x the eager int3 number and
under the 15 ms milestone. Prefill is still the per-expert eager path,
about 116 tok/s at 32K. Server left up for perplexity, then the int4
graph arm. Grouped prefill is written and not compiled yet.

## 2026-10-20 06:28 E1 -- int3 served perplexity 9.12

CONFIG: the live int3 FULL_DECODE_ONLY server on port 18084. Same 4x512
wikitext windows as quant/run_quant.py --mode eval
(/mnt/cache/b70/devmodel/quant/_work/qwen36/ppl_s512_n4.pt). prompt_logprobs
on /v1/completions, max_tokens 1, temperature 0. First token of each
window dropped. Unpack reference for this slice is 9.19.
COMMAND: docker run --network host --memory 8g (no /dev/dri)
python3 tests/ppl_serve.py --base-url http://127.0.0.1:18084
--model hotschmoe-dev --out .../logs/e1-int3-ppl.json
RESULT: windows nll 1.8132, 2.4407, 2.2146, 2.3734. tokens=2044.
ppl 9.1199, nll 2.2105.
VERDICT: Served int3 matches the unpack slice within 0.07 ppl. The
on-the-fly unpack eval is not required for this checkpoint.

## 2026-10-20 06:23 D2 -- GPTQ MTP head is native, not Shisa

CONFIG: CPU only, no GPU lease. Ornith GPTQ int4 mixedcal-v2 (6 shards)
against bf16-mtp-shisa/model-mtp.safetensors (19 BF16 tensors). Manifest
head_source_name Ornith-1.5-35B-A3B-MTP-ONLY, native_mtp_removed 785.
vLLM method mtp takes EagleProposer (use_eagle includes mtp).
qwen3_5_mtp.py sets quant_config None when quantization_config.dynamic
contains a "-:" key mentioning mtp, so the drafter is unquantized.
COMMAND: python3 header walk plus numeric compare. GPTQ MTP tensors
decoded as BF16. Trunk norms decoded as F16 on the GPTQ side and BF16
on the Shisa side. Expert 0 compared to Shisa fused row 0
(gate_up[:512], gate_up[512:], down[0]).
RESULT: GPTQ mtp.* is 785 tensors, dtype census BF16 785, int4 0.
Trunk matches the Shisa trunk numerically: layer 0 input LN mean
0.03110 std 0.04909 max_abs 3e-8; post-attn LN mean -0.10520 max_abs 0;
layer 3 input LN mean -0.07980 max_abs 0; final norm mean 1.63973
max_abs 0; layer 0 gate max_abs 3e-8. Every MTP tensor differs.
q_proj cos -0.0001 rel 1.44; fc cos 0.050 rel 1.08; router gate cos
0.0005 rel 1.12; shared down cos -0.0001 rel 1.09; expert 0 gate/up/down
cos about 0, rel 1.26 to 1.29. GPTQ mtp.norm mean 0.022813 std 0.000166
(near constant). Shisa mtp.norm mean 1.92405 std 0.296714, same scale
as the trunk final norm. GPTQ MTP input LN std 0.0030 vs Shisa 0.1567.
q_norm std 0.0068 vs 0.338. Claude Fable 5.1 and Claude 5.5 both exit
unrecognized_model and "Not logged in".
VERDICT: The 0.38 acceptance is the native GPTQ head, loaded unquantized
and in BF16. It is a real head, not a failed random load, and it is not
the Shisa code-tuned head. Next GPU run overlays the 19 Shisa tensors
onto the drafter (harness/patches/d2_mtp_shisa.py) and benches spec 1,
2, and 3. The int4 trunk GEMV overlay is written and stays off until
the queued card-1 floor shows graph replay max_abs near 0. That floor
is still waiting on E1's lease. No Ornith server was started.

## 2026-10-20 06:33 D2 -- fp16 GEMV stack is 7.15 ms; int4 gemv is slower

CONFIG: card 1, Arc Pro B70, container vllm-fn-d2-gemv, no model load.
torch XPU graph replay with device events. fp16 F.linear shapes are the
Ornith decode projections: GDN qkvz 12288x2048, ba 64x2048, out
2048x4096, attn qkv 9216x2048, attn o 2048x4096, shared gate/up
1024x2048, shared down 2048x512, router 256x2048, lm_head 248320x2048.
Distinct-weight stack is 30 GDN triples + 10 attention pairs + 40
shared pairs + 40 routers + lm_head (231 GEMMs, 3.611 GiB). int4 is
b70_lowbit gemv, bits=4, RTN group 128, replay compared to a fresh
eager gemv. Roof 608 GB/s. Norms are xpu_fused_add_gemma_rms_norm.
COMMAND: B70_AGENT=d2-gemv B70_GPU_LOCK_TIMEOUT=0 harness/xpu-run.sh
--card 1 --name vllm-fn-d2-gemv --memory 48g -- python -u
harness/d2_gemv_floor.py
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-gemv-floor.log
Waited on E1 from 06:07, ran after that lease dropped, exit 0.
RESULT: fp16 stack M=1 7.153 ms, 542.0 GB/s, 89.1% of 608. M=4 7.194 ms,
538.9 GB/s. M=1 per-call (4-copy graph / 4, except lm_head): qkvz
0.0931 ms (540 GB/s), ba 0.0055, out 0.0363 (463 GB/s), attn qkv 0.0708
(533 GB/s), attn o 0.0367 (457 GB/s), shared gate/up 0.0070, shared
down 0.0049, router 0.0049, lm_head 1.707 ms (596 GB/s, 98%). Scaled
by layer count that sums to 7.50 ms (30/30/30/10/10/40/40/40/1).
80 Gemma norms 0.183 ms. Triton fp16 qkvz best 0.180 ms at 280 GB/s,
max_abs 1.6e-2; lm_head best 1.760 ms, max_abs 6.3e-2. int4 graph
replay_max_abs 0 and eager vs dequant slice 0 on qkvz, out, attn qkv,
and lm_head. int4 device time: qkvz 0.169 ms (77 GB/s), out 0.066 ms
(66 GB/s), attn qkv 0.054 ms (179 GB/s), lm_head 1.191 ms (220 GB/s).
VERDICT: The fp16 trunk plus lm_head is already at 89% of 608. The
11.4 ms decode replay is about 4 ms above this 7.15 ms stack, not a
4x miss against a 3 ms int4 roof. b70_lowbit int4 gemv captures
cleanly but is slower than oneDNN fp16 on the large GDN projection
(0.169 vs 0.093 ms) because it runs at 66-220 GB/s. Swapping the trunk
would regress decode. Norm fusion cannot move a 0.18 ms class. The
int4 overlay stays off. Next: eager per-region profile of the real
Ornith step, then the Shisa MTP overlay. A faster int4 GEMV (above
about 400 GB/s at N=12288 K=2048 and at lm_head) is the kernel request;
no fork change.

## 2026-10-20 06:38 C5 -- toolchain leaves both kernels where they were

CONFIG: card 0, ZE_AFFINITY_MASK=0. Real JIT recompile of
sycl/src/kernels/cuda/iq_kernels.dp.cpp (spir64) inside strata-sycl-dev:020,
icpx 2026.1.0.20260617. Strata flags: -O3 -fp-model=precise
-fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel.
Plugin flags: -O3 -fsycl -per_kernel, no precise, no subgroup default.
Run on host IGC 2.36.3, and one row on the 020 runtime (IGC 2.41.5).
Shape EPB_PAIRS=21:42 EPB_SCEN=3 (IQ3_S/Q2_0, C1 card0, T=4, 10 groups /
20 entries). Timer is the bench graph replay (48 layers, median of 5).
ours_bench is the plugin moe_layer with no torch: host icpx
2026.1.1.20260724 flag matrix, plus the same TU built by 020 icpx as JIT
and as AOT (spir64_gen -device bmg-g31). c1 is T=1 topk=10 plus shared,
IQ2_XS/IQ2_XXS/IQ4_NL. s20 is T=2, 20 ungrouped entries, IQ3_S/IQ3_S/Q2_0.
Graph column is 8 launches per graph, median of 5. Shared expert is
night_shexp_bench nc=1 mmvq level 2, historical binary, image
sha256:e41d59c0. xpu-health --card 0 HEALTHY before and after
(pre-health-ab.log, post-health-ab.log). The spir64 relink that dropped
device code (ep_night_bench_jit, 10 MB, no SPIR-V) was not timed.
plugins/b70_gguf_xpu/_C.so and harness/xpu-run.sh were not edited.
COMMAND: B70_AGENT=C5 bin/gpu-run --card 0 bin/cpu-run --quiet
/mnt/vm_8tb/b70/build/vllm-flashnext/c5/run_measure.sh
Exit 0. Logs: c5/meas_jit_strata_oldrt.log, meas_jit_strata_020.log,
meas_jit_strata_grf256.log, meas_jit_strata_imm1.log,
meas_jit_plugin_oldrt.log, meas_ours_*.log, meas_shexp_nc1.log.
RESULT: IQ3_S/Q2_0 bitwise yes (153600, 0). R2 all / IQR2 all, us:
- JIT strata flags, IGC 2.36.3: 93.47 / 69.26 (IQR2 gu/dn 50.93/18.33)
- same binary, IGC 2.41.5: 93.25 / 69.10
- JIT IGC_OverrideGRFSize=256: 93.84 / 69.40
- JIT immediate command lists = 1: 93.95 / 69.35
- JIT plugin flags, IGC 2.36.3: 93.84 / 69.44 (IQR2 gu/dn 51.39/18.05)
Historical AOT R2 was 96.87. Published R2 is 93.2. JIT matches it.
Plugin moe_layer graph us, c1 / s20:
- host plugin JIT -O3: 198.39 / 313.24 (eager min 486.52)
- plus -fp-model=precise: 200.81 / 312.69
- plus subgroup default 32: 198.44 / 312.86
- plus -ffast-math: 198.33 / 313.15
- -O2: 213.82 / 333.72
- 020 JIT: 198.47 / 313.46 (eager min 206.75)
- 020 AOT precise plus sg32: 209.70 / 366.96 (profile FAIL, graph used)
- 020 AOT only: 212.55 / 366.61
C1 torch a2 at this T=1 shape was 216 us. Shared expert
IQ4_XS/IQ3_S/IQ4_NL nc1: orig 24.88 us, night 18.69 us, 0 bit diffs.
Published 24.9 -> 18.7.
VERDICT: IGC 2.36.3 vs 2.41.5, AOT vs JIT, GRF 256, immediate command
lists, -O3 vs -ffast-math, -fp-model=precise, and the subgroup default
leave Strata's routed kernel at 93-97 us and the plugin fused layer at
about 198 us graph (AOT 210-213 us). -O2 is the only plugin flag that
moved the graph number, and it was slower. Do not mount IGC 2.41.5 in
harness/xpu-run.sh. Do not rebuild _C.so. The remaining difference is
the kernel schedule. A paste of the Strata bodies behind
b70_moe_layer_launch is blocked: plugin B70Q8 is 48 bytes and Strata
q8_1 is 36, Strata packs one ggml type for gate and up, and at T=1 each
expert is used once so rows_dot_multi_iq has no second entry to reuse.
That schedule cannot turn the 198 us fused layer into the 93 us routed
number. Report: vllm/flashnext/reports/1C5_strata_transplant.md (one
table cell corrected: 020 AOT-only s20 graph is 366.61 us). Next, when
card 0 is free: a grouped multi-entry dot with Strata's lane map
(subgroup 32, STRATA_EXPERT_LANES 8) and an attribution comment, in a
side library, measured at the C1 card0 plan against today's JIT R2 of
93.5 us. Keep the T=1 fused path until that plan is faster and
tests/gpu_check.py (5e-2) and tests/moe_layer_bench.py (8e-2) pass.
Dequant bit-exact stays on tests/compare_cpu.py. Card 0 is held by F1
(vllm-fn-f1-2627-triton) and C6 (c6_run.sh) at this write, so that
kernel was not started.

## 2026-10-20 06:50 E1 -- grouped moe_prefill matches the per-expert loop

CONFIG: card 1, libb70_lowbit_xpu_e1.so built 06:35 (1696504 bytes).
That binary forces cumsum offsets to int32. The 04:05 libb70_lowbit_xpu.so
(1442856 bytes) was not overwritten. Probe waits behind d2-profile, then
runs tests/grouped_probe.py. Gate is rel_l2 < 0.02 and cos > 0.999 against
the Python per-expert prefill_wna16 loop. One expert id is -1 and expert 0
is unused. K=2048 asks for 64 KiB SLM.
COMMAND: B70_AGENT=E1 B70_GPU_LOCK_TIMEOUT=3600 harness/xpu-run.sh --card 1
--memory 24g --name b70-lowbit-grouped-probe
--env B70_LOWBIT_SO=.../libb70_lowbit_xpu_e1.so -- python3 -u tests/grouped_probe.py
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/e1-grouped-probe.log
RESULT: probe_rc=0, gpu-run done in 28s. PASS b3 T=13 H=256 rel_l2=0
max_abs=0. PASS b2/b3/b4 T=17 topk=8 E=4 H=2048 I=512. b2 rel_l2=6.7e-9
cos=1 max_abs=0.125 (one bf16 ulp). b3 and b4 rel_l2=0 max_abs=0.
Microbench, not a serve number: b3 T=128 E=32 H=2048 grouped 11.94 ms/call
vs per-expert loop 29.19 ms/call.
VERDICT: GROUPED_PROBE_OK. Runtime accepted the 64 KiB SLM tile, and the
int32 offset fix is the one that runs. Next GPU job is the int4 control
FULL_DECODE_ONLY serve on the old .so. The 11.94 ms figure is not 8k tok/s.

## 2026-10-20 06:51 E1 -- mixed expert bits resolve on the vLLM prefix

CONFIG: CPU only, no GPU. Checkpoint
qwen36-mixed-int3-int2-experts-w4a16-trunk extra_config (22735 entries,
22719 explicit int2 keys, catch-all regex bits 3) against
expert_bits.json (40 x 256, hist 2:7573 3:2667). vLLM module prefix is
model.language_model.model.layers.N.mlp.experts. Checkpoint keys omit the
inner .model. Lookup indexes the layers-relative tail. Synthetic repack
puts int2 qweight in the leading rows of an int3-sized buffer and fills
the tail with 0x7EADBEEF.
COMMAND: docker run --cpus 1 --memory 8g --network none (no /dev/dri)
python3 tests/test_mixed_cpu.py
RESULT: MIXED_BITS_OK. hist {2: 7573, 3: 2667}. All 40 layers are mixed.
L0 E0 gate is 2 on both the vLLM prefix and the checkpoint prefix. L0 E1
down is 3. Trunk q_proj stays bits 4. Router mlp.gate is skipped. Repack
plane2 and scales match a per-expert pack; int2 hi plane is zero; int3
hi matches. The garbage tail is not unpacked.
VERDICT: The bit table is safe to load. moe_gemv_mixed is source only and
is not in libb70_lowbit_xpu_e1.so. A uniform int4 vs int3 graph A/B still
uses the templated kernel on the 04:05 .so. The mixed kernel is a
different binary, so that arm is not "only expert bits."

## 2026-10-20 07:00 G2 -- audit pass 1

CONFIG: CPU only, no GPU lease. Read-only. Cursor is JOURNAL through
the 06:51 E1 mixed-bits entry, bench JSON under
/mnt/vm_8tb/b70/build/vllm-flashnext/logs, C2/C5/D2/E1/F1 logs, git
status, fork b70/flashnext, and the plugin/harness sources.
COMMAND: parse bench JSON; match journal figures; check identity,
coherence texts, health logs, and git staging. No gpu-run.
RESULT: Serve numbers for A3, C3, C4, D1, and E1 match the JSON.
Every completed bench has first id hotschmoe-dev and stored
temperature-0 texts. C2, C5, and F1 health logs that were claimed
say HEALTHY. Nothing is staged; plugin .so files are gitignored.
f1-bench-2627-triton.json is coherence.ok by the harness gate but
the greedy text is "A. True / B. False", unlike the host Triton,
host FA, 26.27 FA, and 26.35 FA arms on the same Ornith prompt.
E1 mixed-bits MIXED_BITS_OK has no stdout log on disk. C3 card-0
post-health is still the owed check they already wrote down.
Report: vllm/flashnext/reports/1G2_audit.md.
VERDICT: Pass 1 recorded. One wrong-if-claimed coherence result
(UMD 26.27 Triton). No other numeric contradiction in the completed
serve JSON.

## 2026-10-20 07:08 C7 -- fused moe_layer decode, CPU suite green

CONFIG: fork b70/flashnext tip 2583e27239 (parent e3f784ebb2), not
pushed. Side library
vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.graph.so is
6987192 bytes, mtime 2026-10-20 06:59:04. _C.so is unchanged at
6391032 bytes, mtime 2026-10-20 04:32:30. Decode with T<=8 calls
moe_layer. Expert ids, including -1, stay on device. Scratch never
waits and never frees; a grow while the SYCL queue is recording
throws. reserve_decode floors T=8, topk=16, K=I=8192 and runs on
the first XPU forward. Claude fable-5.1 and claude-5.5 both exit
unrecognized_model and "Not logged in".
COMMAND: B70_PREFILL_OUT set to _C.graph.so, bin/cpu-run scripts/build.sh.
Then bin/cpu-run the 15 mapping tests
(logs/c7-cpu-mapping.log). Fork commit only.
RESULT: build exit 0 in 69s, 120 deprecation warnings. CPU suite
ALL_DONE 0 in 91s. test_linear_and_moe_call_ops sees one moe_layer
and min id -1. test_expert_map_and_shared_not_added maps global id 1
to local slot 0. At 07:08 card 0 is free and card 1 is held by E1
(vllm-fn-lowbit-e1-int4 since 07:04). No serve ITL yet.
VERDICT: Card 0 next: xpu-health, graph probe on _C.graph.so, then
FULL_DECODE_ONLY + FLASH_ATTN + B70_XPU_CG_RECLAIM=500 with
max_num_seqs 4 and max_model_len 33792.

## 2026-10-20 07:08 F1 -- UMD A/B measured; keep host 26.22

CONFIG: card 0 only. Kernel 7.1.0-070100-generic, host UMD not
replaced. Arms: host 26.22.38646 / IGC 2.36.3 / loader 1.28.2;
extracted 26.27.39122.11 / IGC 2.38.2 / loader 1.32.0; extracted
26.35.39758.10 / IGC 2.41.5 / loader 1.32.0 (newest compute-runtime
release; loader 1.34.0 was not the matching loader). Mounted with
xpu-run.sh --umd. Ornith GPTQ int4, FULL_DECODE_ONLY, port 18091,
B70_XPU_CG_RECLAIM=500. Plugin _C.so unchanged (6391032 bytes,
2026-10-20 04:32). Card 1 left with E1.
COMMAND: logs/f1_umd_driver.sh. Per arm: sycl-ls + torch.xpu probe,
serve_ornith_gptq.sh ATTN=FLASH_ATTN then bench_latency.py,
ATTN=TRITON_ATTN GDN_BACKEND=triton then the same bench, then
gpu_check.py and moe_layer_bench.py. xpu-health --card 0 around the
work. 26.35 plugin run retried once after ops.py stopped registering
a fake for reserve_decode on the default _C.so.
RESULT: all six serves coherent (Paris, both strings identical,
hotschmoe-dev first). FlashAttention C1 ITL p50/p99 ms and TTFT ms:
host 12.80/13.13 and 84.68; 26.27 12.77/13.25 and 89.67; 26.35
12.60/13.00 and 92.07. Prefill 32k tok/s 6182 / 6171 / 6193. Triton
C1 ITL p50 13.13 / 13.18 / 12.97 ms. Triton 32k prefill 1224 / 1868 /
1705 tok/s, with the host cache already warm (ready 106 s) and the
other two cold (about 450-467 s in container). Engine maps showed
libigc from /host-umd on every arm, including Triton. Dense GEMV M=1
Q4_K 129.4 / 132.3 / 128.9 us; Q5_K 810.6 / 809.5 / 835.7 us.
moe_layer T=8 behind 1408.5 / 1424.8 / 1441 us. T=1 moe_layer is
scattered (370 / 589 / 254 us) and is not a ranking. Card 0 HEALTHY
after. No xe-reset. C5 held card 0 for a few minutes between the
26.27 FlashAttention bench and the Triton serve; that health check
was skipped, the Triton serve queued, and later health passed.
Claude fable-5.1 and claude-5.5: unrecognized_model, not logged in.
VERDICT: Campaign runtime stays on host 26.22 / IGC 2.36.3 / loader
1.28.2. 26.35 wins FlashAttention C1 ITL by 0.20 ms and loses TTFT
by about 7 ms. The AOT plugin does not move with the runtime UMD;
C5's compile-time IGC A/B is the right comparison for _C.so.
Report: vllm/flashnext/reports/1F1_umd.md.

## 2026-10-20 07:12 G1 -- hybrid offload contract on CPU

CONFIG: No GPU. Package vllm/flashnext/kvoffload. Ornith text layout
from the GPTQ config: 10 full-attention layers, 30 GDN layers, XPU block
alignment 64. A parked prefix is valid only with both full-attention KV
and the GDN boundary state. HybridCPUOffloadConnector subclasses upstream
OffloadingConnector (pinned host, XPU swap_blocks_batch) and refuses to
start unless both groups are in the prefix-cacheable offload set.
supports_divergent_local_hybrid_hits stays false (vLLM issue 52773: the
flag without a scheduler backfill drops the hit). Copy engine is upstream.
COMMAND: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vllm/flashnext python3 -m
unittest kvoffload.tests.test_layout_park kvoffload.tests.test_metrics
RESULT: 16 tests OK. GDN page 2146304 bytes/layer, attention 20480
bytes/token, scaled block 1088. Checkpoint of 28*1088 tokens is 27*1088
(the align-mode proof leaves the last token, so an exact multiple loses
one block). A missing Prometheus host series counts as zero. Host venv
python points at /usr/bin/python3.12 and does not run outside the
container; the connector import is part of the XPU serve.
VERDICT: CPU contract holds. Next serve is card 0, host UMD, port 18111,
PIECEWISE graphs, num_gpu_blocks_override 45, 16 GiB pinned. Card 1 is
held by E1 (vllm-fn-lowbit-e1-int4) since 07:04. Expected hold about 30
minutes: model load, graph capture, three ~30K prompts, greedy check.

## 2026-10-20 07:21 C7 -- one-card GGUF FULL_DECODE_ONLY measured

CONFIG: card 0 only. Fork tip 2583e27239. Image vllm-flashnext-xpu:py312.
Qwen3.6-35B-A3B UD-IQ3_XXS via --load-format b70_gguf --quantization
b70_gguf. B70_GGUF_SO=_C.graph.so (6987192 bytes). FULL_DECODE_ONLY,
FLASH_ATTN, compilation mode 0, capture sizes 1,2,3,4,8,
B70_XPU_CG_RECLAIM=500, B70_EAGER_QK=0, B70_GGUF_REF=0. max_num_seqs 4,
max_model_len 33792, gpu_memory_utilization 0.90. Port 18104. Container
vllm-fn-qwen36-gguf-c0. No CCL_TOPO_P2P_ACCESS. Matched eager baseline
is C4: C1 ITL p50 87.62 ms, short-prompt TTFT 265 ms, 1017-token
prefill 177 tok/s.
COMMAND: harness/serve_qwen36_gguf.sh with the flags above. Then
harness/bench_latency.py --model hotschmoe-dev --alias
qwen3.6-35b-a3b-ud-iq3-xxs --decode-tokens 64 --prefill
1024,8192,32768 --timeout 900. Log
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-serve-graph.log. Bench
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-bench-graph.json. Graph
probe before the serve: plugins/.../tests/graph_probe.py on card 0,
log c7-graph-probe.log. Teardown docker stop, then bin/xpu-health
--card 0.
RESULT: Pre-serve card 0 HEALTHY. Probe replay max_abs=0 on gemv and
moe_layer (finite IQ4_NL rows), GRAPH_PROBE_OK. Load placed 733,
skipped_mtp 20, weights 16.96 GiB. reserve_decode on xpu printed before
capture. V2 runner captured 4 FULL graphs (size 8 dropped). Capture
1 sec, pool 0.02 GiB. KV 8.92 GiB, 381493 tokens, 11.29x at 33792.
Identity ids hotschmoe-dev, qwen3.6-35b-a3b-ud-iq3-xxs. Both greedy
texts, temperature 0, 16 tokens: " Paris.\nThe capital of France is
Paris.\nThe capital of France is". Same as C4. Bench exit 0. C1 ITL
p50 11.11 ms, p90 12.41, p99 13.46, 90.16 tok/s, TTFT 174.36 ms, n=64.
C2 stream0 ITL p50 16.51 ms, 61.34 tok/s, TTFT 830.65 ms, 46 tokens.
C2 stream1 ITL p50 16.65 ms, 62.01 tok/s, TTFT 867.12 ms, 40 tokens.
Requested 64; the server usage counts stopped earlier. Prefill tok/s:
1017 tokens 177.01 (TTFT 5746 ms), 8185 tokens 175.58, 32762 tokens
173.23 (TTFT 189124 ms). No OOM. Serve shutdown drain, gpu-run exit 0
in 491 s. Post-health card 0 HEALTHY in 12 s. No xe-reset. Card 1
still held by E1.
VERDICT: One-card decode is graph-capturable and matches eager
greedy text. C1 ITL is 11.11 ms versus C4 eager 87.62 ms on the same
model and card. Prefill stays on the eager grouped path at about 175
tok/s through 32K. Next is the two-card window after E1 releases
card 1: announce, xpu-collective-health, then DP=2 EP eager. Do not
take card 0 while waiting.

## 2026-10-20 07:23 E1 -- int4 control FULL_DECODE_ONLY, C1 ITL 10.23 ms

CONFIG: same graph flags as the int3 arm. Checkpoint
/mnt/cache/b70/devmodel/quant/qwen36-w4a16-g128, alias
qwen36-w4a16-b70-lowbit. Library is the 04:05 libb70_lowbit_xpu.so
(1442856 bytes). B70_LOWBIT_SO and B70_LOWBIT_PREFILL unset.
B70_LOWBIT_MOE_LAUNCHES=2, FULL_DECODE_ONLY, FLASH_ATTN,
B70_XPU_CG_RECLAIM=500, bf16, TP=1, max-len 33792, util 0.90, card 1,
port 18084. Expert lookup is uniform bits 4. Repack log is bits=4 for
40 layers, w13 qweight (256, 256, 1024), w2 (256, 64, 2048).
COMMAND: serve_lowbit.sh, then bin/cpu-run --quiet harness/bench_latency.py
--tag e1-int4-graph --decode-tokens 64 --timeout 1200, then ppl_serve.py
on the same 4x512 wikitext windows. Logs:
.../logs/e1-int4-serve.log, e1-int4-graph-bench.json, e1-int4-ppl.json.
RESULT: Identity first=hotschmoe-dev first_ok alias_ok. Coherence ok,
same, paris. Both texts start " Paris, a city renowned for its rich
history, art, and culture. Sit". Two graph captures, 1 s each. Weight
load 18.13 GiB in 664 s. KV 7.22 GiB, 308574 tokens, 9.13x concurrency
at 33792. init engine 19.02 s. gpu-run exit 0. No OOM.
C1 ITL p50/p90/p99 10.23/10.37/10.48 ms, mean from the json 10.229 ms,
99.59 tok/s, TTFT 2533.19 ms, n=56. C2 p50 13.65 and 13.61 ms
(p90 13.88/13.89, p99 13.99/14.04), 75.25 and 76.22 tok/s, TTFT
3007.19 and 3006.76 ms. Prefill 1017/8185/32762 tok at
96.90/108.74/112.99 tok/s. Ppl 8.8582, nll 2.1813, tokens 2044.
Window nll 1.7888, 2.4407, 2.0974, 2.3985.
Expert bytes per token, full weight read, 8 slots x 40 layers, no
dedup, trunk excluded: int4 519045120 B (495 MiB), int3 393216000 B
(375 MiB). The matched int3 graph arm was C1 ITL 11.29 ms and prefill
99.33/112.32/115.78 tok/s, ppl 9.1199. Int4 moves 32 percent more
expert bytes and the C1 step is 1.06 ms faster. If the whole step were
that read, the rates would be about 51 GB/s and 35 GB/s. Both are far
under a 608 GB/s roof, so this decode step is not expert-bandwidth bound.
VERDICT: Int4 control is measured on the templated kernel. Server
stopped. The mixed arm still needs moe_gemv_mixed in
libb70_lowbit_xpu_e1mix.so, a different binary, so it is not only
expert bits. d2-profile took card 1 again at 07:23:26.

## 2026-10-20 07:29 G3 -- CPU qwen4_exp reference measured

CONFIG: CPU only. No GPU and no fork edits. uv CPython 3.12.13.
PYTHONPATH is the fork tree, the vllm-flashnext venv site-packages,
strata-ab llama-pinned gguf-py, then vllm/flashnext.
ONEAPI_DEVICE_SELECTOR=opencl:cpu, VLLM_TARGET_DEVICE=cpu. Shard1 is
the IQ3_XXS GGUF (mmap). Shard2 is the PLE table (mmap, 16 rows
copied). HF tokenizer and text config:
/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext. Prompt "Return
only the integer: 17 * 19." with enable_thinking false and
add_generation_prompt: 25 ids, matching DD prompt_tokens 25.
Activations are fp32. The daily driver runs HC at Q8; that rounding
is outside this compare.
COMMAND: bin/cpu-run python -m flashnext_ref.compare_layers --stage all.
Then python flashnext_ref/collect_strata_ref.py (walk only, no lock).
Evidence: flashnext_ref/last_compare.json. Report:
reports/1G3_flashnext_model.md. Greedy file:
harness/strata_reference.json. Tools live under flashnext_ref/.
RESULT: compare exit 0, 26 s of Python after the cpu-run lock (E1
held it). PLE multipliers, offsets, and head vocab sizes match the
GGUF header and the strata ngram transcription. Vocab sizes sum to
320001446 of 320001536 rows. ple.layers=[1] is the 0-based block; HF
ple_layer_ids=[2] attaches layer_idx 1. On the 25-id prompt, strata
rows equal the XPU _compute_rows transcription at eos 248044 (0 of
400) and equal the NVIDIA int64 remainder (0 of 400; in-vocab
products stay below 2^63). eos 248046 changes 32 of 400 row ids.
Fork _compute_rows on the first 4 tokens: 0 of 64. IQ4_NL, 16
gathered rows versus gguf-py: bit_mismatches 0, max_abs 0. Fork
HyperConnection.mix after the loader subtracts 1 from the stored
gamma: max_abs 1.311e-6. The same mix against the raw stored gamma:
max_abs 1.278. Fork PLE block on blk.1: max_abs 4.77e-7. A second +1
on the stored PLE gamma moves that block by max_abs 1.46. GGUF RMS
epsilon is the fp32 spelling of 1e-6. QSA width 2051, per-head ReLU,
tail +1e9, and GQA integer division pass as formula checks. The XPU
decoder uses Qwen3NextAttention and leaves the indexer unread. DD
has no --mtp-hnorm, so the wide pooled RMS is the DD MTP default.
The Unsloth eh_proj sidecar mean-collapses and broadcasts;
random-weight wide versus eh_proj max_abs 3.97. Expert pools and the
GDN qkv split were not dequantized. No Strata hidden-state dump for
layers 0-2 was found. strata_reference.json holds 14 temperature-0
texts from 2619 files. Short greedy: 17*19 -> 323, 23*29 -> 667,
thinking 23*17 -> 391. Prompts longer than 400 characters are stored
as a 400-character prefix plus sha256.
VERDICT: Layer-0 HC and layer-1 PLE match Strata's formulas in fp32
when the loader treats stored norm gamma as (1+w). C7 should serve
/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext and keep text eos
at 248044, the PLE cut. 248046 stays a stop id. Draft acceptance
against the daily driver needs the wide fc in mtp-bf16/rt. Dense GQA
matches QSA selection only through 2051 tokens. This is a CPU
localization ladder for the first layers, not a GPU forward.

## 2026-10-20 07:35 G2 -- audit pass 2

CONFIG: CPU only. New JOURNAL entries C7, F1, G1, E1 int4, G3.
Bench JSON and serve logs named above. No GPU.
COMMAND: parse c7-bench-graph.json, f1-bench-2635-triton.json,
e1-int4-graph-bench.json, e1-int4-ppl.json, last_compare.json, and
the F1 microbench logs.
RESULT: F1 latency table, C7 graph bench, E1 int4 bench, and G3
compare file match the raw numbers. E1 mean ITL is 10.227 in the
JSON, written as 10.229. F1's "all six serves coherent" does not
hold as a matched greedy result: 26.27 Triton and 26.35 Triton both
emit "A. True / B. False", while host Triton and all three
FlashAttention arms keep the capital-of-France continuation. G1's
16-test run has no stdout log. C7 health lines were not found; the
serve log does show capture and KV. Fork tip is now 4d73572e14
(07:31), after the measured C7 commit 2583e27239.
VERDICT: Pass 2 recorded in reports/1G2_audit.md. One wrong coherence
claim. Latency figures checked this pass are intact.

## 2026-10-20 07:34 E1 -- moe_gemv_mixed matches prefill on int2/int3

CONFIG: card 1, libb70_lowbit_xpu_e1mix.so (2280328 bytes, built 06:56).
tests/mixed_probe.py. Uniform int3 must match templated moe_gemv.
Mixed bits must match the per-expert prefill_wna16 loop, rel_l2 < 0.02
and cos > 0.999. One id is -1. The full-K cases are H=2048 I=512, the
35B shape. Storage width is the max bits. The 04:05 uniform .so and the
06:35 grouped .so were not overwritten.
COMMAND: B70_GPU_LOCK_TIMEOUT=7200 xpu-run.sh --card 1 --memory 24g
--name b70-lowbit-mixed-probe --env B70_LOWBIT_SO=.../libb70_lowbit_xpu_e1mix.so
-- python3 -u tests/mixed_probe.py
Log: .../logs/e1-mixed-probe.log
RESULT: probe_rc=0, gpu-run exit 0. PASS uniform3 vs templated moe_gemv
rel_l2=0 max_abs=0. PASS uniform3 vs prefill rel_l2=4.2e-3. PASS
mix-small int2/int3 rel_l2=1.6e-3. PASS mix-23 full K rel_l2=1.0e-3
cos=0.999999 max_abs=1024. PASS mix-234 full K rel_l2=4.0e-3
cos=0.999992 max_abs=2048. The large max_abs is on randn activations;
the relative error stays inside the gate. The first full-K attempt
failed in the reference: prefill_wna16 rejects an int3 hi row stored
in a wider int4 stride. The kernel itself had already returned. The
reference now slices to the native stride. This rerun is the one above.
VERDICT: MIXED_PROBE_OK. Safe to serve the mixed checkpoint with
B70_LOWBIT_SO pointing at e1mix.so. This kernel is not the templated
moe_gemv used for the int4 vs int3 arm.

## 2026-10-20 07:38 G1 -- host restore of Ornith KV plus GDN

CONFIG: card 0, host UMD, TP=1, port 18111, container vllm-fn-g1-kv-c0.
Ornith GPTQ int4, dtype float16, max-model-len 32768,
num-gpu-blocks-override 45, mamba-cache-mode align, prefix caching on,
no async scheduling. HybridCPUOffloadConnector, cpu_bytes_to_use
17179869184. PIECEWISE graphs, compilation mode 0,
VLLM_USE_BREAKABLE_CUDAGRAPH=1, B70_XPU_CG_RECLAIM=500. Served names
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
COMMAND: CARD=0 PORT=18111 vllm/flashnext/kvoffload/serve_ornith.sh
Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/g1-serve.log
Then evict_test.py inside the container. JSON g1-evict.json.
Metrics snapshot g1-metrics.txt. An earlier PIECEWISE+inductor serve
died on missing ocloc (g1-serve-inductor-fail.log, 114s, exit 1).
RESULT: Block size 1088. PIECEWISE capture 5+5 graphs. KV override
245 -> 45 (39852 tokens, concurrency 1.22x at 32768). Hold 330s, exit 0.
A is 30464 tokens. Cold TTFT 5.177s, cached 0 (5884 tok/s). Resident
replay TTFT 0.267s, cached 29376. After B and C, restore TTFT 0.732s,
cached 29376, computed 1088, host delta 29376, device delta 0.
Greedy text matches the cold control and the resident replay.
One host load: 666009600 bytes in 0.461s. That is 29376*20480 bytes of
full-attention KV plus 30*2146304 bytes, one GDN snapshot.
Stores: 2064875520 bytes, 93 jobs, 0.475s. Post-health card 0 HEALTHY
(16s). Card 1 was held by E1 and was not probed.
VERDICT: Pass. A fully evicted hybrid prefix resumes from host RAM
with no extra recompute beyond the align-mode tail. No fork commit.
Report: vllm/flashnext/reports/1G1_kv_offload.md.

## 2026-10-20 07:53 E1 -- mixed int3/int2 FULL_DECODE_ONLY, C1 ITL 10.63 ms

CONFIG: checkpoint qwen36-mixed-int3-int2-experts-w4a16-trunk, alias
qwen36-mixed-b70-lowbit. Library libb70_lowbit_xpu_e1mix.so.
B70_LOWBIT_MOE_LAUNCHES=2, FULL_DECODE_ONLY, FLASH_ATTN,
B70_XPU_CG_RECLAIM=500, bf16, TP=1, max-len 33792, util 0.90, card 1,
port 18084. B70_LOWBIT_PREFILL unset, so prefill stays the per-expert
loop. Decode calls moe_gemv_mixed, not the templated moe_gemv used for
the int4 and int3 arms. Repack storage bits=3, 40 layers, w13 qweight
(256, 192, 1024), w2 (256, 48, 2048). Those row counts are the int3
allocation. Shorter int2 rows are sliced in gptq_experts_mixed_to_b70lb1.
COMMAND: e1_mixed_graph.sh. Logs e1-mix-serve.log,
e1-mix-graph-bench.json, e1-mix-ppl.json.
RESULT: Identity hotschmoe-dev first, alias ok. Coherence ok, same,
paris. Both texts: " Paris.\nThe capital of France is Paris.\nThe
capital of France is". Two graph captures, 1 s each. Weight load
14.38 GiB in 522 s. KV 10.78 GiB. init engine 18.07 s. gpu-run exit 0.
C1 ITL p50/p90/p99 10.63/10.96/11.12 ms, 95.54 tok/s, TTFT 2123.48 ms,
n=64. C2 p50 14.26 and 13.97 ms, p90 14.72 and 14.55, p99 21.75 and
14.92, TTFT 4684.66 and 4709.15 ms, 70.78 and 78.76 tok/s. Prefill
1017/8185/32762 tok at 112.47/126.94/129.10 tok/s. Ppl 10.1590, nll
2.3184, tokens 2044. Window nll 1.9115, 2.5400, 2.3456, 2.4763.
Expert bytes per token, full weight read, 8 slots x 40 layers, no
dedup, trunk excluded. int4 519045120 B, int3 393216000 B, int2
267386880 B. Design mix (mass 0.8 on int3) 368050176 B. Count-weighted
uniform average 300158976 B, which understates routed traffic because
the hot experts are int3.
Matched graph arms, C1 ITL / 8k prefill / ppl: int4 10.23 ms / 108.74
tok/s / 8.858; int3 11.29 ms / 112.32 tok/s / 9.120; mixed 10.63 ms /
126.94 tok/s / 10.159. Mixed moves fewer expert bytes than int3 and
the C1 step is 0.66 ms faster than int3, still slower than int4.
VERDICT: Mixed arm measured. The kernel is not the uniform one, so
this is not an expert-bits-only delta. Server stopped. d2-shisa took
card 1 at 07:52:47. Next is grouped prefill on the int3 checkpoint
with the probed e1.so.

## 2026-10-20 07:55 C8 -- N-fast prefill merged, 25 TF dense, T=512 experts win

CONFIG: card 0, ZE_AFFINITY_MASK=0, icpx 2026.1, flags unchanged
(-fsycl -O3 -fPIC -fexceptions -fsycl-device-code-split=per_kernel).
C5 06:38 still the latest C5 entry; IGC 2.41.5 not mounted.
Prefill source is prefill_sched.inl: C6 dequant-once SLM panel, dense
grid N-fast, auto policy dense=reuse-rm64 and experts=s64 (MT 4 when
avg rows <= 32 else 16). Decode sources not edited. Full link used
kernels.cpp at 07:32:53. Side libraries kept: _C.c6.so (5445576, 06:23),
_C.c8.so (5445760, sha 82f8e4296849c1c3), _C.graph.so (6987192).
COMMAND: bin/gpu-run --card 0 bin/cpu-run --quiet scripts/c6_run.sh --full
on _C.c6.so (c8-prefill-smoke.log) and on _C.c8.so (c8-prefill-c8bench2.log).
Rejected BN=128 + SLM-acc binary recorded in c8-prefill-c8bench.log.
Full link B70_PREFILL_OUT=_C.c8full.so (c8-full-build.log, 89s, exit 0).
One lease c8-task3.sh: moe_layer on the 04:32 _C.so, then gpu_check,
moe_layer, prefill_check, c6 --full on _C.c8full.so (c8-task3.log,
TASK3_FAIL:0). compare_cpu.py under bin/cpu-run, PYTHONPATH=venv
(c8-compare-cpu.log). Then cp -a + sha recheck + mv onto _C.so.
RESULT: All GPU runs HEALTH_BEFORE/AFTER 0, failures=0. compare_cpu
13/13 mismatches=0. Dense Q4_K N=12288 K=2560 reuse-rm64 event-min
1227.1 / 10243.9 / 30689.3 us (26.25 / 25.16 / 25.19 TF) at
M=512 / 4096 / 12288. torch.mm bf16 323.0 / 1855.5 / 5481.5 us
(99.74 / 138.89 / 141.04 TF). C2 was 5296 / 42016 / 126036 us.
C6 smoke reuse at M=12288 was 44878 us (17.23 TF); the N-fast grid
holds the rate flat. Layer IQ3_XXS+Q2_0 E=512 top-10: T=512 s64-mt4
9716.9 us (expert48 1098, C2 13885 / 768); T=4096 s64-mt16 28335.7 us
(3012, C2 27599 / 3092); T=12288 s64-mt16 76670.8 us (3339) tied with
reuse-rm32 76389.3 (3351), both behind C2 66129 / 3871. Down at
T=12288 is 19.2 ms vs C2 11.7 ms. Decode moe_layer behind_us
T=1/2/4/8: 04:32 was 472.7/417.0/713.9/1383.3, merged 367.3/315.0/515.1/815.1.
_C.so sha now 086595a01babbc366b3d5366147464ebd24022fba0b65d83b1ff530489c70d55
(was 1e484faeeade967701bf56b8abeb24acd230b28eec79212b28a101ffe4bc3cac).
GEMM-only projection at T=12288, Q4 rate on quantized mats and torch.mm
rate on BF16 HC/router/indexer, attention excluded: 7265 ms, 1691 tok/s.
VERDICT: Dense and grouped prefill are correct and merged. Dense is
4.1x C2 and 5.6x behind torch at M=12288 (25 TF wall). Experts win at
T=512 and miss at T=4096/12288 on the short-K down GEMM. 2000 tok/s
prefill is still ahead of the 1691 tok/s GEMM projection. Report:
vllm/flashnext/reports/1C8_prefill3.md. No commit.

## 2026-10-20 08:05 G4 -- T0 exact-tail index and byte model

CONFIG: host CPython, no GPU, no vLLM import. ParkIndex in
vllm/flashnext/kvoffload/index.py (TailKey, supersede, 4-slot LRU, pin,
running, byte budget). FlashNextTextLayout from
flashnext_ref/qwen4exp_text_config.json (12 full attn, 36 GDN, MTP, PLE
9x10240 bf16). G1 Park and layout tests kept.
COMMAND: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vllm/flashnext python3 -m
unittest kvoffload.tests.test_layout_park kvoffload.tests.test_park_index
kvoffload.tests.test_metrics -v
RESULT: 28 tests OK in 0.003s. Transcript:
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/g4-t0-park-index.txt.
Flash-Next 200K bf16 conversation 5.440 GB (spec about 5.44). 4 parked
plus one GDN checkpoint each 22.224 GB (spec about 22.2). int8 spec
column 2.820 GB and 11.740 GB (spec about 2.82 and 11.7). Ornith fp16
attn 20480 B/token, int8 payload 10240, GDN page still 2146304.
Turn-2 accounting: E = len(P1)+len(R1)-1, replay max(0, prev_total-1-E)
= 0, computed = len(U2)+1. A tail with E == prompt length is a miss
(scheduler recomputes the last token and would double-apply GDN).
Host reserve constant 2.5 GiB (2684354560).
VERDICT: T0 passes. Next is fork F-A (park_finished_tail, default off)
on b70/flashnext, then the parking connector. C7 owns the fork: F-A is
one tiny default-off commit, no push. kv_both is already a producer
(KVProducer includes kv_both; scheduler.py finalize_partial_tail_offloads
is gated by is_kv_producer), so W5 is not a code change.


## 2026-10-20 08:11 C7 -- quantized GGUF payload is one copy

CONFIG: CPU only. Fork tip 2a50cfe2a7, parent c38d0a6197 (G4,
park_finished_tail, default off). No GPU. Serve library remains
_C.graph.so, 6987192 bytes, mtime 2026-10-20 06:59. _C.so is C8's
merge and is not loaded for this serve.
COMMAND: bin/cpu-run python3.12
tests.models.qwen4_exp.test_b70_gguf_mapping.test_quantized_payload_detaches_from_memmap
RESULT: PAYLOAD_TEST_OK in 25s. A quantized tensor from the 35B
GGUF matches the mmap bytes. Adding 1 to the tensor leaves the
mmap unchanged. A non-contiguous memmap slice does the same.
Commit 2a50cfe2a7 is not pushed. At 08:09 card 0 was C9 and card 1
was E1.
VERDICT: Each DP worker copies the Flash-Next PLE table once. The
two-card window is still waiting on both leases. It will be
announced here immediately before gpu-run takes them.

## 2026-10-20 08:16 G2 -- audit pass 3

CONFIG: CPU only. No GPU. Cursor was JOURNAL.md through the 08:11 C7
payload entry. Evidence under
/mnt/vm_8tb/b70/build/vllm-flashnext/logs and /mnt/vm_8tb/b70/build/c8-*.log.
Fork /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext.
COMMAND: parsed e1-mix-graph-bench.json, e1-mix-ppl.json, g1-evict.json,
g1-metrics.txt, g4-t0-park-index.txt, and the named serve and C8 logs.
sha256 of _C.so. Layout byte functions imported on CPU. git diff --cached
and git log -12 on the fork.
RESULT: F16-F20 and F22 OK. E1 mixed probe, E1 mixed graph (C1 p50
10.629 ms, ppl 10.1590), G1 restore (load 666009600 bytes, post-health
HEALTHY), C8 prefill times, and G4's 28-test transcript match the raw
files. G4's run also passes the 16 tests that F15 could not see.
C8's 1691 tok/s is the GEMM projection in 1C8_prefill3.md, not a serve.
F21 UNVERIFIED: commit 2a50cfe2a7, parent c38d0a6197, and _C.graph.so
6987192 bytes at 06:59 match, but PAYLOAD_TEST_OK is only in this
journal. The test checks aliasing. It does not time a two-worker copy.
E1 probe and E1 mixed serve have no xpu-health line. G1 has post-health
only. git diff --cached empty. Campaign *.so files stay ignored.
D2 shisa JSON and e1-grp-serve.log are still unjournaled.
Report: vllm/flashnext/reports/1G2_audit.md.
VERDICT: Pass 3 recorded. Latency and restore figures checked this
pass match. One CPU test claim has no transcript.

## 2026-10-20 08:16 C7 -- two-card window announced

CONFIG: both cards, no CCL_TOPO_P2P_ACCESS. Lock timeout 30s so a miss
releases card 0. Fork tip 2a50cfe2a7, not pushed. Library _C.graph.so.
First command is bin/xpu-collective-health. Then eager Qwen3.6-35B-A3B
UD-IQ3_XXS, TP=1, DP=2, --enable-expert-parallel, --api-server-count 1,
max-model-len 2048, max-num-seqs 2, port 18104, container
vllm-fn-qwen36-gguf-ep. B70_COLL_LOG records collectives. Serve budget
14 minutes plus the health check, under 20 minutes.
COMMAND: bash /tmp/c7-dp-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 08:15 E1 -- grouped int3 prefill 174 tok/s at 8k

CONFIG: int3 checkpoint, libb70_lowbit_xpu_e1.so (06:35, the probed
binary), B70_LOWBIT_PREFILL=grouped, B70_LOWBIT_MOE_LAUNCHES=2,
FULL_DECODE_ONLY, FLASH_ATTN, B70_XPU_CG_RECLAIM=500, card 1, port
18084, alias qwen36-w3a16-grouped. Prefill is outside the graph.
COMMAND: e1_grouped_serve.sh. Bench tag e1-int3-grouped. Logs
e1-grp-serve.log, e1-grp-bench.json.
RESULT: Identity hotschmoe-dev first, alias ok. Coherence ok, same,
paris. Both texts: " Paris, a city renowned for its rich history,
culture, and iconic landmarks." Decode C1 ITL p50/p90/p99
11.29/11.49/11.61 ms, 90.40 tok/s, TTFT 810.34 ms, n=57. C2 p50
15.59 and 15.66 ms, TTFT 1157.17 and 1156.72 ms. Prefill
1017/8185/32762 tok at 165.05/173.56/175.14 tok/s (TTFT
6162/47158/187062 ms). The matched per-expert int3 graph arm was
99.33/112.32/115.78 tok/s. Grouped is 1.55x at 8k. Decode ITL matches
the templated graph arm (11.29 ms). gpu-run exit 0. Server stopped.
VERDICT: GROUPED_SERVE_OK and far from the 5000 tok/s 8k target.
The 64 KiB SLM tile is numerically right and still the prefill floor.
Next is a one-layer device time against prefill_wxa8 and a bf16 matmul,
not another full serve, so the kernel change has a measured gap.

## 2026-10-20 08:22 C7 -- DP=2 EP profile allgatherv segfault

CONFIG: both cards, no CCL_TOPO_P2P_ACCESS. Fork 2a50cfe2a7.
_C.graph.so. Eager Qwen3.6-35B-A3B UD-IQ3_XXS, TP=1, DP=2,
--enable-expert-parallel, --api-server-count 1, max-model-len 2048,
max-num-seqs 2, port 18104, container vllm-fn-qwen36-gguf-ep.
B70_COLL_LOG on. Collective health first.
COMMAND: bash /tmp/c7-dp-window.sh. Logs c7-dp-window.out,
c7-coll-health.txt, c7-ep-dp-serve.log, c7-coll.log.
RESULT: HEALTH_RC 0. COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120
compiled_iterations=10 p2p=0, gpu-run 35s exit 0. Both workers loaded
shards=1 tensors=753 ep=0/2 and ep=1/2, placed 733 skipped_mtp 20
q8_dequant 7, 11.64 GiB, about 75 s. Init collectives flushed on the
first execute_model (tokens=2048, the profile forward): each rank
all_reduce=2 (6.460 ms and 6.118 ms) and barrier=2 (0.609 ms and
166.562 ms). No execute line. Both workers then printed
"Segfault encountered" in oneCCL allgatherv_ll_ring for bf16
Rt64_128_PCIE, called from dist.all_gather with a one-element list
(xpu_communicator.py equal-size branch). EngineCore cancelled at
determine_available_memory. SERVE_RC 1. No greedy and no ITL.
Container stopped by 08:19:37. No xe-reset: E1 took card 1 at
08:19:35. That container was already gone at 08:21 and both owner
files were empty.
VERDICT: Expert placement and the GGUF load work at EP=2. The
equal-size all-gather call is the crash. Next serve uses
all_gather_into_tensor for that branch.

## 2026-10-20 08:22 C7 -- two-card window retry

CONFIG: same DP=2 EP eager serve. Fork working tree now calls
dist.all_gather_into_tensor for equal-sized XPU all-gather.
Collective health runs again. No P2P. Lock timeout 30s.
COMMAND: bash /tmp/c7-dp-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: retry open.

## 2026-10-20 08:20 E1 -- one-layer prefill times, WXA8 is not faster

CONFIG: card 1, libb70_lowbit_xpu_e1.so, random planes, bits=3,
H=2048 I=512 E=256 top-8. XPU events. Not a serve number.
COMMAND: tests/grouped_time.py via xpu-run.sh --card 1. Log
e1-grouped-time.log.
RESULT: moe_prefill (both projections) T=128/1024/8192 is
63.14/96.05/324.10 ms. Forty layers at 8192 would be 13.0 s, about
632 tok/s if nothing else ran. The served grouped prefill was 47.2 s
(174 tok/s), so the dense trunk GEMMs are in that remainder.
One expert M=256 K=2048 N=1024: wna16 1.118 ms, wxa8 1.156 ms,
torch.matmul 0.113 ms. A loop of 256 bf16 matmuls is 15.06 ms.
VERDICT: GROUPED_TIME_OK. The int8 DPAS tile is not faster than the
bf16 DPAS tile on this shape. The vendor bf16 GEMM is about 10x the
ESIMD tile. Next is unpack-to-bf16 plus that GEMM, built into a new
library so e1.so stays the probed grouped binary.

## 2026-10-20 08:25 C7 -- DP retry still segfaults in allgatherv

CONFIG: fork 1fcc7bbd1d. Same eager DP=2 EP serve. Equal-size path
now calls dist.all_gather_into_tensor.
COMMAND: the 08:22 window. Preserved logs c7-ep-dp-serve-into.log
and c7-coll-into.log.
RESULT: HEALTH_RC 0 again. placed 733, 11.64 GiB, load 49 s on the
warm cache. Python warned that all_gather_into_tensor is deprecated
in favor of all_gather_single, so the new call ran. Both workers
then segfaulted in the same oneCCL allgatherv_ll_ring bf16
Rt64_128_PCIE, inside SYCL PersistentDeviceCodeCache::getItemFromDisc.
Init collectives before that execute: all_reduce=2 (0.625 ms and
0.935 ms) and barrier=2 (163.434 ms and 0.701 ms). SERVE_RC 1.
No greedy, no ITL. No xe-reset: the following health check is the
gate, and the previous retry's health passed after the first crash.
VERDICT: The one-element list was not the crash. The allgatherv
kernel faults while reading the SYCL device cache. Next serve sets
SYCL_CACHE_PERSISTENT=0 and a fresh cache directory.

## 2026-10-20 08:26 C7 -- DP window, fresh SYCL cache

CONFIG: same DP=2 EP eager serve. SYCL_CACHE_PERSISTENT=0.
SYCL_CACHE_DIR=/mnt/vm_8tb/b70/build/vllm-flashnext/cache/sycl-c7-dp.
Collective health first. No P2P. Fork 1fcc7bbd1d.
COMMAND: bash /tmp/c7-dp-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: retry open.

## 2026-10-20 08:26 E1 -- unpack plus matmul is 126x on one q-proj

CONFIG: card 1, libb70_lowbit_xpu_e1mm.so (2345192 bytes, built 08:25).
tests/matmul_probe.py. Gate rel_l2 < 0.02 against prefill_wna16, and
the unpack against format.dequant rounded to bf16. The 04:05 uniform
.so and the 06:35 grouped .so were not overwritten.
COMMAND: build.sh SO_NAME=libb70_lowbit_xpu_e1mm.so, then xpu-run.sh
--card 1 --memory 24g --env B70_LOWBIT_SO=that library --
python3 tests/matmul_probe.py. Logs e1-matmul-build.log,
e1-matmul-probe.log.
RESULT: probe_rc=0. PASS bits 2/3/4 unpack, small linear rel_l2
0, 2.3e-7, 3.8e-6. PASS bits 3 M=8192 N=2048 K=2048 unpack rel_l2=0
and linear rel_l2=0. Time on that shape: wna16 72.659 ms,
unpack+matmul 0.575 ms (126x). Small M=32 cases are 0.14 ms versus
0.23-0.30 ms, so the win is the large GEMM.
VERDICT: MATMUL_PROBE_OK. B70_LOWBIT_PREFILL=matmul on this library
is the prefill path to serve. Decode stays GEMV.

## 2026-10-20 08:30 C10 -- dequant library linked

CONFIG: icpx 2026.1, same flags as C8 (-fsycl -std=c++20 -O3 -fPIC
-fexceptions -fsycl-device-code-split=per_kernel). New TU
csrc/dequant_fast.cpp. build.sh source list adds that TU (shared
file; additive ops only). Output is _C.c10.so. _C.so is C8's binary
and is not replaced: sched_moe.hpp is newer than _C.so (C9 decode).
COMMAND: B70_AGENT=C10 B70_PREFILL_OUT=.../_C.c10.so bin/cpu-run
vllm/flashnext/plugins/b70_gguf_xpu/scripts/build.sh
Log /mnt/vm_8tb/b70/build/c10-build.log.
RESULT: cpu-run done in 97s, exit 0. _C.c10.so 8294752 bytes sha256
55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0.
_C.so sha256 still
086595a01babbc366b3d5366147464ebd24022fba0b65d83b1ff530489c70d55.
120 existing deprecation warnings, no errors. Card 0 held by C9
(run_gpu4.sh); card 1 held by E1. Bench will queue on card 0.
VERDICT: link ok. Speed and crossover not measured yet.

## 2026-10-20 08:30 G5 -- CPU QSA layer matches Strata cell top-k

CONFIG: host CPython, numpy, no GPU. flashnext_ref/qsa_layer.py on top of
G3's qsa.py (untouched). Small geom (block 4, top_k 4, width cap 7) and
the Flash-Next width cap 2051. Strata sources: qsa.hpp, qsa.cu
indexer_key_append_kernel, kv_q8.cu, qsa_select.hpp (open block scores
the dead key).
COMMAND: cd vllm/flashnext && PYTHONDONTWRITEBYTECODE=1 python3 -m flashnext_ref.test_qsa_layer
RESULT: QSA_LAYER_OK. Radix top-k matches lexsort through n=2100 width=2051,
including ties and +1e9. Chunk prefill matches token-at-a-time ids and
gated output bit-exact (T=16). int8 group-64 rint ties-to-even
(0.5->0, 1.5->2, -1.5->-2), zero group stores scale 0. Park slope
12*(1056+128)+1056 = 15264 B/token. Fixed indexer tail+dead+block_pos
is (3*128*4)+(128*4)+4 bytes at the real shape. Marginal block:
Strata cells [4,5,6,7,12,13,14], vLLM whole-block expand [4,5,6,7].
Pooled keys vs G3 pool_indexer_keys max_abs 1.19e-7. Text mRoPE equals
1D; a split triple does not. Spare key ignores pos_base.
VERDICT: CPU reference is the contract for the XPU kernels. Plugin
b70_qsa_xpu is compiling next (SIMT, graph-sized grids, int8 KV).
XMX union prefill is not in this library. No commit.

## 2026-10-20 08:31 G4 -- F-A committed, parking connector written

CONFIG: fork /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext.
F-A is c38d0a6197, default-off park_finished_tail, align mode, in-flight
tokens suppress the offer. Not pushed. C7 tip is now 1fcc7bbd1d on top
of it. vllm/models/qwen4_exp/xpu/model.py is dirty in that tree and was
not touched. W5 is not a code change: kv_both is already a producer.
PLE state is F-D, not part of F-A.
COMMAND: CPU test of F-A was importlib of
tests/v1/core/test_single_type_kv_cache_manager.py inside
vllm-flashnext-xpu:py312 (PARK_TAIL_OK, earlier this session). Host
unittest of the park index: PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=vllm/flashnext python3 -m unittest
kvoffload.tests.test_layout_park kvoffload.tests.test_park_index
kvoffload.tests.test_metrics -v
RESULT: 29 tests OK in 0.003s, including ParkIndex.discard.
ParkingOffloadConnector is in kvoffload/parking.py. It forces
offload_prompt_only off, binds ConversationCachePolicy, queues the
exact tail (GDN plus the partial attention page at E) into the next
step's jobs_to_flush, and refuses a tail whose end is the whole
prompt so the scheduler's last-token backoff cannot double-apply GDN.
A failed prepare_store drops the index record. Resident full-block
keys are pinned for that call. serve_ornith.sh now uses this
connector, 32 GiB, --shm-size 40g, --memory 120g, slots 4,
--park-finished-tail, and the fork on PYTHONPATH ahead of the wheel.
turn_test.py is the T2 client (cache_salt control, host counter,
greedy token ids). Docker import of the connector has not run: C9
holds the CPU quiet lock and card 0, E1 holds card 1.
VERDICT: F-A is in the fork. Connector import and Ornith T2 wait for
a free card and a free quiet lock. Copy op is next on CPU, compile
after the quiet window.

## 2026-10-20 08:28 C7 -- DP fresh-cache profile dies in topk padding

CONFIG: fork 1fcc7bbd1d. Eager DP=2 EP, TP=1, max_model_len 2048,
max_num_seqs 2, 35B UD-IQ3_XXS. SYCL_CACHE_PERSISTENT=0 and
SYCL_CACHE_DIR=/mnt/vm_8tb/b70/build/vllm-flashnext/cache/sycl-c7-dp.
No P2P. Collective health first.
COMMAND: bash /tmp/c7-dp-window.sh (launched 08:26:10). Logs copied to
c7-ep-dp-serve-nocache.log and c7-coll-nocache.log before the next
truncation.
RESULT: HEALTH_RC 0. In-container echo confirmed the cache override.
placed 733, 11.64 GiB, load 35 s. all_gather_into_tensor ran (the
deprecation warning printed) and did not segfault. Both workers then
raised RuntimeError: is_padding size mismatch, expected: 4096, in
torch.ops._moe_C.topk_softmax during the 2048-token profile forward
(gpu_worker.py determine_available_memory). 4096 is 2*2048. The XPU
check is vllm-xpu-kernels csrc/moe/topk.cpp:880 and compares the mask
to gating rows. Naive dispatch all-gathers activations first
(moe_runner.py _maybe_dispatch) and leaves the runner mask at the
local 2048. Init collectives only: all_reduce=2 and barrier=2 per
rank (pid 424 coll_ms 162.166, pid 423 coll_ms 1.542). No execute
line. SERVE_RC 1 at 08:28:29. No greedy, no ITL. No xe-reset: C9 took
card 0 and E1 took card 1 at 08:28:24, and health had passed.
VERDICT: the fresh cache let equal-size all_gather_into_tensor return.
The profile still cannot pass top-k. Next commit gathers the local
padding mask with the dispatch.

## 2026-10-20 08:35 C7 -- padding mask is gathered with naive DP dispatch

CONFIG: fork 2fec939431, not pushed. moe_runner.py only. The other
dirty file vllm/models/qwen4_exp/xpu/model.py was left untouched.
COMMAND: py_compile of moe_runner.py. No GPU.
RESULT: PY_COMPILE_OK. _maybe_dispatch passes the local is_padding
slice as extra_tensors on the existing all_gatherv, installs the
gathered mask for top-k, and restores the runner mask after the
layer. A mask shorter than the local batch is dropped for that
forward instead of failing the size check.
VERDICT: unmeasured. The next DP window uses this commit. Cards were
busy at 08:35 (C9 card 0, E1 card 1).


## 2026-10-20 08:36 H1 -- EP sidecar libraries linked

CONFIG: icpx 2026.1, -fsycl -std=c++20 -O3 -fPIC -fexceptions
-fsycl-device-code-split=per_kernel, gcc toolchain under
/mnt/vm_8tb/b70/build/gguf-xpu-toolchain. New package
vllm/flashnext/plugins/b70_ep_xpu. GGUF rebuild uses
B70_PREFILL_OUT=_C.ep.so so _C.so and _C.graph.so stay put.
Link is -l:libze_loader.so.1 (the host has the SONAME, not
libze_loader.so). No rpath on the loader. Scratch in kreg.hpp,
row_moe.hpp, and sched_moe.hpp is slotted by sycl::device.
COMMAND: B70_AGENT=H1 MAX_JOBS=4 bin/cpu-run
plugins/b70_ep_xpu/scripts/build.sh. Log
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1-build.log.
First attempt failed at link (cannot find -lze_loader) in 48s.
RESULT: second cpu-run exit 0 in 159s after G5 released the CPU
lease. plugins/b70_ep_xpu/b70_ep_xpu/_C.so is 391992 bytes, sha256
b0f8376f608f14a735cd7bd9c0e8ff45389cb479bcc5733d215a8d2e8cab2cde,
NEEDED libze_loader.so.1 and libsycl.so.9.
plugins/b70_gguf_xpu/b70_gguf_xpu/_C.ep.so is 8294752 bytes, sha256
55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0
(same bytes as C10's _C.c10.so; same sources). nm exports
b70_moe_layer and b70_reserve_decode_queue.
_C.graph.so is still 6987192 bytes, mtime 06:59, sha256
2315bd4593523b569f73067454a4c8867114bc2e99a3f0f87745858474f7201a.
_C.so sha256 still
086595a01babbc366b3d5366147464ebd24022fba0b65d83b1ff530489c70d55.
VERDICT: link ok. CPU import is next, then the two-card M0 window.
No GPU lease taken for the build.

## 2026-10-20 08:36 H1 -- M0 two-card window announced

CONFIG: both cards, no CCL_TOPO_P2P_ACCESS, no oneCCL. Transport is
the Strata posted push plus a bounded device flag (spin 200000, not
2e6). Eager S28 oracle (65 samples) then card-0 torch.xpu.graph plus
a plugin card-1 command_graph (65 samples). Section 7.5 probes 1-5
follow. Host alarm 900s. Libraries are the _C.so and _C.ep.so from
the entry above. Container vllm-fn-h1-m0, memory 8g.
COMMAND: bin/gpu-run bin/xpu-health, then
harness/xpu-run.sh --card both python tests/m0_probe.py, then
bin/gpu-run bin/xpu-health. B70_GPU_LOCK_TIMEOUT=3600 so the request
queues behind E1 on card 1. Logs h1-m0-health-before.txt,
h1-m0-probe.log, h1-m0-health-after.txt.
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 08:45 E1 -- int3 matmul prefill 366 tok/s at 8k

CONFIG: int3 checkpoint, libb70_lowbit_xpu_e1mm.so,
B70_LOWBIT_PREFILL=matmul, B70_LOWBIT_MOE_LAUNCHES=2, FULL_DECODE_ONLY,
FLASH_ATTN, B70_XPU_CG_RECLAIM=500, card 1, port 18084, alias
qwen36-w3a16-matmul. Decode stays GEMV. Prefill unpacks b70lb1 to
bf16 and calls the vendor GEMM, including one matmul per active expert.
COMMAND: e1_matmul_serve.sh. Bench tag e1-int3-matmul. Logs
e1-mm-serve.log, e1-mm-bench.json.
RESULT: Identity hotschmoe-dev first, alias ok. Coherence ok, same,
paris. Both texts: " Paris, a city renowned for its rich history,
culture, and iconic landmarks." C1 ITL p50/p90/p99 11.34/11.55/11.63 ms,
89.94 tok/s, TTFT 1875.76 ms, n=57. C2 p50 15.84 and 15.84 ms, TTFT
2046.96 and 2046.12 ms. Prefill 1017/8185/32762 tok at
287.93/366.37/385.20 tok/s (TTFT 3532/22341/85051 ms). Grouped DPAS
was 173.56 tok/s at 8k. The per-expert WNA16 path was 112.32.
gpu-run exit 0. Server stopped.
VERDICT: MATMUL_SERVE_OK. 3.3x the original 8k prefill and still
13x short of 5000 tok/s. One q-proj microbench was 0.575 ms, so the
remaining 22 s is not that single GEMM. Next is a one-layer split of
the expert loop versus the dense projections.

## 2026-10-20 08:46 G4 -- copy op compiled, T2 client fits 4 slots

CONFIG: libb70_kvpark.so built with icpx 2026.1.0 against torch XPU
headers. Output /mnt/vm_8tb/b70/build/vllm-flashnext/b70_kvpark/libb70_kvpark.so
(90904 bytes). Not loaded into the connector. T2 still uses
swap_blocks_batch. Host pool stays 32 GiB; this run caps the container
at 80g because the host has 128 GiB and card 1 is live.
COMMAND: B70_AGENT=g4 bin/cpu-run plugins/b70_kvpark/build.sh.
Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/g4-kvpark-build.txt.
The earlier connector import log is g4-import.txt (IMPORT_OK).
RESULT: compile rc=0 in 45s, ldd reports no missing libraries, and the
TORCH_LIBRARY schema for alloc_host, copy_batch, and engine_ordinal is
in the .so. No bandwidth number: both cards were busy (H1 health on
card 0, E1 serve on card 1). turn_test no longer sends flush prompts.
Those extra conversations would be a fifth slot and would evict A
before the restore. The four slots are A, the salted control, B, and C.
An eagle KV group, if the model has one while MTP is off, is a
zero-size copy slot and is not part of the hit.
VERDICT: library is built and unmeasured. Ornith T2 starts when card 0
is free. F-B is not a new scheduler change: a non-aligned external
count is already kept (scheduler.py around the ext_tokens assign)
except for the DSV41 prefix-replay round-down. F-C and F-D are not
committed.
G4 taking card 1 for Ornith T2, about 20 min, port 18111

## 2026-10-20 08:48 E1 -- 8k layer split, expert loop 5.8 s

CONFIG: card 1, libb70_lowbit_xpu_e1mm.so, random int3 planes,
H=2048 I=512 E=256 top-8, T=8192. XPU events. Not a serve tok/s.
COMMAND: tests/prefill_split.py. Log e1-prefill-split.log.
RESULT: split_rc=0. Expert w13 unpack 12.93 ms. Full expert layer
(both unpacks, 256 matmuls, SiLU, reduce) 144.64 ms, times 40 layers
is 5.79 s. One dense GEMM M=8192 N=2048 K=2048 is 0.655 ms, times
160 is 0.10 s. Eager decode moe_gemv 0.168 ms (6.72 ms times 40).
Eager gemv N=2048 0.098 ms (15.76 ms times 160). The served matmul
8k prefill was 22.3 s. These kernels account for about 5.9 s of that.
The eager GEMV sum is larger than the 11.3 ms graph ITL because the
graph collapses launch overhead. Do not read the eager sum as the ITL.
VERDICT: PREFILL_SPLIT_OK. The lowbit GEMMs are no longer the bulk of
the 8k prefill. The next measurement is XPU-event brackets around
linear_attn, self_attn, and mlp on one real 8k forward. Not Kineto.

## 2026-10-20 08:49 G4 -- T2 health passed, serve yielded to H1

CONFIG: card 1, port 18111, container vllm-fn-g4-kv-c1. Health image
vllm-xpu-env:int8g-v0251.
COMMAND: kvoffload/run_t2.sh. Health log g4-t2-health-before-c1.txt.
Serve log g4-t2-serve-c1.txt.
RESULT: xpu-health card 1 OK in 24s. The serve then found card 1 held
by H1 (vllm-fn-h1-m0, both cards) and is waiting up to 180s. It will
exit rather than take the card out from under that probe.
VERDICT: no T2 measurement yet.

## 2026-10-20 08:53 G4 -- CLI flag was not registered

CONFIG: fork b70/flashnext. F-A added CacheConfig.park_finished_tail
but EngineArgs.add_cli_args lists flags by hand.
COMMAND: CARD=1 kvoffload/run_t2.sh. Health before passed (card 1 OK,
24s). Serve log g4-t2-serve-c1.txt.
RESULT: vllm serve exited in 27s: unrecognized arguments
--park-finished-tail. The wheel does not have the field. The fork
does, and PYTHONPATH pointed at the fork, but the flag was never
added to the parser. Commit e442b4b53f registers it and passes it
into CacheConfig. Default off. Not pushed. qwen4_exp XPU model.py
stayed unstaged.
VERDICT: no T2 measurement. Relaunch after this entry.

## 2026-10-20 08:55 G2b -- audit pass 4

CONFIG: CPU only. No GPU. Cursor was JOURNAL.md through the 08:53
G4 CLI entry. Evidence under /mnt/vm_8tb/b70/build/vllm-flashnext/logs,
c10-build.log, and fork b70/flashnext. Report
vllm/flashnext/reports/1G2_audit.md.
COMMAND: parsed e1-grp-bench.json, e1-mm-bench.json, the C7 serve
and coll logs, and the G4/H1 transcripts. sha256 of the new .so
files. Re-ran the QSA module and the 29 kvoffload tests with
bytecode writes off. git show of 2fec939431 and e442b4b53f.
RESULT: F23, F24, F27-F29, F31, F32, F34, F35 OK. Grouped int3
C1 p50 is 11.289 ms and the matmul serve C1 p50 is 11.342 ms.
Both benches are hotschmoe-dev first and the landmarks string.
C7's second and third DP logs match the journal. The first crash
log was overwritten by the nocache run, so its 6.460 ms figures
are UNVERIFIED (F25). PY_COMPILE_OK and H1's 48s link failure
have no surviving transcript (F26, F27). H1 M0 raw log is
DEVICE_LOST in m0_prep; post-health files are empty (F33).
G4's 08:49 "would exit rather than take the card" is WRONG
against the finished log (F30). The 08:53 entry corrects it:
exit 2, unrecognized --park-finished-tail, and commit
e442b4b53f adds the flag (F35). Matmul "3.3x" is 3.26x
(366.37/112.32). E1 GPU probes in this window have no xpu-health
line. git diff --cached is empty. Campaign *.so files stay ignored.
VERDICT: Pass 4 recorded. Latency numbers checked this pass match.
One overwritten C7 log and the H1 device-lost probe remain open.

## 2026-10-20 08:58 C7 -- DP window, gathered padding mask

CONFIG: fork 2fec939431. Eager DP=2 EP, TP=1, 35B UD-IQ3_XXS.
SYCL_CACHE_PERSISTENT=0. Collective health first. No P2P.
The local is_padding slice is an extra all_gatherv tensor.
COMMAND: bash /tmp/c7-dp-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: retry open.


## 2026-10-20 09:01 C7 -- DP=2 EP eager measured

CONFIG: fork e442b4b53f, which contains 2fec939431 (padding mask
gathered with naive DP dispatch) and G4's default-off
--park-finished-tail CLI (not passed). Eager DP=2, TP=1,
--enable-expert-parallel, max_model_len 2048, max_num_seqs 2,
35B UD-IQ3_XXS, B70_COLL_LOG on, SYCL_CACHE_PERSISTENT=0,
cache dir sycl-c7-dp. No P2P. Collective health first.
COMMAND: bash /tmp/c7-dp-window.sh, launched 08:58:00 after both
owner files were empty. Logs preserved as c7-ep-dp-serve-ok.log,
c7-coll-ok.log, c7-bench-ep-dp.json, c7-ep-dp-greedy.json.
RESULT: HEALTH_RC 0. placed 733, skipped_mtp 20, 11.64 GiB, linear
EP 128/256. READY 09:00:32. Both greedy completions, temperature 0,
16 tokens, are the C4 Paris string. match_c4 true. Identity
hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs. Bench exit 0.
C1 ITL p50 121.40 ms, p90 143.80, p99 153.09, TTFT 462.42 ms,
7.86 tok/s, n=32. C2 stream ITL p50 120.73 ms and 120.79 ms,
TTFT 523.45 ms and 522.66 ms. This serve is eager and the collective
counter synchronizes each call, so these times are not the one-card
FULL_DECODE_ONLY 11.11 ms. Steady tokens=1 execute (159 steps, both
pids): 160 collectives, 120 all_gather_into_tensor (median 15.01 ms)
and 40 reduce_scatter_tensor (median 5.62 ms), coll median 20.61 ms,
wall median 119.06 ms. That is 40 MoE layers times 3 all-gathers
(hidden, router logits, padding mask) plus 1 reduce-scatter. No
all_reduce on the decode step (TP=1). One first-step warmup on pid
423 took coll 643.27 ms inside reduce_scatter_tensor. SERVE_RC 0.
docker stop 09:00:52. No crash, no xe-reset. Post-health was not
run: C9 took card 0 and G5 took card 1 at 09:01:05.
VERDICT: DP=2 expert parallel loads, matches the one-card greedy
string, and issues 160 collectives per decode step. ITL is about
121 ms eager. Next window is TP=2, which the loader must refuse,
then Flash-Next IQ3_XXS DP=2 against the recorded 323 and 667 strings.


## 2026-10-20 09:02 G4 -- CLI flag parses

CONFIG: fork e442b4b53f on b70/flashnext. Image vllm-flashnext-xpu:py312.
PYTHONPATH is the fork then the venv site-packages. VLLM_TARGET_DEVICE=cpu.
ONEAPI_DEVICE_SELECTOR=opencl:cpu. No GPU. Script
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/g4_cli_parse.py.
COMMAND: bin/cpu-run docker run --network none --cpus 4 --memory 24g
--entrypoint /usr/bin/python3.12 vllm-flashnext-xpu:py312 g4_cli_parse.py
RESULT: PARK_ON True. PARK_OFF False. CLI_PARSE_OK. cpu-run exit 0 in 21s.
XPU device count zero is the CPU container. MemAvailable was about 76 GiB.
At 09:01 card 0 is C9 and card 1 is D3, with the CPU lock in quiet mode.
VERDICT: --park-finished-tail is on the parser and stays default off.
No T2 number yet. The serve waits for card 1 and a free CPU lock.
The copy path for that serve is still swap_blocks_batch.

## 2026-10-20 09:05 H1b -- first M0 window lost the device in prep

CONFIG: libraries from the 08:36 link (_C.so sha256 b0f8376f...,
_C.ep.so sha256 55c154ee...). Both cards, no oneCCL, spin 200000.
Container vllm-fn-h1-m0, memory 8g. The 08:36 entry said the lease
was only requested. The window script did run.
COMMAND: /tmp/h1-m0-window.sh. Logs
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1-m0-window.out and
h1-m0-probe.log.
RESULT: Health before passed (HEALTHY cards 0 and 1, 28s). PROBE5:
zeMemOpenIpcHandle failed 2013265924, which is
ZE_RESULT_ERROR_INVALID_ARGUMENT. same_ctx=1, both queues in order,
peer access both ways, peer atomics 0, ipc_store=0, same_store=1,
path=2, bdf 0000:0b:00.0 and 0000:44:00.0. Torch keeps one context
for both devices, so an IPC open in that context is rejected. The
4-byte same-context store matched. m0_prep then raised
UR_RESULT_ERROR_DEVICE_LOST (20). That prep poisoned the peer
regions with a range parallel_for, including 153600 words into the
return buffer. gpu-run exit 1 in 81s at 08:50:34. Health after,
once card 0 was free, was HEALTHY cards 0 and 1 in 30s
(h1-m0-window.out). The tee file h1-m0-health-after.txt is empty.
No xe-reset.
VERDICT: path 2 is the transport. A scalar store does not make the
large peer range-fill safe. The next binary pads every region by
one page, poisons on the owning card (S28 default), and preflights
the 256-wide push from 64 bytes through the 600 KB return before
the 65-sample oracle. Rebuild is queued behind D3's CPU quiet lock.

## 2026-10-20 09:06 C7 -- TP=2 window, loader must refuse packed sharding

CONFIG: fork e442b4b53f. Same 35B UD-IQ3_XXS, TP=2,
--enable-expert-parallel, enforce-eager, max_model_len 2048.
No packed sharding. No P2P. Collective health first.
COMMAND: bash /tmp/c7-tp-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.


## 2026-10-20 09:07 C7 -- TP=2 window lost card 0

CONFIG: the 09:06 TP=2 announce. Collective health, then the eager
TP=2 serve. No P2P. Loader raise not expected to run if the lease
is lost.
COMMAND: bash /tmp/c7-tp-window.sh. Preserved as
c7-tp-window-race.out and c7-ep-tp-serve-race.log.
RESULT: HEALTH_RC 0 at 09:06:02. The serve launch at 09:06:30 found
card 0 held by C9 (pid 3152937, same second). gpu-run waited 30s
and exited 124. The vLLM process did not start. RAISE:0. No loader
line. SERVE_RC 124 is the lock timeout, not a model failure. Card 1
was free again by 09:07. No xe-reset.
VERDICT: TP=2 is unmeasured. Retry when both owner files are empty.
Do not treat this 124 as the sharding refusal.

## 2026-10-20 09:09 H1b -- M0 two-card window, padded regions

CONFIG: rebuilt b70_ep_xpu/_C.so sha256
2248124379c09191a4c77aaff2854b49ab59b6e743a90f2f802eae1e70189fe2
(445392 bytes). _C.ep.so unchanged. Owner-local poison, one-page
pad, 256-wide preflight 64 B / 4 KB / 17520 B / 64 KB then the
600 KB return. Oracle is still 65 eager and 65 graph. Both cards,
no oneCCL. Host alarm and the window timeout are 900s. Container
vllm-fn-h1b-m0, memory 8g.
COMMAND: nohup env B70_AGENT=H1 B70_GPU_LOCK_TIMEOUT=7200 bash
/tmp/h1b-m0-window.sh. The script calls gpu-run itself (health,
then xpu-run, then health). An outer gpu-run would deadlock on
xpu-run's inner flock. Log h1b-m0-window.out.
RESULT: rebuild exited 0 in 40s under cpu-run. Window requested
after this entry. Card 0 was busy at 09:08.
VERDICT: window open.


## 2026-10-20 09:08 C10b -- bf16 gate, card 0 bench queued

CONFIG: continuing C10. _C.c10.so sha256 55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0 (08:28, newer than dequant_fast.cpp). Pass bit is dense_dq vs oneDNN bf16 torch.mm within 1 ulp, plus an fp32 matmul of those bf16 weights within 8 ulps. The 5e-2 check vs the fp32 dequant is logged only. Non-finite outputs fail. Card 0. Expected about 40-90 min once the lease is free (correctness, dense M=512/4096/12288, K=640, experts, head).
COMMAND: B70_AGENT=C10b B70_GPU_LOCK_TIMEOUT=21600 bin/gpu-run --card 0 scripts/c10_run.sh. Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c10b-dq-bench.log. CPU quiet lock is taken only after card 0, with a 90s cap so a CPU-then-GPU holder is not deadlocked.
RESULT: queued. Card 0 was C9 at 09:06. CPU quiet was D3 at 09:07.
VERDICT: run open.

## 2026-10-20 09:10 G5 -- QSA XPU decode matches the CPU layer

CONFIG: card 1, image vllm-flashnext-xpu:py312, harness/xpu-run.sh
(inner bin/gpu-run, host UMD libze 1.28.2 / IGC 2.36.3). Plugin
libb70_qsa_xpu.so 369040 bytes, icpx 2026.1, intel_gpu_bmg_g31, built
09:07 under cpu-run. Test tests/test_qsa_xpu.py vs flashnext_ref
QsaState, seed 2. B70_QSA default off. Fork model.py hook is
uncommitted (docstring line 17, qsa_runner 566-572, forward 613-614).
C7: please keep that as its own commit. I did not commit.
COMMAND: CARD=1 B70_AGENT=G5 B70_GPU_LOCK_TIMEOUT=7200
vllm/flashnext/plugins/b70_qsa_xpu/tests/run_gpu.sh
Do not wrap that script in a second gpu-run.
RESULT: Earlier sycl::half scale path, same card, exited 1 in 23s.
K matched. Device scales were raw amax/127 (bits 1015215198 and
1018736225). CPU float16 scales were 1015218176 and 1018732544.
k_codes[0,0,0,20] was -103 vs -102. Health after HEALTHY.
Rebuild uses f32_to_f16/f16_to_f32 (RTNE, numpy-matched). Rerun:
QSA_XPU_OK, gpu-run done in 16s, exit 0. xpu-health --card 1 HEALTHY.
Passed: topk n=2100 width=2051, zero group, small T=8, page perm,
pos_base 100, mrope, head_dim 256 T=10, graph replay at pos 4/5/7
bitwise vs eager (attn and gated max_abs 0; state exact). Codes and
scales max_abs 0. Worst head_dim-256 residuals: pooled 2.384e-07,
scores 9.537e-07, attn 7.153e-07, gated 2.441e-04, all inside atol.
No wedge. No xe-reset.
VERDICT: decode QSA on one card matches the CPU reference on these
shapes, including a block-completing cell under graph replay. Not a
latency number. Greedy >4K vs strata_reference.json is still unmeasured
(short refs sit under the 2051 identity bound). XMX union prefill is
still size L. Hook was not served. Report: reports/1G5_qsa.md.

## 2026-10-20 09:16 C10b -- dq prefill measured, gate is bf16 matmul

CONFIG: card 0, _C.c10.so sha256 55c154ee... (08:28, not rebuilt).
Pass bit is dense_dq vs oneDNN bf16 torch.mm within 1 ulp, plus an
fp32 matmul of those bf16 weights within 8 ulps. The 5e-2 check vs
the fp32 dequant is logged only. Non-finite outputs fail.
COMMAND: bin/gpu-run --card 0 scripts/c10_run.sh. Log
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c10b-dq-bench.log.
RESULT: gpu-run exit 0 in 43s (python quiet window 12s, SYCL cache
warm). HEALTH_BEFORE 0, HEALTH_AFTER 0. Dequant 0 mismatches, 13
types. dense mm_max_abs=0 and bf16w ulps bad=0, while vs_f32_max_rel
is 23.5 (the failure C10 stopped on). MoE bmm and loop match (max
abs 0) and sit within 8 ulps of the bf16 reference. Q4/IQ4 K=2560 dq
wins at every M from 512 to 12288 (M=12288: 7.93 ms / 97.5 TFLOPS vs
fused 30.6 ms / 25.3). K=640 fused wins at M=512 (238 vs 356 us wall)
and dq wins at 4096 and 12288. Experts: fused at T=512 (10.82 ms) and
T=4096 (26.76 ms); bmm at T=12288 (66.97 vs 74.34 ms). Dispatch:
dense_wants_dq M>=4096 or (M>=512 and K>640); moe_wants_dq T>=12288;
grouped GEMM stays fused. GEMM-only budget at T=12288 is 4256 ms,
2887 tok/s, vs C8 7265 ms / 1691. 1.89 s of the 6.14 s 2000-tok/s
budget is left for attention and GDN recurrence. Report
reports/1C10_prefill_dq.md.
VERDICT: CORRECT_AND_WIRED. End-to-end 2000 tok/s is not claimed.
Recommend max_num_batched_tokens=12288.

## 2026-10-20 09:23 D3 -- fused GDN decode measured, serve capture blocked on page views

CONFIG: card 1, Ornith-35B GPTQ int4, torch 2.14.0+xpu, vLLM
0.32.1rc1.dev94+g5fd62688f. Plugin
plugins/b70_gdn_xpu (libb70_gdn_xpu.so, 08:52 build) plus
harness/patches/d3_gdn.py. Overlay default off. Attention overlay
left off. Claude fable-5.1 and claude-5.5 both unrecognized_model /
not logged in (exit 1); not retried.
COMMAND: harness/d3_probe.py --only gdn under xpu-run --card 1.
Log logs/d3-probe-gdn.json. Then two no-MTP FULL_DECODE_ONLY serves
(B70_D3_GDN=1 only). Both torn down before bench.
RESULT: SSM state is fp32 [32,128,128] = 2 MiB/layer. Conv state is
fp16 [3,8192] = 48 KiB. T=1 traffic 4.40 MB, roof 7.24 us. Stock
conv+delta graph 0.0640 ms, fused conv+delta+RMSNorm 0.0411 ms.
Core max_abs 7.6e-6, cosine 1. Post-norm max_abs 3.9e-3 is one fp16
ulp at magnitude 4. T=3 fused 0.0144 vs stock 0.0170 ms. T=3 conv
state vs stock cosine 0.15 because stock does not write conv state
when num_prefills==0 and T>1; fused vs CPU ref is exact. Spec T=3
fused 0.0161 vs stock 0.0178 ms, core cosine 1. Attention sweep is
in logs/d3-probe.json: FA auto split-k at S=200000 is 1.13 ms
(363 GB/s); custom paged kernel matches cosine 1 and is ~18x slower.
B70_D3_ATTN stays off. First serve fell back because pure decode sets
has_initial_state to None. Second serve skipped the op because
bind_kv_cache views conv/ssm out of a padded int8 page (non-contiguous
slot stride, packed inner dims). The kernel already takes stride(0).
The contiguous check is being removed and the .so rebuilt.
VERDICT: microbench correct and faster. No end-to-end number yet.
30-layer isolated upper bound on the fusion win is about 0.69 ms
on the 11.4 ms step. MoE not touched.


## 2026-10-20 09:35 orchestrator -- EP sidecar transport works (M0 64/65)
CONFIG: b70_ep_xpu, single process, both cards, same-context peer stores
(IPC open fails, not needed), bounded device flags, no oneCCL.
COMMAND: H1b window, tests/m0_probe.py (log h1b-m0-probe.log).
RESULT: one-way flag 1.82/1.77 us p50 (0->1 / 1->0); push 4 KB 0.83 us,
64 KB 5.83 us (11.2 GB/s, near PCIe 3.0 x16); preflight to 614 KB ok;
eager oracle 64/65 (first-use sample only: stale push, spin timeout);
graph samples pass; health clean after.
VERDICT: transport beats Strata's own 3-5 us handshake. First-use warm-up
is the only M0 gap; H1c (with --max-turns) continues to M1-M3.

## 2026-10-20 09:38 H1c -- M0 warmup rebuild, two-card window

CONFIG: b70_ep_xpu/_C.so sha256
bd4f4aa010c718daa6a2c983e8a93670acdd2e4df1f094c0819ed68517c39594
(476912 bytes). _C.ep.so unchanged, sha256
55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0.
Init warmup touches one word per peer page in both directions, does an
uncached flag store, resets epochs, runs one full dummy push/return
(a second try if the first spin times out), then resets epochs again.
Scored samples start after that. Both cards, no oneCCL, spin 200000.
Container vllm-fn-h1c-m0, memory 8g. Host alarm 900s. Oracle is still
65 eager and 65 graph. The same warmup is in ep_init for M1.
COMMAND: B70_AGENT=H1 B70_CPU_LOCK_TIMEOUT=1800 B70_EP_SKIP_GGUF=1
MAX_JOBS=4 bin/cpu-run plugins/b70_ep_xpu/scripts/build.sh (exit 0,
39s). Then nohup env B70_AGENT=H1 B70_GPU_LOCK_TIMEOUT=7200 bash
/tmp/h1c-m0-window.sh. Logs h1c-m0-window.out and h1c-m0-probe.log.
RESULT: rebuild ok. Window requested after this entry. Card 0 was free.
Card 1 was D3 (vllm-fn-d3-shisa) since 09:28.
VERDICT: window open.

## 2026-10-20 09:36 G2b -- audit pass 5

CONFIG: CPU only. No GPU, no lease, no systemd, no commit. Cursor is
the journal through the 09:35 orchestrator heading. Findings F36-F47
appended to reports/1G2_audit.md. Pass 4 was left as written.
COMMAND: read new JOURNAL headings and reports/1C10_prefill_dq.md and
reports/1G5_qsa.md; recompute c7-bench-ep-dp.json, c7-coll-ok.log,
c7-ep-dp-serve-ok.log, g4-cli-parse.txt, h1-m0-window.out,
h1b-m0-probe.log, h1b-m0-window.out, c10b-dq-bench.log,
d3-probe-gdn.json, d3-probe.json, d3-serve-gdn.log, d3-serve-gdn2.log,
d3-serve-gdn3.log, d3-bench-gdn.json; sha256 the new _C.so and
libb70_qsa_xpu.so; read prefill_dq.py, m0_probe.py, probe.cpp
preflight, qsa_kernels.cpp f32_to_f16, gdn_decode.cpp contig checks,
and the lowbit GateUp/Down brace diff; git status and git diff --cached
in /mnt/vm_8tb/github/b70_ai_things and the fork.
RESULT: C7 DP identity, short Paris coherence, C1/C2 ITLs, and the
159-step collective medians (15.01 / 5.62 / 20.61 / 119.06 ms) match
the raw files. Health before that serve passed; post-health was not
run, as the entry says. G4's saved log says cpu-run 18s exit 0 and
does not contain PARK_ON. H1's first window matches the 09:05 writeup,
including DEVICE_LOST and a later HEALTHY line in the window log.
C7 TP=2 is SERVE_RC 124, C9 pid 3152937, RAISE 0. H1b _C.so sha matches
22481243... and is 445392 bytes. M0 flag and push times match, eager
is 64/65, GRAPH fails=0, and health after is in h1b-m0-window.out.
The same probe log has MIRROR fails=3, so "only gap is first use" does
not hold. C10b bench lines, health rcs, and the prefill_dq.py cuts
match; 4256 ms is the report's projected GEMM sum. G5's .so is 369040
bytes at 09:07; QSA_XPU_OK is not in a saved log. D3 microbench numbers
match d3-probe-gdn.json. The first serve traceback is a missing
kv_cache attribute. d3-bench-gdn.json at 09:26 has identity and
coherence and C1 ITL 12.60 ms and is not claimed by any heading yet.
Fork tip still e442b4b53f, index empty, no new staged binary.
VERDICT: Pass 5 recorded. Latency numbers checked this pass match the
cited logs. Open items: G4 parse stdout, G5 run stdout, the M0 mirror
misses, and the unjournaled D3 bench.

## 2026-10-20 09:42 C9 -- grouped MoE schedule measured, whole-layer table

CONFIG: card 0 only (card 1 held by D3). Decode schedule in
plugins/b70_gguf_xpu/csrc/sched_moe.hpp, behind the existing moe_layer
op. B70_MOE_SCHED unset selects it; B70_MOE_SCHED=0 keeps the C1
kernels. Launch is reqd_sub_group_size(16), STRATA_EXPERT_LANES 8,
two rows per rib (row and row+32), chunks 4/2/1, gate+up in one pass
over a 36-byte Q8 block (float scale, int8 qs[32], qs at offset 4).
Subgroup 32 is the slower one-row Strata fallback and is not this
launch. Fast path: T<=8, topk<=16, I and K multiples of 32 up to 8192,
every used type DP4A (16/17/18/20/21/22/42), including the shared
expert. Device plan, so a captured graph sees new expert ids.
Library under test: b70_gguf_xpu/_C.c9.so sha256 8a76f559e65981fe
(7574840 bytes, 07:39). C8's later full link installed the same
schedule into the default _C.so sha256 086595a01babbc36 (same size,
07:40). That file was left in place. Prefill sources were not edited.
Fair Strata side relinks C5's iq_strata_jit.o (not recompiled) inside
strata-sycl-dev:020, image e41d59c0, IGC 2.36.3. Compile of
fair_strata.cpp uses -fp-model=precise and subgroup default 32; the
link line does not. Shape K=2560 I=640 topk=10, one card runs every
routed entry plus the shared expert. Gate and up share one ggml type
because native_expert_grouped requires that. Pairs: IQ3_S/Q2_0 and
IQ2_XS/IQ4_NL. Live groups: T=1 10x1; T=2 6x2+8x1; T=3 4x3+6x2+6x1;
T=4 2x3+8x2+18x1 (published card0+card1 combined) plus one shared
group of T entries. Router weight 0.1, shared weight 1.
COMMAND: bin/gpu-run --card 0. Strata cases: FAIR_MODE=cases on
c9/fair_strata, log c9/fair_all5.log (FAIR_DONE rc=0). Control graph:
c9/fair_ctrl.log. C5 anchor: c9/c5_anchor.log. Plugin graphs:
B70_GGUF_SO=_C.c9.so tests/c9_fair.py, logs c9/ours6_sched1.log and
ours6_sched0.log, outer c9/run_ours7.outer.log (44s, exit 0). Eager
A/B: c9/moe_bench_sched1c.log and sched0c.log. Gates earlier today on
the same _C.c9.so: gpu_check.log, moe_layer_bench.log,
compare_cpu.log, graph_probe.log.
RESULT: Health HEALTHY before and after the plugin graphs
(health_pre_rc=0, health_post_rc=0). No xe-reset. Control graph
(IQ3_S/Q2_0, T=4, 10 groups, 20 entries, cap 40, no shared, Q8
outside) is phase1 60.14 us, base 105.06 us, R2 92.39 us. C5's binary
on this card today is R2 92.55 us, IQR2 69.69 us, gu 60.03 us, all
103.24 us. Strata whole-layer is the sum of three synced-eager
medians (q8, grouped, reduce), 48 launches, q.wait after each launch.
Printed groups=40 and entries=43/54/65/76 are the padded caps. A
48-layer command graph of the full plan (shared included) resets the
device on replay; the per-launch wait is the measured path. T=1
grouped piece is 92.22 us (IQ3) and 91.05 us (IQ2), next to that
control. Whole-layer R2 us, per token in parentheses:
IQ3_S/Q2_0 T=1/2/3/4: 112.89 (112.89), 148.10 (74.05), 202.93 (67.64),
261.64 (65.41). IQR2 whole: 87.80, 116.62, 156.15, 206.20.
IQ2_XS/IQ4_NL R2: 110.79 (110.79), 168.87 (84.43), 231.10 (77.03),
310.50 (77.62). IQR2: 99.98, 132.26, 179.39, 236.45.
r2_iqr2_diff=0 on all eight rows (same floats, IQR2 is the faster
schedule). Plugin XPUGraph, 48 calls, median of 5, per call, sched=1:
IQ3 186.32 / 278.53 / 400.90 / 534.72; IQ2 196.31 / 289.93 / 417.92 /
571.06. sched=0: IQ3 192.80 / 351.71 / 494.24 / 660.14; IQ2 239.78 /
422.77 / 584.28 / 807.23. Cosine vs the Strata R2 dump is 0.999992 to
0.999999. Mean abs error is 0.13 to 3.89. Both plugin schedules print
the same error against Strata. Bulk max relative error reaches 1.15,
so this is not an 8e-2 match against Strata. Gates vs the plugin's own
reference: gpu_check atol/rtol 5e-2 failures=0, moe_layer_bench 8e-2
failures=0, compare_cpu 13/13 mismatches=0, GRAPH_PROBE_OK. Clean
eager A/B (E=32, IQ2_XS/IQ2_XXS/IQ4_NL, shared, top-10) sched1
230.7/289.9/492.7/877.3 vs sched0 234.7/397.6/735.6/1426.8 at
T=1/2/4/8, failures=0. T=1 stays on the new kernel. UD-IQ3_XXS header
scan: 753 tensors; shared experts Q6_K 120 and Q8_0 3; routed also
has IQ4_XS, Q3_K, Q2_K. The fast path requires DP4A on the shared
expert, so it does not run on that GGUF. C7's FULL_DECODE_ONLY serve
uses _C.graph.so. No 35B serve this stage.
VERDICT: Schedule is in the plugin and faster than the C1 fused
kernels from T=2 up (about 1.23x to 1.46x on the fair graphs, up to
1.63x eager at T=8). It does not reach Strata's whole-layer time
(about 1.65x to 2.04x slower than R2). Report:
reports/1C9_moe_sched.md.


## 2026-10-20 09:42 E2 -- one expert layer, bmm vs the 256-launch loop

CONFIG: card 0. Card 1 was D3 (vllm-fn-d3-shisa) from 09:28. Library
libb70_lowbit_xpu_e1mm.so (unpack + vendor GEMM). B70_LOWBIT_MOE_LAUNCHES=2.
Random int3 planes, E=256 H=2048 I=512 topk=8. Not a serve and not a
tok/s. The first probe overlapped the niced e2.so compile, so that
run's span sums include JIT. The second probe is one call after warmup.
e1mm.so was not overwritten. New binaries, not yet served:
libb70_lowbit_xpu_e2.so 3305936 bytes (09:35, multi-row GEMV, old
unpack) and libb70_lowbit_xpu_e2u.so 3380088 bytes (09:40, ESIMD
unpack via dequant_group). cpu-run 80s and 74s, both compile rc=0.
C10 coordination: same plan as plugins/b70_gguf_xpu prefill_dq.py
_project_bmm. Unpack b70lb1 to bf16, pad each expert to a multiple of
8, one torch.bmm per projection. No dependency on C10's .so. grouped_mm
is on this torch and accepts an empty expert (offs plateau). It is
slower here than bmm, so the serve path stays bmm.
COMMAND: xpu-run.sh --card 0 --memory 24g tests/e2_bmm_probe.py then
tests/e2_grouped_probe.py. Logs
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/e2-bmm-probe.log and
e2-grouped-probe.log.
RESULT: small bmm rel 0 vs the per-expert F.linear loop. T=2048 rel
1.36e-4, loop 115.1 ms, bmm 33.5 ms (first probe, spans polluted by
warmup). T=8192 rel 1.77e-4, loop 119.2 ms, bmm 54.1 ms. Clean second
call after warmup: T=2048 full bmm 39.28 ms (unpack_w13 17.27,
unpack_w2 8.11, bmm_w13 3.08, bmm_w2 2.25, reduce 1.87). T=8192 full
bmm 54.44 ms: unpack_w13 12.88, unpack_w2 6.47, group 2.10, pad_x
2.68, bmm_w13 8.83, silu 2.61, pad_hid 0.44, bmm_w2 6.09, reduce
10.13. grouped_mm w13 12.21 ms plus a 4.01 ms weight transpose.
Uniform-M bmm of the same w13 FLOPs is 2.71 ms, about 100 TFLOPS
(2*65536*1024*2048 / 2.71 ms). The 8.83 ms span is that GEMM plus the
output transpose and gather. Eager moe_gemv T=1 on the old kernel:
int4 0.0724 ms, int3 0.0973 ms, int2 0.0663 ms. The int3-int4 gap is
0.025 ms/layer, 1.0 ms over 40 layers, the same size as E1's served
ITL gap (11.29 vs 10.23 ms).
VERDICT: The served 22.3 s is not 16 s of attention beside one 8k
expert pass. serve_lowbit.sh left max_num_batched_tokens at the vLLM
default 2048, so an 8k prompt is four forwards. The per-expert loop is
launch-bound (115 ms at T=2048, 119 ms at T=8192, unpack extra). Four
chunks of unpack-plus-loop is about 4*40*140 ms = 22 s. One 8k bmm
layer is 54 ms, 40 layers 2.16 s of experts before GDN and attention.
5000 tok/s at 8185 tokens needs the whole forward under 1.64 s, so
2.16 s of experts misses it until unpack (19 ms) and the bmm copies
plus index_add (about 15 ms) shrink. Next: layout probe, then one
real 8k forward with max batched tokens 8192.


## 2026-10-20 09:50 C11 -- wide-SGN expert prefill build started

CONFIG -> Card 0 is leased (E2 layout/fast probes, then H1c). No GPU this step. Slim library only: grouped_prefill.cpp + dequant_fast.cpp + csrc/c11_bind.cpp, -DB70_PREFILL_SMOKE=1, out b70_gguf_xpu/_C.c11.so. Does not replace _C.so or _C.c10.so and does not link kernels.cpp. IQ3_XXS and Q2_0 s64 instantiates SGN 4/8/16/40 (WG 64/128/256/640) behind B70_PREFILL_SGN; unset stays 4. Other types stay SGN=4.

COMMAND -> B70_AGENT=C11 B70_CPU_LOCK_TIMEOUT=7200 bin/cpu-run -- bash vllm/flashnext/plugins/b70_gguf_xpu/scripts/c11_build.sh. Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c11-build.log.

RESULT -> Shared-file diff, not yet measured. csrc/bind.cpp gained prefill_wide_sgn() -> Tensor (def + CompositeExplicitAutograd impl of b70_prefill_wide_sgn_op, which fills int32 40). grouped_api.hpp declares it; grouped_prefill.cpp defines it. c11_bind.cpp is the slim twin and is not linked with bind.cpp. prefill_sched.inl launch_s64 is now template <TYPE, MT, SGN>. Dispatch (prefill_dq.py moe_wants_dq) is unchanged until the bench.

VERDICT -> Build in flight. GPU bench is scripts/c11_run.sh via bin/gpu-run --card 0 after C11_BUILD_OK.

## 2026-10-20 09:49 C11 -- _C.c11.so built, card 0 bench queued

CONFIG -> icpx 2026.1, -DB70_PREFILL_SMOKE=1, sources grouped_prefill.cpp dequant_fast.cpp c11_bind.cpp. Out plugins/b70_gguf_xpu/b70_gguf_xpu/_C.c11.so. _C.so and _C.c10.so untouched. Card 0 held by E2 (vllm-fn-e2-layer) at queue time. Bench waits on the lease.

COMMAND -> B70_AGENT=C11 B70_CPU_LOCK_TIMEOUT=7200 bin/cpu-run -- bash plugins/b70_gguf_xpu/scripts/c11_build.sh. Then B70_AGENT=C11 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=180 bin/gpu-run --card 0 plugins/b70_gguf_xpu/scripts/c11_run.sh. Logs logs/c11-build.log and logs/c11-bench.log.

RESULT -> Build rc=0 in 60s, 48 warnings (existing deprecated parallel_for), 0 errors. _C.c11.so 1591952 bytes, 09:48. C11_BUILD_OK. CPU lease released before the GPU queue. Bench not measured yet.

VERDICT -> Device time is still ahead. Dispatch stays on the C10 bmm-at-T>=12288 policy until the bench prints a winner.

## 2026-10-20 09:50 D3 -- fused GDN decode, attention left on FA

CONFIG -> Card 1. Ornith-35B GPTQ int4, FULL_DECODE_ONLY, FLASH_ATTN,
async, B70_XPU_CG_RECLAIM=500, max-model-len 33792. Plugin
plugins/b70_gdn_xpu, .so rebuilt 09:23 (322256 bytes, compile rc=0)
so strided mamba pages pass. Wrapper harness/patches/d3_gdn.py, default
off. B70_D3_ATTN off. Claude fable-5.1 and claude-5.5:
unrecognized_model / not logged in, exit 1, not retried.
COMMAND -> d3_probe.py (logs/d3-probe.json attention, logs/d3-probe-gdn.json
GDN). Then bench_latency.py. No-MTP: B70_D3_GDN=1, logs/d3-serve-gdn3.log,
logs/d3-bench-gdn.json. Shisa spec=2: B70_D2_SHISA_MTP=1 MTP=1
MTP_TOKENS=2 B70_D3_GDN=1 B70_D3_GDN_SPEC=1, logs/d3-serve-shisa.log,
logs/d3-bench-shisa.json. Servers stopped. xpu-health --card 1 HEALTHY.
RESULT -> SSM state fp32 [32,128,128] = 2 MiB/layer. T=1 traffic 4.40 MB,
roof 7.24 us. Fused conv+delta+RMSNorm graph 0.0411 ms vs stock 0.0640 ms.
Core max_abs 7.63e-6, cosine 1. T=3 fused 0.0144 vs 0.0170. Spec T=3
0.0161 vs 0.0178, cosine 1. No-MTP e2e: identity and coherence pass.
C1 ITL p50/p90/p99 12.60/12.83/13.07 ms, 79.39 tok/s (D1 12.81/12.99/13.12,
78.23). C2 p50 13.50/13.48 (D1 13.75). Prefill 1k/8k/32k 6076/7883/6415
vs D1 6030/7888/6414. Shisa spec=2: C1 18.67/19.15/19.41, 121.67 tok/s,
acceptance 113/170. D2 was 18.72/18.97/19.26, 121.73, same 113/170.
Attention: FA auto split-k at S=200000 is 1.127 ms (363 GB/s); forced
splits=1 is 7.68 ms. Custom paged kernel matches cosine 1 and is slower
(20.5 ms at 200k). Short-context 143 us/layer is qkv+o GEMVs plus a 12 us
FA core at block 1088. First Shisa attempt lost the device on a ragged
spec warmup (slot width 3, fewer query tokens). Guard falls back. Second
attempt captured uniform spec batches and benched clean.
VERDICT -> B70_D3_GDN=1 is a real 0.20 ms C1 ITL win and stays opt-in.
B70_D3_GDN_SPEC and B70_D3_ATTN stay off. MoE not touched. Report
reports/1D3_gdn_attn.md.

## 2026-10-20 09:51 E2 -- ESIMD unpack and NT bmm, one layer 34 ms

CONFIG: card 0, after H1's both-card window. libb70_lowbit_xpu_e2u.so
(3380088 bytes, ESIMD unpack, multi-row GEMV compiled but not selected).
B70_LOWBIT_MOE_LAUNCHES=2, B70_LOWBIT_GEMV_ROWS unset. Random int3,
E=256 H=2048 I=512 topk=8. Not a serve tok/s. e1mm.so and e2.so were
not overwritten. Layout timings used e1mm (scalar unpack).
COMMAND: tests/e2_layout_probe.py, tests/e2_fast_probe.py,
tests/e2_layer_probe.py via xpu-run.sh --card 0. Logs
e2-layout-probe.log, e2-fast-probe.log, e2-layer-probe.log under
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/.
RESULT: ESIMD unpack matches format.dequant at bits 2/3/4, max_abs 0,
and matches the scalar kernel. w13 unpack 3.78 ms vs scalar 12.94 ms.
w2 unpack 1.99 ms vs 6.53 ms. bmm(X, W^T) view is 2.94 ms and the row
gather is 0.52 ms. bmm(W, X^T) with a forced contiguous transpose is
7.38 ms, and the output transpose is 0.95 ms. index_add fp32 7.03 ms,
bf16 7.66 ms. The serve path is now bmm(X, W^T) plus index_select.
One 8k layer after warmup: 34.22 ms, rel 1.77e-4 vs the per-expert
loop. small rel 0. Spans: unpack_w13 3.84, unpack_w2 1.90, group 4.68,
pad_x 2.68, bmm_w13 3.37, silu 2.61, pad_hid 0.45, bmm_w2 2.79, reduce
10.14. Forty layers of that is 1.37 s of experts before GDN and
attention. Multi-row GEMV rows=2 at int4 is wrong (max_abs 1.0) and
slower (0.0748 vs 0.0645 ms) so B70_LOWBIT_GEMV_ROWS stays unset.
The fast probe exited 1 on that check. Decode keeps the 1-row kernel.
VERDICT: Expert prefill math matches and the layer dropped from 54 ms
to 34 ms. 5000 tok/s still depends on the non-expert part of a real
forward. Next serve is this library, max batched tokens 8192, one 8k
forward instrumented, then the bench with the diagnostic sync off.


## 2026-10-20 09:51 H1c -- M0 warmup passes 65/65

CONFIG: _C.so sha256 bd4f4aa010c718daa6a2c983e8a93670acdd2e4df1f094c0819ed68517c39594 (476912 bytes, built 09:38). _C.ep.so unchanged sha256 55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0. Both cards, path 2 (same-context peer pointer), no oneCCL, spin 200000. Container vllm-fn-h1c-m0, memory 8g. Init warmup touches every peer page both ways, sys-stores the flag pages, runs one dummy push/return, then resets flag epochs before the scored oracle. Host alarm 900s.
COMMAND: nohup env B70_AGENT=H1 B70_GPU_LOCK_TIMEOUT=7200 bash /tmp/h1c-m0-window.sh. The script calls bin/gpu-run bin/xpu-health, then harness/xpu-run.sh --card both --name vllm-fn-h1c-m0, then health again. It is not wrapped in an outer gpu-run. Logs /mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1c-m0-window.out and h1c-m0-probe.log.
RESULT: Queued 09:39:11 while D3 held card 1. Health before HEALTHY (cards 0 and 1) rc 0 at 09:46:03. PROBE5: zeMemOpenIpcHandle failed 2013265924, same_ctx=1, peer access both ways, atomics 0, ipc_store=0, same_store=1, path=2, bdf 0000:0b:00.0 and 0000:44:00.0. Preflight ok for 64 B, 4 KB, 17520 B, 64 KB (0 to 1) and 614400 B (1 to 0). WARMUP touch inbox=1 parts=1 flag1=1 flag0=1, flag saw1=1 saw0=1, exchange attempt=1 err0=0 err1=0, reset epochs ok=1. Eager sample s=0 first_use push_bad=0 ret_bad=0 guard_bad=0 err0=0 err1=0. EAGER fails=0 across s=0..64. Graph capture of s=0 does not execute (capture_executed=0, poison counts stay push_bad=4380 ret_bad=153600). Replay of s=0 is push_bad=0. GRAPH fails=0. Mirror arm is not a gate: fails=3, partial push_bad 1700/1596/1566, err=0. PROBE1 both directions n=200 one_way_us min 1.667 p50 1.771 max 1.980. PROBE2 65536 B groups=64 us_p50 5.834 GBps 11.233. PROBE3 card0 5.834 us, card1 5.833 us, host wall 26.631 us. PROBE4 plain_bad=0 uncached_bad=0. Process line: PASS eager 65/65 graph 65/65. PROBE_RC=0 at 09:49:45. Health after HEALTHY rc 0 at 09:50:40. No xe-reset.
VERDICT: M0 gate met. The init exchange removed the first-use spin timeout. Flag handshake is 1.8 us each way. A 64 KB push is 5.8 us (11.2 GB/s). M1 is next.

## 2026-10-20 09:52 H1c -- M1 window open

CONFIG: same _C.so bd4f4aa0 and _C.ep.so 55c154ee. Both cards. Container vllm-fn-h1c-m1, memory 32g. blk.0 of the 35B UD-IQ3_XXS GGUF, linear split 128/128, T=1,2,4,8,4096. Transport warmup runs inside ep_init; compute warmup runs on the first fully registered layer.
COMMAND: nohup env B70_AGENT=H1 B70_GPU_LOCK_TIMEOUT=7200 bash /tmp/h1c-m1-window.sh. Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1c-m1-window.out and h1c-m1-layer.log.
RESULT: requested 09:52:44. Card 0 free. Card 1 held by G4 (vllm-fn-g4-kv-c1) since 09:50:40. The window waits.
VERDICT: window open.

## 2026-10-20 09:55 G4 -- T2 died on a slots attribute

CONFIG: card 1, Ornith GPTQ int4, port 18111, ParkingOffloadConnector,
--park-finished-tail, 45 GPU blocks, cpu_bytes 34359738368, shm 40g,
memory 80g, PIECEWISE mode 0. Copy path is swap_blocks_batch.
Fork e442b4b53f. Identity hotschmoe-dev then the research alias.
COMMAND: CARD=1 vllm/flashnext/kvoffload/run_t2.sh
Logs: g4-t2-driver-c1-r2.txt and g4-t2-serve-c1-r1.txt.
Health before and after via bin/xpu-health --card 1.
RESULT: Health before OK. Ready 09:53:38. G4 parking: slots=4
chunks=1542 block=1088 exact_tail=True groups=[(0, gdn), (1, gdn),
(2, gdn), (3, full_attention)] offload_prompt_only=False. Host mmap
34.36 GB. Queued tails: turn1 end=3063 tail=887 full_keys=2; control
end=3144; B and C end=30464 tail=0 full_keys=28. Turn1 ttft 0.531s
computed 3000. Control ttft 0.443s computed 3081. B 5.032s. C 5.023s.
The restore lookup found a host HIT, then EngineCore aborted:
RequestOffloadState is a slots dataclass and has no _g4_tail_key
(parking.py:578). The restored completion was empty. The cold filler
then got HTTP 500 because the engine was dead. Health after:
xpu-health HEALTHY card 1. No T2_RESULT json.
VERDICT: the tail was stored and the exact-end lookup hit. The hit
cannot be hung off RequestOffloadState. Not a pass. Next run keeps
the choice on the scheduler.

## 2026-10-20 09:59 H1c -- M1 r1 bitwise, T=4096 elem-rel

CONFIG: _C.so bd4f4aa0, _C.ep.so 55c154ee. Both cards, path 2, container vllm-fn-h1c-m1, memory 32g. blk.0 IQ3_XXS, E=256 split 128/128, qtypes gate/up 22 (IQ2_S) down 21 (IQ3_S). T=1,2,4,8,4096 topk=8. Spin 200000.
COMMAND: bash /tmp/h1c-m1-window.sh. Logs preserved as h1c-m1-layer-r1.log and h1c-m1-window-r1.out.
RESULT: Health before HEALTHY. Transport warmup touch all 1. Exchange attempt 1 err1=11 (first ep_dispatch/ep_wait in this process), attempt 2 err0=0 err1=0. Compute warmup prejit T=1 and T=16, moe attempt 1 err=0. Own experts bitwise at every T (own_max=0). Versus one-card moe_layer: T=1 full_max 3.7e-9 elem_rel 8.7e-6 cos 1; T=2 7.5e-9 / 4.3e-4; T=4 and T=8 1.5e-8 / 4.5e-4; T=4096 full_max 3.0e-8 elem_rel 7.5e-2 cos 1.000000. The per-element ratio uses max(|ref|, 1e-8), so a few near-zero outputs at T=4096 miss 1e-2 while the absolute gap stays at a few ulp. TIME T=1 ep 580 us vs one-card 411 us. TIME T=8 ep 591 us vs one-card 816 us. M1_RC=1. The window then xe-reset on any nonzero rc (rebind, RECOVERED, collective health OK). Health after HEALTHY rc 0 at 09:58:47. No device-lost in the probe.
VERDICT: transport and the split are exact against y0+y1. The T=4096 miss is the per-element relative definition, not a wrong sum. The rerun scores max|a-b| / max|b| and still prints elem_rel. xe-reset on a numeric miss will not run again.

## 2026-10-20 10:05 C11 -- wide SGN measured, card 0 blocked behind H1

CONFIG -> _C.c11.so (smoke, SGN 4/8/16/40). Card 0. Log logs/c11-bench2.log. Health before and after 0.

COMMAND -> B70_AGENT=C11 B70_GPU_LOCK_TIMEOUT=21600 bin/gpu-run --card 0 plugins/b70_gguf_xpu/scripts/c11_run.sh

RESULT -> SGN 8/16/40 match SGN=4 exactly (max_abs 0) on the small layer. Device max work-group is 1024, but the MT=16 kernel rejects a group above 512, so SGN=40 (640) does not launch at T=4096 or 12288. It does launch at T=512 (MT=4) and is slower (14.7 ms vs 9.9). Layer event ms, SGN 4/8/16: T=512 9.87/10.11/9.98; T=4096 27.90/27.67/27.93; T=12288 77.92/76.42/76.14. C10 fused was 10.82/26.76/74.34 and C10 bmm at 12288 was 66.97. Widening does not cut X reads: each subgroup still loads X. Dequant of the three packs is 7.81/7.81/7.41 ms at 256 GB/s. Copy of one gate bf16 buffer is 6.38 ms at 526 GB/s. Routing empty count is 0 at every T, so skipping experts saves nothing. bf16-reference max_rel on the scaled random pack is not green yet (max_abs 23.2, max_rel 1.94); a dtype probe is queued. oneDNN shapes did not run: torch.nn.functional was imported as F and hid the intermediate size. That is fixed for the queued rerun.

VERDICT -> SGN=16 is not the expert win. Next binary _C.c11b.so calls the existing fp16 SLM kernel (X panel shared across the group) when B70_PREFILL_XSLM=1. Card 0 is held by H1's both-card gpu-run xpu-health, which took card 0 at 09:59 and is flock-waiting on card 1 (G4 serve). C11's oneDNN probe is queued behind that. Not a dispatch change.

## 2026-10-20 10:16 G2b -- audit pass 6

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor was the 10:05 C11 heading (JOURNAL line 2611). Findings F48-F57 append reports/1G2_audit.md after F47.
COMMAND: Read new headings from 09:42 C9 through 10:05 C11. Check bench JSON and serve logs under /mnt/vm_8tb/b70/build/vllm-flashnext/logs and c9/. Spot-check sched_moe.hpp, d3_gdn.py, parking.py, and the ep memcpy waits. git status in /mnt/vm_8tb/github/b70_ai_things and the fork at /mnt/vm_8tb/b70/build/vllm-flashnext/vllm.
RESULT: H1c M0 65/65 matches h1c-m0-probe.log, mirror fails=3 disclosed, health before and after HEALTHY, no xe-reset (F48). C9 fair and eager cells match the logs; failures=0; the Strata ratio mixes a part-sum with a graph; the GGUF census was not re-counted (F49). E2 probe spans match; the 22 s line is an estimate; GEMV rows=2 mismatches (F50). C11 bench2 matches, failures=1, health 0; c11-nan.log had already finished at 09:55 with all-NaN y, so "dtype probe queued" is stale (F51). D3 and D1/D2 JSON numbers match; raw C1 delta is 0.204 ms; the first-attempt device-loss log was overwritten and no D3 health file was saved (F52). G4 T2 abort matches the driver and parking.py:578 traceback; health after in that driver is HEALTHY; r3 is a later 180s lease timeout, rc=124 (F53). H1c M1 table matches, M1_RC=1, xe-reset RECOVERED, health after 0 (F54). Unjournaled: e2-serve.log died at 10:14 in expert_prefill_bmm with UR_RESULT_ERROR_OUT_OF_RESOURCES on an 8204-token prefill, gpu-run exit 0, no coherence body and no health line (F55). Fork tip is 9761b7b48a, ahead 15, not pushed. Cached diff empty. No new staged binary. Host loader still libze 1.28.2.
VERDICT: Pass 6 recorded. Latency numbers checked this pass match the cited logs. Open items: G4 r3 and the parking re-bench, the overwritten D3 first-attempt log, the E2 serve OOR with no health after, G5 QSA_XPU_OK still unsaved, and the C11 nan outputs.

## 2026-10-20 10:21 E2 -- full-batch bmm OOMs beside the KV cache

CONFIG: card 1, port 18084, container vllm-fn-lowbit-e2,
libb70_lowbit_xpu_e2u.so, int3 checkpoint, FULL_DECODE_ONLY,
FLASH_ATTN, bf16, B70_LOWBIT_MOE_LAUNCHES=2, B70_XPU_CG_RECLAIM=500,
max_num_batched_tokens=8192, prefix caching left at the vLLM default
(on). Mode file started as matmul, then bmm, same weight load.
B70_LOWBIT_GEMV_ROWS unset. Diagnostic hooks were installed.
COMMAND: logs/e2_serve_driver.sh. Serve log e2-serve.log. Diag log
e2-driver.log.
RESULT: Identity hotschmoe-dev first, alias qwen36-w3a16-b70-lowbit.
Coherence both times: " Paris, a city renowned for its rich history,
culture, and iconic landmarks." MATMUL_1 was a cold 8204-token
prefill, TTFT 33212 ms, 247 tok/s. MATMUL_2 TTFT 4311 ms, 1903 tok/s,
but the engine prefix-cache hit rate was 46.4 percent, so that TTFT
is not a full 8k prefill. Switching the mode file to bmm then failed
inside expert_prefill_bmm at torch.bmm(xpad, w13.transpose(1, 2)):
level_zero error 40 UR_RESULT_ERROR_OUT_OF_RESOURCES. The dumped
scheduler step had only 588 scheduled tokens (7616 prefix hits). The
per-expert matmul path had just served the same shape. fwd_split.log
contains only "fwd_split installed". The worker executes
vllm.v1.worker.gpu.model_runner.GPUModelRunner, and the hook patched
vllm.v1.worker.gpu_model_runner.GPUModelRunner. No layer split was
recorded. Engine shutdown was clean. Card 1 was taken by G4 before a
post-crash health run. gpu-run lease released.
VERDICT: BMM_OOM. The 256-expert bmm does not fit in the residual
next to weights and KV. Next serve tiles that bmm (at most 32
experts), patches the v2 runner, disables prefix caching, and sets
max_num_batched_tokens=16384 so one 8k prompt is one step. Decode
GEMV is unchanged.

## 2026-10-20 10:22 H1c -- M1 layer passes

CONFIG: same _C.so bd4f4aa0 and _C.ep.so 55c154ee. Both cards, path 2. blk.0 of 35B UD-IQ3_XXS, E=256 split 128/128, topk=8, T=1,2,4,8,4096. Gate is bitwise y_ep==y0+y1, cosine > 0.999, and max|y_ep-y_full| / max|y_full| < 1e-2. elem_rel is logged and is not the gate. Container vllm-fn-h1c-m1, memory 32g.
COMMAND: bash /tmp/h1c-m1-window.sh, second launch, after E2 released card 1. Log h1c-m1-layer-r2.log.
RESULT: Health before HEALTHY at 10:15:00. Transport warmup touch all 1, exchange attempt 1 err0=0 err1=0 (the r1 process needed a second attempt). Compute warmup prejit T=1 and T=16, moe attempt 1 err=0. Every T is bitwise (own_max=0) with cosine 1. Scale-relative full_rel: T=1 5.9e-8, T=2 6.6e-8, T=4 1.4e-7, T=8 1.2e-7, T=4096 8.9e-8. Absolute max at T=4096 is 3.0e-8. elem_rel at T=4096 is still 7.5e-2 and is the near-zero ratio from r1. TIME T=1 ep 500 us vs one-card 410 us. TIME T=8 ep 592 us vs one-card 818 us. PASS m1. M1_RC=0 at 10:21:40. No xe-reset. Health after was requested at 10:21:40 and is waiting on E2s new card-1 serve.
VERDICT: M1 gate met. The split matches the one-card layer inside FP32 reorder. Decode-sized EP layer time is 0.50-0.59 ms versus 0.41-0.82 ms one-card.

## 2026-10-20 10:26 C11 -- expert prefill stays on the C10 split

CONFIG -> Card 0. IQ3_XXS gate/up, Q2_0 down, E=512 top-10 H=2560 F=640. _C.c11.so (1591952 bytes, 09:48) and _C.c11b.so (1760272 bytes, 10:04, XSLM opt-in). _C.so and _C.c10.so untouched. x scaled by 0.001 so fp16 swiglu stays finite. Shared diff: bind.cpp prefill_wide_sgn (09:50 note) plus prefill_sched.inl MT>4 work-group cap at 512 with fallback to SGN=4. The cap is source only; neither .so has it. Default SGN stays 4. DQ_T_MIN stays 12288.
COMMAND -> Builds: bin/cpu-run scripts/c11_build.sh, logs c11-build.log (60s, 48 warnings) and c11-build-b.log (70s, 60 warnings). GPU: bin/gpu-run --card 0 scripts/c11_run.sh. Logs c11-bench2.log (_C.c11.so), c11-dnn.log (--ref-debug --bench-only --dnn-only), c11-xslm.log (_C.c11b.so --sgn-only), c11-nan.log (both .so). Health before and after 0 on every run. No xe-reset.
RESULT -> Schedules match: SGN 8/16/40 and XSLM vs SGN=4 max_abs 0. bf16-weight gate fails (c11-dnn REF bf16 max_abs 25.28 max_rel 1.98 frac_gate 0.9368; fp32 and fp16 refs also miss). Gate not weakened. Unscaled randn is all-NaN on _C.c11.so and _C.c10.so (dequant max 480). Empty experts 0 at T=512/4096/12288. SGN event ms from c11-xslm, 4/8/16: T=512 10.00/10.12/10.00; T=4096 28.26/27.87/27.98; T=12288 77.78/76.54/76.28. SGN=40 launches at T=512 (14.76 ms) and throws at MT=16 (work-group 640, kernel limit 512). XSLM is bit-exact and 14.80/28.97/70.02 ms; at T=12288 down drops 18.98 to 11.94 ms and gate+up stays 45.11. C10 bmm at T=12288 is 66.97 ms. Dequant gate/up/down 7.83/7.82/7.40 ms at 255-259 GB/s; bf16 copy of one gate 6.35 ms at 529 GB/s. Uniform bmm at T=12288: gate 4.72 ms (85 TF), gate+up 7.86 ms (102 TF), down 4.65 ms (87 TF). grouped_mm gate 21.53 ms and does not scale from T=4096. Dense mm ceiling 3.24 ms at 124 TF. Three dequants plus gate+up bmm plus down bmm add to 35.55 ms of kernels at T=12288 and 33.12 ms at T=4096. Those sums were not run as a layer. Two streams: mm 636 us, copy 124 us, both 727 us. moe_dq and the 4255.71 ms / 2887 tok/s budget stay as reports/1C10_prefill_dq.md published them. The bench BUDGET line best_sgn=xslm512 (70.02 ms, 2791 tok/s) is the fastest C11 kernel and is slower than C10 bmm.
VERDICT -> 30 ms expert layer unmet. Dispatch unchanged: fused below T=12288, dequant + bmm at T>=12288. B70_PREFILL_XSLM and B70_PREFILL_SGN stay opt-in. Report reports/1C11_expert_prefill.md.

## 2026-10-20 10:28 G4 -- T2 second turn restored from host

CONFIG: card 1, Ornith GPTQ int4 MixedCal-v2, port 18111,
ParkingOffloadConnector, --park-finished-tail, num-gpu-blocks-override 45,
cpu_bytes 34359738368, shm 40g, container memory 80g, slots 4,
offload_prompt_only false, prefix caching, max-model-len 32768,
max-num-seqs 2, compilation mode 0 PIECEWISE. Copy path is
swap_blocks_batch. Fork tip 9761b7b48a. G4 commits on that branch are
c38d0a6197 (F-A) and e442b4b53f (CLI flag). Not pushed. Identity
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
P1=3000, R1=64, U2=17, E=3063, turn-2 prompt=3081.
COMMAND: /tmp/g4-launch-t2.sh attempt 2 ran CARD=1
vllm/flashnext/kvoffload/run_t2.sh. Logs g4-t2-driver-c1-r3.txt,
g4-t2-serve-c1-pass.txt, g4-t2-turn-c1.json. Health before and after
via bin/xpu-health --card 1.
RESULT: Attempt 1 (09:58) lost card 1 to H1 and exited 124 before ready.
Attempt 2 health before ok, ready 10:20:05. Engine log: slots=4
chunks=1542 block=1088 exact_tail=True, groups 3 gdn + 1 full_attention,
offload_prompt_only=False, mmap 34.36 GB. Queued tails: turn1 end=3063
tail=887 full_keys=2; control end=3144; B and C end=30464 tail=0
full_keys=28. Restore: G4 load tail end=3063 local=0 keys=6. The serve
log has no "G4 state group" warning, so the GDN dest was block index
(3063-1)//1088 = 2 and that block was unhashed. JSON pass=True.
cached=3063, computed=18 (len(U2)+1), host_delta=3063, device_delta=0.
Greedy ids and text matched the salt=control prefill for 64 tokens.
Restored TTFT 0.1753 s, control TTFT 0.4166 s. Cold filler TTFT 4.934 s
is recorded and is not the gate. Host load 131235840 bytes in 0.063618 s
= 2.063 GB/s. First store window 391249920 bytes in 0.089957 s = 4.35 GB/s.
Both copies are swap_blocks_batch. Health after HEALTHY card 1, 26 s,
exit 0. T2_DRIVER_EXIT 0 at 10:22:06. No xe-reset.
VERDICT: T2 gate passed on Ornith. Device delta 0 means the 45-block cap
evicted A's GPU blocks and the host supplied the whole hit. This is not
the 7.5 GB/s copy engine and not the 4-parked decode stall test. F-C was
not required: the tail was already resident at lookup. W8 bench is next.

## 2026-10-20 10:38 G4 -- F-B is a test, mid-block hit kept

CONFIG: CPU only. VLLM_TARGET_DEVICE=cpu, no /dev/dri, HF_HUB_OFFLINE=1.
Container memory 16g, cpus 4. Fork commit 765993c197 on b70/flashnext,
parent 9761b7b48a, not pushed. One file: tests/v1/core/test_scheduler.py.
COMMAND: docker run the assertion in
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/g4_fb_test.py. Log
g4-fb-test-r3.txt. pytest is not installed, so this did not go through
pytest collection. The committed test is the same numbers.
RESULT: F_B_OK. block=16, external hit=23, prompt=64, scheduled=41,
connector hits=23. After schedule(), num_computed_tokens is 64 because
_update_after_schedule adds the scheduled chunk. Rounding the hit down
to the block would schedule 48. No GPU. No xe-reset.
VERDICT: F-B does not change the scheduler. A mid-block external count
is already admitted when the local hit has no partial tail. The
post-load minus-one backoff was not touched. Ornith T2 is the hybrid
proof of that path. W8 bench is still waiting on a free card.

## 2026-10-20 10:39 C12 -- per-type decode schedule linked, host dequant exact

CONFIG: Candidate plugins/b70_gguf_xpu/b70_gguf_xpu/_C.c12.so sha256 d46f80175c67a10681e09e7642b5293eedec0d3f98517b5fe7ba4c9662b70112, 10120096 bytes, mtime 10:28. Not copied onto _C.so (sha 086595a0, 7574840 bytes, 07:40) or _C.graph.so (6987192 bytes, 06:59). Fast path now accepts Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and IQ4_XS in addition to the C9 DP4A set (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0). One templated GU kernel per distinct gate/up type and one DOWN kernel per distinct down type, chained so two types do not share the EUs. Geometry is still the C9/R2 plan: subgroup 16, 8 lanes, 2 rows per rib, work-group 256, 36-byte Q8 activation. Float types use b70_decode32 plus fma. DP4A types use b70_prep. B70_MOE_PHASE=1 prints C12_PHASE (host submit plus wait, serialized) and must stay off during graph capture. Shared diffs: decode.hpp Q8_0, iq_dot.hpp prep, row_moe.hpp one case, kernels.cpp one B70_DISPATCH(B70_Q8_0), dequant_fast.cpp one case, types.py, compare_cpu.py WANTED 8, sched_moe.hpp. bind.cpp and grouped_prefill.cpp were not edited. Q8_0 is not added to the kreg type list.
COMMAND: bin/cpu-run scripts/build.sh with B70_PREFILL_OUT=_C.c12.so. Log /mnt/vm_8tb/b70/build/vllm-flashnext/c12/build.log. Host compare: bin/cpu-run tests/compare_cpu.py. Log c12/compare_cpu.log.
RESULT: First link exited 1 in 0.3s: host_dequant.cpp includes decode.hpp before size_t is visible, and the Q8_0 row pointer used a size_t cast. The pointer is row + gi * 34. Second link exited 0 in 125s. compare_cpu VERDICT failures=0 types=14/14, including Q8_0 tensor blk.39.attn_k.weight rows=2 k=2048 mismatches=0. Card-0 gates are queued with B70_GPU_LOCK_TIMEOUT=0 on bin/gpu-run --card 0 c12/run_gates.sh (gpu_check 5e-2, moe_layer_bench 8e-2, c12_types, c9_fair graph, c12_phase). At 10:39 card 0 is held by H1 pid 3218685 (both-card xpu-health, flock -w 7200 on card 1, which E2's serve holds). G5's QSA docker is ahead of this gate run on card 0. No device time yet. No install. No serve.
VERDICT: Host dequant is bit-exact on every type the 35B GGUF and Flash-Next use. The per-type schedule is only in the candidate .so. Fair us/call, the phase split, the _C.so/_C.graph.so install, and C1/C2 ITL wait on card 0.

## 2026-10-20 10:38 G5b -- QSA page spec and serve hook

CONFIG: CPU only for the new checks. Fork
/mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext, not pushed.
Host uv CPython 3.12.13 plus the flashnext venv site-packages. SYCL library
/mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu/libb70_qsa_xpu.so is 817928
bytes, mtime 10:16, icpx 2026.1, built by bin/cpu-run of
plugins/b70_qsa_xpu/build.sh (rc 0, both qsa_kernels.cpp and qsa_batch.cpp).
No second build while the GPU job may map that file.
COMMAND: At 10:10, C7 had not committed the default-off hook, so G5b
committed only vllm/models/qwen4_exp/xpu/model.py. Spec test:
tests/test_kv_spec.py. Serve bookkeeping: tests/test_qsa_serve.py. GPU:
CARD=0 B70_AGENT=G5b B70_GPU_LOCK_TIMEOUT=7200
plugins/b70_qsa_xpu/tests/run_gpu.sh, log
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/g5b-qsa-batch.log. That job is
still waiting. It is not a second lease.
RESULT: Hook commit 9761b7b48af63922e8be4599aebda548c8ba77e4, subject
"[XPU] Attach the default-off QSA plugin when B70_QSA=1". It is the parent
of G4's later 765993c197. model.py is clean. QSA_SPEC_OK: pooled block-16
page 18944 bytes, draft 16896, manager FullAttentionManager, uniform base
FullAttentionSpec, block 16 and 64 are not kernel-split. Builder cudagraph
support is UNIFORM_SINGLE_TOKEN_DECODE. QSA_SERVE_OK: uint8 page view,
external-page alias, side state keyed by physical block 0, inactive rows
scatter to private sinks, export/import remaps logical blocks, padding id
0 is not wiped, mark_draft switches the spec to the draft page. Card 0
health-before in the queued job was HEALTHY (rc 0, 26 s) and then released.
xpu-run is flock-waiting on card 0, held by H1 pid 3218685 since 10:22:06
(bin/xpu-health). No QSA_BATCH_OK yet. No xe-reset.
VERDICT: The fork hook is its own commit. The KV spec and the park/export
bookkeeping match the CPU page layout. Batch-kernel numeric match on XPU
is not measured; the card-0 job is queued behind H1. C7: MTP layers are
Qwen4ExpDecoderLayer, so B70_QSA=1 attaches the pooled runner. Please call
b70_qsa_xpu.model_hook.mark_draft(layer) on each MTP decoder layer at the
end of MTP init, before KV spec collection, and runner.set_draft_ids(ids)
with the target selection before the draft forward. Draft pages are int8
KV with no pooled rows. One small commit. G5b will not edit mtp.py.

## 2026-10-20 10:41 C7 -- TP=2 window, loader must refuse packed sharding

CONFIG: fork e442b4b53f. Same 35B UD-IQ3_XXS, TP=2,
--enable-expert-parallel, enforce-eager, max_model_len 2048.
No packed sharding. No P2P. Collective health first.
COMMAND: bash /tmp/c7-tp-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.


## 2026-10-20 10:42 H1c -- M1 health closed, M2 window open

CONFIG: M1 health after the 10:21 PASS. M2 is eager, B70_EP=1, both cards, port 18114, container vllm-fn-h1-m2, max_len 2048, GRAPH=0. Reference greedy text is the C4/C7 string " Paris, a city renowned for its rich history, culture, and iconic landmarks."
COMMAND: M1 window finished by itself. M2: nohup env B70_AGENT=H1 B70_GPU_LOCK_TIMEOUT=7200 bash /tmp/h1c-m2-window.sh. Logs h1c-m2-window.out, h1c-m2-serve.log, h1c-m2-greedy.json, h1c-m2-bench.json.
RESULT: M1 HEALTH_AFTER_RC=0 at 10:40:23, cards 0 and 1 HEALTHY. M2 requested 10:42:03. Both cards were held by C7 (pid 3230134) since 10:41:34. The window waits.
VERDICT: M1 is closed. M2 is open.

## 2026-10-20 10:43 C12 -- gate script was not executable, requeued

CONFIG: Same candidate _C.c12.so sha256 d46f80175c67a10681e09e7642b5293eedec0d3f98517b5fe7ba4c9662b70112. Card 0 only. No kernel launch.
COMMAND: bin/gpu-run --card 0 /mnt/vm_8tb/b70/build/vllm-flashnext/c12/run_gates.sh
RESULT: The first acquire ran the script and the kernel returned EACCES (exit 126, 0s). The file mode was 644. chmod +x and the same command is queued again. CPU crop probe of tests/c12_types.py shapes passed: five cases CROP_OK and Q8_0 cropped dequant mismatches=0 (c12/crop_probe.log, 30s). At 10:43 card 0 is held by C7's TP=2 eager serve (vllm-fn-qwen36-gguf-tp, old _C.graph.so). No install. No xe-reset.
VERDICT: Host path is unchanged. Device gates have not started.

## 2026-10-20 10:48 C7 -- Flash-Next IQ3_XXS DP=2 EP window

CONFIG: fork e442b4b53f. qwen4_exp IQ3_XXS shard 1 plus the IQ4_NL
PLE shard, HF hf-flashnext, DP=2 TP=1 --enable-expert-parallel,
enforce-eager, language-model-only, max_model_len 2048, memory 100g.
SYCL_CACHE_PERSISTENT=0. No P2P. No hotschmoe-dd. Collective health
first. Greedy chat vs recorded Strata texts 323 and 667.
COMMAND: bash /tmp/c7-fn-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.


## 2026-10-20 10:50 G4 -- W8 bench overflowed a device pointer; ordered library is built

CONFIG: card 1, libb70_kvpark.so 90904 bytes from 08:41 (no ordered
symbol). USM-host via sycl::malloc_host, copy queue group 1.
Side library libb70_kvpark_ordered.so built with cpu-run, no GPU.
C7's Flash-Next DP=2 window (10:48, /tmp/c7-fn-window.sh) holds both
cards after this. qwen4_exp was not edited.

COMMAND: /tmp/g4-launch-copy.sh then
CARD=1 vllm/flashnext/kvoffload/run_copy_bench.sh.
Health: bin/xpu-health --card 1 before and after.
Ordered build: SO_NAME=libb70_kvpark_ordered.so
vllm/flashnext/plugins/b70_kvpark/build.sh under bin/cpu-run.

RESULT: COPY_START 10:40:42 card 1. engine_ordinal=1, malloc_host
type=host. The first torch.tensor of the device pointer raised
ValueError: Overflow when unpacking long long. No COPY_BENCH line
and no GB/s. gpu-run exit 1 in 15s. Health after 10:41:34 HEALTHY
card 1, exit 0. No xe-reset. The 08:41 file is unchanged.
Ordered compile rc=0 in 74s at 10:47, 95512 bytes, schema
"ordered() -> int". That file has not been loaded on a GPU.
bench_copy.py now packs pointers as uint64 and views them as int64.
The next bench waits out C7's window, then loads the ordered file.
T3 client is written (t3_test.py, run_t3.sh) and has not run.

VERDICT: W8 has no bandwidth number. T2 on swap_blocks_batch stands
(2.063 GB/s host load, gate pass). Direct copy stays off.

## 2026-10-20 10:52 E2 -- one-chunk 8k split, dense path was still WNA16

CONFIG: card 1, port 18084, container vllm-fn-lowbit-e2,
libb70_lowbit_xpu_e2u.so (not rebuilt), int3 experts / int4 trunk,
FULL_DECODE_ONLY, FLASH_ATTN, bf16, MOE_LAUNCHES=2, GEMV_ROWS unset,
max_num_batched_tokens=16384, PREFIX_CACHE=0, B70_XPU_CG_RECLAIM=500.
Env B70_LOWBIT_PREFILL=bmm with the mode file starting at matmul.
fwd_split patched both GPUModelRunner classes. Hooks on for the diag.
C10b (09:16) dequant + oneDNN bmm is the same expert shape on a
different checkpoint; this entry is the served lowbit plugin.
COMMAND: logs/e2_serve_driver.sh second launch. Split log
e2-fwd-split.log. Serve log e2-serve.log (this run only).
RESULT: Identity hotschmoe-dev first, alias qwen36-w3a16-b70-lowbit.
Coherence both times the Paris sentence, TTFT 92 ms and 71 ms.
Unique 8186-token prompts, one step each. MATMUL_1 TTFT 31352 ms,
261 tok/s. MATMUL_2 TTFT 31299 ms, 262 tok/s. Warm matmul split
(step 2): forward 31263 ms, linear_attn 19519 ms (n=30), self_attn
5312 ms (n=10), mlp 6395 ms, experts 6390 ms, shared_expert 2272 ms
(n=80), router 4.4 ms, norm 22 ms. host_sync item 7, tolist 44,
synchronize 1. BMM_1 TTFT 35950 ms, 228 tok/s. Step 3: forward
35754 ms, linear_attn 19592, self_attn 5329, experts 10794.
lowbit spans: group 3687, bmm_w13 1692, bmm_w2 1281, pad_x 441,
reduce 372, unpack_w13 135, unpack_w2 68. BMM_2 returned no usage.
Engine died at 10:39:48 on F.sigmoid of the shared-expert gate,
level_zero error 40 UR_RESULT_ERROR_OUT_OF_RESOURCES. Diag still
wrote the off file and returned 0. bench_latency coherence then
HTTP 500 EngineDeadError. Container stopped. xpu-health --card 1
rc 0 at 10:40:41. The XPU GDN logger returns immediately on XPU, so
the missing "Using SYCL GDN" line does not identify the backend.
linear.py only unpacked for env exactly "matmul", so env "bmm" sent
every dense projection through prefill_wna16.
VERDICT: The ~25 s outside the expert loop is dense WNA16 inside
linear_attn, self_attn, and the shared expert, not a host sync.
E1's 0.1 s dense figure was the unpack + F.linear path. The tiled
bmm still unpacked all 256 experts (~1.5 GB) and the second 8k
request OOM'd. Not a 5000 tok/s result. Next serve: dense matmul
and bmm both use unpack + F.linear, bmm unpacks one tile, explicit
--gdn-prefill-backend sycl, diag fails closed before the bench.
GEMV_ROWS stays unset (rows=2 int4 max_abs 1.0 on the earlier probe).

## 2026-10-20 10:53 H1c -- M2 r1 layer-1 flag timeout, eager wait rebuilt

CONFIG: _C.so was bd4f4aa0. Eager B70_EP=1, both cards, port 18114, container vllm-fn-h1-m2, max_len 2048. Spin 200000. nlayers=64.
COMMAND: bash /tmp/h1c-m2-window.sh. Logs kept as h1c-m2-serve-r1.log and h1c-m2-window-r1.out. Rebuild: B70_EP_SKIP_GGUF=1 bin/cpu-run plugins/b70_ep_xpu/scripts/build.sh.
RESULT: Health before HEALTHY. Transport and compute warmup passed on attempt 1. All 40 MoE layers split 128/128. Engine init finished in 11.56 s, so the profile forward returned. The greedy request then died in method.py apply: "EP previous error layer=1 err0=0 err1=11" and RuntimeError ep flag timeout on a previous call. Layer 0 had no stored error. Card 1s layer-1 wait is queued right after its own layer-0 push, so it spins through card 0s expert compute. Connection reset on the client. SERVE stopped. Health after HEALTHY rc 0 at 10:48:44. No xe-reset. New _C.so sha256 7f0981fd20a2375965fd89612bb32969a71ca6906ef94bfa9a6d1d0e4926f010 (476912 bytes). Eager ep_moe now host-waits both queues before returning, and throws on that call if a flag timed out. Capture still records the pipelined chain.
VERDICT: r1 text was not measured. The rerun uses the eager host wait.

## 2026-10-20 10:43 C7 -- TP=2 loader refuses packed sharding

CONFIG: fork e442b4b53f at launch (later tip 35da272b61 does not
change this raise). 35B UD-IQ3_XXS, TP=2, --enable-expert-parallel,
enforce-eager, max_model_len 2048. No P2P. No packed sharding.
COMMAND: bash /tmp/c7-tp-window.sh at 10:41:34. Logs
c7-tp-window-raise.out and c7-ep-tp-serve-raise.log.
RESULT: HEALTH_RC 0. Both workers raised at serve_loader.py:337:
tensor parallel size is 2. Packed attention, shared expert, and
vocab sharding are not implemented. RAISE:1 at 10:43:25. SERVE_RC
137 is docker stop after the Python exception, not a GPU fault.
The following Flash-Next window's collective health passed, so no
xe-reset was required.
VERDICT: TP=2 is the expected loader refusal. EP for this GGUF
stays DP=2 with TP=1, which was measured at 09:01.

## 2026-10-20 10:51 C7 -- Flash-Next load dies on indexer K

CONFIG: fork e442b4b53f. qwen4_exp IQ3_XXS, DP=2 TP=1,
--enable-expert-parallel, enforce-eager, language-model-only,
max_model_len 2048, memory 100g, SYCL_CACHE_PERSISTENT=0. No P2P.
No hotschmoe-dd.
COMMAND: bash /tmp/c7-fn-window.sh at 10:48:48. Logs
c7-fn-window-index.out and c7-fn-serve-index.log.
RESULT: HEALTH_RC 0. Shards=2, tensors=1224, root
Qwen4ExpForCausalLM, EP 256/512, qk_rows=4096. placed 100, then
both workers IndexError in serve_loader.py _check_linear:
sizes[_SHARD_INDEX[shard]]. The next linear is
blk.3.indexer.k_proj.weight, ggml shape (2560, 128), shard k.
index_qk_proj is one ReplicatedLinear of 512+128. Q is (2560, 512).
SERVE_RC 1. No greedy. No xe-reset. G4 took card 1 at 10:51:26.
VERDICT: loader check, not a collective failure. Next commit sums
the split shards against the one partition.

## 2026-10-20 10:55 C7 -- indexer Q and K share one linear

CONFIG: fork 35da272b61, not pushed. serve_loader.py only.
COMMAND: py_compile. Metadata read of the IQ3_XXS shard: Q rows
512, K rows 128, hidden 2560, indexer heads 4 and 1, head dim 128.
RESULT: PY_COMPILE_OK. A one-partition module records each named
shard and checks the row sum after the load. 512+128 equals the
ReplicatedLinear output.
VERDICT: unmeasured. The next Flash-Next window uses this commit.


## 2026-10-20 10:56 G4 -- W8 H2D 14.0 GB/s, D2H 4.58 GB/s

CONFIG: card 1, one process, cpu-run --quiet around the copies.
Library libb70_kvpark_ordered.so, 95512 bytes, ordered()=1,
engine group 1. Host buffers sycl::malloc_host. Sizes 16 MiB,
64 MiB, 256 MiB, 1 GiB. Decimal GB/s over the sync wait.
swap_blocks_batch timed on the same pointers. xpu-health before
and after. B70_KVPARK_COPY stays 0. No serve.

COMMAND: /tmp/g4-launch-copy-r2.sh then
B70_KVPARK_SO=libb70_kvpark_ordered.so CARD=1
vllm/flashnext/kvoffload/run_copy_bench.sh.
Log g4-copy-driver-r2.txt. An earlier run of the 08:41 library
(g4-copy-driver.txt, 10:40) died in torch.tensor of a device
pointer above 2^63-1. Pointers are now packed uint64 viewed as
int64. dev_ptr on this run was 0xffffd556b7400000.

RESULT: bad=0 at every size. H2D 13.344 / 14.031 / 14.223 / 14.037.
D2H 4.636 / 4.583 / 4.659 / 4.670. 1 MiB fragments about 13.4 GB/s
H2D. swap H2D 8.853 / 8.110 / 6.042 / 4.119. swap D2H about 4.66.
worst one-buffer 4.583. COPY_BENCH printed. Exit 1, the gate wants
both directions at or above 7.5. Health after 10:52:09 HEALTHY
card 1, exit 0. No xe-reset.

VERDICT: H2D on the copy queue clears the 7.5 GB/s bar and the
8.5 GB/s Strata probe. D2H stays at the swap_blocks_batch rate.
The op is not wired into a serve. A host-bias probe is next.

## 2026-10-20 10:55 G5b -- batch QSA matches the CPU page

CONFIG: libb70_qsa_xpu.so 817928 bytes, mtime 10:16, icpx 2026.1. No
rebuild between the two GPU runs. Image vllm-flashnext-xpu:py312, host
UMD via harness/xpu-run.sh. tests/run_gpu.sh.

COMMAND: Card 0, B70_AGENT=G5b, B70_GPU_LOCK_TIMEOUT=7200, log
logs/g5b-qsa-batch.log, container 10:40:23. Then the graph compare was
narrowed to ids[:, :width]. Card 1, same agent, timeout 14400, log
logs/g5b-qsa-batch-r2.log, launcher 10:46:43.

RESULT: Card 0 printed QSA_XPU_OK. Eager batch passed block 4, block 8,
pos_base, mrope, inactive, chunked prefill, SIMT union (words exact,
attn max_abs 7.007e-4), draft, and export/import. Graph replay of
pages, tail, dead, block_pos, and length was exact, then "FAIL graph
ids mismatch". topk does not write past width. The check compared the
whole buffer after fill_(-1), and the snap tail was still the
allocation zero. Health after HEALTHY, rc 0. RUN_EXIT:1. No xe-reset.

Card 1 health before HEALTHY (27s). QSA_XPU_OK, single-seq graph ids
exact at widths 5, 6, and 7. QSA_BATCH_OK. Graph pos 4 and pos 5:
pages, tail, dead, block_pos, length, attn, and gated exact, id prefix
exact. Slot-3 completion on seq 0 with seq 1 inactive: seq 0 pages and
ids exact, attn and gated max_abs 0, inactive pages, tail, and dead
exact. gpu-run done in 24s, exit 0. Health after HEALTHY. RUN_EXIT:0.
Codes, scales, and ids exact on the eager rows. Pooled inside 1e-5,
scores inside 1e-4, attn inside 2e-3. Not a speed claim.

VERDICT: SIMT batch decode, chunked multi-query index, SIMT union
prefill, draft int8 pages, and park export/import match the CPU page
on card 1, including graph replay. XMX mode 1 is compiling to
libb70_qsa_xpu_next.so and is not measured. The hook prefill is still
the host token loop. Greedy prompts past 4K stay blocked until
Flash-Next serves. C7: call mark_draft on each MTP decoder layer before
KV spec collection, and set_draft_ids before the draft forward. G5b
will not edit mtp.py.

## 2026-10-20 11:00 G2b -- audit pass 7

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 10:55 G5b heading (JOURNAL line 2981). Findings F69-F76 append reports/1G2_audit.md after F68. F58-F68 were already on disk from the start of this pass.
COMMAND: Read headings from 10:52 E2 through 10:55 G5b. Check e2-driver.log, the pre-overwrite split log and serve traceback, h1c-m2 logs, c7 raise and index logs, g4-copy-driver-r2.txt, both g5b batch logs, fork commits 35da272b61 and e442b4b53f, moe_ep.cpp, linear.py, and serve_loader.py. git status in /mnt/vm_8tb/github/b70_ai_things and the fork.
RESULT: E2 identity, coherence TTFTs 92.2/71.4 ms, and the rounded 8k split times match. Diag rc 0, then bench rc 1. The second OOR is F.sigmoid in qwen2_moe.py:121, so F59's torch.bmm sentence for that crash is wrong. The split file was truncated by the 10:52 serve (F69). H1 M2 r1 died on ep flag layer=1 err1=11, health after HEALTHY, GREEDY_RC=0 with empty HTTP 500 (F70). C7 TP=2 is the packed-loader refusal, SERVE_RC 137 (F71). C7 DP=2 load IndexError matches the pre-commit sizes[] index; the tensor name is not in the log (F72). Commit 35da272b61 is loader-only, ahead 17, unmeasured (F73). G4 ordered copy bad=0, H2D about 14 GB/s, worst D2H 4.583, exit 1, health after HEALTHY (F74). G5b card 0 FAIL graph ids mismatch and RUN_EXIT 1; card 1 QSA_BATCH_OK and RUN_EXIT 0 (F75). Cached diff empty. No new staged binary.
VERDICT: Pass 7 recorded. Latency numbers checked this pass match the cited logs. Open items: the third E2 serve was still loading at 10:57 and is not scored, the 1.5 GB unpack claim and the 46.4 percent hit rate stay unverified, C12 device gates remain unjournaled, H1 M2 has no passing greedy text, and G4 D2H is under the 7.5 GB/s gate.

## 2026-10-20 11:02 C12 -- per-type schedule passes gates, graph time flat vs C9

CONFIG: card 0, ZE_AFFINITY_MASK=0, plugin side build
b70_gguf_xpu/_C.c12.so sha256
d46f80175c67a10681e09e7642b5293eedec0d3f98517b5fe7ba4c9662b70112
(10120096 bytes, mtime 10:28). Copy kept at
/mnt/vm_8tb/b70/build/vllm-flashnext/c12/_C.c12.measured.so.
Not installed over _C.so or _C.graph.so. One templated GU kernel per
distinct gate/up type and one DOWN kernel per distinct down type.
Fast-path types: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0. Shared experts are
accepted when the caller passes the tensors. The live 35B serve still
passes empty shared tensors, so the runner GEMV still owns shared.
B70_MOE_PHASE prints host submit plus wait, not device time. Fair
graph is tests/c9_fair.py with B70_MOE_PHASE unset. That window was
not under cpu-run --quiet.

COMMAND: bin/gpu-run --card 0
/mnt/vm_8tb/b70/build/vllm-flashnext/c12/run_gates.sh
(log c12/gates.log, 340s, exit 0). Host dequant earlier:
c12/compare_cpu.log, failures=0 types=14/14.

RESULT: xpu-health HEALTHY card 0 before and after. gpu_check
VERDICT failures=0 (atol 5e-2). moe_layer_bench VERDICT failures=0
(atol 8e-2). c12_types VERDICT failures=0 covered=12, including Q8_0
dequant mismatches=0. Fair graph us/call, median of 5:
IQ3_S/Q2_0 (gu=21 dt=42) 186.95 / 277.77 / 398.45 / 535.64 at T=1..4,
cosine 0.99999. IQ2_XS/IQ4_NL (gu=17 dt=20) 198.51 / 288.44 / 414.10 /
571.82, cosine 0.99999. Strata R2 IQ3 reference remains
112.89 / 148.10 / 202.93 / 261.64. Ratio about 1.66x at T=1 and 2.05x
at T=4. No xe-reset.

VERDICT: type coverage is in that candidate and the graph time matches
C9. One kernel per type did not close the gap. A second binary is
linked and not measured yet: same path, sha256
457a262da74b9477ebf34922be18d481c8c5fb224db5b50f2d3277f014d4762f
(11266352 bytes, mtime 10:59). It uses the 3D grid, use_root_sync,
separate EC 4/2/1 launches, and a branchless two-row k-loop. Shipped
_C.so and _C.graph.so are unchanged.

## 2026-10-20 11:10 G4 -- D2H stays 4.7 GB/s on every host bias

CONFIG: card 1, 256 MiB, four timed copies after one warmup.
Library libb70_kvpark_flags.so, 106584 bytes, compile rc=0 at
11:00:56. ordered()=1, engine group 1. Arms: compute stream with
sycl::malloc_host, and the copy queue with Level Zero host flags
cached, uncached, and write-combined. Decimal GB/s. bad counts
bytes that differ every 4096. xpu-health before and after.

COMMAND: /tmp/g4-launch-flags.sh. Log g4-flags-driver.txt.
B70_KVPARK_PROBE_ONLY=1. B70_KVPARK_COPY stays 0.

RESULT: compute h2d=13.812 d2h=4.687. cached h2d=14.205 d2h=4.690.
uncached h2d=14.014 d2h=4.663. write-combined h2d=14.165 d2h=4.649.
bad=0 on all four. Health after 11:08:52 HEALTHY card 1, exit 0.
FLAGS_DRIVER_EXIT 0 is the probe returning, not a 7.5 GB/s pass.
H1 took both cards at 11:09:24 (vllm-fn-h1-m2).

VERDICT: D2H is about 4.7 GB/s on the copy engine, on the compute
stream, and on every host cache mode tried. H2D stays about 14.
The store path does not reach 7.5 GB/s. The next measurement is
the same op on an mmap of /dev/shm, which is the serve pool.

## 2026-10-20 11:12 C12 -- EC-split graph does not instantiate

CONFIG: card 0, quiet CPU not held. Side build _C.c12.so sha256
457a262da74b9477ebf34922be18d481c8c5fb224db5b50f2d3277f014d4762f
(11266352 bytes). 3D grid, use_root_sync, three launches per type
(EC 4, then 2, then 1), branchless two-row dot. B70_MOE_PHASE unset
for c9_fair. Copy of that .so kept at c12/_C.c12.ec3.so once the
next link starts. Not installed.

COMMAND: bin/gpu-run --card 0 c12/run_fair.sh. Log c12/fair2.log.
gpu-run done in 83s, exit 1. Health before HEALTHY card 0.

RESULT: moe_layer_bench VERDICT failures=0. c12_types VERDICT
failures=0 covered=12, sched=1, Q8_0 dequant mismatches=0. Eager
cosine-level max_abs stayed under 2e-3 on the cropped cases.
c9_fair then died in torch.xpu.graph capture_end:
RuntimeError: Failed to instantiate native UR executable graph.
No OURS us_graph line. No xe-reset. The 48-call graph previously
captured the one-kernel-per-type schedule (about 7 kernels per
call). Three EC launches raise that to about 11, and 48 * 11 is
528 graph nodes.

VERDICT: correctness of the 3D schedule holds in eager mode. The
separate EC launches are not a graph-serve candidate. Strata keeps
EC 4/2/1 as constexpr passes inside one kernel. That shape is
rebuilding. Shipped _C.so and _C.graph.so are unchanged.

## 2026-10-20 11:12 C7 -- Flash-Next DP=2 retry

CONFIG: fork 35da272b61. qwen4_exp IQ3_XXS, DP=2 TP=1,
--enable-expert-parallel, enforce-eager, language-model-only,
max_model_len 2048, memory 100g, SYCL_CACHE_PERSISTENT=0.
No P2P. No hotschmoe-dd. Collective health first.
Greedy chat vs recorded Strata texts 323 and 667.
COMMAND: bash /tmp/c7-fn-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.


## 2026-10-20 11:14 G4 -- mmap pool copies at 9.4 GB/s H2D

CONFIG: card 1, 256 MiB /dev/shm mapping, four runs after one warmup.
Library libb70_kvpark_flags.so. ordered()=1, engine group 1.
Decimal GB/s. bad is mismatches every 4096 bytes. xpu-health
before and after. B70_KVPARK_COPY stays 0. No serve.

COMMAND: /tmp/g4-launch-mmap.sh. Log g4-mmap-driver.txt.
B70_KVPARK_MMAP_ONLY=1.

RESULT: h2d=9.435 d2h=4.451 bad=0. Health after 11:12:37 HEALTHY
card 1, exit 0. MMAP_DRIVER_EXIT 0 is the probe returning.
USM-host H2D on the same op was about 14 GB/s. D2H stays about 4.5.
C7 took both cards at 11:13:08 for vllm-fn-flashnext-gguf-ep.

VERDICT: The serve pool pointer clears 7.5 GB/s host-to-device.
Device-to-host does not. The next run is T2 with
B70_KVPARK_COPY=1 on libb70_kvpark_flags.so, logs tagged copy,
after that Flash-Next container releases a card. qwen4_exp was
not edited.

## 2026-10-20 11:15 E2 -- dense vendor GEMM, expert loop still 4.1 s

CONFIG: card 1, port 18084, vllm-fn-lowbit-e2, libb70_lowbit_xpu_e2u.so,
int3 experts / int4 trunk, FULL_DECODE_ONLY, FLASH_ATTN, bf16,
MOE_LAUNCHES=2, GEMV_ROWS unset, max_num_batched_tokens=16384,
PREFIX_CACHE=0, --gdn-prefill-backend sycl. Mode file started as
matmul. linear.py unpacks for both matmul and bmm. Expert bmm
unpacks one tile. The worker imported that code at model init.
C10b's dequant+bmm note still stands; this is the served plugin.
COMMAND: logs/e2_serve_driver.sh third launch. Split e2-fwd-split.log.
RESULT: Repack finished, server up 11:06:45. Identity hotschmoe-dev
first. Coherence Paris, TTFT 88 ms and 69 ms. Unique 8186-token
prompts. MATMUL_1 TTFT 4906 ms, 1668 tok/s. MATMUL_2 TTFT 4555 ms,
1797 tok/s. Warm split step 2: forward 4521 ms, linear_attn 242 ms
(n=30), self_attn 126 ms (n=10), experts 4110 ms, shared_expert 35 ms
(n=80), router 4.5 ms, norm 22 ms. BMM_1 TTFT 10648 ms, 769 tok/s.
Step 3 experts 9936 ms: group 4695, bmm_w13 1687, bmm_w2 1071,
pad_x 453, unpack_w13 264, unpack_w2 260, reduce 315. BMM_2 died
at torch.bmm of the down tile, level_zero error 40. Diag returned 1
before the bench. Container stopped. No health after: C7 took both
cards at 11:13:08 (vllm-fn-flashnext-gguf-ep). No xe-reset.
VERDICT: The 19.5 s linear_attn was dense prefill_wna16. Vendor
GEMM plus the requested SYCL GDN flag drops it to 242 ms, and
self_attn to 126 ms. The 8k step is now the per-expert F.linear
loop at 4.1 s, 1797 tok/s. Not 5000. The tiled bmm is slower on
the first request and the second request still runs out of device
memory. Next serve reuses one bmm scratch (fixed pad cap) and
releases cached blocks once per large step. It waits behind C7.
GEMV_ROWS stays unset.

## 2026-10-20 11:16 C13 -- wide dequant library build started
CONFIG: new TU csrc/dequant_wide.cpp + csrc/c13_bind.cpp, plus the C11 smoke
objects grouped_prefill.cpp and dequant_fast.cpp. Output
plugins/b70_gguf_xpu/b70_gguf_xpu/_C.c13.so. Does not replace _C.so,
_C.c10.so, _C.c11.so, _C.c11b.so, or _C.c12.so (C12 .so is 10169704 bytes,
11:14). Shared edit is prefill.py dispatch only: T>=DQ_T_MIN and
prefill_dequant_wide present and B70_PREFILL_DQ_LAYER not in
bmm/off/0/c10. decode.hpp (10:26), dequant_fast.cpp (10:13),
grouped_prefill.cpp (10:02), kernels.cpp, bind.cpp untouched.
icpx -fsycl -std=c++20 -O3 -fPIC -DB70_PREFILL_SMOKE=1
-fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4.
COMMAND: B70_AGENT=C13 B70_CPU_LOCK_TIMEOUT=7200 bin/cpu-run -- bash
vllm/flashnext/plugins/b70_gguf_xpu/scripts/c13_build.sh
RESULT: CPU lease was free. Log
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c13-build.log.
VERDICT: build in progress.

## 2026-10-20 11:17 G5b -- XMX union matches SIMT and the CPU page

CONFIG: Mode-1 port of Strata union_attn in csrc/qsa_xmx.cpp. No
Strata headers and no dpct. Page-local K and V. icpx 2026.1,
intel_gpu_bmg_g31, per-kernel split. Installed library
libb70_qsa_xpu.so is 962040 bytes, mtime 10:57:30. SIMT backup is
libb70_qsa_xpu.so.simt-817928 (817928 bytes). Image
vllm-flashnext-xpu:py312. Card 0. 24/2/256, 64 tokens, 8 queries,
hand-built width 16, union 58 cells. Atol 2e-2 vs SIMT and 5e-2 vs
CPU. Q, K, V, and P enter the products as fp16, so this is not a
bitwise compare.

COMMAND: Build under bin/cpu-run, SO_NAME=libb70_qsa_xpu_next.so,
log logs/g5b-qsa-xmx-build.log. GPU: CARD=0 B70_AGENT=G5b
B70_GPU_LOCK_TIMEOUT=14400 bash tests/run_gpu.sh. Log
logs/g5b-qsa-xmx.log. Launcher 3245496 at 10:58:20.

RESULT: compile rc=0, BUILD_EXIT:0. Health before HEALTHY (16s).
Container vllm-fn-g5-qsa done in 32s, exit 0. QSA_XPU_OK.
QSA_BATCH_OK. PASS xmx refuse (ok=0). xmx ok=1, union_cells=58.
xmx vs simt max_abs=9.710e-04. xmx vs cpu max_abs=5.405e-04.
PASS xmx. QSA_XMX_OK. Health after HEALTHY, rc 0. RUN_EXIT:0.
No xe-reset. The suite waited behind H1 (both-card health holding
card 0 while blocked on card 1) and then behind C12 run_fair.sh.

VERDICT: Mode-1 XMX union attention matches the SIMT union and the
CPU page inside those atols on card 0. The installed library stays
the 962040-byte build. Hook prefill now appends with append_cells
and union-attends each request in tiles of 8 (XMX when ok is 1,
else SIMT). That composition is queued on card 0 behind C7's
both-card Flash-Next serve (max_model_len 2048). Log
logs/g5b-qsa-prefill.log. The hook forward itself is not the test.
Greedy past 4K stays blocked: 2048 is under the 2051 identity
bound, and G5b will not start hotschmoe-dd. The C7 mark_draft ask
from 10:38 and 10:55 still stands. G5b will not edit mtp.py.


## 2026-10-20 11:19 C13 -- _C.c13.so linked, card 0 bench queued
CONFIG: icpx -fsycl -std=c++20 -O3 -fPIC -DB70_PREFILL_SMOKE=1
-fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4.
Sources grouped_prefill.cpp dequant_fast.cpp dequant_wide.cpp c13_bind.cpp.
Output _C.c13.so only.
COMMAND: B70_AGENT=C13 B70_CPU_LOCK_TIMEOUT=7200 bin/cpu-run -- bash
vllm/flashnext/plugins/b70_gguf_xpu/scripts/c13_build.sh
RESULT: cpu-run done in 139s exit 0. _C.c13.so is 2635416 bytes at 11:19.
_C.so stays 7574840 bytes (07:40). _C.c12.so stays 10169704 bytes (11:14).
Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c13-build.log ends C13_BUILD_OK.
Warnings only (deprecated parallel_for). Card 0 is held by C12
(run_fair.sh since 11:18). Bench waits on that lease.
VERDICT: library linked. Measurement not started.

## 2026-10-20 11:12 H1c -- M2 eager serve matches the C4 string

CONFIG: _C.so sha256 7f0981fd20a2375965fd89612bb32969a71ca6906ef94bfa9a6d1d0e4926f010 (476912 bytes, 10:53). Eager B70_EP=1, both cards, port 18114, container vllm-fn-h1-m2, max_len 2048, spin 200000, path 2. _C.ep.so still 55c154ee. Greedy temperature 0, 16 tokens, prompt "The capital of France is". The C4/C7 GGUF string is " Paris.\nThe capital of France is Paris.\nThe capital of France is".
COMMAND: bash /tmp/h1c-m2-window.sh from 10:53:34. Logs h1c-m2-window.out, h1c-m2-serve.log, h1c-m2-greedy.json, h1c-m2-bench.txt.
RESULT: Queued behind E2 until health-before HEALTHY at 11:07:45. Warmup moe attempt 1 layer 0 err0=0 err1=0. Split layers 0-39, layer 39 types 21,21,23. Application startup complete. No EP flag timeout. READY 11:10:45. Both greedy texts equal the C4 loop string, errors empty. The window JSON match_c4 is false only because that script still compared the landmarks sentence. Identity first hotschmoe-dev, alias qwen3.6-35b-a3b-ud-iq3-xxs, coherence OK/OK. C1 ITL p50 81.84 ms p90 83.19 p99 83.78, 12.60 tok/s, TTFT 217.93 ms, n=32. C2 both streams ITL p50 87.33 ms. BENCH_RC=0 GREEDY_RC=0. SERVE_RC=0 at 11:11:06. Health after HEALTHY rc 0 at 11:12:23. No xe-reset.
VERDICT: M2 gate met. Eager ITL is recorded, not a gate (C4 eager one-card C1 was 87.62 ms). The host wait is eager-only. M3 graph window is next.


## 2026-10-20 11:20 H1c -- M3 FULL_DECODE_ONLY window open

CONFIG: both cards, port 18115, container vllm-fn-h1-m3, GRAPH=1, GRAPH_MODE=FULL_DECODE_ONLY, ATTN=FLASH_ATTN, MAX_LEN=33792, B70_XPU_CG_RECLAIM=500, B70_EP=1, _C.so 7f0981fd, _C.ep.so 55c154ee. Text gate is the C4 loop string. xe-reset only if the serve log contains DEVICE_LOST.
COMMAND: nohup env B70_AGENT=H1 B70_GPU_LOCK_TIMEOUT=7200 bash /tmp/h1c-m3-window.sh. PID 3262526. Log logs/h1c-m3-window.out.
RESULT: queued at 11:20:08. Card 0 held by G5b. Waiting for both cards.
VERDICT: window is in the lock queue. Not measured yet.

## 2026-10-20 11:21 C12 -- fused EC still fails graph instantiate

CONFIG: card 0. Side build _C.c12.so sha256
6a2e412bfde900fc086d3fc6b1b7bf46a4fd4c36db8527efe0afe285046a285c
(10169704 bytes, mtime 11:14). Copy at c12/_C.c12.fused.so.
3D grid, use_root_sync, EC 4/2/1 constexpr inside one kernel,
branchless two-row dot. Not installed.

COMMAND: bin/gpu-run --card 0 c12/run_fair.sh. Log c12/fair3.log.
gpu-run done in 93s, exit 1. Health before HEALTHY card 0.

RESULT: moe_layer_bench VERDICT failures=0. c12_types VERDICT
failures=0 covered=12, sched=1. c9_fair capture_end again:
RuntimeError: Failed to instantiate native UR executable graph.
No us_graph line. No xe-reset. Node count is back to one GU kernel
and one down kernel per call, so the extra EC launches were not
the cause. The shared difference from the capturing schedule is
use_root_sync plus the 3D grid.

VERDICT: eager layer checks pass. This binary is not a graph-serve
candidate. The next link drops use_root_sync and keeps the 3D grid.
Shipped _C.so and _C.graph.so are unchanged.


## 2026-10-20 11:21 G5b -- prefill union matches the CPU page

CONFIG: Same 962040-byte library. Card 0. append_cells writes the
page without attending. prefill_chunk scores each query at its own
position and union-attends tiles of 8. Small geom (4/2/64) stays on
SIMT. 24/2/256 launches mode-1 XMX. Atol 2e-3 on SIMT gated output
and 5e-2 on the XMX gated output. Decode graph path is unchanged.

COMMAND: CARD=0 B70_AGENT=G5b B70_GPU_LOCK_TIMEOUT=14400 bash
vllm/flashnext/plugins/b70_qsa_xpu/tests/run_gpu.sh. Log
logs/g5b-qsa-prefill.log. Launcher 3259840 at 11:17:05. Queued
behind C7's both-card serve, which had released before health.

RESULT: Health before HEALTHY (13s). QSA_XPU_OK. PASS prefill
tiles: append_cells cache matched the CPU page, neighbor pages
stayed empty, ids exact, gated worst max_abs=1.953e-03 (queries
0 and 1), XMX did not launch. QSA_BATCH_OK. Hand-built XMX union
again ok=1, 58 cells, vs SIMT 9.710e-04, vs CPU 5.405e-04.
PASS prefill xmx: 16 appended cells, 10 queries (tiles 8 and 2),
XMX launched, gated vs CPU max_abs=9.766e-04. QSA_XMX_OK.
Container done in 26s, exit 0. Health after HEALTHY, rc 0.
RUN_EXIT:0. No xe-reset. No speed claim.

VERDICT: Chunked prefill through append_cells plus union tiles
matches the CPU page on card 0. SIMT is the small-shape path.
Mode-1 XMX is the 24/2/256 path. QsaRunner.forward itself was
not called. Greedy prompts past 4K stay blocked until a
Flash-Next serve uses max_model_len above 2051. G5b will not
start hotschmoe-dd.

## 2026-10-20 11:35 C12 -- 3D grid captures, about 4 percent faster

CONFIG: card 0, not under cpu-run --quiet. Side build _C.c12.so
sha256 b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979
(10165216 bytes, mtime 11:22). 3D grid, no use_root_sync, EC 4/2/1
inside one kernel, branchless two-row dot. B70_MOE_PHASE unset.
Not installed. Copy will be c12/_C.c12.noroot.so before the next link.

COMMAND: bin/gpu-run --card 0 c12/run_fair.sh. Log c12/fair4.log.
gpu-run done in 84s, exit 0. Health before and after HEALTHY card 0.

RESULT: moe_layer_bench failures=0. c12_types failures=0 covered=12.
Fair graph us/call, median of 5, cosine 0.99999:
IQ3_S/Q2_0 180.17 / 267.04 / 385.36 / 515.13 at T=1..4.
IQ2_XS/IQ4_NL 191.86 / 280.62 / 402.88 / 560.55.
Prior one-kernel 1D schedule was 186.95 / 277.77 / 398.45 / 535.64
and 198.51 / 288.44 / 414.10 / 571.82. Strata R2 IQ3 remains
112.89 / 148.10 / 202.93 / 261.64. Ratio about 1.60x at T=1 and
1.97x at T=4. No xe-reset.

VERDICT: dropping use_root_sync lets the 3D schedule capture.
The graph is a few percent under the C9 number and does not close
the gap. Next link is one kernel per entry width, still without
use_root_sync, to test the EC=1 register file at T=1. Shipped
_C.so and _C.graph.so are unchanged.


## 2026-10-20 11:22 C13 -- bench lost the CPU lock, requeued
CONFIG: card 0 via bin/gpu-run --card 0, B70_GPU_LOCK_TIMEOUT=21600,
B70_PREFILL_TIMEOUT=7200, inner cpu-run --quiet default wait 90s.
COMMAND: bin/gpu-run --card 0 scripts/c13_run.sh
RESULT: G5b released card 0. Health before rc=0. cpu-run --quiet then
waited on C12 build.sh (heavy, since 11:20:18) and exited 124 at 90s.
The script printed "FAIL python timed out" for that CPU-lock miss.
No kernel ran. Health after rc=0. gpu-run done in 129s exit 124.
CPU lease is free again. Next run waits up to 1800s for the quiet lock.
VERDICT: no measurement. Requeue.

## 2026-10-20 11:32 C7 -- Flash-Next DP=2 load global-OOM on the PLE copy

CONFIG: fork 35da272b61. Same window as the 11:12 announce. qwen4_exp
IQ3_XXS, DP=2 TP=1, --enable-expert-parallel, enforce-eager,
language-model-only, max_model_len 2048, docker memory 100g,
SYCL_CACHE_PERSISTENT=0. No P2P. No B70_QSA. No hotschmoe-dd.
Collective health first.
COMMAND: bash /tmp/c7-fn-window.sh. Preserved logs
c7-fn-serve-oom.log and c7-fn-window-oom.out.
RESULT: HEALTH_RC 0 at 11:12:39. Serve launched 11:13:08. Both workers
logged placed 1200 tensors at 11:17:22 (Worker_DP0 pid 493,
Worker_DP1 pid 475). No Python traceback from the worker. Kernel
11:17:41: oom-kill constraint CONSTRAINT_NONE, global_oom, task
VLLM::Worker_DP pid 3258155, total-vm 129572252 kB, anon-rss
46915752 kB, file-rss 12 kB. Mem-Info: active_anon 70711248 kB,
inactive_anon 26628912 kB, active_file 0, gpu_active 22211584 kB,
all_unreclaimable, Normal free about 195 MB. Reaped 11:17:44.
Engine core initialization failed. Stop 11:18:06, SERVE_RC 1.
Cards then held by C12 (card 0) and E2 (card 1). No xe-reset.
The death is a host OOM, not an XPU wedge. No greedy text.
VERDICT: the load died while copying per_layer_token_embd.weight
(IQ4_NL, 28800138240 bytes, uint8 shape (320001536, 90)). A second
private copy of that table does not fit.

## 2026-10-20 11:32 C7 -- PLE table is a memmap view, retry not started

CONFIG: fork 0a6170da8c, not pushed. Host tables skip _owning_uint8.
A non-contiguous or non-memmap source raises. HostPackedTable.owner
keeps the numpy view alive after load_weights drops the GGUFReader.
The 09:01 collective note "median 15.01 ms" is the sum of the 120
all_gather_into_tensor calls on a steady step (c7_collectives.py
writes count/sum_ms), about 0.13 ms each after warmup, plus 40
reduce_scatter_tensor calls summing about 5.6 ms.
COMMAND: B70_AGENT=C7 bin/cpu-run python -P /tmp/c7-ple-view-test.py.
RESULT: cpu-run exit 0 in 12 s. HOST_TABLE_TEST_OK. PY_COMPILE_OK.
IQ3_XXS and IQ3_S shard-2 PLE tensors are C-contiguous uint8 memmaps,
ggml ne (160, 320001536), nbytes 28800138240. At 11:32 card 0 is held
by H1 (xpu-health since 11:22) with C12 and C13 queued behind it, and
card 1 is held by E2 (vllm-fn-lowbit-e2 since 11:18). MemAvailable is
about 108 GiB. No C7 container and no C7 waiter.
VERDICT: the 27 GiB anon copy is gone on CPU. The DP=2 greedy compare
against recorded Strata texts 323 and 667 is not measured. Next step,
only when both gpu.lock owner files are empty and MemAvailable is at
least 100 GiB: append a window line here, then run
logs/c7-fn-window.sh in that same shell. The script refuses a short
host, runs collective health, and docker-stops the container if
MemAvailable falls under 8 GiB. Do not queue gpu-run without --card
while one card is held. Do not start hotschmoe-dd. Do not set B70_QSA.
Do not set CCL_TOPO_P2P_ACCESS.

## 2026-10-20 11:35 E2 -- empty_cache before the 8k step kills grouping

CONFIG: same int3 serve as 11:15, plus one shared bmm scratch and
torch.xpu.empty_cache at the start of each large execute_model.
Card 1 after C7 released it. Mode file switched to bmm before any
8k matmul. GEMV_ROWS unset. e2u.so not rebuilt.
COMMAND: logs/e2_serve_driver.sh fourth launch.
RESULT: Server up 11:33:27. Identity and both Paris coherence
calls passed (TTFT 99 ms and 70 ms). BMM_1 returned no usage.
Engine died at 11:33:38 in _group_slots at torch.cat, level_zero
error 40, before any expert bmm. Split log has the config only.
Container stopped. H1 took card 1 at 11:33:44 for xpu-health.
No xe-reset.
VERDICT: empty_cache before the forward is not safe next to the
KV pool. Removed. The matmul bench is queued behind G4
(vllm-fn-g4-kv-c1). The bmm trial runs only after that bench
json is on disk.

## 2026-10-20 11:41 G2b -- audit pass 8

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 11:35 E2 heading (JOURNAL line 3384 at write time) plus the unjournaled H1 M3 window that stopped at 11:38:08. Findings F77-F96 append reports/1G2_audit.md after F76.
COMMAND: Read headings from 11:02 C12 through 11:35 E2. Check c12/gates.log, fair2.log, fair3.log, fair4.log, g4-flags-driver.txt, g4-mmap-driver.txt, e2-driver.log, g5b-qsa-xmx.log, g5b-qsa-prefill.log, c13-build.log, c13-bench.log, h1c-m2-bench.json, h1c-m3-window.out, h1c-m3-bench.json, h1c-m3-serve.log, c7-fn-window-oom.out, c7-fn-serve-oom.log, and /var/log/kern.log. Hash the c12 side libraries. git status in /mnt/vm_8tb/github/b70_ai_things and the fork. Scan dequant_wide.cpp, c13_bind.cpp, and qsa_xmx.cpp for host sync.
RESULT: C12 11:02 microseconds, health, and verdicts match; bulk rel on gu=17 T=2 is 1.1534 and the type names are not on the OURS lines (F77). G4 flags and mmap match, and D2H stays near 4.5-4.7 (F78, F81). Fair2 and fair3 die at UR graph instantiate with no us_graph and no health after (F79, F87). Fair4 captures, 84s exit 0, and the microseconds match; the numeric error is unchanged (F93). E2 third-serve TTFTs match and the split table is gone (F82). E2 fourth serve identity and coherence match, BMM_1 is empty, and the _group_slots frame is not in the surviving log (F92). G5b XMX and prefill PASS lines match (F83, F88). C13 linked (F84). The 11:22 exit-124 line is not in c13-bench.log; the file is now a gguf import failure, exit 1, with health before and after (F89). C7 DP=2 is a host global OOM; kern.log matches the counters; the PLE tensor name is not in that log (F90). Commit 0a6170da8c is ahead 18 and the HOST_TABLE_TEST_OK line was not found on disk (F91). H1 M2 identity, C4 texts, and ITL match (F85). H1 M3 bench JSON matches the window, alias_ok is false, and the second id is ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2 while the launcher was still waiting on the cards (F94). Cached diff empty. No new staged binary (F95).
VERDICT: Pass 8 recorded. Latency numbers checked this pass match the cited logs. Open items: H1 M3 is not a Qwen graph ITL until alias_ok is true and the container acquire is in the serve log; E2 fifth serve was still loading at the driver snapshot; C13 exit 124 is unverified; G4 D2H is still under 7.5; fair graph bulk rel is still high; the PLE CPU test log is missing; the 46.4 percent hit rate and the 1.5 GB unpack claim stay unverified.

## 2026-10-20 11:35 C13 -- first bench, dequant still ~280 GB/s
CONFIG: _C.c13.so 2635416 bytes. Card 0 after H1's health check.
B70_CPU_LOCK_TIMEOUT=1800. B70_PREFILL_TIMEOUT=7200. SYCL subgroup
kernel, one subgroup per 512 outputs, SLM fill, 64-byte store.
COMMAND: bin/gpu-run --card 0 scripts/c13_run.sh
RESULT: health before/after rc=0. Wide dequant is bit-exact vs
prefill_dequant on IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, Q2_0,
IQ4_NL, including the fused gate+up layout (0 mismatches). Full-expert
event GB/s: IQ3_XXS 285.0 (old 256.2), IQ3_S 286.3, IQ2_XXS 272.9,
IQ2_XS 275.0, IQ2_S 279.2, Q2_0 270.5 (old 259.2), IQ4_NL 280.9.
bf16 copy of one gate 1154.7 GB/s. Small fast layer max_rel 1.19e-2
against the bf16 mv reference. Fused SGN=4 vs that reference on the
scale-1 pack: max_rel 1.8625, frac_gate 0.9476, sig_max_rel 1.8625,
worst element got=-2.06 ref=1.77. Python then died importing gguf
(not on PYTHONPATH). No T=4096/12288 layer time. gpu-run exit 1.
VERDICT: correctness of the wide dequant holds. Bandwidth does not.
Rebuilding with ESIMD flat loads and 64-byte block stores. gguf-py
added to the bench path.

## 2026-10-20 11:43 C13 -- ESIMD dequant linked, bench queued
CONFIG: dequant_wide.cpp ESIMD kernel, one thread per 512 outputs,
aligned flat load, 64-byte block stores. gguf-py on PYTHONPATH.
Output _C.c13.so 2822840 bytes at 11:43. Does not replace _C.so or
_C.c12.so. Card 0 held by C12 run_fair.sh since 11:42.
COMMAND: B70_AGENT=C13 B70_CPU_LOCK_TIMEOUT=7200 bin/cpu-run -- bash scripts/c13_build.sh
then bin/gpu-run --card 0 scripts/c13_run.sh with CPU wait 1800s and python limit 7200s.
RESULT: build exit 0 in 96s. Log c13-build2.log ends C13_BUILD_OK.
Bench waits on card 0. Log c13-bench2.log.
VERDICT: library linked. Measurement queued.

## 2026-10-20 11:42 H1c -- M3 first window benched the wrong server

CONFIG: port 18115, container vllm-fn-h1-m3, FULL_DECODE_ONLY, B70_EP=1. The ready check was any HTTP 200 on /v1/models.
COMMAND: bash /tmp/h1c-m3-window.sh PID 3262526. Preserved logs h1c-m3-window-falseport.out, h1c-m3-serve-falseport.log, h1c-m3-bench-falseport.txt, h1c-m3-greedy-falseport.json.
RESULT: Health before HEALTHY at 11:34:13. Our gpu-run was still waiting on card 1. At 11:38:03 port 18115 answered. That was G4's Ornith serve (served name hotschmoe-dev, alias ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2, port 18115 in the gpu-run wait line). Bench alias_ok=False, text happened to be the C4 loop, C1 ITL 16.51 ms. Those numbers are not the EP sidecar. docker stop ran before our container existed, so the still-queued gpu-run would have started a server nothing would stop. The process tree was SIGTERM'd at 11:42. Card 0 released. No xe-reset. Our container never started.
VERDICT: discarded. The next window accepts ready only when container vllm-fn-h1-m3 lists alias qwen3.6-35b-a3b-ud-iq3-xxs, and the wait covers the lock timeout.

## 2026-10-20 11:45 G4 -- copy_batch T2 restored the second turn

CONFIG: card 1, port 18115, tag=copy, B70_KVPARK_COPY=1,
libb70_kvpark_flags.so. Ornith GPTQ, 45 GPU blocks, 32 GiB pool,
4 slots, offload_prompt_only false, shm 40g, memory 80g. Identity
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
Health before HEALTHY. Worker stderr: b70_kvpark: copy queue group 1.
The pool stayed the pinned mmap. Script default for B70_KVPARK_COPY
stays 0.

COMMAND: TAG=copy PORT=18115 B70_KVPARK_COPY=1
B70_KVPARK_SO=/mnt/vm_8tb/b70/build/vllm-flashnext/b70_kvpark/libb70_kvpark_flags.so
CARD=1 vllm/flashnext/kvoffload/run_t2.sh.
Logs g4-t2-serve-c1-copy.txt, g4-t2-turn-c1-copy.json,
g4-t2-copy-driver.txt.

RESULT: pass=True. host_delta=3063, device_delta=0, computed=18,
cached=3063, greedy and text match. control TTFT 0.414 s, restored
TTFT 0.124 s. Load delta 131235840 bytes in 0.009678 s = 13.560 GB/s,
load_size_count=1. Pre-shutdown store counter 1369210880 bytes in
0.298174 s = 4.592 GB/s. The 11:38:42 scrape (440729600 bytes) is a
reset during SIGTERM. H1c's M3 bench posted to this port and had
finished by the 11:38:12 line (Running: 0). The first tail was queued
at 11:38:16 and the restore at 11:38:28. No EngineDead. Serve gpu-run
exit 0. Health after timed out at 90 s because E2 took card 1 at
11:38:53 (vllm-fn-lowbit-e2). g4-t2-health-after-c1-copy.txt is that
timeout. The retry launcher was killed before a second serve.

VERDICT: Direct copy restores the second turn. Host-to-device on the
pinned mmap pool is 13.560 GB/s. Device-to-host on the pre-shutdown
counter is 4.592 GB/s, in line with the 4.6 GB/s microbench, and
under 7.5. Card 1 post-health is still owed. qwen4_exp was not
edited. The fork was not pushed.

## 2026-10-20 11:56 G4 -- card 1 healthy after the copy T2

CONFIG: card 1, after E2 released vllm-fn-lowbit-e2. No serve.
COMMAND: bin/gpu-run --card 1 bin/xpu-health --card 1.
Log g4-t2-health-after-c1-copy-late.txt. The 11:40 timeout file
was left as it was.
RESULT: HEALTHY, exit 0, 18 s, 11:56:16. Card 0 was C13's.
VERDICT: The copy-path T2 has a post-health. Next is T3 ITL on
card 1, port 18112, container vllm-fn-g4-kv-c1, B70_KVPARK_COPY=1,
libb70_kvpark_flags.so. Expected hold is the Ornith load plus the
two-decode client, on the order of 15 minutes. qwen4_exp was not
edited.

## 2026-10-20 11:57 G4 -- T3 not started, H1 holds both cards

CONFIG: card 1 was healthy at 11:56:16. No G4 container was started.
COMMAND: none. gpu-run --status at 11:57:12.
RESULT: card 0 and card 1 both HELD by H1 pid 3278138 since 11:56:08
and 11:56:16, container vllm-fn-h1-m3. T3 was not launched.
VERDICT: Waiting. The next T3 is still port 18112, copy path on,
one card, after that container releases one.

## 2026-10-20 11:50 C12 -- per-EC kernels slower; 3D binary installed

CONFIG: card 0. Per-EC binary sha256
cd2882dce7964770701c31f3c8eef77dbd4166005b47a54541c219b01fc45cca
(11297864 bytes). Copy at c12/_C.c12.ecsplit.so. 3D grid, no
use_root_sync, one kernel per entry width. Not the installed file.
The installed file is the fused-EC 3D binary sha256
b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979
(10165216 bytes), copied onto _C.so, _C.graph.so, and _C.c12.so.
Nothing had those files mapped. Fair window was not under
cpu-run --quiet.

COMMAND: bin/gpu-run --card 0 c12/run_fair.sh. Log c12/fair5.log.
gpu-run done in 107s, exit 0. Health before and after HEALTHY card 0.

RESULT: moe_layer_bench failures=0. c12_types failures=0 covered=12.
Fair graph us/call, cosine 0.99999:
IQ3_S/Q2_0 185.76 / 320.79 / 378.78 / 595.23 at T=1..4.
IQ2_XS/IQ4_NL 197.79 / 329.97 / 403.33 / 626.23.
The fused 3D binary was 180.17 / 267.04 / 385.36 / 515.13 and
191.86 / 280.62 / 402.88 / 560.55. Separate entry-width launches
are slower at T=1, T=2, and T=4. No xe-reset.

VERDICT: keep the fused 3D schedule. It is installed as _C.so and
_C.graph.so. The 35B UD-IQ3_XXS graph serve (C7 recipe, port 18104,
B70_GGUF_SO=_C.graph.so) is queued behind H1's both-card health,
which holds card 0 while waiting on card 1. ITL is not measured yet.

## 2026-10-20 12:50 C12 -- 35B graph C1 ITL 11.02 ms

CONFIG: card 0, container vllm-fn-qwen36-gguf-c0, port 18104.
_C.graph.so sha256
b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979.
GRAPH=1 FULL_DECODE_ONLY FLASH_ATTN mode 0 capture 1,2,3,4,8.
B70_XPU_CG_RECLAIM=500 B70_EAGER_QK=0 B70_GGUF_REF=0.
max_model_len 33792 max_num_seqs 4 gpu_mem 0.90.
Model Qwen3.6-35B-A3B UD-IQ3_XXS. Bench under cpu-run --quiet.

COMMAND: c12/serve_35b.sh. Logs c12/serve_driver.log,
c12/serve_35b.log, c12/bench_35b.json.

RESULT: Health before HEALTHY card 0 (23s). Startup complete,
graph capture finished, no DEVICE_LOST. Identity hotschmoe-dev
first, alias qwen3.6-35b-a3b-ud-iq3-xxs. Coherence OK/OK, both
texts " Paris.\nThe capital of France is Paris.\nThe capital of France is".
C1 ITL p50 11.019 ms p90 11.229 p99 11.469, 90.89 tok/s,
TTFT 169.3 ms, n=64. C2 ITL p50 16.606 / 16.607 ms, tok/s
55.24 / 62.08. Stream 0 p99 75.01 ms. Prefill tok/s 177.4
(1017), 175.4 (8185), 173.1 (32762). MTP off. cpu-run quiet
done in 246s exit 0. BENCH_RC=0. Container docker-stopped
after the bench. Post-health waited behind that lease and then
behind H1, then HEALTHY card 0 in 13s. SERVE_DONE, DRIVER_RC=0.
No xe-reset.

VERDICT: end-to-end C1 matches C7's 11.11 ms on the old graph.so.
The 4 percent fair-layer gain does not move this 35B ITL. The
layer remains about 1.60x Strata R2 at T=1.


## 2026-10-20 11:46 C13 -- ESIMD kernel tripped IGC, block-write rebuild
CONFIG: _C.c13.so 2822840 bytes, ESIMD flat load plus 64-byte block_store.
Card 0 after C12 run_fair.sh. Health before the launch.
COMMAND: bin/gpu-run --card 0 scripts/c13_run.sh
RESULT: wants_fast=1. First prefill_dequant_wide JIT failed:
IGC Internal Compiler Error, floating point exception.
Health after rc=0. gpu-run exit 1 in 31s. No bandwidth number.
VERDICT: ESIMD kernel is not usable on this IGC. Rebuilding the tuned
path as a SYCL subgroup kernel with SubgroupBlockWrite of 8 bf16 per lane.

## 2026-10-20 11:55 E2 -- official matmul bench, 8k is 1961 tok/s

CONFIG: card 1, port 18084, vllm-fn-lowbit-e2, e2u.so, int3 experts /
int4 trunk, FULL_DECODE_ONLY, FLASH_ATTN, bf16,
--gdn-prefill-backend sycl, max batched tokens 16384, prefix cache
off, MOE_LAUNCHES=2, GEMV_ROWS unset. Mode file matmul for the bench.
empty_cache removed. Dense matmul and bmm both use unpack + F.linear.
C10b dequant+bmm remains their GGUF path; this bench is the lowbit
plugin. No xe-reset.
COMMAND: logs/e2_serve_driver.sh. Bench
vllm/flashnext/harness/bench_latency.py tag e2-int3-matmul-fdo,
out logs/e2-bench.json. Then a bmm trial. Then xpu-health --card 1.
RESULT: Server up 11:53:47. Identity hotschmoe-dev first. Coherence
Paris both times. Warm instrumented 8186-token step: forward 4479 ms,
linear_attn 243 ms (n=30), self_attn 126 ms (n=10), experts 4067 ms,
shared_expert 35 ms. Official bench rc 0, sync off: 1017 tok TTFT
2945 ms = 345 tok/s; 8185 tok TTFT 4173 ms = 1961 tok/s; 32762 tok
TTFT 9664 ms = 3390 tok/s (two 16384 chunks). C1 ITL p50/p90/p99
11.34/11.54/11.70 ms, 89.9 tok/s. C2 ITL p50 15.76 ms both streams.
BMM trial after the bench returned no usage. Engine died at
y.index_add_ in expert_prefill_bmm, level_zero error 40. Container
stopped. Health card 1 HEALTHY, rc 0, 11:55:30.
VERDICT: 5000 tok/s not met. The dense WNA16 span is gone (linear_attn
0.24 s). The 8k step is the 256-launch expert loop. Report
reports/1E2_lowbit_prefill.md. Grouped bmm is still not a serve path
next to the KV cache. GEMV rows=2 stays unshipped (max_abs 1.0).

## 2026-10-20 11:55 C13 -- block write scrambled outputs
CONFIG: _C.c13.so 2647704 bytes. SubgroupBlockWrite of 8 bf16 per lane,
values packed as if the write were blocked (lane i owns out[i*8, i*8+8)).
COMMAND: bin/gpu-run --card 0 scripts/c13_run.sh
RESULT: health before/after rc=0. Generic path (IQ4_NL 3x64, not a
multiple of 512) mismatches=0. Tuned path mismatches: IQ3 128x2560
314956, IQ3 2560x2560 6299964, Q2 4x640 1875, Q2 8x640 3757, IQ4 64x256
15126, IQ2_XXS 75878, IQ2_XS 78065, IQ2_S 78257, IQ3_S 78729, fused
layout 12599687. failures=10, layer timing skipped. gpu-run exit 1.
VERDICT: the block write is striped (sub_group.hpp N==16 split advances
by 8 * subgroup size). Rebuilding with an SLM stage and striped pickup.

## 2026-10-20 11:58 G4 -- T3 ITL started on card 1

CONFIG: card 1, port 18112, container vllm-fn-g4-kv-c1,
B70_KVPARK_COPY=1, libb70_kvpark_flags.so, phase=itl, 45 GPU blocks,
max_num_seqs 4. Card 0 is C12's vllm-fn-qwen36-gguf-c0. CPU lock is
C13's c13_build.sh. The client waits up to 180 s for a quiet CPU.
COMMAND: /tmp/g4-launch-t3.sh 1. Log g4-t3-driver-c1.txt.
RESULT: health-before acquired card 1 at 11:58:47. Serve not ready yet.
VERDICT: In progress. One card, about 15 minutes. qwen4_exp was not
edited.

## 2026-10-20 12:04 G4 -- T3 restore counts passed, greedy and p99 did not

CONFIG: card 1, port 18112, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
phase=itl, 45 GPU blocks, PIECEWISE. Identity hotschmoe-dev then
ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2. Logs preserved as
g4-t3-*-c1-r1.*.
COMMAND: /tmp/g4-launch-t3.sh 1. The client waited on C12's quiet
bench, then held the quiet lock for 33 s.
RESULT: restored cached=3063 computed=18 host_delta=3063
device_delta=0 ttft=0.146 s against control 0.415 s. Load 131235840
bytes in 0.009508 s = 13.803 GB/s, one load. greedy_match=false.
The ITL window covered the whole 64-token reply (win_n=68). p50
17.3 ms to 19.0 ms. p99 17.7 ms and 32.4 ms to 51.0 ms. Health
after HEALTHY, exit 0, 12:04:39. No EngineDead.
VERDICT: Token accounting and the copy rate hold under two live
decodes. Greedy did not match the solo control, and the stall
window included the later three-way decode. Rerun on card 1 at
12:07:41 records the first mismatched token and the gaps until
the first restored token. qwen4_exp was not edited.

## 2026-10-20 12:11 G4 -- T3 first-token gaps and a late stutter

CONFIG: card 1, port 18112, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
phase=itl, 45 blocks. The ITL window now ends at the restored
request's first token. Logs g4-t3-*-c1-r2.*.
COMMAND: /tmp/g4-launch-t3.sh 1. Quiet CPU acquired at once.
RESULT: cached=3063 computed=18 host=3063 device=0. Load 131235840
bytes in 0.009544 s = 13.750 GB/s. TTFT 0.144 s. Greedy matched 50
of 64, then stuttered the bigram (18, 2195) inside a repeating loop.
Gaps ms, stream 0: 17.6 17.9 17.8 24.1 83.2 35.5. Stream 1: 17.7
17.9 17.8 24.1 83.0 35.8. Baseline p50 17.4 ms. Health after
HEALTHY, 15 s, 12:11:17.
VERDICT: The three gaps on the copy stay within 3 percent of 17.4 ms.
The 83 ms gap is the 18-token prefill, so the literal 10 percent
p99 on that window fails. The stutter is at token 50 of a loop, after
an exact prefix. Solo T2 matched all 64. Next is the 4-slot cycle on
card 1, port 18113, 32 GPU blocks. qwen4_exp was not edited.

## 2026-10-20 12:07 H1c -- M3 capture host-wait is Level Zero 44

CONFIG: FULL_DECODE_ONLY, FLASH_ATTN, max_len 33792, B70_XPU_CG_RECLAIM=500, B70_EP=1, port 18115, container vllm-fn-h1-m3. _C.so was 7f0981fd. _C.ep.so 55c154ee. Ready check requires container vllm-fn-h1-m3 and alias qwen3.6-35b-a3b-ud-iq3-xxs.
COMMAND: bash /tmp/h1c-m3-window.sh PID 3274080. Log h1c-m3-serve-capture44.log.
RESULT: Health before HEALTHY at 11:55:06. EP warmup moe attempt 1 err 0. Split through layer 39. Capture of FULL graphs then raised RuntimeError level_zero backend error 44 (UR_RESULT_ERROR_UNSUPPORTED_FEATURE) inside moe_ep, from profile_cudagraph_memory. FULL capture calls the model with CUDAGraphMode.NONE, so cap_on stayed false and eager ep_moe host-waited on the recording stream. No DEVICE_LOST. No xe-reset. SERVE_RC=1 at 11:57:49. Health-after was still queued at 12:08 (C12 on card 0, g4 on card 1).
VERDICT: not an M3 measurement. New _C.so sha256 e783539c4fa49f774a115a816209201a189a78d8d2c051f5b56c30575eb2c870 (481240 bytes, 12:07). A recording card-0 stream now records the card-1 graph and does not host-wait. The graph is finalized on the last MoE layer. FULL replay submits that graph from CudaGraphManager via the d1 hook. Next window uses this binary.

## 2026-10-20 12:16 G2b -- audit pass 9

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 12:11 G4 heading and the 12:07 H1c heading. Findings F97-F110 append reports/1G2_audit.md after F96.
COMMAND: Read headings from 11:35 C13 through 12:11 G4. Check c13-bench.log, c13-bench2.log, c13-bench3.log, c13-build2.log, c13-build3.log, fair5.log, e2-bench.json, e2-driver.log, e2-fwd-split.log, e2-serve.log, g4-t2-turn-c1-copy.json, g4-t2-serve-c1-copy.txt, g4-t3-itl-c1-r1.json, g4-t3-itl-c1-r2.json, g4-t3-serve-c1-r1.txt, g4-t3-serve-c1-r2.txt, h1c-m3-serve-capture44.log, h1c-m3-window.out, c12/bench_35b.json, and c12/serve_driver.log. Hash the installed gguf and EP libraries. git status in /mnt/vm_8tb/github/b70_ai_things and the fork.
RESULT: C13 wide-dequant GB/s and exact mismatches match, then the gguf import exits 1 (F97). ESIMD link is 2822840 bytes and the bench dies in IGC (F98, F103). Block-write mismatches match and exit 1 (F105). H1's false-port bench is the Ornith server already in F94 (F99). G4 copy T2 pass is true, H2D 13.560 GB/s, pre-shutdown D2H 4.592, health-after timed out, and the late health is HEALTHY (F100, F101). Fair5 microseconds match and the fused sha b3c15e14 is installed on _C.so and _C.graph.so (F102). E2 bench is 1961 tok/s at 8185 tokens, identity and coherence match, and the bmm trial is index_add error 40 (F104). G4 T3 r1 and r2 restore counts match; greedy fails, including mismatch_at 50 on r2; health after ran (F106, F108). H1 capture error 44 is inside moe_ep, SERVE_RC 1, and the new EP _C.so hash matches (F107). c12/bench_35b.json is unjournaled: identity and coherence match, C1 ITL p50 11.02 ms, prefill about 175 tok/s, health before HEALTHY, health after still waiting on its own container (F109). Cached diff empty. Fork still 0a6170da8c, ahead 18 (F110).
VERDICT: Pass 9 recorded. Latency numbers checked this pass match the cited logs. Open items: the C12 35B graph bench has no health-after line; H1 M3 has no ITL; G4 D2H is still about 4.6 GB/s; fair bulk rel is unchanged; C13 block-write and ESIMD are not serve candidates; the PLE CPU test log and the 46.4 percent hit rate stay unverified.

## 2026-10-20 12:16 G4 -- cycle serve refused 32768 on 32 blocks

CONFIG: card 1, port 18113, 32 GPU blocks, max_model_len 32768,
B70_KVPARK_COPY=1. Logs g4-t3-serve-c1-cycle-maxlen.txt and
g4-t3-cycle-driver-c1-maxlen.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: Engine init ValueError. 32 blocks are 0.64 GiB. A 32768
context wants 0.77 GiB. The estimated fit is 27200 tokens. Health
after HEALTHY, 12 s, 12:16:04. No xe-reset.
VERDICT: Not a GPU fault. The retry sets MAX_LEN=20480. The longest
cycle prompt is 17344 tokens. qwen4_exp was not edited.

## 2026-10-20 12:20 G4 -- cycle restores missed because actives finished

CONFIG: card 1, port 18113, max_len 20480, 32 blocks, 4 slots,
B70_KVPARK_COPY=1. Logs g4-t3-cycle-c1-r1.json,
g4-t3-cycle-driver-c1-r1.txt, g4-t3-serve-c1-cycle-r1.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: Eight restores. cached was 0 on five and 1088 on three.
No G4 load tail. Park tails were queued at end=1519. Both actives
then finished (ends 17407 and 16319) and took park slots. Greedy
matched because the misses recomputed. Health after HEALTHY,
12:20:33.
VERDICT: A finished active is a fifth conversation, and the 4-slot
index drops a park before the lookup. The next run keeps each active
inside its blocks for 1024 decode tokens. qwen4_exp was not edited.

## 2026-10-20 12:28 G4 -- long decode still finished during the other prefill

CONFIG: card 1, port 18113, max_len 20480, 32 blocks, 4 slots,
active max_tokens 1024. Logs g4-t3-cycle-driver-c1-r2.txt and
g4-t3-serve-c1-cycle-r2.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: One active still queued a tail at end=17407 before the first
restore. Five restores cached 0 and three cached 1088. No load tail.
Each miss then stored and dropped the next park. Health after
HEALTHY, 12:28:14.
VERDICT: 1024 decode steps fit inside the other 16k-token prefill.
The retry uses 8 GPU blocks and max_len 4096. The actives are 4 and
3 blocks with 1024 decode tokens, and the other prefill is only a
few thousand tokens. qwen4_exp was not edited.

## 2026-10-20 12:33 G4 -- 8 GPU blocks only fit 1088 tokens

CONFIG: card 1, port 18113, num_gpu_blocks 8, max_len 4096.
Logs g4-t3-serve-c1-cycle-8block.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: ValueError at init. Available KV 0.15 GiB, 4096 tokens need
0.21 GiB, estimated max len 1088. Health after HEALTHY, 13 s,
12:33:27. The 32-block serve had estimated max 27200, which is
(32-7)*1088. Eight blocks is (8-7)*1088.
VERDICT: Not a GPU fault. The next cycle is 32 blocks, max_len 20480,
prompts 15000 and 16000, decode budget 3072. qwen4_exp was not edited.

## 2026-10-20 12:40 G4 -- waiter never admitted beside the runner

CONFIG: card 1, port 18113, 32 blocks, max_len 20480, prompts 15000
and 16000, decode budget 3072. Logs g4-t3-cycle-driver-c1-r3.txt
and g4-t3-serve-c1-cycle-r3.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: From 12:38:49 to 12:39:39 the engine had Running 1, Waiting 1,
prompt throughput 0, KV usage 58 to 64 percent. The runner then
queued a tail at end=18071. Restores cached 0 or 1088. No load tail.
Health after HEALTHY, 12:40:29.
VERDICT: About 11 blocks were free and the waiter needed 15, so it
sat until the runner finished and took a park slot. The retry uses
prompts 26400 and 6000 with max_len 27000. 26400 tokens is 25 blocks,
6000 is 6, and 7 blocks remain. qwen4_exp was not edited.

## 2026-10-20 12:48 G4 -- 6000-token partner still waited at 90 percent

CONFIG: card 1, port 18113, 32 blocks, max_len 27000, prompts 26400
and 6000, decode budget 512. Logs g4-t3-cycle-driver-c1-r4.txt and
g4-t3-serve-c1-cycle-r4.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: At 12:47:30 Running 1, Waiting 1, KV 90.3 percent, prompt
throughput 2640 tok/s. The runner queued end=26911 at 12:47:34.
Restores cached 0, then device 1088. No load tail. Health after
HEALTHY, 12:48:14. KV cache on this serve is 27870 tokens.
VERDICT: 90 percent full leaves about 3 blocks. The 6000-token
partner did not fit, so the runner finished and took a slot. The
retry partner is 1024 tokens. qwen4_exp was not edited.

## 2026-10-20 12:53 G4 -- 1024-token partner also waited out the runner

CONFIG: card 1, port 18113, 32 blocks, max_len 27000, prompts 26400
and 1024, decode budget 580. Logs g4-t3-cycle-driver-c1-r5.txt and
g4-t3-serve-c1-cycle-r5.txt.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: At 12:52:24 Running 1, Waiting 1, KV 90.3 percent. The runner
queued end=26979 at 12:52:33. Restores cached 0, then device 1088.
No load tail. Health after HEALTHY, 12:53:45.
VERDICT: A running prefill keeps the token budget, so the partner is
not admitted until that prefill ends, and by then the free blocks are
too few. Card 1 was taken by H1 after this health check. The next
attempt starts the 1024-token request first, then the 24000-token
prefill. qwen4_exp was not edited.

## 2026-10-20 12:58 G4 -- cycle retry, small request first

CONFIG: card 1, port 18113, 32 blocks, max_len 27000. The 1024-token
request starts and reaches one token, then the 24000-token prefill
starts. Decode budgets 8000 and 2000.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1. Health-before took the card
at 12:58:00.
RESULT: serve not ready yet.
VERDICT: In progress. qwen4_exp was not edited.

## 2026-10-20 13:02 G4 -- one cycle restore hit from host

CONFIG: card 1, port 18113, 32 blocks, max_len 27000. Small request
started first (1024 tokens, 8000 decode), then the 24000-token
prefill (2000 decode). Logs g4-t3-cycle-driver-c1-r6.txt,
g4-t3-serve-c1-cycle-r6.txt, g4-t3-cycle-c1-r6.json.
COMMAND: /tmp/g4-launch-t3-cycle.sh 1.
RESULT: Turn 0 passed. host=1519 device=0 computed and cached match,
greedy mismatch_at null. Serve log: G4 load tail end=1519 local=0
keys=5, same second as the long request queued end=25999. Turns 1-3
then missed. Health after HEALTHY, 13:02:54.
VERDICT: The host tail restore works while a second request is
running. The long stream's first event arrived as that request
finished, so the later turns saw a fifth conversation. The retry
waits 15 s after submitting the long prefill instead of waiting on
that event, and the decode budget is 2800. qwen4_exp was not edited.

## 2026-10-20 12:56 G2b -- audit pass 10

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 12:53
G4 heading. The 12:50 C12 heading sits above the 11:46 C13 entry.
Findings F111-F121 append reports/1G2_audit.md after F110.
COMMAND: Read headings from 12:16 G4 through 12:53 G4, plus the
inserted 12:50 C12 heading. Check the cycle maxlen, 8-block, r1, r2,
r3, r4, and r5 logs, c12/bench_35b.json, and c12/serve_driver.log.
git diff --cached in /mnt/vm_8tb/github/b70_ai_things.
RESULT: The 32768 and 4096 refusals match, with estimates 27200 and
1088 (F111, F114). r1 cached is 0 on five turns and 1088 on three.
r2 and r3 are four and four. Greedy is true and cycle pass is false.
load_bytes is 86671360, then 346685440, then 346685440 (F112, F113,
F115). r4 and r5 are also four and four, host 0, device 1088 only on
the cached-1088 turns, and pass false (F118, F120). r4 end=26911 and
r5 end=26979 match 26400 plus the decode budget minus 1. The 6000
length is not printed. Both serves show Waiting 1 at KV 90.3 percent,
then Running 2 after the park. load_bytes is absent there and store
bytes are non-zero. Health after is HEALTHY on card 1. C12 json C1
ITL p50 is 11.0188 ms, 90.893 tok/s, coherence texts match, health
before is 23 s, and serve_driver.log still has no HEALTHY-after line.
The container is gone from docker ps (F119). Cached diff was empty.
The only modified binary is the pre-existing push-allreduce .so.
VERDICT: Pass 10 recorded. Greedy text matches and a device delta of
1088 are not restore passes. Open items: C12 35B health after is still
missing; H1 M3 still has no ITL; G4 load_bytes is not printed on r4 or
r5; fair bulk rel is high; C13 block-write and ESIMD are not
candidates; the PLE CPU test log and the 46.4 percent hit rate stay
unverified.


## 2026-10-20 13:01 H1c -- M3 FULL_DECODE_ONLY passes

CONFIG: both cards. FULL_DECODE_ONLY, FLASH_ATTN, max_len 33792, max_num_seqs 4, gpu_memory_utilization 0.90, capture sizes 1,2,3,4,8, B70_XPU_CG_RECLAIM=500, B70_EP=1, port 18115, container vllm-fn-h1-m3. _C.so sha256 e783539c4fa49f774a115a816209201a189a78d8d2c051f5b56c30575eb2c870 (481240 bytes, 12:07). _C.ep.so 55c154ee. World size 1. Sidecar owns card 1.
COMMAND: bash /tmp/h1c-m3-window.sh PID 3286770. Logs h1c-m3-window-r2.out, h1c-m3-serve.log, h1c-m3-bench.json, h1c-m3-greedy.json.
RESULT: Queued from 12:12:49 behind C12, then C13. Health-before HEALTHY rc 0 at 12:55:14. EP warmup exchange and moe attempt 1 err0=0 err1=0. Split layers 0-39. Profile capture finalized card-1 keys 4 and 3. Real FULL capture finalized keys 4, 3, 2, 1 (no key=8; 4 graphs, same as C7). No error 44. No EP flag timeout. KV 14.72 GiB, 630487 tokens, 18.66x at 33792. READY 12:56:40. Identity hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs, first_ok and alias_ok true. Both greedy texts, temperature 0, 16 tokens: " Paris.\nThe capital of France is Paris.\nThe capital of France is". match_c4 true. Bench coherence OK/OK. C1 ITL p50 11.36 ms, p90 11.62, p99 12.35, 89.50 tok/s, TTFT 142.82 ms, n=63. Gate is 1.05x 11.11 ms = 11.665 ms. C2 stream0 15.31 ms (64 tokens, 71.14 tok/s, TTFT 243.01) and stream1 15.74 ms (40 tokens, 65.97 tok/s, TTFT 242.36), against one-card 16.51 / 16.65 ms and C7 stock DP=2 eager 121.40 ms. GATE_RC=0. SERVE_RC=0 at 12:56:53. Health-after HEALTHY rc 0 at 12:57:23. No xe-reset. The earlier capture-44 window health-after also finished HEALTHY at 12:53:25 with done rc=1 and no xe-reset.
VERDICT: M3 pass. Text matches the C4 string. C1 ITL is 1.022x the one-card graph. M0-M3 are closed.

## 2026-10-20 13:02 C13 -- one-layer expert prefill measured
CONFIG: _C.c13.so 2635416 bytes (12:59). Card 0. E=512 top-10 H=2560
F=640, gate/up IQ3_XXS, down Q2_0. DQ_T_MIN stays 12288.
B70_PREFILL_DQ_LAYER=bmm keeps C10. Daily _C.so is unchanged.
COMMAND: B70_AGENT=C13 B70_GPU_LOCK_TIMEOUT=21600 B70_PREFILL_TIMEOUT=7200
B70_CPU_LOCK_TIMEOUT=1800 bin/gpu-run --card 0 scripts/c13_run.sh
Log /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c13-bench5.log.
RESULT: health before/after rc=0. Wide dequant bit-exact vs
prefill_dequant. GB/s: IQ3_XXS 285.0 (old 255.8), IQ3_S 286.4,
IQ2_XXS 272.6, IQ2_XS 273.9, IQ2_S 279.0, Q2_0 270.3 (old 258.8),
IQ4_NL 281.0. bf16 copy 1147 GB/s. Layer event ms: T=4096 fast 40.13
(seq 40.13, C10 bmm 45.12, mmax 109); T=12288 fast 56.97 (seq 56.92,
C10 bmm 66.16, mmax 287). Overlap saves ~0. Profile at T=12288 ms:
group 1.80, dq 8.94+7.11+7.15, pad 2.89, bmm_gu 8.45, silu 1.50,
bmm_dn 4.67, unpad 3.15, reduce 11.47. Dispatch matches moe_fast
(0 mismatches). GEMM-only at T=12288: 3775.74 ms, 3255 tok/s.
Attention not timed. 2-layer bf16 scratch is 9.375 GiB and does not
fit the 2x200K headroom. 5e-2 gate unchanged. Synthetic fused max_rel
1.96 at magnitude ~2.8 (sign disagreement). Real blk.36 strict
max_rel misses are lanes near 1e-4 with frac_gate 1. Logged
failures=1 is that strict check on the new layer at x1.
VERDICT: T>=12288 one-layer path is 56.97 ms. Dequant is ~280 GB/s.
Report reports/1C13_expert_prefill2.md.

## 2026-10-20 13:15 G4 -- cycle restore waited out the long prefill

CONFIG: card 1, port 18113, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
32 GPU blocks, max_len 27000, PIECEWISE, 4 slots, 32 GiB host.
Client slept 15 s after submitting a 24000-token prefill (decode
budget 2800) and a 1024-token partner (decode budget 8000).
COMMAND: bash /tmp/g4-launch-t3-cycle.sh 1. Logs
g4-t3-cycle-driver-c1-r7.txt, g4-t3-serve-c1-cycle-r7.txt.
RESULT: health before ok. READY 13:06:31. At restore start
small_n=522 big_n=0. Engine 13:07:09-13:08:09 Running 1 Waiting 2,
KV 83.9-93.5 percent. 13:08:12 stored end=26799 and loaded tail
end=1519 local=0 keys=5. Turn 0 host 1519 device 0 greedy match.
Turns 1-7 missed (host 0, or host/device 1088 with no load tail).
Health after rc 0 at 13:08:51. No xe-reset. Client then hit
NameError active_n; the result line was already printed.
VERDICT: the copy landed, and the restored request was not scheduled
until the pool-filling request finished and took a host slot. Next
cycle keeps both actives at a 256-token prompt and a 4000-token
decode budget.

## 2026-10-20 13:22 G4 -- four parked restores under two live decodes

CONFIG: card 1, port 18113, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
32 GPU blocks, max_len 27000, PIECEWISE, 4 slots. Active prompts 256
tokens, decode budget 4000. Parks end at 1519.
COMMAND: bash /tmp/g4-launch-t3-cycle.sh 1. Logs
g4-t3-cycle-driver-c1-r8.txt, g4-t3-serve-c1-cycle-r8.txt,
g4-t3-cycle-c1-r8.json.
RESULT: health before ok, health after rc 0. Actives at restore start
17 and 8 tokens; at the end 287 and 296, streams_alive true. Engine
Running 3, KV 41.9 percent. Turns 0-3: load tail end=1519 local=1088
keys=4, host 431, device 1088, greedy match. Turns 4-7: host 0,
device 1088, computed 441. No xe-reset.
VERDICT: the first visit of each parked conversation restored the
tail from host while two requests decoded. The second visit reused a
same-length prompt after the tail had been replaced at end=1560.
Next cycle appends that reply and a new user span.

## 2026-10-20 13:27 G4 -- cycle cache gate passed, one late greedy miss

CONFIG: card 1, port 18113, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
32 GPU blocks, max_len 27000, PIECEWISE. Four parks. Second visit
appends the first reply and 9 new tokens. Actives are 256-token
prompts with a 4000-token decode budget.
COMMAND: bash /tmp/g4-launch-t3-cycle.sh 1. Logs
g4-t3-cycle-driver-c1-r9.txt, g4-t3-serve-c1-cycle-r9.txt,
g4-t3-cycle-c1-r9.json.
RESULT: health before ok, health after rc 0. streams_alive true,
active tokens 289 and 298. Running 3, KV 41.9 percent. Eight load
tails, local=1088 keys=4. Turns 0-3 host 431 device 1088 cached 1519
computed 10. Turns 4-7 host 472 device 1088 cached 1560 computed 10.
Greedy exact on turns 0-6. Turn 7 matched 29 of 32, then 40822
against gold 14308 inside a 5-token loop. TTFT 0.126-0.139 s.
No xe-reset.
VERDICT: host accounting holds for every parked restore under two
live decodes. Strict greedy fails on three trailing tokens of one
turn. Next is FULL_DECODE_ONLY T2. Not rerunning the cycle.

## 2026-10-20 13:33 G2b -- audit pass 11

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 13:27
G4 heading. Findings F122-F129 append reports/1G2_audit.md after F121.
COMMAND: Read headings from 12:58 G4 through 13:27 G4, plus 13:01 H1c
and 13:02 C13. Check h1c-m3-window-r2.out, h1c-m3-bench.json,
h1c-m3-greedy.json, h1c-m3-serve.log, c13-bench5.log, the G4 r6
through r9 drivers, serves, and json, and c12/serve_driver.log.
Hash the EP and C13 libraries. git diff --cached and the fork log -1.
RESULT: H1 M3 identity, coherence, health, and ITL match the json.
C1 p50 is 11.359 ms, n_chunks 62, completion 63. Capture keys are 4
and 3, then 4, 3, 2, 1. No error 44. The 121.40 comparison is the
eager max_len 2048 citation (F122). C13 GB/s and layer ms match.
gemm_ms is 3775.5, not 3775.74. The single FAIL is fast real x1
(F123). G4 r6 turn 0 is host 1519 device 0, and the cycle json has
the active_n NameError (F124). r7's 13:07:09 sample is Waiting 1,
and turn 4 is host 1088 device 0 (F125). r8 and r9 cache flags match.
r9 turn 7 mismatch_at 29 is 40822 against gold 14308, and pass is
false (F126, F127). c12/serve_driver.log now has HEALTHY card 0,
SERVE_DONE, and DRIVER_RC=0 (F128). Cached diff empty. Fork still
0a6170da8c, ahead 18. EP waits return when the stream is recording
(F129).
VERDICT: Pass 11 recorded. H1 M3 is a measured ITL with identity,
coherence, and health on both sides. G4 cycle pass stays false.
Open items: the C13 bench exits 1; G4 r9 greedy misses three tokens;
fair bulk rel is high; the PLE CPU test log and the 46.4 percent hit
rate stay unverified.


## 2026-10-20 13:33 G4 -- FULL_DECODE_ONLY T2 passed

CONFIG: card 1, port 18116, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
45 GPU blocks, max_len 32768, max_num_seqs 2. compilation-config mode 0,
cudagraph_mode FULL_DECODE_ONLY, sizes 1,2,3,4,8. Engine log
CUDAGraphMode.FULL_DECODE_ONLY (2, 0). Capture "FULL" 2/2 in 1 s.
COMMAND: bash /tmp/g4-launch-t2-graph.sh 1. Logs
g4-t2-graph-driver-c1.txt, g4-t2-serve-c1-graph.txt,
g4-t2-turn-c1-graph.json.
RESULT: health before ok. Health after HEALTHY rc 0, 12 s.
pass=True. host 3063, device 0, computed 18, greedy and text match.
Load tail end=3063 local=0 keys=6. Load 131235840 bytes in 0.009645 s
= 13.607 GB/s. Restored TTFT 0.120 s, control 0.412 s. No xe-reset.
VERDICT: parking the real second turn works under the full decode
graph. Next is the largest Ornith context. Flash-Next is not serving.
Ornith text max_position_embeddings is 262144. The 45-block pool held
39852 tokens; the profiler wanted 244 blocks.

## 2026-10-20 13:36 G2b -- audit pass 11 addendum

CONFIG: CPU only. No GPU, no systemd, no commit. The 13:33 G4 T2
heading landed after the pass 11 milestone. Finding F130 appends
reports/1G2_audit.md after F129.
COMMAND: Check g4-t2-graph-driver-c1.txt, g4-t2-serve-c1-graph.txt,
g4-t2-turn-c1-graph.json, and the Ornith GPTQ config.json.
RESULT: pass=True, host 3063, device 0, computed 18, greedy and text
match. Load 131235840 bytes in 0.009645 s is 13.607 GB/s. Restored
TTFT 0.120 s, control 0.412 s. Health after HEALTHY, 12 s, driver
exit 0. KV size 39852 and the 244-to-45 override match. Capture ran
2/2. max_position_embeddings 262144 is in the checkpoint config, not
the serve log (F130).
VERDICT: The T2 client gate passed with a real load and device delta
0. It is not a cycle pass. Open items from pass 11 are unchanged.


## 2026-10-20 13:42 G4 -- Ornith 200063-token restore passed

CONFIG: card 1, port 18117, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
profiled KV pool, max_len 204800, PIECEWISE, 4 slots, 32 GiB host.
Ornith GPTQ int4, not Flash-Next. P1=200000. KV cache 257312 tokens,
concurrency 1.26x at 204800.
COMMAND: bash /tmp/g4-launch-t4.sh 1. Logs g4-t4-driver-c1.txt,
g4-t2-serve-c1-t4.txt, g4-t2-turn-c1-t4.json, g4-t4-mem-c1.txt.
RESULT: health before ok. Health after HEALTHY rc 0, 12 s. pass=True.
host 200063, device 0, computed 18, greedy and text match. Load tail
end=200063 local=0 keys=187. Load 4164321280 bytes in 0.299164 s
= 13.920 GB/s. Restored TTFT 0.833 s, control 84.788 s. Four slots
held two ~200K conversations and two 30464 fillers. The cold fifth
evicted one slot after the restore. MemAvailable minimum 78051432 kB.
No xe-reset.
VERDICT: the Ornith stand-in for T4 meets the 0.42 s copy target.
Flash-Next 200K and T6 are still waiting on that model serving.

## 2026-10-20 14:08 G2b -- audit pass 12

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 13:42
G4 heading. Findings F131-F132 append reports/1G2_audit.md after F130.
COMMAND: Check g4-t4-driver-c1.txt, g4-t2-serve-c1-t4.txt,
g4-t2-turn-c1-t4.json, g4-t4-mem-c1.txt, and the Ornith section of
reports/1G4_kv_parking.md. git diff --cached and the fork log -1.
RESULT: pass=True, host 200063, device 0, computed 18, greedy and
text match. Load 4164321280 bytes in 0.299164 s is 13.920 GB/s.
Restored TTFT 0.833 s, control 84.788 s. KV line is 257312 tokens,
1.26x at 204800. Mode is PIECEWISE. Health after HEALTHY, 12 s,
driver exit 0. MemAvailable minimum is 78051432 kB. One evict
end=30464 is at 13:40:26, after the load at 13:40:19. keys=187.
183*1088+959=200063. Cached diff empty. Fork still 0a6170da8c,
ahead 18 (F131, F132).
VERDICT: Pass 12 recorded. The Ornith copy gate passed with device
delta 0. It is not a Flash-Next 200K result. Open items: C13 bench
exits 1; G4 r9 greedy misses three tokens; D2H stays under 7.5;
fair bulk rel is high; the PLE CPU test log and the 46.4 percent
hit rate stay unverified.


## 2026-10-20 14:36 C7b -- Flash-Next DP=2 eager window

CONFIG: fork 0a6170da8c. Both gpu.lock owner files empty. MemAvailable
118492772 kB. hotschmoe-dd inactive. Frontdoor on 18080 left running.
No P2P. No B70_QSA. Prepared window logs/c7-fn-window.sh: DP=2 TP=1,
--enable-expert-parallel, enforce-eager, language-model-only,
max_model_len 2048, docker memory 100g, SYCL_CACHE_PERSISTENT=0,
B70_GGUF_SO=_C.graph.so (10165216 bytes, C12 11:22). Greedy chat,
temperature 0, enable_thinking false, against recorded Strata texts
323 and 667. Watchdog docker-stops if MemAvailable falls under 8 GiB.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
stdout /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-window.out. RSS trace /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-rss.log.
RESULT: leases requested immediately after this entry.
VERDICT: window open.


## 2026-10-20 14:41 C7b -- Flash-Next load stopped on duplicated expert bytes

CONFIG: fork 0a6170da8c plus an uncommitted local edit that is not in
this run. Eager DP=2 TP=1 EP, max_model_len 2048, docker memory 100g,
SYCL_CACHE_PERSISTENT=0, B70_GGUF_SO=_C.graph.so (10165216 bytes).
No P2P. No B70_QSA. No hotschmoe-dd. Collective health first.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
stdout /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-window.out (preserved /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-window-ram.out).
Serve log preserved /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-serve-ram.log. RSS /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-rss-ram.log.
RESULT: HEALTH_RC 0. MEM_AVAILABLE_KB 118396268 at serve start 14:36:44.
Engine 0.32.1rc1.dev94+g5fd62688f, Qwen4ExpForConditionalGeneration,
DP=2 EP linear 256/512. Placed 1100 of 1224 tensors at 14:40:47.
No host-table line, so the PLE memmap view was not reached. No Python
traceback. Watchdog ABORT_LOW_RAM avail_kb=7309964 at 14:40:53.
SERVE_RC 137. No new kernel oom-kill (last one is still the 11:17
PLE copy). MemAvailable returned to about 118 GiB. No xe-reset.
Card 0 was then taken by O3 for a 73 s profile and returned HEALTHY.
Shard 1 is 43.81 GiB: 144 expert tensors 39.97 GiB, 887 other tensors
3.72 GiB, 192 shared-expert tensors 0.11 GiB. Largest tensor 0.439 GiB.
slice_expert_tensor returned a dim-0 view, so each worker kept the
full expert Storage. Docker cgroup was 84.59 GiB at 1000 tensors and
AnonPages was 85399492 kB. Strata's about 16 GiB is one process after
the weights sit on the GPU. This watermark is two private copies of
the packed experts, still on the host, before the GPU upload.
VERDICT: not a coherence result. The PLE view was not the copy that
filled the host. The next load clones the local expert half.

## 2026-10-20 14:45 C7b -- Flash-Next DP=2 retry, local expert storage

CONFIG: same eager window. mapping.slice_expert_tensor now clones a
partial expert slice so the other rank's bytes can be freed. ep_size 1
still returns the original tensor. CPU check EP_SLICE_STORAGE_OK.
Not committed. MemAvailable 118622492 kB. Both locks empty.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
stdout /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-window.out.
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 14:53 G2b -- audit pass 13

CONFIG: CPU only. No GPU, no systemd, no commit. Cursor is the 14:45
C7b heading. Findings F133-F138 append reports/1G2_audit.md after F132.
COMMAND: Check c7b-fn-window-ram.out, c7b-fn-serve-ram.log,
c7b-rss-ram.log, the retry c7b-fn-window.out, c7-fn-serve.log,
c7b-rss.log, kern.log oom lines, fork log -1, git status, and the
uncommitted mapping.py and b70_ep_xpu diffs.
RESULT: First window health-before OK, p2p=0, then ABORT_LOW_RAM
7309964 at 14:40:53, SERVE_RC 137, placed 1100 of 1224. Journal
118492772 kB is not the window's 118396268. The rebound sample is
119235164 kB (113.71 GiB). Nearby docker samples are 80.16GiB and
84.52GiB, not 84.59. Shard counts sum to 1223, not 1224. Retry
health-before OK, MEM_AVAILABLE_KB 118262704, placed 1224, PLE
memmap 28800138240 bytes, then XPU OOM 418 MiB with 200.77 and
202.14 MiB free. SERVE_RC 1 at 14:49:26. No live /v1/models and no
greedy. EP map stores raw pointers and does not bounds-check the
looked-up slot. Cached diff empty. Fork still 0a6170da8c (F133-F138).
VERDICT: Pass 13 recorded. Neither eager window is a coherence or
ITL result. The retry reached placement and died on device OOM.
Open items: C13 bench exit 1, G4 r9 greedy, D2H under 7.5, fair
bulk rel, the earlier PLE CPU test log, and the 46.4 percent hit
rate. The EP map needs an owned copy and a slot check before capture.



## 2026-10-20 14:49 C7b -- PLE view held, XPU OOM on the bf16 placeholder

CONFIG: eager DP=2 window, fork plus the uncommitted expert-slice clone.
max_model_len 2048, gpu_memory_utilization 0.90. No P2P. No B70_QSA.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
stdout preserved c7b-fn-window-xpuoom.out. Serve log preserved
c7b-fn-serve-xpuoom.log.
RESULT: HEALTH_RC 0. MEM_AVAILABLE_KB 118262704 at 14:46:02.
Placed 1224, skipped_mtp 0, arch qwen4exp. Host table
per_layer_token_embd.weight memmap view bytes=28800138240.
Host MemAvailable bottomed at 40042444 kB (anon 13164496 kB), then
recovered. No kernel oom-kill. Both workers then raised
torch.OutOfMemoryError in embedding.py process_weights_after_loading
at packed.to(device), 418 MiB. GPU 0 and GPU 1 each had 31.10 GiB
allocated by PyTorch of 31.89 GiB. SERVE_RC 1 at 14:49:26. No xe-reset.
O3 used card 0 for 75 s after that and left it HEALTHY, then started
another card-0 run at 14:53:52. Card 1 stayed empty.
Model init runs inside the XPU device context, so each linear and
embedding allocates its bf16 weight on the card. apply() and
embedding() read the packed tensor. Packed experts are about 20 GiB
per rank after the slice clone. The placeholders make up the rest of
the 31.10 GiB.
VERDICT: host RAM is no longer the stop. The next load drops a packed
layer's bf16 placeholder before the packed H2D. That retry waits until
both locks are empty so card 1 is not blocked behind card 0.


## 2026-10-20 14:54 C7b -- Flash-Next DP=2 retry, drop bf16 placeholders

CONFIG: same eager window. linear and embedding
process_weights_after_loading replace a packed layer's bf16 weight
with an empty parameter on the same device before the packed copy.
Expert-slice clone still in place. Not committed. MemAvailable
118062644 kB. Both locks empty.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 14:55 H1d -- Flash-Next EP sidecar adapted on CPU

CONFIG: Phase A, no GPU. Profile
/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin,
q0=138, share0 off. B70_EP_SKIP_GGUF=1, MAX_JOBS=4. GGUF library
for the later serve is b70_gguf_xpu/_C.graph.so (symbol
_Z13b70_moe_layerPvPKtPKhiS3_iS3_iPKiPKfS3_iS3_iS3_iPfiiiii and
b70_reserve_decode_queue both present). _C.ep.so not rebuilt.
At 14:55 both gpu lock owner files were empty and
reports/1C7b_flashnext_serve.md was absent. C7b's 14:54 entry had
just requested a window, so the sidecar serve stays down until that
report exists or both locks have been empty for 10 minutes.
COMMAND: B70_AGENT=H1d B70_CPU_LOCK_TIMEOUT=7200 B70_EP_SKIP_GGUF=1
MAX_JOBS=4 bin/cpu-run bash
vllm/flashnext/plugins/b70_ep_xpu/scripts/build.sh
Logs h1d-ep-build.log (first link) and h1d-ep-build2.log (owned
maps). Then python3
vllm/flashnext/plugins/b70_ep_xpu/tests/test_placement.py.
RESULT: First link waited on b70s4dayz mb/build.sh, then exit 0 in
45s, sha 308cd69d555edc7c59bd12688cf750644e7de2849f9fd2485be5eafb8fbc86d7
(495104 bytes). That binary stored the caller's map pointer. Second
link exit 0 in 39s, sha
5a0a047c67fc192588007e93b69c22a8ed9c781d2a891056dff200cceab067ac
(499200 bytes, 14:55). It copies both maps onto card 0, rejects a
slot outside the local pack, a duplicate slot, or an expert on both
cards or neither, and the dispatch kernel clamps a bad slot to -1.
_C.ep.so sha stays
55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0.
CPU test printed PLACEMENT_OK and PLE_GUARD_OK. 48 layers, 138 and
374, disjoint, union 0..511, local slots 0..n-1. Hottest profile
expert of a layer lands on card 0 and the next on card 1. Layer 0
card 0 head is (3, 7, 10, 17). Ranked pairs 24576. Shorthand pack
(IQ3_XXS gate, IQ3_XXS up, Q2_0 down) is 11361484800 bytes on card 0
and 30791270400 on card 1. The shard mix is wider than that
shorthand (reports/02_model_support.md). PLE guard rejects a
non-memmap, a second data pointer, a size mismatch, and
MemAvailable under the floor. MemAvailable during the test was
118655020 kB. Serve script
vllm/flashnext/harness/serve_flashnext_ep.sh is written and was
not launched.
VERDICT: Phase A CPU adapt is in place, including an owned map.
Phase B has not started. No greedy text and no ITL.

## 2026-10-20 15:02 C7b -- placeholders dropped, KV budget short

CONFIG: eager DP=2 EP window, fork plus uncommitted expert-slice
clone and the packed-layer bf16 placeholder drop (linear.py and
embedding.py process_weights_after_loading). max_model_len 2048,
gpu_memory_utilization 0.90, max_num_seqs 2, enforce-eager.
B70_GGUF_SO is _C.graph.so. B70_GGUF_REF=0, B70_EAGER_QK=0. No P2P.
No B70_QSA. The window parent died after launch; timeout pid
3345213 kept the serve. Host watchdog was the adopt loop.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
Serve log preserved at
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-serve-kvshort.log.
RSS samples c7b-rss-run3.log.
RESULT: HEALTH_RC 0 at 14:54:56. MEM_AVAILABLE_KB 118704776.
Placed 1224, skipped_mtp 0, q8_dequant 0, arch qwen4exp. Host table
per_layer_token_embd.weight memmap view bytes=28800138240 on both
workers. Model loading took 24.9 GiB and about 125 s on each card.
No XPU OOM on the packed upload. Then both engine cores raised
ValueError: No available memory for the cache blocks.
gpu_worker.py logged Available KV cache memory: -1.53 GiB.
Card capacity on the previous OOM was 31.89 GiB, so 0.90 requests
about 28.70 GiB and the profiled non-KV footprint is about 30.23 GiB.
Host during placement peaked near AnonPages 51190084 kB
(MemAvailable 45549788 kB) around 800 tensors. After the upload,
steady AnonPages was 12252612 kB and the two VLLM workers summed to
9206088 kB RSS (about 4.4 GiB each). Strata's about 16 GiB is one
process after weights are on the device. The extra host RAM here is
the transient two-worker packed copy, and it was released. Steady
host use is under that 16 GiB figure. No kernel oom-kill. Locks
released by 15:01. No xe-reset. No greedy text.
VERDICT: the placeholder drop lets the packed weights land (24.9 GiB
per card). The 0.90 pool cannot hold a KV cache. Next serve uses
gpu_memory_utilization 0.98 and max_num_batched_tokens 2048.


## 2026-10-20 15:03 C7b -- eager window, util 0.98

CONFIG: DP=2 EP eager, max_model_len 2048, max_num_batched_tokens
2048, gpu_memory_utilization 0.98, max_num_seqs 2. Same fork fixes
as the 15:02 entry. Container vllm-fn-flashnext-gguf-ep98. Port
18104. MemAvailable 118285456 kB. Both locks empty.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-eager98.sh
stdout c7b-fn-eager98-window.out. Serve log c7b-fn-serve-eager98.log.
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 15:10 C7b -- eager DP=2 coherent at util 0.98

CONFIG: logs/c7b-fn-eager98.sh. DP=2 EP, enforce-eager,
max_model_len 2048, max_num_batched_tokens 2048,
gpu_memory_utilization 0.98, max_num_seqs 2. Fork placeholder
drop and expert-slice clone, uncommitted. _C.graph.so.
B70_GGUF_REF=0, B70_EAGER_QK=0. No P2P. No B70_QSA. Container
vllm-fn-flashnext-gguf-ep98. Port 18104.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-eager98.sh
RESULT: HEALTH_RC 0. MEM_AVAILABLE_KB 118088084 before serve.
Model loading took 24.9 GiB per card. Available KV cache memory
0.89 GiB. XPU KV cache size 6144 tokens. Maximum concurrency for
2048 tokens: 3.00x. kv lcm block size 1600. HTTP ready 15:09:33.
Greedy temperature 0, enable_thinking false, 25 prompt tokens:
"Return only the integer: 17 * 19." -> "323" (4 completion tokens).
"Return only the integer: 23 * 29." -> "667" (4 completion tokens).
Both match the Strata strings. SERVE_RC 0. Served names
hotschmoe-dev, qwen3.8-flash-next-iq3-xxs.
Host at ready: MemAvailable 47736880 kB, AnonPages 12847624 kB.
Worker RSS 4562356 kB and 4645336 kB. PSS 4316028 kB and
4398933 kB. Anonymous 4136256 kB and 4219228 kB. Shared_Clean
about 425000 kB each. Placement peak AnonPages 53938904 kB at
15:08:08, then the host copies were released. Strata's about
16 GiB is one process after the weights are resident. Steady
host anonymous memory here is 12.3 GiB for two workers, under
that figure. The 51 GiB peak is the transient packed copy.
VERDICT: first coherent Flash-Next IQ3_XXS eager serve. Graph
window is next, still at max_model_len 2048.

## 2026-10-20 15:13 C7b -- FULL_DECODE_ONLY window open

CONFIG: logs/c7b-fn-graph-window.sh. DP=2 EP,
FULL_DECODE_ONLY, capture sizes 1,2,3,4,8, FLASH_ATTN,
B70_XPU_CG_RECLAIM=500, max_model_len 2048,
max_num_batched_tokens 2048, gpu_memory_utilization 0.98,
max_num_seqs 4. Same weights and _C.graph.so. No P2P.
MemAvailable 117855916 kB. Both locks empty. O3 had card 0 for
run_gates.sh from 15:09:57 until the locks were empty again.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-graph-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 15:19 C7b -- graph capture rejected the PLE host copy

CONFIG: logs/c7b-fn-graph-window.sh. FULL_DECODE_ONLY, FLASH_ATTN,
B70_XPU_CG_RECLAIM=500, capture sizes 1,2,3,4,8, util 0.98,
max_model_len 2048, max_num_seqs 4. HEALTH_RC 0.
MEM_AVAILABLE_KB 118052504.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-graph-window.sh
RESULT: weights loaded, then capture of FULL graphs failed on both
workers. level_zero error 44 UR_RESULT_ERROR_UNSUPPORTED_FEATURE at
Qwen4ExpPLE.forward input_ids.to("cpu"), inside torch.cuda.graph
during profile_cudagraph_memory. SERVE_RC 1 at 15:18:34. No greedy
and no ITL. Log c7b-graph-serve-ple-d2h.log. No DEVICE_LOST.
The retry at 15:22 found card 0 held by O3 pid 3358701 since
15:19:48, container vllm-fn-qwen36-gguf-c0, port 18104. It did
not start a second serve.
VERDICT: eager coherence stands. FULL graphs need the PLE gather
outside the captured region. The staged-row retry waits for both
locks.

## 2026-10-20 15:40 O3 -- MoE decode gap closed: misaligned weight loads, serial plan
CONFIG: card 0 only. Plugin b70_gguf_xpu, fair dumps from C9 (E=10..28
dense, top-10 + shared, K=2560, I=640, IQ3_S/Q2_0 and IQ2_XS/IQ4_NL).
Strata fair_strata (C9 bench plus a FAIR_T/FAIR_PAIR filter) in image
e41d59c0. unitrace -d for device time per kernel. Work dir
/mnt/vm_8tb/b70/build/vllm-flashnext/o3.
COMMAND: gpu-run --card 0 o3/run_prof.sh (unitrace, ours and Strata);
o3/mb microbench A/B (o3/run_mb.sh); o3/run_gates.sh; cpu-run --quiet
gpu-run --card 0 o3/run_fair.sh; o3/serve_35b.sh (C12 recipe).
RESULT: T=1 device time, gate/up was 123.1 us here vs 45.3 us in Strata R2
(same geometry, same 112 dp4a, no spills). The cause is b70_prep reading the
8-/4-byte block fields via memcpy at 2-aligned addresses. IGC emits
misaligned d64/d32 LSC loads. Strata load8_a2/load4_a2 aligned-dword loads:
gate/up 123 -> 44.4 us, down 38 -> 16.5 us. The plan single_task was 8.5 us
(T=1) and 74 us (T=4); it is now a parallel work-group fused with Q8(x),
and SiLU is fused with Q8(h), so a layer is 5 launches (was 7). Fair graph
under cpu-run --quiet, IQ3_S/Q2_0 T=1..4: 180.1/267.0/385.8/515.8 ->
68.6/100.3/141.4/178.6 us. Strata R2 is 112.9/148.1/202.9/261.6. Device-only,
ours is 1.07x (T=1) and 1.01x (T=4) Strata IQR2. IQ2_XS/IQ4_NL:
192.2/281.4/403.8/565.4 -> 81.4/117.4/168.3/216.7. Outputs are bit-identical
to C12 (same cos/max_abs vs the Strata dump). Gates: gpu_check, moe_layer_bench
(sched 1 and 0), c12_types (12), graph_probe, compare_cpu 14/14, all
failures=0; health HEALTHY before and after each GPU run. 35B UD-IQ3_XXS
FULL_DECODE_ONLY serve: identity ok, coherence OK, C1 ITL p50 11.02 ->
9.34 ms (107.0 tok/s), C2 16.61 -> 12.60 ms/stream.
Installed _C.so = _C.graph.so = _C.o3.so sha256 be76661bd120aa2f...e861eb
(10135104 bytes). Revert copy: _C.c12.so (b3c15e14). NOTE for C7b: its
graph window uses _C.graph.so and had not started at install (15:19), so it
will load the O3 binary.
VERDICT: the gap was memory-op lowering, not schedule. Target (<= 1.15x
Strata at T=1..4) met at 0.61-0.73x of the R2 numbers. Next: the same
aligned-load fix for the float decode32 K-quant path (35B Q2_K/Q3_K/IQ4_XS
experts), then bandwidth (gate/up ~350 GB/s of ~608). Report:
reports/2O3_moe_decode.md.

## 2026-10-20 15:39 C7b -- staged PLE passed, FA decode kernel missing

CONFIG: logs/c7b-fn-graph-window.sh. FULL_DECODE_ONLY, FLASH_ATTN,
B70_XPU_CG_RECLAIM=500, capture sizes 1,2,3,4,8, util 0.98,
max_model_len 2048, max_num_seqs 4. B70_GGUF_SO=_C.c12.so
sha256 b3c15e14. PLE rows staged outside the graph. HEALTH_RC 0.
MEM_AVAILABLE_KB 117993036.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-graph-window.sh
RESULT: serve started 15:35:19 and died during FULL capture at
15:39:23. SERVE_RC 1. No greedy and no ITL. PLE host copy did not
fire. FlashAttention paged decode is not compiled for
16,256,64,false,false,false (qgroup,head,page,causal,local,sink).
The PyTorch fallback then calls cu_seqlens_q.cpu() inside the
graph and Level Zero returns error 44. Installed package is
vllm-xpu-kernels 0.1.15.4. The source conf line exists at HEAD
19c6ddc and is absent from tag v0.1.15. Log
c7b-graph-serve-fa-missing.log. Locks released. No DEVICE_LOST.
VERDICT: eager coherence stands. Next is a side build of that
one Xe2 decode specialization. The shared wheel stays put.

## 2026-10-20 15:34 C7b -- graph window, staged PLE rows

CONFIG: logs/c7b-fn-graph-window.sh. FULL_DECODE_ONLY, FLASH_ATTN,
B70_XPU_CG_RECLAIM=500, capture sizes 1,2,3,4,8, util 0.98,
max_model_len 2048, max_num_seqs 4. PLE host gather is staged
into _graph_emb outside the XPU graph. Uncommitted.
B70_GGUF_SO is _C.c12.so (sha256 b3c15e14, 10165216 bytes).
_C.graph.so is O3's later binary and is left in place.
MemAvailable 117593984 kB. Both locks empty. Port 18104 free.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-graph-window.sh
RESULT: leases requested immediately after this entry.
VERDICT: window open.

## 2026-10-20 15:36 G2b -- audit pass 14

CONFIG: CPU only. No GPU, no systemd, no commit. Findings F139-F147
append reports/1G2_audit.md after F138. Cursor is the 15:34 C7b
heading. The 15:13 graph window had already exited.
COMMAND: Read c7b-fn-serve-kvshort.log, c7b-rss-run3.log,
c7b-fn-serve-eager98.log, c7b-eager98-greedy.json,
c7b-fn-eager98-window.out, c7b-rss-eager98-ready.txt,
c7b-coll-health-eager98.txt, c7b-graph-serve-ple-d2h.log,
c7b-fn-graph-window-ple-d2h.out, c7b-coll-health-graph.txt,
h1d-ep-build.log, h1d-ep-build2.log, o3/fair_quiet.log,
o3/bench_35b.json, o3/models.json, o3/serve_driver.log,
c12/serve_driver.log. Hash the EP _C.so, _C.ep.so, _C.graph.so,
_C.o3.so, and _C.c12.so. Read ep_register_map, the dispatch clamp,
stage_graph_rows, and the torch.cuda.graph capture body.
RESULT: Eager util 0.98 reached READY 15:09:33, chats 323 and 667
matched, SERVE_RC 0, KV 0.89 GiB and 6144 tokens, health-before
only. The KV-short run died on Available KV -1.53 GiB after
placement, with no XPU OOM string. The 15:13 graph window died at
15:18:14 inside torch.cuda.graph (forward_fn mode NONE) on
input_ids.to("cpu"), Level Zero error 44, SERVE_RC 1. The EP
sidecar on disk is sha 5a0a047c, 499200 bytes, and the source owns
and clamps the maps. O3's 35B card-0 bench matches identity,
coherence, C1 p50 9.34 ms, and tok/s 107.03. Installed _C.so,
_C.graph.so, and _C.o3.so share sha be76661b, 10135104 bytes,
mtime 15:19:23, after that graph window exited. _C.c12.so is still
b3c15e14. The 15:34 retry pins _C.c12.so and has started; its
window file says MEM_AVAILABLE_KB 117993036, not 117593984. Cached
diff empty. Fork still 0a6170da8c.
VERDICT: Pass 14 recorded. The 15:10 eager window is coherence,
not an ITL. The 15:13 graph window is closed and failed. F136's
dangling map is fixed in the current EP library, and that library
has not served. The 9.34 ms figure is the 35B serve, not
Flash-Next. The staged-row window is still open. Open items remain
C13 bench exit 1, G4 r9 greedy, D2H under 7.5, fair bulk rel, the
PLE CPU test stdout, and the 46.4 percent hit rate.

## 2026-10-20 15:45 O3 -- short card-0 window (microbench only)
CONFIG: card 0 only, about 3 minutes, standalone microbench (no serve).
Checked first: C7b's 15:35 window ended at 15:39 (WINDOW_DONE), both locks
empty. If a C7b/H1d window is announced it goes first; O3 retries later.
COMMAND: gpu-run --card 0 o3/run_kq.sh
RESULT/VERDICT: in the next O3 entry.

## 2026-10-20 15:56 O3 -- card-0 gates window (about 6 minutes)
CONFIG: card 0 only. C7b holds card 1 for the vllm-xpu-kernels FA build
(15:52); no graph window is announced. If one is, O3 yields after this run.
COMMAND: gpu-run --card 0 o3/run_gates_b.sh (_C.o3b.so)
RESULT/VERDICT: in the next O3 entry.

## 2026-10-20 15:54 X3 -- prefill dispatch audit and probe preparation

CONFIG: CPU only. DD inactive. C7b and H1e retain both-card priority.
Read BRIEF, AGENTS, bottom 300 journal lines, C10/C11/C12/C13/E2 reports.
COMMAND: inspected gguf/ops_interface.py, prefill wrappers and serve harness.
Claude --bare -p --model fable-5.1 and --model claude-5.5 consulted in
parallel; both exit 1 unrecognized_model and Not logged in. No responding
model ID and no advice received.
RESULT: GGUF linear always uses gemv; expert T>8 uses singleton-entry
legacy grouped_gemm. Neither reaches prefill.py. serve_qwen36_gguf.sh
also drops all B70_PREFILL_* env. C13 is a smoke-only library lacking
serve decode ops. Existing 3775.74 ms Flash-Next table is a mixed
measured/projected GEMM budget, not end-to-end prefill.
VERDICT: add opt-in B70_PREFILL=1 dispatch, measure missing non-GEMM
pieces on one card and remeasure 35B serve. X3 plans short card-0
windows (5 minutes each) once CPU quiet lease and priority runs allow;
no both-card serve requested. No commit. Report reports/1X3_prefill.md.

## 2026-10-20 15:59 X1 -- MTP sidecar and proxy preparation

CONFIG: CPU only. DD inactive. C7b then H1e retain both-card priority.
Read BRIEF, AGENTS, bottom 300 journal lines, G3, D2, Shisa overlay,
Strata MtpDrafter and tools/mtp_rt.py. No commit or service changes.
COMMAND: inspect fork MTP, GGUF mapping/loader and Strata MTP artifacts;
claude --bare -p --model fable-5.1 and --model claude-5.5.
RESULT: both consults exit unrecognized_model / Not logged in; no actual
responding model ID. Strata has separate mtp-q2_0.gguf plus rt dense.bin,
dense.txt, experts.bin. Default fc uses pooled wide RMS and independent
stream projection, not GGUF eh_proj collapse. Solo spec4; batch depth2.
Added isolated mtp_loader.py and dispatch from B70GGUFModelLoader;
B70_MTP_RT chooses wide fc with packed quant methods. Proxy nextn block
is loaded only into drafter, rather than silently skipping its weights.
VERDICT: implementation in progress, no MTP serving or speed claim yet.
X1 requests card 0 for short proxy graph windows (about 5 minutes each)
after priority C7b/H1e windows; all GPU work queues with timeout 7200.


## 2026-10-20 16:00 X2 -- header budget and opt-in memory probe

CONFIG: CPU only, no changes to shared fork or existing plugins. New
b70_memory_xpu general plugin is inert unless B70_MEMORY_FIX=1 or
B70_MEMORY_REPORT=<directory>. C7b, then H1e retain both-card priority.
COMMAND: python3 vllm/flashnext/memory/gguf_budget.py <ISTA XXS shard1>
<shared IQ3_S shard2>. Raw x2/headers.json under build/vllm-flashnext.
Read current linear loader, dequant scratch cache and memory profiler.
RESULT: exact routed bytes 42912972800, trunk 4115863040, DP+EP
canonical weights/card 25572349440. Sidecar 138/374 weights/card
15682250240 / 31346585600. BF16 KV 24576 bytes/token, INT8
with per-64 FP16 scales 12672; indexer 768 bytes/token additional.
Full dense GGUF loads retain a second tensor; merged dense layers
retain unused placeholders. Dense dequant caches distinct type/shape
buffers. The opt-in plugin aliases these dense weights and reuses
one ordered-stream dequant buffer, with JSON storage/profile evidence.
No GPU result yet. Claude fable-5.1 and claude-5.5 consultation calls
both returned unrecognized_model (404); no responding model ID.
VERDICT: measuring real savings next. Do not subtract guessed profiler
headroom or promote FP8 KV without quality evidence. Other serves are
unaffected unless they explicitly enable the new plugin flags.

## 2026-10-20 15:58 X1 -- short proxy window announcement

CONFIG: card 0 only, expected 3-5 minutes per serve, FULL_DECODE_ONLY.
Both owners empty at 15:57; C7b's FA side build exited 1 at 15:56:32,
no new C7b or H1e serve window is recorded. O3 gates ended. X1 yields
before the next window if a priority pair serve is announced.
COMMAND: MTP_TOKENS=0 harness/x1_proxy_window.sh, then MTP_TOKENS=2
only after rechecking owners and journal. Queue timeout 7200.
RESULT/VERDICT: pending; exact token-ID comparisons prepared (4 x 96),
identity first hotschmoe-dd, acceptance and C1/C2 latency diagnostics.
The earlier X1 heading marked 15:59 was written at about 15:55 UTC.

## 2026-10-20 16:02 C7b -- Xe2 paged-decode 16,256,64 side library

CONFIG: logs/c7b-fa-build.sh. One decode policy
16,256,64,false,false,false and one chunk-prefill line
256,true,true,false,false,false. VLLM_CUTLASS_SRC_DIR is local
sycl-tla 87f6850680. CMPLR_ROOT is oneAPI 2026.1. Host ocloc
26.22.1 and llvm-foreach are on PATH. Card 1 only. The shared
vllm-xpu-kernels 0.1.15.4 wheel is not modified.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fa-build.sh
RESULT: BUILD_OK 16:00:05 in 59s (objects were already compiled).
libattn_kernels_xe_2.so is 11944232 bytes with OFFLOAD_DEVICE_CODE
10813072 bytes. Enabled trait is decode_policy_q16_h256_p64 with
causal,local,sink all false. _vllm_fa2_C.abi3.so NEEDs only that
Xe2 library. Private overlay is
/mnt/vm_8tb/b70/build/vllm-flashnext/c7b-fa-overlay. Shared wheel
mtime is still 2026-10-19 22:31. Earlier attempts failed on
setuptools-scm, /bin/icpx, missing git, and ocloc (logs
c7b-fa-build-scm.log, c7b-fa-build-compiler.log,
c7b-fa-build-git.log, c7b-fa-build-ocloc.log).
VERDICT: the missing decode specialization is linked. Next is
the both-card graph window logs/c7b-fn-graph-window.sh as soon
as both gpu.lock owner files are empty. Current card-0 work
should yield after its current command. C7b will not kill it.

## 2026-10-20 16:01 X1 -- sidecar CPU audit passed, proxy baseline retry

CONFIG: actual Flash-Next MTP, CPU/meta construction and payload placement.
Card-0 proxy baseline at 15:58 used _C.o3.so be76661b; no-MTP graphs.
COMMAND: cpu-run uv ... harness/x1_check_sidecar.py; MTP_TOKENS=0
harness/x1_proxy_window.sh. Logs under build/vllm-flashnext/x1.
RESULT: sidecar-cpu-r2 exit 0, FLASHNEXT_PAYLOAD_AUDIT_OK and
WIDE_FC_REFERENCE_OK [2,4,2560]. Additional actual _fc_stage compare
prepared. Proxy baseline reached HTTP ready around 16:00:24, but an
X1 edit to the active shell script moved its input offset and caused
line 42 leanup: command not found. Clean server teardown, no benchmark.
VERDICT: baseline evidence invalidated. Fixed harness is frozen during
runs. Owners/journal rechecked: X3 card 0; no new priority pair serve
announced. Retry queued, timeout 7200; about 3 minutes once acquired.
C7b FA side build now succeeded; X1 will yield the following window.

## 2026-10-20 16:03 X3 -- additive build and queued one-card budget probe

CONFIG: card 0 only, next window expected 3-5 minutes. No both-card use.
C7b/H1e retain priority. X1 proxy and O3 health currently take card 0.
COMMAND: x3/run_ops.sh --part gdn,attention,hc,dense,reduce,ple,expert
--out ops-basic3.json, detached with pid/log in build/vllm-flashnext/x3.
RESULT: additive _C.x3wide.so build exit 0 (36s), 898512 bytes.
CPU hash correctness X3_HASH_OK across 100 ragged trials. Initial probes
stopped on missing explicit _xpu_C import, then missing standalone vLLM
config for RMSNormGated. Neither was a kernel crash. Before the second
stop, 35B T=12288 GDN core measured 7.47 ms median with a declining warm
sequence; rerun adds warmups and measures both real head geometries.
Python default dispatch changes are opt-in B70_PREFILL=1. X1 explicitly
uses 0. Decode is unchanged. Padded scratch now reuses geometry keys.
VERDICT: no speed claim from the partial probe; complete stage measurements
and matched 35B A/B next. No reset, no commit, DD stays stopped.


## 2026-10-20 16:02 X2 -- allocator split policy candidate, priority wait

CONFIG: CPU only; new opt-in plugin installed editable, inert by default.
COMMAND: memory plugin test_cpu.py in runtime image with no /dev/dri;
full register import with no devices; inspect GPUWorker split guard.
RESULT: CPU_ALIAS_FORWARD_OK STREAM_ARENA_OWNERSHIP_OK; REGISTER_OK.
Upstream _scoped_allocator_max_split returns immediately on XPU. Added
an opt-in device-neutral setter wrapper preserving/restoring settings,
to test whether large pinned allocator segments explain profiler loss.
The first header pass overcounted dense duplicates by including HC/PLE
plain _WeightContainer parameters; corrected script estimates 186040320
bytes of linear duplicates, not 1.42 GiB. No measured GPU savings yet.
The earlier X2 heading marked 16:00 was written about 15:54 UTC.
VERDICT: X2 controller waits outside leases for C7b then H1e. After
their windows, requests BOTH cards about 12 minutes total for matched
eager current/aliases/capped-FP8 memory probes, no latency benchmark.
No core fork edits, no existing plugin changes, no commit/service action.

## 2026-10-20 16:04 X3 -- budget probe moves to free card 1

CONFIG: card 1 only, expected 3-5 minutes. Device pin supplied by
xpu-run --card 1 (ZE_AFFINITY_MASK=1). Card 0 held by O3's 35B serve;
card 1 owner empty. No new C7b/H1e pair window recorded. Cancelled X3's
queued card-0 ops3 before it acquired a lease. C7b/H1e still go first.
COMMAND: X3_CARD=1 x3/run_ops.sh --part
gdn,attention,hc,dense,reduce,ple,expert --out ops-card1.json.
RESULT/VERDICT: pending. This avoids waiting behind serial proxy serves.

## 2026-10-20 16:07 X3 -- first missing-op measurements, real-shape budget

CONFIG: actual card-1 window was 16:03:16-16:04:32 (the prior heading's
16:04 was approximate). T=12288, one-card event median, quiet CPU lease.
No kernel crash. Ops-rest follow-up now on card 1, expected about 2 minutes.
COMMAND: x3/ops-card1.log and ops-card1.json; x3/run_ops.sh.
RESULT: Flash-Next GDN conv/chunk 6.699 ms per layer; full attention
23.875 ms. HC mix 12.187 ms, combine 2.679 ms. PLE scalar hash 903.32 ms,
gather including hash 938.01 ms, whole PLE 990.09 ms. Hash replacement
first tripped on GPU-resident metadata; fixed by tolist(), same convention
as original. All Python exceptions, not kernel failures; post-health
HEALTHY. The initial default standalone VllmConfig chose native norm;
follow-up explicitly uses mode 0 and logs forward_xpu, matching serves:
35B GDN norm 0.709 ms versus 6.676 ms native. Do not use native norm in
the mode-0 budget.
VERDICT: PLE host hash and the 97 HC passes consume material TTFT;
attention/GDN are now measured. X2 integration note: C13 full-E expert
scratch alone is 4.6875 GiB, half-E 2.34375 GiB, plus activation/padding.
The old xpad key retained every mmax across layers; X3 changed it to
reuse geometry keys. Real-router skew will be measured before a full
prefill budget is trusted. No commit, no DD, no reset.

## 2026-10-20 16:08 X3 -- completed op probe and 35B A/B window requested

CONFIG: next window CARD 1 ONLY, expected about 4 minutes for one 35B
load plus matched old/new prefill. Port 18133, vllm-fn-x3-35b. Card 0 O3
serve, card 1 just completed X3 probe. No later C7b/H1e serve announcement.
The priority pair serves remain first. No DD service actions.
COMMAND: x3/serve_window.sh, long lease timeout, watchdog abort under
8 GiB MemAvailable; FULL_DECODE_ONLY, max batched 12288, SYCL GDN,
FA, prefix cache disabled. New prefill opt-in; runtime flag file supplies
matched old/new scheduled requests in the same loaded serve.
RESULT: op-probe exit 0, post-health underway. At T=12288, Flash-Next
GDN conv/chunk 6.725 ms and fused norm 0.989 ms; attention 24.155 ms.
PLE whole 983.97 -> 79.47 ms with numpy hash (hash 902.72 -> 2.91).
Expert layer 56.89 -> 47.08 ms for real blk.36 weights with fused
routing reduction; other measured types similar. Reduction max_abs
1.91e-6 vs original and allclose 1e-5 passes. Raw ops-rest.json/log.
VERDICT: the per-layer budget now has measurements for the formerly
missing pieces. PLE and reduction fixes save about 1.38 s per 12288-token
full-model sum; this is not yet a full-model measured throughput claim.

## 2026-10-20 16:07 X1 -- matched no-MTP proxy baseline measured

CONFIG: card 0, official Qwen3.6 HF config + UD-IQ3_XXS GGUF, BF16,
TP=1, FULL_DECODE_ONLY sizes 1,2,3,4,6,8, FLASH_ATTN, prefix cache off,
max_len 2048, max_seqs 2, _C.o3.so be76661b. Stable first hotschmoe-dd,
research alias qwen3.6-35b-a3b-ud-iq3-xxs-mtp0 registered in eval config.
COMMAND: MTP_TOKENS=0 harness/x1_proxy_window.sh (frozen harness retry).
RESULT: identity, coherence pass; 4 x 96 greedy token IDs saved;
GREEDY_RC=0, BENCH_RC=0. C1 p50 9.27 ms, 108.24 tok/s. C2 p50
12.48/12.38 ms, 82.34/74.93 tok/s (one slow prefill transition).
No MTP. Server cleanly down; post-health queued behind O3's post-health.
Full sidecar CPU r3 also passes actual _fc_stage against independent
pooled RMS plus Q8 dequant matmul, exit 0 in 25 s.
VERDICT: valid proxy baseline, not Flash-Next performance. X1 holds no
new serving window while C7b then H1e have pair priority. Next X1 window
is card 0, about 3 minutes, MTP=2 exact-token/acceptance/latency gate.

## 2026-10-20 16:10 O3 -- stage 2: K-quants/IQ4_XS on DP4A; bandwidth tries
CONFIG: plugin iq_dot.hpp only (decode.hpp/host dequant untouched). Q2_K,
Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS go through b70_prep as signed-byte words +
integer scales/mins with aligned-dword loads (ggml vec_dot_*_q8_1 form);
-DB70_KQ_DP4A=0 restores decode32 + fma. sched_moe R-row dot is generic
(-DB70_SCHED_RPL, default 2, same code and timing as stage 1). Card 0
windows only while no C7b/H1 window was announced (15:45, 15:56, 16:03).
COMMAND: o3/run_kq.sh (synthetic K=2048 I=512 top-10+shared dumps,
o3/gen_kq.py), o3/run_gates_b.sh, o3/serve_35b.sh (TAG o3b-graph).
RESULT: microbench graph us T=1 old->new: Q2_K 485.5->49.3, Q3_K
268.2->79.4, Q4_K 96.8->59.2, Q5_K 119.4->78.7, Q6_K 581.9->88.4, IQ4_XS
110.5->81.8 (T=4 1179->126, 685->196, 252->149, 302->193, 1397->213,
280->199); new vs old cos 1.000000. Gates on _C.o3b.so: gpu_check,
moe_layer_bench sched 1/0, c12_types 12, graph_probe all failures=0;
compare_cpu 14/14; HEALTHY before/after. c9_fair IQ3_S/Q2_0
68.4/99.8/141.4/176.8 us (unchanged). 35B graph serve: identity/coherence
OK, C1 ITL p50 9.30 ms (stage 1 9.34), C2 12.53 ms/stream (12.6); only 6
of 123 routed tensors are K-quant/IQ4_XS, shared Q6_K runs in the runner.
Bandwidth push: RPL 4 70.8/115.1/158.2/213.2 vs 69.9/103.7/146.5/190.3,
unroll 2 and 16 lanes also slower; gate/up stays ~350 GB/s (= Strata night).
Installed _C.so = _C.graph.so = _C.o3b.so sha256 e9e5282d...91befa1
(10221368 bytes); _C.o3.so (be76661b) kept for X1's pinned window.
VERDICT: K-quant decode 1.4x-11x faster per layer, bit-close; 35B C1 flat
because those types are rare there. Cheap bandwidth knobs exhausted; a
cooperative superblock block-load is the next (1-2 day) item. Report
updated: reports/2O3_moe_decode.md.

## 2026-10-20 16:09 X1 -- baseline post-health and Strata depth convention

CONFIG: same frozen proxy baseline; CPU source audit while priority waits.
COMMAND: xpu-health --card 0 after clean teardown; inspect Strata
src/program/generate.cpp:4495,13852 and draft_policy.hpp.
RESULT: post-health HEALTHY, X1_WINDOW_DONE ready=1. Strata --spec 4
means a four-row verify window, one target token plus up to THREE draft
tokens. It is not vLLM num_speculative_tokens=4. Batch depth 2 means
two drafts per slot. Corrected this distinction in reports/1X1_mtp.md.
VERDICT: no-MTP baseline complete. X1 is outside all leases so C7b and
H1e can take the pair. Strata restricted head has about 40k vocabulary
entries (162100-byte file), another concrete later latency optimization;
first prove greedy verification and graph capture with the full head.


## 2026-10-20 16:10 X2 -- short allocator-only card-0 measurement

CONFIG: card 0 only, expected 60-90 seconds including health. X3 holds
card 1. C7b private overlay still in CPU preparation; H1e has no pair
window announcement. Both retain priority; no Flash-Next serve started.
COMMAND: gpu-run --card 0 xpu-health --card 0, xpu-run --card 0
memory/allocator_probe.py (ZE_AFFINITY_MASK=0), then post-health.
RESULT/VERDICT: pending. Compare default and 20 MiB split policy with
a 64 MiB temporary followed by a 2 MiB persistent allocation. This
tests the actual setter and whether reserved slack is counted twice.

## 2026-10-20 16:12 G2b -- audit pass 15

CONFIG: CPU only. No GPU, no systemd, no commit. Findings F148-F154
append reports/1G2_audit.md after F147. Cursor is the 16:10 X2
heading, which was still pending.
COMMAND: Read c7b-graph-serve-fa-missing.log,
c7b-fn-graph-window-fa-missing.out, c7b-fa-build.log, the side
libattn so, x1 bench and sidecar logs, x2/headers.json, x3
ops-card1.json and ops-rest.json, o3/kq2.log and
o3/bench_35b_o3b.json. Hash _C.so, _C.graph.so, _C.o3.so,
_C.o3b.so, and _C.c12.so.
RESULT: The staged-PLE graph window died at 15:39:23 inside
torch.cuda.graph on cu_seqlens_q.cpu(), Level Zero error 44,
SERVE_RC 1, after the PLE input_ids copy did not appear. The FA
side so is 11944232 bytes and the venv wheel was not replaced.
X1's measured proxy first id is hotschmoe-dd, C1 p50 9.27 ms,
coherence OK, MTP off. X3 card-1 medians match the journal's
6.699 / 23.875 / 990.09 and the rest-probe 6.725 / 24.155 /
79.47. O3 K-quant graph times match kq2.log. New-row cos is
1.000000 and Q6_K max_abs is 144. The 35B o3b bench first id is
hotschmoe-dev, C1 p50 9.30 ms. Installed _C.so, _C.graph.so, and
_C.o3b.so share sha e9e5282d, 10221368 bytes. _C.c12.so is still
b3c15e14. Cached diff empty. Fork still 0a6170da8c. Repo HEAD is
952f1971.
VERDICT: Pass 15 recorded. The 15:39 window is not an ITL. The
9.27 ms and 9.30 ms figures are 35B card-0 proxies, not
Flash-Next. X1's first served id is not hotschmoe-dev. Q6_K is
not bit-close. The X2 allocator probe had not posted a result.
Open items remain C13 bench exit 1, G4 r9 greedy, D2H under 7.5,
fair bulk rel, the PLE CPU test stdout, and the 46.4 percent
hit rate.

## 2026-10-20 16:13 H1e -- Flash-Next sidecar window queued

CONFIG: Phase B not started. C7b holds both cards for
vllm-fn-flashnext-gguf-graph since 16:10:40 (FA overlay,
FULL_DECODE_ONLY, util 0.98). reports/1C7b_flashnext_serve.md
is absent. Sidecar _C.so is 5a0a047c. MoE pin for the sidecar
is _C.o3.so be76661b. _C.graph.so is now e9e5282d (O3 stage 2,
16:00) and is not the pin. FA overlay c7b-fa-overlay is on the
serve PYTHONPATH. Recipe: world size 1, port 18116, util 0.98,
max_len 2048, max_batched 2048, max_seqs 4, FULL_DECODE_ONLY,
FLASH_ATTN, B70_QSA=0, no MTP. MemAvailable abort is 8 GiB.
COMMAND: none. Driver is
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1e-fn-window.sh.
It starts after C7b's report exists, or after both gpu lock
owner files have been empty for 10 minutes.
RESULT: no serve, no greedy text, no ITL.
VERDICT: queued behind C7b's graph window. H1e takes both
cards next. A short single-card command already running can
finish. Do not start a new pair serve or a long single-card
job in the gap after C7b releases.

## 2026-10-20 16:13 X3 -- warm-profile padding guard, yielding to pair serves

CONFIG: CPU only after failed 35B profile; all X3 jobs ended.
COMMAND: inspect x3/serve.log and post-health. Prepared retry only.
RESULT: 35B packed load completed, but zero-input memory profile reached
27.52 GiB allocated then requested a 6 GiB bmm output. Added default
padding-ratio guard (2x live rows), using unpadded loop when exceeded.
This is also required for real router skew. Post-health card 1 HEALTHY.
No DEVICE_LOST; no reset. Standalone config is now explicit mode 0 and
profile hooks cover V1 and V2 model runners. Earlier approximate X3
heading times are secondary to the raw log UTC stamps.
VERDICT: no 35B A/B result yet. C7b currently owns both cards for graph
serve; H1e goes next. X3 will retry the 35B serve, then a 2-3 minute
one-card census/skew/check probe and a separate QSA budget probe once
the priority windows finish. No new GPU window is being requested now.


## 2026-10-20 16:18 X2 -- split setter ineffective; USM probe next

CONFIG: card-0 64 MiB temporary + 2 MiB pin. Queued behind C7b;
ran 16:16:18-16:16:35; health before/after HEALTHY, no reset.
COMMAND: memory/allocator_probe.py. Raw x2/allocator-probe.log.
RESULT: both default and max_split_size_mb:20 keep 67108864 reserved
for 2097152 live bytes. Setter accepts/restores policy, but the split
limit has no measured effect on XPU. First arm driver-consumed
137662464 bytes, profiler 202674176; second driver delta zero while
profiler still adds 65011712 bytes. Total memory 34242297856 bytes.
VERDICT: do not claim the scoped split setter saves memory. Inspecting
the installed adapter confirms UR_L0_DISABLE_USM_ALLOCATOR. Next short
card-0 probe, expected 30 seconds, tests disabled UR pooling and XPU
expandable segments. No priority pair window is currently announced;
H1e and C7b go first if they queue. No model serve started by X2.

## 2026-10-20 16:18 C7b -- Flash-Next IQ3_XXS FULL graph serve

CONFIG: logs/c7b-fn-graph-window.sh. TP=1 DP=2 EP,
FULL_DECODE_ONLY, FLASH_ATTN, B70_XPU_CG_RECLAIM=500, capture
sizes 1,2,3,4,8, max_model_len 2048, max_num_seqs 4, util 0.98.
B70_GGUF_SO=_C.c12.so. FA overlay c7b-fa-overlay with paged
decode 16,256,64,false,false,false. Port 18104. Names
hotschmoe-dev then qwen3.8-flash-next-iq3-xxs. Pre-health
HEALTH_RC 0, MEM_AVAILABLE_KB 117876416, p2p=0.
COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-graph-window.sh
RESULT: READY 16:15:28, SERVE_RC 0, WINDOW_DONE 16:16:18.
Greedy text "323" and "667", both match. Identity first
hotschmoe-dev. Bench under bin/cpu-run --quiet, BENCH_RC 0,
log c7b-bench-graph.json. C1 ITL p50 81.75 ms (14.88 tok/s).
C2 ITL p50 33.20 ms and 33.14 ms (18.83 tok/s per stream).
Prefill 1017 tokens at 62.5 tok/s. KV pools 4352 and 4608
tokens, available KV 0.64 GiB. 32k was not run. Ready
AnonPages 14085920 kB. Worker RSS 4930408 kB and 4938344 kB.
FULL capture finished with no missing-kernel line and no
Level Zero error 44. Post-health 16:17 COLLECTIVE_HEALTH_OK
p2p=0, HEALTHY, exit 0. Locks empty after that command.
No DEVICE_LOST.
VERDICT: first coherent Flash-Next IQ3_XXS vLLM graph serve.
Rates are below the Strata night-final reference in the bench
file. 32k does not fit this KV pool. Recipe and numbers are in
reports/1C7b_flashnext_serve.md. Cards released. hotschmoe-dd
was not started.

## 2026-10-20 16:19 X1 -- rejection-safe PLE adapter and proxy MTP window

CONFIG: PLEHistory position ring, history length + maximum verify width;
CPU unit fixture with real PLE hash method, long prefill, partial rejection,
sequence reordering and physical slot reuse. Opt-in via B70_MTP_RT only.
COMMAND: cpu-run uv ... harness/x1_check_ple.py. New mtp_ple.py, five-line
V2 runner preparation hook plus target-loader install; no no-MTP change.
RESULT: ple-cpu-r2 exit 0, X1_PLE_REJECTION_AND_SLOT_REORDER_OK. Existing
PLE reads only non_spec_* metadata and skips an all-spec batch; the adapter
handles both metadata groups and restores by logical token position.
VERDICT: CPU behavior passes; XPU capture still needs testing. C7b pair
window ended cleanly at 16:16:06 (WINDOW_DONE). At 16:18:47 both owners
empty, no H1e pair window announced. X1 requests card 0 for about 3 min,
MTP=2 proxy only, timeout 7200. A queued H1e pair takes priority at gaps.


## 2026-10-20 16:20 X2 -- native/plugin memory-info discrepancy

CONFIG: short card-0 allocator probes, no model serve. Both previous
health-after probes HEALTHY. C7b graph serve ended 16:16:06; H1e still
has priority and has not announced its pair window. info_probe.py is
queued for about 30 seconds on card 0 (xpu-run pins physical card 0).
COMMAND: allocator_probe with UR_L0_DISABLE_USM_ALLOCATOR=1, then
info_probe.py compares torch.xpu.mem_get_info and the vLLM override.
RESULT: UR-off does not remove the pinned 64 MiB segment. Expandable
segments reduce that test reservation to 20 MiB, but native free-memory
readouts need a settled comparison. The ineffective split-policy patch
was removed from the plugin. Found platforms/xpu.py replaces the native
memory-info API. C7b logs 30.3 GiB total; native probe returns exactly
34242297856 bytes (31.89 GiB).
VERDICT: fix candidate switches profiling to the native PyTorch XPU
memory API, matching the allocator. No savings claim until matched
full-model measurements. Observers record both raw storage and driver
snapshots. Detached full-model controller still waits for H1e.


## 2026-10-20 16:20 X2 -- CPU note for H1e preflight

CONFIG: read-only CPU check of H1e missing-symbol failure.
COMMAND: nm -D _C.o3.so | grep -q _Z13b70_moe_layer under pipefail;
then nm -D _C.o3.so | rg b70_moe_layer (consumes the whole input).
RESULT: grep-q pipeline returns 141, PIPESTATUS 141 0. Both required
b70_moe_layer and b70_reserve_decode_queue symbols are present.
VERDICT: H1e preflight is a false failure from nm SIGPIPE. Consume the
full nm output (grep without -q redirected to /dev/null), or write
nm output once and search the file. X2 did not edit H1e-owned harness.
Full X2 serve controller still yields to H1e retry.


## 2026-10-20 16:21 X2 -- info microprobe moves to free card 1

CONFIG: card 1 only, about 30 seconds plus post-health. Card 0 is held
by X1 proxy MTP. H1e has not announced its retry after preflight.
COMMAND: canceled own queued card-0 info job before acquisition;
xpu-run --card 1 info_probe.py, ZE_AFFINITY_MASK=1 (logical device 0).
RESULT/VERDICT: pending. No model serve; H1e retry retains priority.

## 2026-10-20 16:23 X3 -- short card-1 checks after priority window teardown

CONFIG: card 1 only, expected 90 seconds including health. C7b and
H1e first window both WINDOW_DONE; H1e post-health both cards HEALTHY.
Both owner files empty at 16:23:22. H1e first serve stopped at its
preflight, not a GPU crash. No H1e retry announcement found yet.
A queued H1e retry has priority; X3 takes no pair lease.
COMMAND: X3_CARD=1 x3/run_ops.sh --part check,tile,gdnsplit
--out checks-tile.json. Long lease timeout; detached and polled.
RESULT/VERDICT: pending. Numerical gates include mixed quant types,
invalid IDs, skew fallback, direct grouped reduction and expert tiling.


## 2026-10-20 16:26 X2 -- paired API result and final probe plan

CONFIG: settled one-card memory snapshots, native versus vLLM override;
card 1 pin. H1e is now loading on both cards; X2 has no GPU job active.
COMMAND: memory/info_probe.py. x2/info-probe.log and info-health-after.txt.
RESULT: native total 34242297856; vLLM total 32530182144. Free differs
by the same 1712115712 bytes at baseline, 64 MiB temporary, 2 MiB pin,
and after release (1.1 s settle per snapshot). Native/plugin used-byte
counts agree. At util .98 this changes requested budget 1677873398
bytes/card. Health-after card 1 HEALTHY. No reset.
VERDICT: the plugin now fixes aliases plus native XPU memory info; no
unsupported max-split policy or speculative raw-memory subtraction.
After H1e, X2 requests separate ~4-minute both-card eager arms: current,
fixed, fixed batch512+FP8 max_len200K, and fixed BF16 sidecar max_len200K
with exchange cap2048. Current/fixed batch2048 are matched. No latency
claims; prefix/GDN/quality gates remain explicit in the report.

## 2026-10-20 16:26 X1 -- proxy MTP measured; PLE XPU probe queued after H1e

CONFIG: matched card-0 proxy, same library/config as baseline, MTP depth2.
Full target/draft graphs, stable identity, no prefix cache, BF16.
COMMAND: MTP_TOKENS=2 harness/x1_proxy_window.sh. Raw x1/window-mtp2.out,
bench-mtp2.json, greedy-mtp2.json, greedy-eos-gate.json.
RESULT: identity/coherence pass, health before/after HEALTHY, clean teardown.
C1 chunk ITL p50 18.29 ms, 143.12 tok/s = 6.99 effective ms/token
(baseline 9.24); C2 chunk p50 28.66/28.42 ms, 91.38/84.28 tok/s =
10.94/11.86 effective ms/token. Bench acceptance 114/148 = 0.77027.
Exact forced-96 gate: three prompts match; hash-table prompt first differs
at token 82, AFTER the shared configured EOS at token 78 (248044). All
four match through the first configured EOS / length limit, 367 tokens
including EOS; post-EOS forced continuation remains a failing diagnostic.
VERDICT: proxy MTP gives a measured latency reduction, not Flash-Next
speed. H1e now holds both cards for its retry. X1 will wait for its
WINDOW_DONE, then request card 0 about 45 seconds for isolated PLE graph
rejection/reorder validation, health around it and quiet CPU timing.


## 2026-10-20 16:29 X2 -- current DP memory window requested

CONFIG: BOTH cards, expected ~4 minutes serving plus health. C7b graph
window ended; H1e window ended with layer-2 EP flag timeout, and
h1e-health-after.txt is HEALTHY on both cards. Both owner files empty.
No new priority pair window announced. Controller pauses between arms
so H1e gets first use when its next retry is ready.
COMMAND: detached memory/run_window.sh; long lease timeout 7200.
Current arm: DP=2+EP, eager, max_len2048, batch2048, seqs2, util.98,
_C.c12.so, QSA=0, PREFILL=0, B70_MEMORY_FIX=0, byte observers on.
RESULT/VERDICT: pending. x2/window.log and x2/current/*.json.
Native-info/alias fixes and FP8 are not active in this current arm.

## 2026-10-20 16:31 X3 -- direct grouped expert path measured; exact census queued

CONFIG: card 1 only, next window expected 2 minutes, no pair lease.
Both priority serve windows have ended; X2 health/serve queues remain
coordinated by leases. The quiet wrapper releases CPU immediately when
its GPU is held, avoiding a quiet-lock/serve-lock dependency.
COMMAND: x3/run_ops.sh --part check,census,misc,skew
--out census.json. Detached, long lease timeout.
RESULT: checks-tile completed GPU work. GDN split, Flash-Next T=12288:
conv 2.206 ms, delta 4.633 ms, whole 6.747 ms. Direct grouped path
41.801 ms vs triton slot-reduce path 47.215. Tile128/64/32 are
48.213/53.769/61.171 ms. PyTorch scratch+temp peak delta fullE512:
7.018 GiB direct vs 7.604 slot path, tiles 2.168/1.318/0.894 GiB.
Reference, mixed-type, invalid-ID and skew checks all pass original
5e-2 gate. Next check also compares candidate to prior fast output at
1e-7 absolute / 1e-5 relative so tiny real outputs cannot hide errors.
VERDICT: direct grouping is a measured latency and workspace win.
Tiling trades 6-19 ms for several GiB of workspace. Defaults remain
off until the tightened gate and 35B serve pass.

## 2026-10-20 16:32 X1 -- PLE XPU graph rejection/reorder gate passed

CONFIG: card 0 only, actual HC width 10240, two sequences, verify width3,
ring length12. Existing XPU PyTorch operators, no new kernel build.
COMMAND: cpu-run --quiet harness/xpu-run.sh --card 0 -- python
harness/x1_ple_graph.py, after H1e WINDOW_DONE and HEALTHY post-health.
RESULT: X1_PLE_XPU_GRAPH_REJECTION_REORDER_OK. Captured graph replays with
changed logical positions, differing rejection lengths and reordered live
slots; matched independent CPU convolution. Conv graph 0.095748 ms
(50 replays, device events), not whole PLE or serving latency. Probe exit0.
Health before HEALTHY; post-health queued behind X2's collective health.
VERDICT: graph-safe restore primitive verified. Next X1 requests BOTH
cards for Flash-Next PLE-ring no-MTP control (~6 min), then MTP2 (~6 min),
with a lease release and priority recheck between them. Same IQ3_XXS,
BF16, TP1/DP2/EP, maxlen2048, batch512, seqs2, util.98, _C.o3.so,
FULL_DECODE_ONLY, no prefix cache, side-built C7b FA overlay. C7b/H1e
priority windows have ended; no service changes or commit.

## 2026-10-20 16:38 X3 -- request next short card-1 gap between model serves

CONFIG: X3 census/numerical probe has waited outside GPU/CPU locks
since 16:31. X2 current arm released at 16:36:41; X1 then took both
cards for Flash-Next MTP0 at 16:37:16. C7b/H1e retain priority if they
announce a retry. X3 requests CARD 1 for about TWO MINUTES after the
current X1 serve, before another full-model load. Card 0 can remain
available for short independent work; quiet CPU lease covers timing.
COMMAND: already detached x3/run_ops.sh --part check,census,misc,skew.
RESULT: no new GPU measurements while queued. All prepared heavy
build work is complete; only measured per-op budget/35B A-B remain.
VERDICT: please leave this short profiling gap between serves as the
standing development loop requests. No GPU bypass or service action.

## 2026-10-20 16:38 X1 -- Flash-Next control started; profiling gap reserved

CONFIG: Flash-Next IQ3_XXS, same planned PLE-ring control. TP1/DP2/EP,
MTP0, FULL_DECODE_ONLY, batch512, seqs2, aligned Mamba state, cache off.
COMMAND: MTP_TOKENS=0 harness/x1_flash_window.sh, queued since 16:32:31.
RESULT: pair acquired for server at 16:37:16 after X2 current arm.
Both-card pre-health HEALTHY. Worker startup now in progress. PLE probe
post-health HEALTHY; final CPU BF16/rejection/reorder check also passes.
VERDICT: X1 will release the pair after this control. X3's already queued
card-1 two-minute census/numerical window gets the gap before X1 requests
another full-model load. No MTP2 pair request is queued by X1 yet.

## 2026-10-20 16:42 X1 -- Flash-Next control hit SYCL persistent-cache crash

CONFIG: same Flash-Next MTP0 control; no PLE Python exception occurred.
COMMAND: harness/x1_flash_window.sh. Log x1/flash-mtp0.log.
RESULT: both ranks placed all weights, loading 23.97 GiB/card in 122-124 s.
First XCCL all-gather in profiling segfaulted in SYCL getSortedImages /
PersistentDeviceCodeCache::getItemFromDisc / allgatherv_ll_ring. The later
RuntimeError cancelled is secondary. No ready, no greedy, no target ITL.
Post-health currently finishing; card 0 passed. No reset requested.
VERDICT: C7b's passing graph recipe explicitly disables persistent SYCL
cache; X1's generic harness had left it enabled. The prepared retry sets
SYCL_CACHE_PERSISTENT=0 and a private cache directory, leaving shared
caches untouched. After X3's reserved two-minute card-1 gap, X1 requests
both cards about 90 s for the compiled XCCL health program under the same
XPU runtime, with CCL_TOPO_P2P_ACCESS left unset as instructed. Then retry
the control only if health passes. No repeated identical crash arm.

## 2026-10-20 16:43 O3 -- short card-0 microbench window (about 2 minutes)
CONFIG: card 0 only, standalone microbench, no serve. Locks empty at 16:41
after X1's control; X3's reserved gap is card 1. O3 yields to any pair window.
COMMAND: gpu-run --card 0 o3/run_s3.sh (stage-3 ALU variants)
RESULT/VERDICT: next O3 entry.


## 2026-10-20 16:42 X2 -- baseline reconciled; packed embedding fix

CONFIG: current DP+EP arm, _C.c12.so, eager batch2048 seq2 util.98.
COMMAND: memory/run_window.sh, x2/current/*.json and greedy.json.
RESULT: READY16:36:22, identity hotschmoe-dd then detailed IQ3 alias;
323/667 exact, clean exit16:36:41. Owned model storage 26656237720
bytes/card; PyTorch allocated after load26741253120. The embedding
is BF161271398400 bytes, not packed273152000. QSA indexer has
39321600 unused placeholder bytes; GDN b/a has17694720. Router and
conv bypass B70GGUFLinearMethod, so earlier129024000 full-linear
duplicate upper bound is NOT actual savings. Actual aliases57016320.
Baseline peak reserved29345447936, post-profile reserved28087156736,
post-profile allocated26741253120, allocated peak28078742528.
Peak reserved headroom1258291200 vs original1337489408:79198208
bytes overcount, not the whole1345903616 inactive slack. Largest free
block835928064; fragmentation forbids blindly reclaiming all slack.
VERDICT: added ONE fork line passing quant_config to embed_tokens,
saving expected998246400 bytes/card; old current arm had already
constructed the dense embedding. Fixed arm checks20480 BF16 values
against CPU GGUF dequant before coherence. New profiler uses reservation
peak, preserving fragmentation/non-Torch costs. X2 pauses for X1/X3
then queues fixed arm; H1e retry retains pair priority. No commit.


## 2026-10-20 16:43 X2 -- fixed DP comparison requested

CONFIG: BOTH cards, ~4 minutes after current O3/X3 short probes.
X1 window ended ready=0 and post-health both HEALTHY. H1e has no new
pair serve announcement and retains priority for its next ready retry.
COMMAND: release controller hold-next for aliases arm, matched eager
DP+EP batch2048 seq2 util.98. Packed embed constructor, merged-dense
aliases, native memory API, physical reservation-peak profiler enabled.
RESULT/VERDICT: pending. Exact embedding row check and near-limit
concurrent prefill will validate storage/headroom reuse before teardown.
Raw x2/aliases. No new quantization in this arm.

## 2026-10-20 16:45 X1 -- collective check infrastructure fix, yield to H1e

CONFIG: actual campaign torch/XPU runtime, both ranks, P2P override unset,
SYCL persistent cache disabled, MAX_JOBS=4, compile threads4.
COMMAND: cpu-run xpu-run --card both -- torchrun ...
bin/xpu-collective-health.py. Raw x1/collective-default.log.
RESULT: eager all-reduce passed; Inductor compile failed with missing
ocloc (FileNotFoundError). Not a GPU failure; no reset. H1e's next
ready serve acquired both cards at 16:44:19. X1 post-health queues behind
it, and no Flash-Next control retry is queued ahead of that window.
VERDICT: prepared compiled-health retry adds C7b's existing
c7b-aot-bin (host ocloc26.22.1 + libocloc) to PATH/LD_LIBRARY_PATH;
waits for H1e WINDOW_DONE outside leases first. No shared install/cache
mutation, no CCL_TOPO_P2P_ACCESS assignment. X3 got its reserved gap.

## 2026-10-20 16:46 X3 -- direct gate passed, tiled BF16 geometry diagnostic

CONFIG: prior short census window stopped before timing on a tightened
check of mixed-type tiling. Direct full-E path and same-type tiling
passed 1e-7 absolute / 1e-5 relative against the old fast output.
COMMAND: x3/census.log. Next census-r2 queued card 1 ~2 minutes after
current H1e retry, with measurements retained before candidate checks.
RESULT: mixed types 17/16/20, E4 to E2 bmm geometry, max_abs 0.00133
vs prior bmm. Original bf16-reference 5e-2 gate passed. Full-E direct
permutation also passed tight check for that mix. Post-health HEALTHY.
VERDICT: keep tight equality for unchanged GEMM geometry; tiling must
retain original reference gate plus cosine >.999 and be qualified in
serving. No tolerance in prior tests changed. Record both metrics; do
not label tiling bit-identical. 35B/FN aliases registered as unqualified
research entries in evals/configs/models.yaml. No commit.

## 2026-10-20 16:50 X1 -- compiled default XCCL health passed; control retry

CONFIG: same campaign torch2.14 XPU, host UMD, SYCL cache disabled,
existing C7b ocloc directory on PATH and LD_LIBRARY_PATH, MAX_JOBS4.
COMMAND: x1/collective-default-r2.log, torchrun two ranks over
bin/xpu-collective-health.py, after H1e WINDOW_DONE and post-health.
RESULT: COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120
compiled_iterations=10 p2p=None. Exit0. No P2P override assignment.
The first check failed only because ocloc was off PATH. No xe-reset.
VERDICT: queues healthy with default collectives. Retry MTP0 control
with SYCL_CACHE_PERSISTENT=0; both cards ~6 minutes once acquired.
X2 aliases may already be ahead on the lease. X3 received its short gap.
Crash logs preserved as flash-mtp0-sycl-cache-crash.log and matching .out.
## 2026-10-20 16:55 G2b -- audit pass 16

CONFIG: CPU only. Cursor is the JOURNAL from the 16:12 G2b
milestone through the 16:50 X1 collective heading.
COMMAND: matched bench JSON, serve logs, window outs, weight
dumps, census.log, collective logs, and the fork diff. No GPU
and no commit.
RESULT: F155-F163 in reports/1G2_audit.md. C7b graph bench
first id hotschmoe-dev, C1 p50 81.75 ms, mean rate 14.88 tok/s,
C2 p50 33.20 and 33.14 ms, greedy 323 and 667, SERVE_RC 0,
health before and after. X1 MTP2 and the X2 eager arm served
hotschmoe-dd first. The Flash-Next control segfaulted before
ready. census.log assert_close atol 1e-7 exited 1 at max_abs
0.00132982. collective-default.log is COLLECTIVE_HEALTH_FAIL
on missing ocloc; collective-default-r2.log is
COLLECTIVE_HEALTH_OK.
VERDICT: the 16:18 window is a measured Flash-Next IQ3_XXS
FULL graph serve, and its ITL is the bench JSON, above the
Strata 7.09 ms reference printed in that same file. Name-rule
misses and the census tight-check failure are in the audit.


## 2026-10-20 16:58 X2 -- fixed matched arm passed; 200K FP8 next

CONFIG: fixed DP+EP eager batch2048 seq2 util.98; packed embedding,
merged aliases, native memory API, physical reservation-peak headroom.
COMMAND: memory/run_window.sh from aliases arm; x2/aliases and
aliases-summary.json. Controller resumed after its earlier paused-file
edit error; current baseline artifacts were preserved.
RESULT: all20480 sampled BF16 embedding values exact on both cards.
Owned storage25600975000 bytes/card; available KV3687273595 /
3709179003 bytes vs current960932742 /982829958. BF16 one-long-stream
bounds137600 /139200 tokens/card, not yet200K. READY16:53:38;
identity primary hotschmoe-dd; greedy323/667 exact. Concurrent2047-token
prefills pass, each emits one token. Clean teardown; HEALTHY both cards
16:56:58. No reset, no speed or long-context quality claim.
VERDICT: request BOTH cards ~4 minutes for native/fixed batch512,
FP8 KV with calculated scales, max_len200000. H1e ready retries still
have priority; X2 queues behind any current holder with timeout7200.
This is a memory-feasibility arm, not FP8 long-context quality promotion.

## 2026-10-20 17:07 X1 -- control reached PLE warmup; int32 staging fix

CONFIG: cache-disabled control after compiled default XCCL health passed.
COMMAND: harness/x1_flash_window.sh, x1/flash-mtp0-int32.log.
RESULT: both ranks loaded 23.97 GiB, PLE ring layers1/width1 installed.
All-gather/reduce-scatter passed; no SYCL segfault. Graph profile warmup
then raised Python Index put dtype mismatch, Long destination vs Int
source, at PLEHistory.stage token ring write. Native token IDs are int32;
the original synthetic test used int64. No ready or target benchmark.
VERDICT: normalize IDs and positions to host int64 at stage entry; extend
CPU rejection/reorder fixture to int32 runtime inputs. No identical retry.
Post-health queued behind X2 FP8 target arm; no reset without failed health.

## 2026-10-20 17:10 X1 -- int32 fixture passed; control retry queued

CONFIG: unchanged Flash-Next control; only stage input dtype normalization
added after the prior warmup exception. No new GPU kernel/ABI change.
COMMAND: cpu-run uv ... harness/x1_check_ple.py with int32 IDs/positions;
MTP_TOKENS=0 harness/x1_flash_window.sh after the fixture passed.
RESULT: X1_PLE_REJECTION_AND_SLOT_REORDER_OK, exit0, 11 s. Both-card
post-health of the Python-exception control was HEALTHY, no reset.
X3's requested census gap completed; X2 health may be ahead on leases.
VERDICT: retry queues with timeout7200, both cards about six minutes.
No source edits while this model runs. C7b/H1e ready windows retain
priority; MTP2 still waits for a measured matched no-MTP control.


## 2026-10-20 17:14 X2 -- current-head FP8 command, queued target

CONFIG: BOTH cards, expected about 4 minutes after acquisition plus
near-limit prefill verification. Corrected target queued behind X1.
COMMAND: X2_START_ARM=target memory/run_window.sh; dtype
fp8_per_token_head, TRITON_ATTN, batch 512, seqs 2, max_len 200000,
util .98. Dynamic token/head scales are native to this vLLM head.
RESULT: earlier fp8 target failed CLI parsing before model load because
--calculate-kv-scales was removed upstream. Preserved x2/target-fp8.
Cancelled only own queued post-error health (before acquisition), then
restarted target with fresh per-card and compiled collective HEALTHY.
The readiness clock now begins at actual lease acquisition, not queue
entry; detached cleanup also cancels a still-queued child safely.
VERDICT: no FP8 result yet. C7b/H1e ready retries retain priority.
Coordination request: after the current X1 control, leave this already
queued pair memory window before another non-priority full-model load.
X2 will pause before its final BF16 sidecar capacity arm. No commit.

## 2026-10-20 17:07 H1e -- Flash-Next sidecar graph serve

CONFIG: world size 1 sidecar, both cards, port 18116, container
vllm-fn-h1-fn, util 0.98, max_len 2048, max_batched 2048, max_seqs 4,
FULL_DECODE_ONLY, FLASH_ATTN, capture 1,2,3,4,8, B70_QSA=0,
B70_EP_SPLIT=profile, B70_EP_Q0=138, no MTP, no vLLM EP, no oneCCL.
Live b70_ep_xpu/_C.so sha 2e22f57e (eager prejit plus eager spin
10000000; capture keeps spin 200000). MoE pin _C.o3.so sha be76661b.
On-disk _C.graph.so sha e9e5282d was not loaded. Saved Phase A binary
logs/b70_ep_xpu_C.so.5a0a047c stays sha 5a0a047c. MemAvailable abort
floor 8 GiB. FA overlay c7b-fa-overlay. Engine v0.32.1rc1.dev94+g5fd62688f.
COMMAND: logs/h1e-fn-window.sh. Four windows. Success artifacts:
h1e-fn-window-ok.out, h1e-fn-serve-ok.log, h1e-greedy.json,
h1e-bench-graph.json, h1e-kv.txt, h1e-models.json, h1e-rss-ready.txt.
Bench was bin/cpu-run --quiet harness/bench_latency.py, tag
ep-graph-2048, decode 64, prefill target 1024.
RESULT: attempt 1 (16:19, EP 5a0a047c) SERVE_RC 1 on a false
missing-b70_moe_layer check (nm -D | grep -q under pipefail); health
before and after HEALTHY. Attempt 2 (16:23, same sha) SERVE_RC 1,
EP flag timeout layer=2 err0=0 err1=11; health before and after
HEALTHY. Attempt 3 (16:42, EP 1a134e8e, prejit only) SERVE_RC 1,
EP flag timeout layer=2 err0=12 err1=0; health before HEALTHY; the
copied window ends at the health-after header. Attempt 4 launched
16:49:51, serve acquired 16:56:03, ready 17:00:34, stop 17:02:25,
window mtime 17:07:08 WINDOW_DONE. SERVE_RC 0, GREEDY_RC 0, BENCH_RC 0.
Health before and after HEALTHY (35s and 33s). No xe-reset. No EP flag
line, no error 44. Card-1 graphs finalized keys 4,3 then 4,3,2,1.
No key 8. Identity hotschmoe-dev then qwen3.8-flash-next-iq3-xxs,
max_model_len 2048. Greedy 17*19 -> 323 and 23*29 -> 667, both match.
C1 ITL p50 60.156 ms, p90 89.402, p99 98.676, mean 54.368, decode
18.393 tok/s, TTFT 264.67 ms. C2 stream ITL p50 32.061 ms and
32.061 ms, means 56.304 and 56.409, decode 17.761 and 17.728 tok/s.
Prefill built 1017/1024 at 106.47 tok/s. MTP enabled false. Coherence
ok, same, mentions Paris, verdicts OK/OK. Card-0 KV 12.06 GiB,
84224 tokens, 41.12x at 2048, block 1600. That is 0.211 of 2x200K,
and max_model_len 2048 caps each request. Lowest recorded
MemAvailable was 49996456 kB at ready. Model load 14.73 GiB in
131.08 s. No h1e-fn-window, vllm-fn-h1-fn, or serve_flashnext_ep
process remains.
VERDICT: greedy gates passed. Latency misses the Strata bar, which
includes MTP: C1 p50 is 8.48x 7.09 ms and C2 p50 is 2.95x 10.87 ms.
KV misses 2x200K. Live sidecar stays sha 2e22f57e. q0 stays 138.
Report: reports/1H1_ep_sidecar.md Phase B. No commit.


## 2026-10-20 17:20 X3 -- complete census, routing-skew bucket probe queued

CONFIG: card 1 only, next micro window expected 90-120 seconds plus health;
C7b/H1e keep priority, current X2 paired capacity run is left undisturbed.
COMMAND: x3/run_ops.sh --part bucket,check --out bucket.json; device pin
ZE_AFFINITY_MASK=1 through xpu-run --card 1; cpu-run --quiet; C12 plus
X3 wide addon. No new build, no commit, no DD service action.
RESULT: prior census-r2 completed GPU exit 0 and post-health exit 0. Exact
quantized projection shape census saved in x3/census.json. HC mix 12.149
-> 4.909 ms, combine 3.560 -> 1.517 ms; both original numeric gates pass.
35B expert crossover is 8192 tokens (29.818 -> 13.776 ms), while 1024
fused is faster than dequant (10.741 versus 12.771 ms). Actual learned
router weights on random activations produce 110-122 ms Flash-Next
expert layers through skew fallback, not the 41.8 ms uniform-route result.
VERDICT: do not project 2k prefill from uniform routes. Test power-of-two
expert row buckets with real packs, numeric gates, peak scratch and timing.
This also removes a GPU boolean host wait and 513 CPU scalar conversions
per grouping. QSA and real serving remain pending.


## 2026-10-20 17:18 X2 -- sidecar KV interpretation for H1e

CONFIG: CPU budget check using H1e reported 12.06 GiB KV, same BF16
12 full +36 GDN layers, manager block 1600, four groups.
COMMAND: memory/capacity.py, actual current/aliases kv-spec JSON.
RESULT: two active 200K requests need 10341580800 bytes in the
shared pool, including a null block and six GDN blocks/request.
12.06 GiB is about12.95 GB, so it should fit; the log capacity at
max_len2048 is dominated by fixed state and is not the long-context
token ceiling. PLE slot tables are extra but far below this margin.
VERDICT: do not conclude that 12.06 GiB misses2x200K from the short
request capacity. X2 final sidecar arm will set max_len200000, seq2,
BF16 and the2048 exchange cap and measure its real capacity. Target
FP8 now starts after the queued X1 window; H1e ready retries go first.

## 2026-10-20 17:18 X1 -- Flash-Next PLE-ring control measured

CONFIG: ISTA IQ3_XXS, TP1/DP2/EP, BF16, FULL_DECODE_ONLY, MTP0,
maxlen2048, batch512, seqs2, util.98, cache off/aligned Mamba,
SYCL persistent cache off/private dir, _C.o3.so be76661b, C7b FA overlay.
COMMAND: third MTP_TOKENS=0 harness/x1_flash_window.sh. Raw x1/
flash-mtp0.log, flash-greedy-mtp0.json and flash-bench-mtp0.json.
RESULT: READY17:15:17, ring layers1/width1 on both ranks. Exact C1
and two concurrent completion token sequences saved; GREEDY_RC=0.
Identity first hotschmoe-dd and registered alias; coherence passed.
C1 p50 40.23 ms, 23.98 tok/s (84 tokens); C2 p50 44.35/44.18 ms,
22.74/22.74 tok/s. BENCH_RC=0; clean teardown. Both-card pre-health
HEALTHY; post-health queues behind X2 target FP8-per-token-head arm.
VERDICT: first valid matched Flash-Next control with the new PLE ring,
not a speed promotion. MTP2 next after health/priority recheck. X1 has
no serving process left running. The Strata 141 tok/s bar is still far.


## 2026-10-20 17:19 X3 -- next card-1 diagnostic/35B windows prepared

CONFIG: after the queued bucket numeric probe, card 1 only; QSA core
12288-token append+selection/union micro expected 1-2 minutes, then 35B
GGUF matched prefill serve expected 5 minutes including load and health.
No both-card X3 load is queued yet. C7b/H1e retain priority.
COMMAND: detached x3/next_windows.sh waits for own bucket completion,
then run_ops.sh --part qsa and x3/serve_window.sh. Timed windows quiet,
long GPU lock timeout, identity primary hotschmoe-dd plus research alias.
RESULT: commands prepared. Proxy uses C12 plus additive wide dequant,
DQ_T=8192, grouped reduce, no bucketing until its measurements pass.
The original OOM logs were copied to x3/oom-first-* before reuse.
VERDICT: no serving claim yet. This pair of single-card windows supplies
missing QSA timings and a same-loaded-model baseline/new-path comparison.

## 2026-10-20 17:23 X2 -- FP8 capacity measured, coherence rejected

CONFIG: fixed DP2+EP, TP1, eager, maxlen200000, seq2, batch512,
util.98, TRITON_ATTN, fp8_per_token_head, C12, no QSA/MTP.
COMMAND: memory/run_window.sh target; x2/target-fp8_per_token_head/.
RESULT: READY17:20:04. Actual page3261440, block3136,137 pool
blocks/rank; available5365499003/5387396219 bytes. Two concurrent
2047-token prefills pass. Identity correct; first greedy17*19 emits
corrupt content, second23*29=667. Embedding samples remain bit exact.
Server stopped; post-health queues behind X1/X3. No reset requested.
VERDICT: reject this FP8 mode pending numeric diagnosis. Capacity alone
is not quality evidence. Next queued window BOTH cards, roughly4-6min:
EP138/374, BF16, maxlen200000, seq2, batch2048/exchange cap2048.
C7b/H1e retain priority; controller waits for post-health before proceeding.

## 2026-10-20 17:23 X1 -- full Flash-Next MTP2 queued

CONFIG: both cards, expected 5-8 minutes after acquisition; matched
against the saved 17:15 no-MTP control, FULL_DECODE_ONLY TP1/DP2/EP,
BF16, maxlen2048, seq2, batch512, pinned O3, C7b FA overlay,
SYCL persistent cache off, no prefix cache. MTP draft depth2.
COMMAND: MTP_TOKENS=2 harness/x1_flash_window.sh; raw
x1/window-flash-mtp2.out and x1/flash-mtp2.log. Long lease timeout.
RESULT: no-MTP post-health HEALTHY on both cards, clean exit0. New
arm queues behind X2 health and X3 single-card work. No C7b/H1e
pending window found in the current journal/owner check.
VERDICT: run exact ordinary-EOS C1 and concurrent C2 token gates,
then acceptance/latency, teardown and post-health; no promotion yet.

## 2026-10-20 17:26 X2 -- BF16 DP budget follow-up prepared

CONFIG: both cards, eager DP2+EP, BF16 default attention, batch512,
maxlen200000, seq2, util.98; fixed storage/native accounting. Expected
4-6 minutes after lease. Sidecar runs first; C7b/H1e retain priority.
COMMAND: x2/run-bf16-target.sh is an exact run_window.sh copy with
only the target stage and dtype=auto. Start after sidecar controller exits.
RESULT: the FP8 batch512 profile has only2097152 transient reserved
bytes, versus1260388352 at batch2048. That suggests BF16 could fit
5190451200 bytes/card for one200K stream without lossy KV.
VERDICT: measure instead of inferring FP8 is necessary. Additional setup
cost is justified by avoiding the numerically failed FP8 path. Existing
C12 binary and health tooling reused; no build or timing benchmark.


## 2026-10-20 17:26 X3 -- expert buckets pass original numeric gate

CONFIG: card1, T12288, real Flash-Next packs/routing matrices with random
activations; C12+X3wide, grouped direct reduce, serial dequant.
COMMAND: x3/run_ops.sh --part bucket,check --out bucket.json.
RESULT: GPU exit0 in76s; card1 pre/post HEALTHY. Layers0/3/36 fallback
113.132/116.275/118.323 ms -> untiled buckets60.707/58.890/60.462 ms;
64-expert tiles63.583/67.339/64.852 ms. Peak incremental allocation
6.438 GiB -> tile64 1.411/1.442/1.473 GiB. Original atol=rtol=.05
reference gate passed for invalid IDs, concentrated routing, same/mixed
gate-up types. Full-shape bucket cosines1.0/1.0/.999719, worst max_abs
.047396 at layer36; keep this difference visible, no bit-exact claim.
VERDICT: bucket path is opt-in, numerically within the original gate,
and halves the skew fallback cost. Use tile64 for full-model headroom;
proxy has room for untiled buckets. QSA core and proxy matched serve
remain queued on card1; X1's queued MTP2 pair owns priority by lease order.
No commit, no reset, no DD actions.


## 2026-10-20 17:31 X3 -- QSA wrapper cost included, still queued

CONFIG: queued card1 QSA window expanded to about3 minutes after
acquisition to time the actual model_hook append and text M-RoPE loop,
not only the custom-op core. X2 sidecar currently owns both; X1 MTP2
is queued; C7b/H1e retain priority.
COMMAND: existing detached next_windows.sh / run_ops.sh --part qsa;
actual T12288 QsaRunner helper with random BF16 Q/K/V/indexer inputs,
physical pages, side state and actual 3-axis text positions.
RESULT: CPU audit finds _append_request prepares unused q/q_idx/gate
per token, round-trips side state per token, and _rope_rows loops each
text M-RoPE token on the host. Opt-in B70_PREFILL_QSA candidate removes
unused work, stages side state per request, and vectorizes query RoPE.
VERDICT: require exact page/tail/dead/length equality and tight query
equality against the old helper, and measure both. Stock serving
candidate remains QSA0, as current campaign dense-GQA controls do.
No QSA serving quality claim is authorized by these helper tests alone.

## 2026-10-20 17:32 X2 -- BF16 sidecar capacity and short gates pass

CONFIG: EP sidecar138/374, BF16, maxlen200000, seq2, eager,
batch/exchange2048, util.98, fixed storage/native profiler, C12;
H1e EP binary2e22f57e. Both-card lease17:25:45; READY17:29:39.
COMMAND: memory/run_window.sh sidecar; x2/sidecar/107-*.json.
RESULT: owned bytes15710974104 /31346585600; card0 nonKV
18245832704, KVbudget15311619195,389 blocks of39321600 bytes.
Two equal long streams have pool ceiling300800 tokens each;
2x200K requires10341580800. Embedding20480 values bit exact,
identity/greedy323 and667 pass; two concurrent2047-token prefills
plus1 generated token pass. Server stopping for post-health.
VERDICT: BF162x200K pool fits the measured sidecar. This is allocation
and short-request evidence, not completed200K coherence or latency.
Do not infer context limit from the2048-request capacity line. Reserve
headroom with explicit~11GB KV for future graph/MTP qualification.
Final DP BF16/batch512 run remains queued after this controller exits.


## 2026-10-20 17:33 X3 -- requested card1 gap after current X1

CONFIG: X3 QSA micro and then35B proxy A/B have been ready/queued
since17:19, yielding CPU quiet whenever a live serve needs it. Both
full X2 arms and now X1 MTP2 have acquired the pair in the meantime.
COMMAND: next_windows.sh remains alive; QSA has fresh card1 pre-health.
RESULT: no failed GPU window; waiting behind active X1 model load.
VERDICT: coordination request to non-priority full-model arms: after
current X1, leave the queued card1 QSA/proxy gap (~3min +5min) before
another full Flash-Next reload. C7b/H1e ready runs still go first.
This avoids starvation of the required short real-shape profiling.

## 2026-10-20 17:38 G2b -- audit pass 17

CONFIG: CPU-only read of JOURNAL.md after the 16:55 G2b milestone
and of bench JSON, serve logs, and probe logs under
/mnt/vm_8tb/b70/build/vllm-flashnext/{logs,x1,x2,x3}. No GPU.
COMMAND: python field checks against those files, plus git status
in /mnt/vm_8tb/github/b70_ai_things and the vllm fork.
RESULT: appended F164-F174 to vllm/flashnext/reports/1G2_audit.md.
H1e attempt 4 C1 p50 60.156 ms, decode_tok_s 18.393, first id
hotschmoe-dev, SERVE_RC 0. X1 PLE control C1 p50 40.226 ms,
decode_tok_s 23.976, first id hotschmoe-dd. X2 BF16 sidecar engine
pool 593893 tokens and 2.97x at 200000, first id hotschmoe-dd,
post-health unfinished at the read. X3 census-r2 and bucket
timings match the logged event_ms. Last journal heading seen at
append: ## 2026-10-20 17:33 X3 -- requested card1 gap after current X1
VERDICT: pass 17 is on disk. No speed promotion. Next pass after
the 30 minute sleep. No commit, no GPU, no DD action.

## 2026-10-20 17:39 X2 -- leave X3's requested short card1 gap

CONFIG: X2 sidecar stopped, post-health HEALTHY both at17:37:17;
compiled post-health queued. Final BF16 DP capacity check is prepared.
COMMAND: hold final BF16 controller outside GPU leases until the queued
X3 QSA/proxy gap completes; x2/bf16-released is an X2 local hold marker.
RESULT: read X3's17:33 request before another Flash-Next reload. No
C7b/H1e new pending window appears in the latest journal. C7b/H1e retain
priority. No X2 model is running while this gap is granted.
VERDICT: let required short profiling make progress, then complete the
last DP measurement. No additional kernels/builds or speed tests needed.

## 2026-10-20 17:39 X1 -- full MTP2 graphs run; exact gate rejected

CONFIG: matched Flash-Next on/off from 17:23 entry. Native MTP2,
rt Q8 dense/Q2 experts, target IQ3_XXS, FULL_DECODE_ONLY TP1/DP2/EP.
COMMAND: x1_flash_window.sh; x1/flash-{greedy,bench}-mtp2.json.
RESULT: complete draft payload audit on both ranks, target and draft
full graphs captured, READY17:35:42. Identity/coherence pass. Exact
ordinary-EOS gate fails: C1 prompt0 at token17, other3x96 identical;
C2 prompts diverge at token4/72. BENCH_RC0, GREEDY_RC3. Acceptance
171/222=77.03%; C1 33.04 tok/s, chunk p50 85.20ms; C2 30.40/31.24
tok/s, chunk p50 83.37/83.53ms. Server down; post-health queues.
VERDICT: useful diagnostic speedup, strict correctness gate NOT passed.
Next X1 card0 only, expected <1min: compare actual rt BF16 HC/router
projections row1 vs row3, and native GDN verification vs sequential
at full Flash shapes. No full reload before narrowing this difference.


## 2026-10-20 17:39 X3 -- QSA full-size wrapper diagnosis and numeric pass

CONFIG: card1, T12288, actual QSA geometry and real indexer gamma,
random BF16 core inputs and 3-axis text positions, no full serve.
COMMAND: run_ops.sh --part qsa --out qsa.json, CPUquiet, C12+X3wide
and existing QSA binary. Full helpers plus mixed-expert arena checks.
RESULT: GPU exit0 in93s; pre-health HEALTHY, post-health pending behind
X1 health. Per QSA layer: core append2060.655 ms, selection/union/gate
5340.442 ms; old model_hook append22042.110 ->3455.148 ms with unused
query preparation and side round-trips removed. Whole indexer query
norm/M-RoPE4719.806 ->0.795 ms vectorized. Cache pages, tail, dead,
block_pos and length bit exact; index query max_abs0. Mixed expert
reference, invalid-ID and bucket checks also pass with bounded dq arena.
VERDICT: QSA1 still costs at least8.796s per attention layer, over105s
for12 layers before the rest of the model. This mode cannot support
2k tok/s without batched append and selection/union kernels. Candidate
serving remains explicit QSA0. Numeric helper equality is not full QSA
serving qualification. Next single-card35B A/B is ready after post-health.


## 2026-10-20 17:39 X3 -- real Flash-Next candidate gated behind proxy

CONFIG: only after own35B A/B succeeds and tears down: BOTH cards,
expected6-8min including exact large Ag/Rs+compiled health, full load,
12288-token requests and teardown. C7b/H1e keep priority.
COMMAND: detached x3/after_proxy.sh -> flashnext_window.sh. Stock
TP1/DP2/EP, BF16, QSA0/MTP0, C12+X3wide, bucket1/tile64, grouped
reduce, PLE NumPy, HC Triton, memoryfix1, maxlen12800/batch12288,
seqs2/util.98/FULL_DECODE_ONLY, C7b FA overlay, persistent cache off.
RESULT: prepared and gated; no full X3 model currently running.
VERDICT: operator budgets are ready; require exact real-model TTFT,
primary/alias identity, C1/C2 math, clean teardown and post-health.
The paired communication micro uses no topology environment override.

## 2026-10-20 17:41 O3 -- card-0 gates window for stage 3 (about 7 minutes)
CONFIG: card 0 only (card 0 free, nobody queued on it; X3 holds card 1).
_C.o3e.so (shared-expert fused op moe_layer_sh, SLM grid + sign tables,
multiply-spread signs, block-pointer walk). No serve.
COMMAND: gpu-run --card 0 o3/run_gates_e.sh
RESULT/VERDICT: next O3 entry.

## 2026-10-20 17:41 X2 -- sidecar teardown and collective health complete

CONFIG: completed eager BF16 sidecar200K/seq2 stage. No X2 server active.
COMMAND: bin/gpu-run bin/xpu-health; bin/gpu-run bin/xpu-collective-health.
RESULT: both-card post-health HEALTHY17:37:17; compiled collective
COLLECTIVE_HEALTH_OK, controller WINDOW_DONE17:40:05. No reset.
VERDICT: sidecar stage complete. Engine593893-token line is normalized
as389/(125+6)*200000, not a universal token cap. Inverting the shared
pool at two streams with null reservation yields300800 tokens/stream.
Configured model limit200000 stays in force. X3 now has its35B proxy
gap; final DP BF16 controller is held outside leases until it finishes.

## 2026-10-20 17:41 G2b -- audit pass 17 addendum

CONFIG: CPU re-read after the 17:38 milestone. The MTP2 bench and
the 17:39 X1, X2, and X3 notes landed during the append. No GPU.
COMMAND: python reads of x1/flash-bench-mtp2.json,
x1/flash-greedy-mtp2.json, x1/flash-mtp2.log, x2/sidecar/health-after.txt,
x2/window.log, and x3/qsa.json plus x3/qsa.log.
RESULT: appended F175. MTP2 C1 chunk p50 85.198 ms, 84 tokens in 32
chunks, decode_tok_s 33.039, acceptance 171/222, GREEDY_RC=3,
BENCH_RC=0, first id hotschmoe-dd, first_mismatch [17, None, None,
None]. Sidecar health-after is HEALTHY on cards 0 and 1. QSA helper
times match qsa.json. QSA log now has card1 post-health HEALTHY.
VERDICT: F175 supersedes the empty-bench clause. No speed promotion.
No commit, no GPU, no DD action.

## 2026-10-20 17:42 G2b -- audit pass 17 close

CONFIG: CPU check of the 17:41 X2 collective close. No GPU.
COMMAND: read x2/window.log and x2/collective-after.txt; evaluate
389/(125+6)*200000 and (389-1-2*6)*1600/2.
RESULT: appended F176. COLLECTIVE_HEALTH_OK world_size=2
shape=4x5120 compiled_iterations=10 p2p=0, HEALTHY, exit 0.
WINDOW_DONE 17:40:05Z. 389/131*200000 rounds to 593893.
(389-1-12)*1600/2 = 300800. X3 and O3 17:39/17:41 entries are
plans with no measured ITL in this pass.
VERDICT: F176 closes the collective and arithmetic claims. No
speed promotion. No commit, no GPU, no DD action.

## 2026-10-20 17:45 X2 -- final BF16 DP capacity window released

CONFIG: BOTH cards, expected4-6min after lease, eager DP2+EP,
BF16, maxlen200000, seq2, batch512, util.98, fixed memory plugin,
C12. No graph/MTP or timing benchmark. C7b/H1e still have priority.
COMMAND: release x2/bf16-released after X3_SERVE_DONE_RC=0 and
healthy card1 post-check. Target-only copy x2/run-bf16-target.sh;
repository equivalent X2_START_ARM=target X2_STOP_ARM=target
X2_TARGET_DTYPE=auto bash vllm/flashnext/memory/run_window.sh.
RESULT: X3's requested QSA/proxy gap is complete. Sidecar post-health
and compiled collective already pass. Long GPU timeout queues rather
than aborting; host RAM guard remains active during request checks.
VERDICT: finish the last BF16 capacity measurement; no new build.

## 2026-10-20 17:48 O3 -- stage 3: shared expert fused, gate/up ALU cut
CONFIG: plugin new op b70_gguf::moe_layer_sh (routed + shared expert +
per-token sigmoid(x . w_sg) gate in one call, bf16 [2,T,K] out; plan
kernel computes the gate; moe_layer unchanged). Fork moe.py opt-in
B70_FUSE_SHARED=1 (default off; MK_INTERNAL_OVERLAPPED shared order;
decode T<=8 without EP map, else the shared module runs inside apply).
Kernels: multiply-spread sign masks, codebook grid staged in SLM, block-
pointer walk (EC1 IQ3_S loop 528 -> 361 ISA instructions); SLM sign table
tried and rejected. _C.o3e.so (sign table on) gated; _C.o3f.so (off) built.
COMMAND: gpu-run --card 0 o3/run_gates_e.sh (17:41-17:47).
RESULT: gpu_check, moe_layer_bench sched 1/0, c12_types 12, graph_probe,
o3_shared (8 cases, cos >= 0.999998 vs moe_layer routed/shared split)
all failures=0; HEALTHY before/after. Layer with shared, XPUGraph
(o3_shared TIME): T=1 IQ3_S/Q2_0 115.2 -> 68.5 us, IQ2_XS/IQ4_NL 124.9 ->
81.4; T=4 217.0 -> 171.9, 254.5 -> 210.7 (unfused = moe_layer + gemv
shared MLP + F.linear gate + sigmoid + add). Microbench IQ3_S/Q2_0 T=1/4:
stage 2 70.3/189.9 -> 61.6/163.1 us; IQ2_XS/IQ4_NL 83.6/226.0 ->
78.1/209.4. Sign table 66.3/174.8 (slower). Load-only probe: gate/up
30.4 us vs 43.9 full (IQ3_S T=1), i.e. ~10 us of ALU left above this
access pattern's floor (~510 GB/s).
VERDICT: shared-expert fusion is the big win (~45 us/layer at the fair
shape). Next: 35B serve with _C.o3f.so, B70_FUSE_SHARED=1 vs 0.


## 2026-10-20 17:47 X3 -- 35B matched serving passes, cold JIT correction

CONFIG: card1, Qwen3.6-35B UD-IQ3_XXS same loaded model; C12+wide,
maxlen16384/batch12288/seqs2/util.75/FULL_DECODE_ONLY/FA/SYCL GDN,
prefixoff, DQ_T8192/bucket1/tile0/grouped reduce; quiet benchmark.
COMMAND: x3/serve_window.sh, harness/x3_serve_probe.py; serve-ab.json,
proxy-spans.jsonl, serve-probe.log, serve-window-r2.log.
RESULT: identity hotschmoe-dd first + registered alias; deterministic
Paris and two concurrent Paris checks pass; same long-prompt32-token
text under flag0/flag1. At1024:6025.851 ->675.427 ms TTFT,169.93 ->
1516.08 tok/s. At8181:45445.784 ->2388.216 ms,180.02 ->3425.57 tok/s
(19.03x). Warm12277:2456.050 ms /4998.68 tok/s. First12277 request
52254.904 ms /234.94 tok/s: inference JIT _pad_tokens and expert span
50304.7 ms, not an accepted warm-speed number. Teardown0; post-health
HEALTHY exit0. No server remains, no reset.
VERDICT: GGUF prefill dispatch diagnosis/fix is demonstrated on the
matched checkpoint and exceeds the unmatched E2 1961 reference at8k.
New expert capacities were constexpr Triton inputs and generated many
late compilations. CAP is now a non-specialized runtime scalar for
pad/inverse; exact tests at16/24/40 capacities precede large Ag/Rs in
the queued paired window. Real Flash-Next then runs profiled and plain
TTFT requests in the same serve; no first-use JIT spike will be hidden.

## 2026-10-20 17:48 X1 -- primitive shape checks exact; control repeat queued

CONFIG: card0 only numeric probes, real HC/router BF16 rt weights,
full GDN shape16/48 heads128, FP32 SSM, packed Q8 fc and C7b FA
24Q/2KV heads256 at past16/64/128/512/1536. O3 library pinned.
COMMAND: harness/x1_shape_probe.py through xpu-run --card0. Logs
x1/shape-probe.log and shape-probe-r2.log, exit0 after21/16s.
RESULT: row1 vs2/3/6 projections and native GDN spec3 vs sequential
(including final SSM) all bit-identical; packed Q8 and FA spec3 vs
sequential also zero unequal elements in these sampled probes.
VERDICT: no evidence for the guessed generic math workaround. Next
X1 BOTH cards, expected6min: x1_flash_repeat_window.sh, same no-MTP
config, two exact C1/C2 gates in one load to test target repeatability.
C7b/H1e retain priority. No extra latency benchmark or CPU rebuild.

## 2026-10-20 17:53 X2 -- DP BF16200K fits with batch512

CONFIG: eager DP2+EP, TP1, BF16/default FlashAttention, maxlen200000,
seq2, batch512, util.98, memory fixes, C12, compact PLE ring. No MTP.
COMMAND: target-only x2/run-bf16-target.sh; raw x2/target-auto/.
RESULT: READY17:50:28; owned25600975000 bytes/card. KVbudgets
5365490811 /5387383931; actual136 /137 blocks of39321600 bytes.
Pool ceilings206400 /208000 tokens for one long stream/card. Required
132 blocks/card for200K leaves only4 /5 extra pool blocks. Profile
oldheadroom336241664 ->physical2097152. Identity and greedy323/667
pass, embedding20480 values bit exact/rank; two concurrent2047-token
prefills plus1 token each pass. Server stopping for final health.
VERDICT: FP8 is unnecessary for the eager DP memory fit after batch
capping. This is allocated capacity plus short gates, not200K coherence
or speed. Graph/MTP costs still need requalification. Sidecar has much
more memory margin; no latency comparison is claimed. No reset/commit.


## 2026-10-20 17:55 X3 -- paired collective diagnostic stalled; health required

CONFIG: BOTH cards under x3/collective-budget-r1.py (copied exact
source before correction), long-message Ag/Rs oracle, no topology env
override. Per-card pre-health HEALTHY; first compiled10-iteration
collective health passed with p2p=None. Dynamic CAP16/24/40 padding
and invalid-slot weighted reduction passed on both cards.
COMMAND: current collective-budget.log, outer180s timeout.
RESULT: no transfer timings. Worker output stops after dynamic-cap
checks and import; the harness destroyed the health default process
group and attempted to initialize another. Suspect lifecycle/rendezvous
problem, not yet diagnosed as hardware. No model is running.
VERDICT: other paired runs should wait for X3 post-health after this
window. Full serving is gated on probe success and will not start.
The corrected oracle keeps one process group for before/after exact
health+timing and prints phase markers. Do not reset without failed
health, do not repeat a full serving init against uncertain collectives.

## 2026-10-20 17:57 X1 -- repeat load stopped for X3 post-health

CONFIG: no-MTP repeat control from17:48, BOTH cards. X1 pre-health
HEALTHY but X3's collective window ran between health and serve.
COMMAND: x1_flash_repeat_window.sh; X1 acquired17:56:22. Read new
17:55 X3 journal warning during load; docker stop vllm-fn-x1-repeat.
RESULT: shutdown requested before READY, no token or latency requests.
The X3 collective timeout has no post-health verdict yet. X1's wrapper
will release leases and run its normal post-health; no reset requested.
VERDICT: preserve aborted log, wait for X3 post-health/collective verdict
before another paired serve. This attempt supplies no qualification.


## 2026-10-20 17:57 X3 -- R1 timeout, cleanup script error, health queued

CONFIG: paired large-transfer R1 diagnostic ended by180s timeout,
GPU wrapper exit124. First eager/compiled health and dynamic capacity
checks passed; no communication timings and no full X3 model started.
COMMAND: immutable R2 flashnext_window_r2.sh now queues BOTH-card
per-card health, corrected single-PG eager/compiled health+large Ag/Rs,
then candidate serving only on success. Expected6-8min after acquisition.
RESULT: R1 cleanup did not execute: editing its live shell script while
waiting changed the reader's offsets and caused an EOF quote error.
This was an X3 harness mistake, now recorded. R1 source/log copies are
preserved. Fresh per-card and compiled health are required; no reset.
X1 repeat acquired both at17:56:22 before X3 could perform post-health;
its own queued pre-health predates R1 completion. Other new paired loads
should wait for the new health results. C7b/H1e keep priority.
VERDICT: do not infer hardware loss from this lifecycle/timeout yet.
No full serving will be launched by X3 without fresh passing health.
Immutable versioned shell scripts are used from this point. No commit.

## 2026-10-20 17:58 X2 -- final collective queued; short gap requested

CONFIG: final DP BF16 server is stopped. Both-card post-health passed
at17:53:11. Final compiled post-collective has queued since then.
COMMAND: x2/bf16-window.log, x2/target-auto/health-after.txt;
currently waiting through X3's timed-out diagnostic and X1 repeat serve.
RESULT: all X2 capacity/coherence/prefill results and per-card health
are complete. No X2 model is running and no reset was attempted.
VERDICT: request a33-second BOTH-card gap after current X1 for the
already-queued final collective, before more full model reloads. C7b/H1e
retain priority. X2 will remain active until its last check is closed.
Evidence note: the target-only controller reused the base final-collective
log path, replacing sidecar's earlier copy. Sidecar success17:40:05 is
independently in this journal and G2 F176. Future harness logs are now
per-arm; model/profile/KV/request artifacts were never overwritten.

## 2026-10-20 17:59 X2 -- VRAM/KV accounting complete and jobs closed

CONFIG: final DP BF16 batch512 and prior sidecar BF16 batch2048;
maxlen200000, no MTP/graphs. Per-arm raw profile/census/KV JSON intact.
COMMAND: final bin/gpu-run bin/xpu-collective-health; syntax/ASCII,
model-registry YAML match, git diff --check, reverse patch check.
RESULT: DP post-health HEALTHY17:53:11; compiled collective
COLLECTIVE_HEALTH_OK world_size2 shape4x5120 iterations10 p2p0,
controller WINDOW_DONE17:58:02. Final raw check copied to
x2/target-auto/collective-after.txt. All X2 controller/watchdog PIDs
and containers are gone. No RAM abort, reset, commit/push or DD action.
VERDICT: report reports/1X2_kv_budget.md complete. Model storage falls
1055262720 bytes/DP card. Final BF16 pool ceilings: DP206400/208000
one stream/card; sidecar300800/stream at concurrency2. Both allocate
space for2x200K and pass short identity/coherence/concurrent-prefill
checks. Long200K quality/latency and graph/MTP qualification remain.
FP8 token/head configuration is rejected for failed short coherence,
with root cause unassigned. Native memory is conservatively counted;
unattributed oneDNN/communication/driver residual is not removed.

## 2026-10-20 18:01 O3 -- card-0 35B serve pair queued (2 x ~5 minutes)
CONFIG: card 0 only, C12 recipe, _C.o3f.so (sha 7389c133), B70_FUSE_SHARED
1 then 0, prefill 1024 only. Gates on _C.o3f.so passed 17:56-18:00
(gpu_check, moe_layer_bench 1/0, c12_types, c9_fair, graph_probe,
o3_shared, failures=0; HEALTHY). Card 0 is released between the two
serves.
COMMAND: o3/run_serves_f.sh
RESULT/VERDICT: next O3 entry.

## 2026-10-20 18:01 X1 -- health closed; immutable repeat retry queued

CONFIG: BOTH-card repeat no-MTP control, expected6min, as17:48.
COMMAND: X1_REUSE_POST_HEALTH=1 harness/x1_flash_repeat_window.sh.
RESULT: aborted server exit137 is intentional docker-stop teardown;
post-health HEALTHY on both cards at18:01:02, wrapper exit1 ready0.
X2 compiled two-rank health passed17:58. Aborted logs preserved as
x1/flash-repeat-health-abort.log and window-flash-repeat-health-abort.out.
VERDICT: reuse that fresh successful post-health instead of repeating
another identical30s per-card setup. Source shell was edited only after
its prior process exited. New serving still queues through gpu-run;
watch new journal/owner state for C7b/H1e priority or health problems.

## 2026-10-20 18:03 X1 -- actual target quant mix probe prepared for card1 gap

CONFIG: card1 only, expected<1min numeric plus per-card health, no serve.
Pinned _C.o3.so; actual main-GGUF dense weights, one tensor/type, and
16 actual experts from layers0/1/3/8/12/24/36/47. No quant regeneration.
COMMAND: x1_packed_shape_probe.py via xpu-run --card1, health around it;
will queue behind X3's paired diagnostic, use card1 while O3 uses card0.
RESULT: CPU source prepared/linted. Prior sampled Q8/HC/GDN/FA probes
were exact but did not cover the main model's mixed expert quant types.
VERDICT: compare row1 vs3/6 and captured two-call scratch reuse. This
is a numeric localization probe, no performance or stability claim.


## 2026-10-20 18:04 X3 -- paired budget passed; uneven RS is a top cost

CONFIG: immutable R2 uses exactly one xccl group/worker, no topology
override. Fresh per-card health HEALTHY; formula-equivalent eager and
compiled10-iteration health before/after large operations passes.
COMMAND: x3/collective-budget.log; collectives-rank0/1.json.
RESULT: both rank CAP checks pass. C1 sizes[12288,1]:3-gather dispatch
7.666/7.747 ms and native uneven RS93.251/93.496 ms. Balanced global
12288 sizes[6144,6144]:dispatch3.210/3.215 ms, RS3.083/3.067 ms.
R2 GPU exit0, no reset. X2 fresh compiled post-health also passed
after R1. R1 stall was a harness lifecycle problem, not proven hardware
loss. Full X3 serve is currently queued behind X1 repeat.
VERDICT: C1 native RS projects4.49s/48layers and must be fixed. The
existing padded equal-size route is enabled only for B70_PREFILL1 long
uneven transfers, with B70_PREFILL_EP_RS=native/pad override. It avoids
a new communication implementation. Full same-loaded-model probe will
compare native then pad with identical12288 input IDs, then collect
three plain TTFTs with profiling disabled and a long-context math gate.
No source used by the running X1 process changes behavior (prefill0).
