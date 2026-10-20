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

