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
