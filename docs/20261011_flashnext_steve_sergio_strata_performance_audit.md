# Current primary performance recipes and the next aggressive controls

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: User target is at least 2000 input tok/s and 50 output tok/s for the
selected original Unsloth UD-Q4_K_XL on TWO B70s. This is a target, not an
observed result. Primary weights/quantization are not interchangeable with
27B dense, Coder, IQ2/IQ3 or official FP8. Current43 API mode1 has a closed
coherent eight-call baseline; that is not full-model mathematical qualification.
Current tiny54/524 and GEN4 captures do not establish the requested workload.

COMMAND: Read primary repositories on 2026-10-11; no downloads of weights,
builds, GPU/provider calls or configuration changes. GitHub API branch heads:

| Repository | Revision | Commit UTC |
|---|---|---|
| [Steve](https://github.com/steveseguin/b70-optimization-lab/tree/3111b5885a79b14655d572aa870ceb66bef56821) | 3111b5885a79b14655d572aa870ceb66bef56821 | Oct11 20:43:14 |
| [Sergio](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/tree/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd) | 1a76136ea1b3f96bba3b20e42041a6fcc95f56fd | Oct7 21:06:37 |
| [Strata](https://github.com/Niko1221/Strata/tree/7695859aa13afb248bdb4d36f9b2e95707db079d) | 7695859aa13afb248bdb4d36f9b2e95707db079d | Oct11 14:19:59 |

RESULT: Latest relevant published cells, not a hardware scaling ranking:

| Source/model | Hardware/quant/draft | Measured cell and actual recipe |
|---|---|---|
| Steve dense Qwen3.8-27B official FP8 | 2 B70, FP16 activations/KV, MTP1 | P512 2856.96 input tok/s; P2048 3679.25; strict natural-suite decode54.747. Context/batch4096, one sequence, cache0, R304 image. Prefill uses server scheduled-to-first-token time; timing outputs128 forced, separate natural suite. [Record](https://github.com/steveseguin/b70-optimization-lab/blob/3111b5885a79b14655d572aa870ceb66bef56821/experiments/qwen38-27b-b70/notes/2026-09-14-fp8-prefill-focus-results.md) |
| Steve dense27B AutoRound INT4 | 2 B70, W4A16/g128, MTP4 | Best117.528 decode; R304117.24/116.95. MTP0 same R30450.07/50.06. FULL_DECODE_ONLY graphs, 67248-row draft shortlist; GPTQ dispatch and fixed-K oneDNN. Different tensors from our GGUF. [Recipe](https://github.com/steveseguin/b70-optimization-lab/blob/3111b5885a79b14655d572aa870ceb66bef56821/repro/qwen38-27b-autoround-int4-b70/README.md) |
| Steve actual Flash-Next125B-A6B | 4 B70, official FP8/BF16 KV, lossless MTP1 | 46.854250 decode; P512 prefill unmeasured. Exact serial GDN rows execute inside the extension rather than Python per-row dispatch. Published as originating-lab replay, not a portable fresh installer. [Recipe](https://github.com/steveseguin/b70-optimization-lab/blob/3111b5885a79b14655d572aa870ceb66bef56821/repro/qwen38-flash-next-fp8-tp4-mtp1-exactgdn-b70-47tps-20260913/README.md) |
| Sergio actual Flash-Next | 2 B70/x8+x8, AtomicChat mixed IQ3_S gate/up and IQ4_NL down, quantized KV | New-tree no-spec26.86 decode; MTP3 fused multirow MoE code33.25/prose30.68 (P512/G128). F16 no-spec P9096 prefill585.9. Separate draft Q4_K_M2.44GiB; compile FP32 decode/F16 prefill. Sampling1.0/top_p.95/top_k20/ignore_eos, PLE prewarmed. [Recipe](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd/docs/qwen38-flash-next/QWEN38-FLASH-NEXT-LLAMACPP.md) |
| Strata full Flash-Next IQ2_XS | 1 B70, MTP draft, 18329 resident/6247 mirrored experts | 60.6 decode after2184 prompt; 549 prefill. Coder IQ1_M is pruned:78.2 short decode/790 prefill at2184. NO_HOST/device-plan, AOT, auto prefill, no borrowing. These lighter checkpoints are not UDQ4 controls. [Intel record](https://github.com/Niko1221/Strata/blob/7695859aa13afb248bdb4d36f9b2e95707db079d/docs/INTEL.md) |

Strata's SAME UDQ4 files require compatibility BF16 packing in the public
recipe:195 Q8_0 small projections are rounded. Its Windows RTX5070/64GB
40GiB-budget example is7-8.5 decode and160 prefill at15873 input, MTP enabled.
We retain original Q8/F32 math, so that converted pack is not a substitute.
[UDQ4 contract](https://github.com/Niko1221/Strata/blob/7695859aa13afb248bdb4d36f9b2e95707db079d/docs/UNSLOTH_Q4.md).
Sergio's current preferred dense27B EXL3 route is another quantization/engine:
4bpw trellis, one B70, MTP3; published18.4->24.6 are accepted-step chunk rates,
not directly generated-token rates. Its README's older50/1700 dense headline
must not be assigned to our Flash-Next.
[Current route](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd/docs/qwen38-27b/EXL3-XPU.md).

Three concrete controls worth actual matched runs, in order:

1. Dispatch a real batched prompt. Observed local54 prompt uses27 serial verify
windows, not prefill GEMM; short-read0 is the ready direct test. Then test exact
P2048 with chunk512 versus2048, independently admitted memory/outputs. Current
SYCL prefill projects T rows through oneMKL and dequantizes experts; it has no
KQuant MMQ option wired in sycl/CMakeLists.txt. Upstream CUDA's quant-native
KQuant/MMQ advantage is a concrete port target, not an env flag already working
on our engine. Preserve primary quantized tensors and check intermediates;
F16 chunk changes can alter arithmetic. Long524 already takes batched prefill,
so eliminating short windows cannot explain its85.7 input tok/s ceiling.

2. Reduce per-row native HC/queue work without changing math. Actual43
sycl/src/core/verify.cpp:996-1017 submits native HC separately for every row,
with reused gate scratch; adjacent QSA projections:1163-1174 have multirow
GEMV/MMVQ paths. Prototype a loaded-context row-batched HC launch with private
per-row scratch and the identical reduction order. Measure launch count,
actual device duration and request wait, not summed nested host scopes.
Steve's exact GDN extension and Sergio's fused multirow MoE demonstrate the
kind of dispatch elimination; they do not prove our HC is the bottleneck.

3. Use both cards for actual residency, not just split presence. Compare fixed
32/16 versus24/24 placement with the same primary artifact/mode/context/prompt.
Record real selected-expert misses, selected bytes, copy durations and per-card
resident counts. Strata's lighter fully resident/high-resident cases differ
materially from our one-card startup8092/24576 resident case, but51.66GB startup
mirror is not per-token traffic and does not quantify the slowdown. Balanced
placement is a hypothesis, requiring the actual trace and full49 topology
checks. Keep MTP as a separate next lever: the original publisher sidecar may
amortize decode after exact target verification; no upstream acceptance rate
or draft arithmetic transfers to our model.

Representative benchmark, not tiny API replies:

- First fixed P2048, context4096, then P8192/context16384 and P32768/context65536
  only after capacity admission. Native P counts include the actual current
  template; save exact IDs using current tokenizer. At least three diverse
  prose/code/document inputs, cacheOFF, temperature0, repeat complete outputs.
- Decode cap512, natural EOS honored. Report eligible actual >=256-output
  requests separately; early EOS is a result, not filled with invented tokens.
  A forced256-token engine screen may be explicitly diagnostic, never the
  natural-completion service result. Report MTP0 and actual MTP acceptance
  separately; keep useful generated-token counts rather than chunk rates.
- Native host prompt time and client TTFT remain separate. Show cold/warm
  request results, actual source/config/identity, teardown and health; output
  differences prevent an optimization promotion, not descriptive reporting.

Memory: no requested chunk is a guarantee. Current upstream counts actual
per-stage buffers and checks need+512MiB against free device memory before
allocation, then may lower the chunk. Source records2MiB allocation granules
for owned buffers and byte-budgeted expert rings, whose slot count depends on
THIS pack's blob size. Retain reserve2048MiB for the larger chunk experiment,
record requested AND resolved chunk/ring plus free memory on EACH card, and
reject a comparison if it silently falls back. Do not reuse Coder/Q2 ring-slot
budgets for bigger Q4/Q5/Q8 blobs or claim64GiB all usable for weights.
[Allocator](https://github.com/Niko1221/Strata/blob/7695859aa13afb248bdb4d36f9b2e95707db079d/sycl/src/prefill/prefill.cpp#L1823),
[per-card admission](https://github.com/Niko1221/Strata/blob/7695859aa13afb248bdb4d36f9b2e95707db079d/sycl/src/program/generate.cpp#L6703).

VERDICT: None of the relevant Flash-Next Intel records establishes2000/50 on
our exact two-card UDQ4 target. This is not a reason to accept the current gap.
The three tests above directly attack dispatch, kernel work and placement;
observe the first results before choosing the substantial kernel port. No
performance attribution or optimization success is claimed by this audit.
