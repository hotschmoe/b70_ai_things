# Native int4 DPAS on Battlemage (Xe2 / Arc B70)? -- NVFP4 kernel research

Research date: 2026-07-04. Scope: can NVFP4 on B70 go BELOW the current int8-XMX
repack and hit a NATIVE int4 DPAS/XMX fast path, feeding the 4-bit data more
directly? READ + WEB study only (no GPU touched). Every hardware claim is cited.
Honest-uncertainty markers: CONFIRMED / REPORTED / UNCONFIRMED.

Prereqs / context in this repo: `vllm/nvfp4/NVFP4_XPU.md` (the working int8-XMX
repack path), `kernels/int8_gemm_w8a16.h` (the oneDNN K-group kernel), memory
notes `b70-int8-xmx-roofline` and `zml-int8-decode-layout-bound`.

--------------------------------------------------------------------------------
## 0. TL;DR verdict

- The HARDWARE DPAS on Xe2/Battlemage DOES have int4 and int2 matrix modes
  (inherited from Xe-HPG). CONFIRMED at the marketing/arch-doc level.
- But NO shipping SOFTWARE STACK exposes a TRUE int4 x int4 -> int32 DPAS GEMM
  for Intel GPU: oneDNN treats s4/u4 as WEIGHT-STORAGE-ONLY (decompress to
  f16/bf16/int8 before the MAC); SYCL joint_matrix lists only int8/bf16/fp16/tf32
  in its supported-combinations table (no 4-bit); Triton-XPU `tl.dot` bottoms out
  at int8. So a native int4 GEMM today means hand-written ESIMD/GRF assembly.
- For LLM DECODE (M=1 GEMV) native int4 DPAS would NOT help anyway: decode is
  weight-bandwidth bound, ~30-300x below the compute roofline. The only decode
  lever is READING FEWER WEIGHT BYTES -- keep the 4-bit data 4-bit in VRAM and
  widen/dequant in registers just before the MAC. That is a bandwidth play that
  works WITHOUT any native int4 DPAS.
- Native int4 DPAS is a COMPUTE play: it only helps PREFILL / large-M (the
  compute-bound region, crossover ~M=150-300 on B70), and even there no library
  can emit it, so the ROI is poor versus the already-good int8-XMX prefill path.
- RECOMMENDATION: prototype option (b) -- a FUSED dequant-in-register GEMV that
  keeps NVFP4 weights 4-bit in VRAM (w4a16-style, E2M1 LUT + E4M3 block scale in
  registers). It ~halves decode weight bytes vs the current int8 repack, is the
  real roofline lever, and needs no native int4 DPAS. Keep the int8-XMX repack as
  the prefill/large-M path. Do NOT chase native int4 DPAS.

--------------------------------------------------------------------------------
## 1. DPAS ISA capability per architecture

DPAS = Dot Product Accumulate Systolic, the instruction that drives the XMX
systolic array. Its operands: `dpas.<sdepth>x<rcount> dst src0 src1 src2`, where
`sdepth` = systolic depth (chain length over successive registers) and `rcount` =
repeat count (number of DPAS ops with dst/src0 advancing). [Intel Xe GPU
Optimization Guide, "Intel Xe GPU Architecture"; the intel.com HTML page 403s to
scrapers but the sdepth/rcount definition is quoted in Intel's oneAPI GPU
Optimization Guide.]

### Hardware precision support table (matrix A/B operand types)

| Arch (product)                    | int2 | int4 | int8 | fp16 | bf16 | tf32 | fp8 | Accum |
|-----------------------------------|------|------|------|------|------|------|-----|-------|
| Xe-HPG  (Alchemist / DG2, Arc A)  | yes  | yes  | yes  | yes  | yes  | no   | no  | s16/s32, fp32 |
| Xe2     (Battlemage / Lunar Lake, Arc B70) | yes | yes | yes | yes | yes | yes | UNCONF | s32, fp32 |
| Xe-HPC  (Ponte Vecchio, GPU Max)  | UNCONF | UNCONF | yes | yes | yes | tf32 | yes(bf8/hf8) | s32, fp32 |
| Xe3     (Panther Lake)            | yes(likely) | yes(likely) | yes | yes | yes | yes | UNCONF | s32, fp32 |

Sources and notes:

- Xe-HPG: "The DPAS instruction was added to the Xe-core in Xe-HPG, supporting
  FP16, BF16, INT8, INT4 and INT2 multiply, with either 16 or 32 bits accumulate."
  [Intel, "Introduction to the Xe-HPG Architecture",
  https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/technology/xe-hpg-microarchitecture.html
  and the Xe-HPG white paper PDF
  https://cdrdv2-public.intel.com/758302/introduction-to-the-xe-hpg-architecture-white-paper.pdf ]
  CONFIRMED.

- Xe2/Battlemage: "On XVE and XMX units, it is possible to perform matrix
  operations with the FP16, BFloat16, INT8, INT4 and INT2 data types," and Xe2
  adds TF32. [HWCooling, "Batttlemage: Details of Intel Xe2 GPU architecture",
  https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/ ]
  Corroborated by chipsandcheese ("lower precision data types all the way down to
  INT2 ... INT8, INT4, and INT2"),
  https://chipsandcheese.com/p/lunar-lakes-igpu-debut-of-intels ,
  and by Intel VTune's instruction-group page literally named "XVE
  FP16/BF16/INT8/INT4/INT2 XMX Instructions",
  https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/xve-fp16-bf16-int8-int4-int2-xmx-instructions.html
  CONFIRMED that the modes EXIST in silicon. The int2/int4 sub-mode operand
  signedness details (pure s4xs4 vs mixed) are UNCONFIRMED from primary ISA docs.

- REPORTED nuance (Xe2 mixed-precision int DPAS): "Xe2 systems offer high compute
  throughput via int2 x int8 -> int32 DPAS instructions (same throughput as
  int8 x int8 -> int32)." [attributed to the Intel LLM-inference paper, arXiv
  2508.06753, via search snippet]. This suggests the low-bit int DPAS modes may be
  MIXED (one narrow, one int8) rather than symmetric int4xint4. Treat the exact
  int4 operand pairing as UNCONFIRMED.

- Throughput multipliers (Xe2): XMX = 2048-bit unit, "four times the throughput of
  the 512-bit vector engines for the same data types"; int8 rated ~4096 ops/clk vs
  fp16 ~2048 ops/clk (int8 = 2x fp16). Our roofline notes:
  int8 = 2x bf16, int4 = 4x bf16 = 2x int8, int2 = 8x bf16. B70 dense int8
  ~367 TOPS -> int4 compute ceiling ~734 TOPS. [chipsandcheese Xe2 article above;
  memory `b70-int8-xmx-roofline`.] The 4x-per-halving-of-bits pattern is
  CONFIRMED at the arch level; exact B70 int4 TOPS is a derived estimate.

Bottom line for (1): the int4/int2 DPAS MODES physically exist on Xe2. The blocker
is not silicon; it is that no programmable interface below exposes them for GEMM.

--------------------------------------------------------------------------------
## 2. oneDNN support (the stack behind vllm-xpu-kernels)

oneDNN version behind our kernels: the vllm-xpu-kernels / torch-2.12-XPU stack
bundles oneDNN ~3.7-3.9 (exact tag UNCONFIRMED from the repo; the int4 grouped-
scale weight-decompression features referenced below landed across oneDNN
3.5-3.7). Re-verify with `ONEDNN_VERBOSE=1` at runtime if it becomes load-bearing.

Findings (primary docs):

- Data-types doc: "s4 / u4 data types are only supported as a storage data type
  for weights argument in case of weight-only quantization." In the Intel-Graphics
  support table s4/u4 show only conversion support (".") across Xe-LP..Xe3, NOT
  hardware-compute support. int8 (s8/u8) shows full hardware support ("+") on
  Xe-LP and newer. [https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html]
  CONFIRMED.

- Matmul doc data-type table (GPU): 4-bit weights (u4/s4) are only valid paired
  with a FLOATING or int8 source; there is NO {s4 src, s4 weights} row. The two
  integer-input rows are:
    src {u8,s8} x wei {u8,s8,u4,s4} -> dst {u8,s8,s32,f32,f16,bf16}
    src {f32,bf16,f16} x wei {u8,s8,u4,s4} -> dst {f32,bf16,f16}
  i.e. 4-bit ALWAYS sits on the weights side and is DECOMPRESSED; accumulation for
  integer inputs is s32, but the narrow operand fed to DPAS is int8 or f16, never
  int4. [https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html] CONFIRMED.

- Release notes confirm the FEATURE is "int8 activations with grouped scales and
  int8 OR int4 COMPRESSED weights" and "f16/bf16 matmul with int8 or int4
  weights" -- i.e. weight-decompression, not int4 compute. [oneDNN release notes,
  https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-deep-neural-network-library-release-notes.html]
  CONFIRMED.

Verdict (2): oneDNN has NO true int4 x int4 -> int32 DPAS GEMM for Intel GPU.
It only has int4-WEIGHT decompression: the 4-bit weight is widened to int8 or f16
in the kernel prologue and the DPAS itself runs int8 or f16. This is exactly the
"widen in registers" pattern -- and it already keeps the weight 4-bit in DRAM,
which is the bandwidth win we care about (see section 5/6).

IMPORTANT NVFP4-specific caveat: oneDNN's s4/u4 path interprets the 4-bit code as
a two's-complement INTEGER (range -8..7 for s4). NVFP4 codes are E2M1 FLOATING
(value set +/-{0,0.5,1,1.5,2,3,4,6}). oneDNN cannot decode E2M1 as s4. And the
E2M1*2 = int8 trick we use for the int8 path needs values up to +/-12, which do
NOT fit s4's -8..7. So NVFP4 CANNOT be losslessly repacked into oneDNN s4. Any
4-bit-in-VRAM NVFP4 path therefore needs a CUSTOM E2M1-LUT dequant kernel; the
oneDNN s4 decompression primitive is not reusable for NVFP4 as-is.

--------------------------------------------------------------------------------
## 3. Triton-XPU (intel-xpu-backend-for-triton)

- `tt.dot` / `tl.dot` lowers operands through the DPAS layout
  (`DotOperandEncodingAttr` -> DPAS MMA). The documented, tuned path is int8 and
  the float types (fp16/bf16/tf32); the backend's tensor-descriptor / 2D-block-IO
  path is built for those. [intel-xpu-backend-for-triton docs/ARCHITECTURE.md,
  https://github.com/intel/intel-xpu-backend-for-triton ]
- No supported `tl.dot` path takes packed-4-bit operands directly. Triton-level
  4-bit is handled the same way as CUDA Triton int4 kernels: load packed 4-bit,
  UNPACK/dequant in-register to int8 or fp16, then `tl.dot` at int8/fp16.
  Memory note `zml-int8-decode-layout-bound` records Intel Triton emitting an
  rcount=1 DPAS tile at M=1 with no throughput benefit -- so Triton's value here
  is the load/unpack/epilogue plumbing, not a 4-bit DPAS.
- Smallest true `tl.dot` operand: int8. CONFIRMED (docs + our own prior probing);
  a native 4-bit `tl.dot` is UNCONFIRMED-absent (not documented as existing).

Triton-XPU is nonetheless the most practical vehicle for the RECOMMENDED
dequant-in-register GEMV (section 6): it can express "load 4-bit, LUT-dequant,
MAC" without hand ESIMD.

--------------------------------------------------------------------------------
## 4. SYCL joint_matrix / ESIMD

- SYCL joint_matrix supported-combinations table (the authoritative programmable
  surface): for Intel XMX (DG2, PVC, and Xe2/Battlemage) the A/B element types are
  uint8/sint8 (-> sint32 accum), fp16 (-> fp32/fp16), bf16 (-> fp32/bf16), and
  tf32 (-> fp32). NO int4/int2/s4/u4 entry appears anywhere in the XMX
  combinations. [intel/llvm
  sycl_ext_oneapi_matrix.asciidoc,
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc ]
  CONFIRMED: joint_matrix cannot express a 4-bit matmul on Xe2; int8 is the floor.
- ESIMD: the `__esimd`/`lsc` dpas intrinsics expose the DPAS precisions the
  compiler knows about. Whether the ESIMD dpas intrinsic accepts an s4/u4
  precision enum on Xe2 is UNCONFIRMED -- not found documented. Given the silicon
  has int4/int2 modes (section 1), an ESIMD/inline-GRF asm path is the ONLY
  plausible route to a native 4-bit DPAS today, and it would be hand-written and
  unsupported. High effort, uncertain payoff.

Verdict (4): the two "portable" surfaces (joint_matrix, Triton) both floor at
int8. Native 4-bit DPAS is reachable, if at all, only through unsupported
ESIMD/asm. UNCONFIRMED whether even ESIMD exposes s4/u4 on Xe2.

--------------------------------------------------------------------------------
## 5. The key practical question: does native int4 even help DECODE?

LLM decode is M=1 (a GEMV): one activation vector times each weight matrix. Its
arithmetic intensity is ~1 MAC per weight element, so it is WEIGHT-BANDWIDTH
BOUND, not compute bound. Our own measurements (memory `zml-int8-decode-layout-
bound`, `b70-int8-xmx-roofline`) put M=1..8 int8 matmul 30-300x BELOW the compute
roofline; the roofline crossover to compute-bound is ~M=300 (int8) / ~M=150
(int4) on B70. The Intel LLM-inference paper agrees: "the decode phase becomes
memory-bandwidth limited," and they DEQUANTIZE int4 weights to higher precision
before the matrix multiply rather than using a native int4 DPAS GEMM. [arXiv
2508.06753, https://arxiv.org/pdf/2508.06753 ] CONFIRMED direction.

B70 decode roofline (608 GB/s, weights-read bound), 8B-param dense model:

| Weight format in VRAM | Bytes read/token | Decode ceiling | vs int8 |
|-----------------------|------------------|----------------|---------|
| bf16 (16 GB)          | ~16 GB           | ~38 tok/s      | 0.5x    |
| int8 / NVFP4->s8 repack (9.6 GB) | ~9.6 GB | ~63 tok/s   | 1.0x    |
| 4-bit-in-VRAM (~4.8 GB) | ~4.8 GB        | ~127 tok/s     | ~2x     |

(Rough ceilings = VRAM_bytes / 608 GB/s; ignores KV + activations, so real
numbers are lower, but the RATIOS hold. Our current NVFP4 int8xmx path serves
31.7 tok/s on the 8B, weight-BW bound, matching "int8 ceiling minus overheads.")

Consequences:

- A native int4 DPAS (more int OPS) does NOTHING for decode: the DPAS is already
  idle 30-300x of the time waiting on weight reads. More FLOPS on an idle unit is
  worthless. CONFIRMED by roofline.
- The ONLY decode lever is HALVING WEIGHT BYTES READ: keep the weights 4-bit in
  DRAM and widen/dequant in registers just before the MAC. This roughly DOUBLES
  the decode ceiling vs our current int8 repack. It works with the int8 (or fp16)
  DPAS -- native int4 DPAS is NOT required.
- Precedent on THIS box: llama.cpp #21517->#21527 got Qwen3-27B Q8_0 decode from
  4.88 -> 15.24 tok/s (3.1x, 21%->66% of 608 GB/s) via a WEIGHT-LAYOUT REORDER
  alone, no XMX change. The lever is layout + fewer bytes, not the matrix unit.
  [memory `zml-int8-decode-layout-bound`; llama.cpp PR #21527.]

VERDICT (5):
- (a) DECODE: invest in the BANDWIDTH play -- 4-bit weights resident in VRAM,
  register dequant. Native int4 DPAS is irrelevant to decode.
- (b) PREFILL / large-M: this is where a native int4 DPAS COULD help (compute-
  bound region), but (i) no software stack emits it, and (ii) our int8-XMX repack
  already runs prefill at ~50% of the 367-TOPS int8 ceiling. A native int4 GEMM's
  theoretical 2x-over-int8 prefill is not worth a hand-ESIMD kernel today.

--------------------------------------------------------------------------------
## 6. Single most promising next kernel to prototype

Options on the table:
  (a) native int4 DPAS GEMM (compute play)
  (b) fused dequant-in-register GEMV, weights 4-bit in VRAM (bandwidth play, w4a16)
  (c) the existing int8-XMX repack (weights int8 in VRAM)

Ranked recommendation:

RANK 1 -- (b) FUSED NVFP4-4bit-in-VRAM DEQUANT GEMV (the decode bandwidth play).
  Keep the weight as packed E2M1 nibbles (K/2 bytes) + the per-16 E4M3 block scale
  (K/16 bytes) + f32 global scale RESIDENT in VRAM. Kernel: coalesced Xe2 load of
  the 4-bit block, in-register E2M1 LUT -> bf16/int8, multiply by the E4M3 block
  scale, MAC into f32/s32. This ~halves weight bytes/token vs the int8 repack ->
  ~2x the decode ceiling (~63 -> up toward ~127 tok/s pre-overhead on the 8B).
  - Why it wins: decode is the dominant serving cost and is purely weight-BW
    bound; this is the ONLY lever that moves it. Needs NO native int4 DPAS.
  - Vehicle: Triton-XPU (`load 4-bit -> unpack/LUT -> dequant -> tl.dot at int8/
    fp16`) for fastest bring-up; or a layout-first ESIMD GEMV (port the llama.cpp
    #21527 reorder idea, proven 3.1x on this box) for the ceiling.
  - NVFP4 caveat baked in: must be a CUSTOM E2M1-LUT dequant -- oneDNN's s4 path
    does NOT decode E2M1 and NVFP4's +/-12 int range does not fit s4 (section 2).
  - Group-scale K=16 must be honored (same K-group fix already made in
    `int8_gemm_w8a16.h` -> infer {grp_k=16, grp_n=1} from scale shape).

RANK 2 -- (c) KEEP the int8-XMX repack for PREFILL / large-M. It already exists,
  is bit-exact, uses INT8 XMX, and sits ~50% of the int8 compute ceiling. Make the
  serve path HYBRID: 4-bit-in-VRAM GEMV at decode (rank 1) + int8-XMX GEMM at
  prefill/verify. This mirrors the W4A8 hybrid pattern already used elsewhere in
  the repo (int4 decode + int8 prefill).

RANK 3 -- (a) native int4 DPAS: DO NOT prototype now. It only helps the compute-
  bound prefill region, no oneDNN/SYCL/Triton path emits it, and it would require
  unsupported hand ESIMD/asm for a benefit the int8-XMX prefill path already
  largely captures. Revisit ONLY if (i) a future oneDNN/joint_matrix release adds
  a real s4xs4->s32 GPU GEMM, or (ii) prefill throughput becomes the proven
  bottleneck AND decode is already at its 4-bit BW ceiling.

One-line rationale tied to the roofline: at M=1 the matrix unit is starved for
bytes, so spend the engineering on reading HALF the bytes (4-bit resident +
register dequant), not on a faster-but-idle int4 DPAS. Native int4 compute is a
prefill luxury the software stack can't even deliver yet.

--------------------------------------------------------------------------------
## 7. Support matrix (software stacks) -- can it emit a TRUE int4 GEMM?

| Stack                    | int8 GEMM to DPAS | int4-WEIGHT decompress | TRUE int4xint4 DPAS GEMM |
|--------------------------|-------------------|------------------------|--------------------------|
| oneDNN (Intel GPU matmul)| YES               | YES (s4/u4 weight, compute int8/f16) | NO (s4/u4 = storage only) |
| SYCL joint_matrix        | YES               | n/a (no 4-bit type)    | NO (int8 is the floor)   |
| Triton-XPU tl.dot        | YES               | via manual unpack->int8/f16 | NO (int8 floor)     |
| ESIMD / raw GRF asm      | YES               | manual                 | UNCONFIRMED (silicon has the mode; no documented s4/u4 dpas intrinsic) |

Hardware (silicon) has int4/int2 DPAS modes on Xe-HPG and Xe2 (section 1). Every
portable/supported software surface floors at int8 for the ACTUAL matmul.

--------------------------------------------------------------------------------
## 8. Sources

- oneDNN Data Types: https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html
- oneDNN Matmul: https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html
- oneDNN release notes (int4 weight decompression): https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-deep-neural-network-library-release-notes.html
- SYCL joint_matrix combinations: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc
- Intel Xe-HPG architecture (DPAS int4/int2): https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/technology/xe-hpg-microarchitecture.html ; white paper https://cdrdv2-public.intel.com/758302/introduction-to-the-xe-hpg-architecture-white-paper.pdf
- HWCooling Xe2/Battlemage architecture: https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/
- chipsandcheese Lunar Lake Xe2 iGPU: https://chipsandcheese.com/p/lunar-lakes-igpu-debut-of-intels
- Intel VTune "XVE FP16/BF16/INT8/INT4/INT2 XMX Instructions": https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/xve-fp16-bf16-int8-int4-int2-xmx-instructions.html
- Intel Xe GPU Optimization Guide (DPAS sdepth/rcount, arch): https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/intel-xe-gpu-architecture.html
- "Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs", arXiv 2508.06753: https://arxiv.org/pdf/2508.06753
- intel-xpu-backend-for-triton: https://github.com/intel/intel-xpu-backend-for-triton
- Internal: vllm/nvfp4/NVFP4_XPU.md, kernels/int8_gemm_w8a16.h, memory b70-int8-xmx-roofline, zml-int8-decode-layout-bound, w4a8-onednn-kernel-sglang
</content>
</invoke>
