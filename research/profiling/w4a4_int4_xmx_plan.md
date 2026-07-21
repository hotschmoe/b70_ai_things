# W4A4 int4-XMX on B70 (Xe2 / Arc Pro B70) -- kernel design, accuracy finding, staged plan

Date: 2026-07-21. Author: int4-XMX kernel agent (no GPU; write + CPU compile-check
+ CPU fake-quant). ASCII only. CONFIRMED / MEASURED / EXPECTED markers used.

Scope: the genuine "B70 int4 = 2x DPAS" prefill path for **W4A4** (int4 weights x
int4 activations) on qwen3.6-27b. Prefill is COMPUTE-bound (decode is BW-bound),
so int4 XMX is exactly where the 2x-over-int8 MAC rate can show up.

Deliverables (all under `vllm/nvfp4/proto_int4/w4a4/`):
- `s4s4_gemm_microbench.cpp` -- tiled ESIMD s4xs4->s32 GEMM on the real gate/up
  prefill shape, per-token act scale + per-channel weight scale, fp32 accumulate,
  in-file correctness + TOPS. `build_gemm.sh` / `run_gemm.sh`.
- `w4a4_accuracy_probe.py` -- CPU fake-quant W4A4 accuracy design (with/without
  Hadamard rotation) on a real qwen3.6-27b gate_proj slice.
- this plan.

Builds on the PROVEN atom: `proto_int4/int4_dpas.cpp` + `INT4_DPAS_PIONEER.md`
already showed a native `esimd::xmx::dpas<...,s4,s4>` is bit-exact (0/128 mismatch)
and runs at a MEASURED 2.0x the int8 MAC rate on this B70 (disasm: `dpas.s4.s4.8.8`
with native `:s4` register operands, not an int8 widen). This work extends that
single-tile atom into a real tiled GEMM mainloop and settles the accuracy question.

--------------------------------------------------------------------------------
## 0. Headline verdict (go / no-go)

**W4A4 is a SHOWCASE / research microbench, NOT a serve path -- yet.** Two facts:

1. SPEED is real and reachable. The s4xs4 DPAS atom is 2.0x int8 MAC rate
   (MEASURED, PIONEER doc 5a); a tiled GEMM built on it should approach the ~734
   TOPS int4 ceiling for prefill (int8 tops ~367, bf16 ~183). The tiled kernel
   here COMPILES AOT for BMG-G31 and emits native `dpas.s4.s4.8.8` in the mainloop
   (CONFIRMED below). Coordinator measures the TOPS.

2. ACCURACY is the blocker, and it is severe. CPU fake-quant on a real qwen3.6-27b
   gate_proj (MEASURED, probe below):
   - W4A4 **no rotation**: output cosine **0.796**, SNR **4.1 dB**, relerr 0.62 --
     BROKEN (naive int4 activations are catastrophic; matches SpinQuant/QuaRot lit).
   - W4A4 **+ block-Hadamard (parameter-free QuaRot R)**: cosine **0.973**, SNR
     **12.6 dB** (+8.4 dB) -- rotation is MANDATORY and recovers most of it, but
     still below the ~20 dB "safe for code evals" bar.
   - W4A4 + Hadamard + per-group-128 weight scale: cosine **0.988**, SNR 16.0 dB
     -- reaches W4A16-RTN level, still short of W8A8 (cosine 0.999, 26.9 dB).

Conclusion: even a well-rotated W4A4 only lands around W4A16/W4A8 accuracy while
costing an ONLINE Hadamard kernel we do not have, and decode stays weight-BW bound
so the 2x prefill TOPS does not translate to a 2x serve win. **Do NOT build a W4A4
serve path now. Keep the tiled s4xs4 GEMM as the Intel "native int4 DPAS 2x" TOPS
showcase, and as the ready building block for if/when a FlatQuant-grade rotation +
online-FWHT kernel make W4A4 accuracy competitive.** Prioritize W4A8 (int4 weight x
int8 act) for a real serve path -- same INT8-XMX datapath, ~lossless-ish, no online
rotation, already partly built (`w4a8/`).

--------------------------------------------------------------------------------
## 1. The kernel: tiled s4xs4 -> s32 GEMM (`s4s4_gemm_microbench.cpp`)

### 1.1 What it is
A real tiled GEMM mainloop (not a single DPAS tile) on the fused gate/up prefill
shape `out[M,N] = dequant(sum_k Aq[M,k] * Wq[N,k])`, M in {512,1024,2048},
N=34816, K=5120. Symmetric signed int4 in [-8,7]: per-token activation scale
(per row m), per-channel weight scale (per output col n), fp32 accumulate.

### 1.2 DPAS tiling
- DPAS atom: SystolicDepth=8, RepeatCount(M)=8, ExecSize(N)=16, **s4 K-depth=64**
  (OpsPerChannel = min(32/4, 8) = 8, K = 8*8 = 64). Twice the int8 K-depth (32) --
  this is where the 2x MAC/instr comes from. One DPAS = 8x16 x K64 = 8192 MACs.
- Each ESIMD work-item computes an **8 x (16*NSUB)** output tile (default NSUB=4 =>
  8x64). The 8-row A tile (64 int32) is loaded once per K-step and REUSED across
  NSUB independent B subtiles (128 int32 each) -> A-reuse + **NSUB-way accumulator
  ILP** to hide DPAS latency (this is the pattern that reached ~508 TOPS in
  `bench.cpp` 5b; a single dependent chain is latency-bound at ~116 GMAC/s).
- K-mainloop walks numKt = K/64 = 80 DPAS steps, accumulating into NSUB int32
  accumulators. Epilogue: `out = acc * act_scale[row] * wt_scale[col]` (vectorized
  per row), store fp32.

### 1.3 Data layout (offline-packable, contiguous device loads)
- Weights Wq[N,K] -> **VNNI B-tiles** `Btile[(kt*numNt + nt)*128 + (kk/8)*16 + nn]`
  (B[k,n]=Wq[n,k], 64x16 tile, 8 int4 per dword). Offline weight repack -- free for
  prefill (weights static).
- Activations Aq[M,K] -> blocked `Atile[(kt*M + m)*8 + kk/8]` so the 8-row A operand
  is 64 contiguous int32. This IS a prologue repack cost per prefill (the probe
  packs on host); a production kernel folds the per-token act-quant + nibble-pack
  into the attention epilogue (same fusion the W8A8 path wants).
- Both layouts give fully contiguous `copy_from` on device (no gather).

### 1.4 Correctness (in-file, two references)
- KERNEL correctness: GPU fp32 output vs a CPU **int-exact** reference of the SAME
  quantized operands (`sum Aq*Wq`, then dequant). This isolates kernel/layout
  correctness from quant error -- EXPECTED relerr < 1e-3 (the int MACs are exact;
  only float dequant rounding differs). This is the load-bearing gate (PASS/FAIL).
- QUANT accuracy: GPU output vs the original fp32 GEMM -- the W4A4-no-rotation error
  signal (EXPECTED large, ~0.5-0.6 relerr; see the probe for the real analysis).

### 1.5 Compile status (CONFIRMED, no GPU)
`build_gemm.sh` AOT-builds for `intel_gpu_bmg_g31` in `vllm-xpu-env:int8g-v0240`:
```
compile rc=0   Build succeeded.
native dpas mnemonics: 8x "dpas.s4.s4.8.8"  (mainloop, NSUB unrolled)
s4 register operands:  dpas.8x8 (16|M0) r114:d null:d r10:s4 r4.0:s4   <- native :s4, dst :d(int32)
```
So the tiled mainloop emits the SAME native s4xs4 DPAS the atom proved -- not an
int8 widen. Binary at `/mnt/vm_8tb/b70/int4_dpas_build/w4a4/s4s4_gemm`.

### 1.6 TOPS: expected vs ceiling (coordinator MEASURES)
- Ceiling: int4 compute ~734 TOPS (2x int8's 367, 4x bf16's 183) at the arch level.
- The atom microbench (`bench.cpp`) hit s4 ~508 TOPS (69% of ceiling) in the 4-chain
  ILP variant and a clean 2.0x-int8 ratio in the controlled single-chain variant.
- EXPECTED for this real-shape tiled GEMM: below the register-resident microbench
  (it now pays real memory traffic for A/B loads + the epilogue), but it should
  clearly beat the int8 GEMM's measured prefill (~50% of 367 = ~180 TOPS effective)
  if memory-bound effects don't dominate. Tuning levers if it undershoots: larger
  N-tile for more A-reuse, GRF-large mode (raise NSUB), 2D-block prefetch of B,
  double-buffer the K-loop. The honest headline stays the MEASURED 2.0x MAC-rate
  ratio; absolute GEMM TOPS is the coordinator's number.

--------------------------------------------------------------------------------
## 2. The accuracy finding (`w4a4_accuracy_probe.py`) -- MEASURED

CPU fake-quant, real qwen3.6-27b `layers.2.mlp.gate_proj` bf16 slice [1024,5120],
synthetic activations (Gaussian + 1% x12 channel outliers = the realistic A4
killer), block-Hadamard block size 256. Output metric on Y = X @ W^T:

```
config          relerr    cosine   SNR(dB)   maxAbsErr
W16A16          0.0000   1.00000    241.4     0.0000     (sanity)
W8A8            0.0454   0.99897     26.85    0.351      <- ~lossless, the bar
W4A16           0.1658   0.98654     15.61    3.131      (int4 weight-only, RTN)
W4A4            0.6210   0.79614      4.14    4.128      <- BROKEN (no rotation)
W4A4+Had        0.2358   0.97333     12.55    1.705      <- rotation MANDATORY
W4A4+Had+g128   0.1577   0.98781     16.04    0.907      <- +group scale ~ W4A16
```
Mechanism (MEASURED): activation outlier ratio (max/median |x|) = **73.3 raw ->
10.3 after block-Hadamard**. The Hadamard mixes each block so per-channel outliers
spread across the block, shrinking the int4 activation range -- exactly why cosine
jumps 0.796 -> 0.973. Robust across a random-weight cross-check (same numbers).

Honest caveats:
- Activations are SYNTHETIC. A real HumanEval+ delta needs a calibration-trace X
  (coordinator has the model). This probe fixes the DESIGN, not the eval number.
- The W4A16/W4A4 weight quant here is plain **RTN** (round-to-nearest). Real
  W4A16-GPTQ/AutoRound has error compensation and is materially better than the
  0.166 relerr shown -- so RTN OVERSTATES weight error and understates the relative
  cost of the ACTIVATION quant. The activation conclusion (rotation mandatory) is
  unaffected and is the load-bearing result.
- Block-Hadamard (size 256) is the parameter-free QuaRot R3/R4 online transform, a
  LOWER bound on rotation quality. Learned rotations (SpinQuant) or affine
  transforms (FlatQuant) recover more -- literature: QuaRot ~4pt zero-shot gap /
  fragile on Llama-3-70B; FlatQuant <1% even at 70B. Getting W4A4 to W8A8 quality
  needs FlatQuant-grade offline optimization on top of the online Hadamard.

--------------------------------------------------------------------------------
## 3. The rotation recipe (which R's fuse offline vs need an online kernel)

QuaRot/SpinQuant/FlatQuant decompose the rotation into four R's:
- **R1 (residual/embedding-level) and R2 (inside attention V/O)**: these commute
  with the linear weights and **FUSE OFFLINE** into the weight matrices -- zero
  runtime cost, no kernel. Do these at quant time.
- **R3 (queries/keys, pre-RoPE) and R4 (down_proj input / MLP)**: these sit on an
  ACTIVATION path that changes per token, so they need an **ONLINE Hadamard (fast
  Walsh-Hadamard transform, FWHT) kernel** on the fast path, applied per fixed-size
  block (128/256, a power of 2). This is the piece WE DO NOT HAVE.

So the W4A4 serve path needs, beyond the s4xs4 GEMM:
1. An **online FWHT kernel** (block 128/256) fused into the GEMM prologue -- rotate
   the int4 activation block right before quant+pack. On B70 this is a cheap ESIMD
   butterfly (log2(B) stages of add/sub) but must be FUSED to not spill to scalar
   cores (the classic W4A4 systems trap -- MIT ships W4A8KV4 precisely to avoid it).
2. FlatQuant/SpinQuant offline calibration to close the last accuracy gap (the
   parameter-free Hadamard alone lands at ~W4A16, not W8A8).

Scope of the missing online-FWHT kernel: small (a fused butterfly over a 128/256
block per token), but it is a NEW kernel + a quant-time rotation-fusion pipeline,
and it only buys W4A4 the right to MATCH W4A8. That ROI is why W4A4 stays deferred.

--------------------------------------------------------------------------------
## 4. Staged plan (microbench -> accuracy -> full layer -> serve) + gates

- **S0 -- atom (DONE, PIONEER doc):** native s4xs4 DPAS bit-exact + 2.0x int8. [done]
- **S1 -- tiled GEMM microbench (THIS work, compile CONFIRMED):** real gate/up
  shape, scales, correctness, TOPS. GATE: coordinator runs `run_gemm.sh` -> PASS
  (relerr<1e-3) AND TOPS > int8 prefill effective. If PASS, the SHOWCASE is real.
- **S2 -- accuracy design (THIS work, MEASURED):** rotation is mandatory; parameter-
  free Hadamard = +8.4 dB, lands ~W4A16. GATE (to proceed past showcase): a real
  calibration-trace HumanEval+ with FlatQuant-grade rotation must reach W4A8 parity
  (>= w4a16-gptq 0.848). EXPECTED: hard; this is the go/no-go that currently says
  NO for a serve path.
- **S3 -- online FWHT kernel (ONLY if S2 clears):** fused ESIMD block-Hadamard in
  the GEMM prologue; validate rotated-GEMM bit-exactness + that fusion keeps the
  2x TOPS (no scalar spill).
- **S4 -- full-layer / serve (ONLY if S2+S3 clear):** wire s4xs4 GEMM + online FWHT
  as a prefill path (decode stays on the 4-bit-in-VRAM w4a16 BW path -- int4 compute
  is irrelevant to decode). Concurrency sweep vs W4A8/W8A8 before any shelf entry.

**Current gate status: STOP at end of S1/S2 as a showcase.** S3/S4 blocked on the S2
accuracy go/no-go, which the synthetic probe already suggests fails without
FlatQuant-grade work. Reassess only if (a) prefill throughput becomes the proven
serve bottleneck AND decode is already at its 4-bit BW ceiling, or (b) a future
oneDNN/joint_matrix exposes s4xs4 so the online-FWHT is the only bespoke piece.

--------------------------------------------------------------------------------
## 5. Files
- `vllm/nvfp4/proto_int4/w4a4/s4s4_gemm_microbench.cpp` -- tiled s4xs4 GEMM (compiles, dpas.s4.s4)
- `vllm/nvfp4/proto_int4/w4a4/build_gemm.sh` / `run_gemm.sh` -- AOT build (no GPU) / GPU run
- `vllm/nvfp4/proto_int4/w4a4/w4a4_accuracy_probe.py` -- CPU fake-quant accuracy (rotation mandatory)
- Prior: `vllm/nvfp4/proto_int4/int4_dpas.cpp` + `INT4_DPAS_PIONEER.md` (the proven atom),
  `INT4_DPAS_RESEARCH.md` (feasibility survey), `docs/literature/11_int4_fp4_landscape_w4a8_roadmap.md`
  (W4A4 vs W4A8 literature: QuaRot/SpinQuant/FlatQuant, why MIT ships W4A8KV4 not W4A4).
