# Native int4 (and int2) DPAS on Intel Arc B70 (Xe2 / Battlemage): REACHED + VALIDATED

Date: 2026-07-04. Card: 1 only (shared lease), TP=1, ZE_AFFINITY_MASK=1.
Toolchain image: `vllm-xpu-env:int8g-v0240` (oneAPI DPC++ 2025.3.3, IGC/VC,
ocloc AOT for `intel_gpu_bmg_g31`). Prototype: `vllm/nvfp4/proto_int4/`.
Companion feasibility survey: `vllm/nvfp4/INT4_DPAS_RESEARCH.md`.

ASCII only. CONFIRMED / MEASURED markers used for load-bearing claims.

--------------------------------------------------------------------------------
## 0. VERDICT (headline)

YES -- a NATIVE int4 x int4 -> int32 DPAS (and int2 x int2 -> int32) is reachable
from a compilable, supported API on this Xe2 silicon: SYCL **ESIMD**
`sycl::ext::intel::esimd::xmx::dpas` with `dpas_argument_type::s4` / `::s2`.

Proven end to end on the B70:
- (a) COMPILES: the ESIMD `dpas<...,s4,s4>` and `<...,s2,s2>` build clean, AOT for
  BMG-G31 (rc=0), no upcast diagnostic.
- (b) EMITS A DISTINCT NATIVE ENCODING: the AOT Gen-ISA machine asm shows
  `dpas.8x8 (16|M0) r16:d null:d r8:s4 r3.0:s4` (and `:s2`), versus the int8
  control's `r8:b r3.0:b`. The source operands carry a native `:s4` / `:s2`
  register type into an `:d` (int32) accumulator -- NOT an int8 widen. vISA shows
  `dpas.s4.s4.8.8` / `dpas.s2.s2.8.8` distinctly. (disasm in
  `proto_int4/disasm_evidence.txt`.)
- (c) NUMERICALLY CORRECT: on the B70, a full DPAS tile validates BIT-EXACT
  (0/128 mismatches) against a CPU integer matmul reference, for s8 (control), s4,
  and s2. Since the s8 control validates the identical host packing + register
  layout code, the s4/s2 passes are trustworthy true int MACs.
- (d) FASTER THAN int8 (MEASURED): in a controlled apples-to-apples microbench
  (identical register footprint, only the precision enum + K-depth differ), native
  s4 DPAS runs at the SAME instruction rate as s8 while doing 2x the MACs per
  instruction -> a clean **2.0x the int8 MAC rate**, matching the hardware
  "int4 = 2x int8" spec and refuting a silent upcast.

This CORRECTS the prior survey's line ("no programmable interface below exposes
them ... UNCONFIRMED whether even ESIMD exposes s4/u4"): ESIMD DOES expose them,
and on this box it compiles, emits native s4/s2 DPAS, runs, and validates.

Practical caveat unchanged from the survey: LLM DECODE (M=1 GEMV) is
weight-bandwidth bound, so a faster int4 COMPUTE does not help decode; native int4
DPAS is a PREFILL / large-M compute lever. See section 6.

--------------------------------------------------------------------------------
## 1. The ESIMD dpas_argument_type enum (verbatim)

From `/opt/venv/include/sycl/ext/intel/esimd/xmx/common.hpp` in
`vllm-xpu-env:int8g-v0240` (identical under
`/opt/intel/oneapi/compiler/2025.3/include/...`):

```
enum class dpas_argument_type {
  Invalid = 0,
  u1 __SYCL_DEPRECATED("u1 is reserved/unsupported") = 1, // unsigned 1 bit
  s1 __SYCL_DEPRECATED("s1 is reserved/unsupported") = 2, // signed 1 bit
  u2 = 3,                                                 // unsigned 2 bits
  s2 = 4,                                                 // signed 2 bits
  u4 = 5,                                                 // unsigned 4 bits
  s4 = 6,                                                 // signed 4 bits
  u8 = 7,                                                 // unsigned 8 bits
  s8 = 8,                                                 // signed 8 bits
  bf16 = 9,                                               // bfloat 16
  fp16 = 10,                                              // half float
  tf32 = 12, // tensorfloat 32
};
```

So the front end names s4/u4/s2/u2 (u1/s1 are explicitly reserved/unsupported).

And the integer branch of the dpas type-check in
`/opt/venv/include/sycl/ext/intel/esimd/xmx/dpas.hpp` explicitly permits ANY
pairing of the integer precisions for BOTH operands into an int32 accumulator
(verbatim static_assert message):

```
" Result |   C   |        B         |           A      \n"
" ud, d  | ud, d | ub,b,u4,s4,u2,s2 | ub,b,u4,s4,u2,s2 \n"
```

i.e. symmetric s4xs4 and s2xs2 are accepted by the header. (The header deduces
K-depth from precision: OpsPerChannel = min(32/bits, 8), K = SystolicDepth(8) *
OpsPerChannel, so s4 -> K=64, s8 -> K=32, s2 -> K=64 with OpsPerChannel capped
at 8.)

--------------------------------------------------------------------------------
## 2. What compiled (compile-only, no GPU)

`proto_int4/build.sh` builds one ESIMD kernel `int4_dpas.cpp` at PREC={8,4,2},
AOT `-fsycl-targets=intel_gpu_bmg_g31`, with `IGC_ShaderDumpEnable=1`.

| PREC | precision | K  | compile | native DPAS emitted |
|------|-----------|----|---------|---------------------|
| 8    | s8 (ctrl) | 32 | rc=0    | `dpas.s8.s8.8.8`     |
| 4    | s4        | 64 | rc=0    | `dpas.s4.s4.8.8`     |
| 2    | s2        | 64 | rc=0    | `dpas.s2.s2.8.8`     |

All three build succeeded; no "unsupported precision" / upcast diagnostic.

--------------------------------------------------------------------------------
## 3. Disassembly ground-truth (the crux evidence)

Final Gen-ISA machine asm (BMG-G31 AOT), the single DPAS instruction per build
(full lines in `proto_int4/disasm_evidence.txt`):

```
s8 :  dpas.8x8 (16|M0)   r16:d  null:d   r8:b    r3.0:b    {$2}
s4 :  dpas.8x8 (16|M0)   r16:d  null:d   r8:s4   r3.0:s4   {$2}
s2 :  dpas.8x8 (16|M0)   r10:d  null:d   r6:s2   r3.0:s2   {$2}
```

vISA:

```
s8 :  dpas.s8.s8.8.8 (M1, 16) V49.0 %null.0 V41.0 V39(0,0)
s4 :  dpas.s4.s4.8.8 (M1, 16) V49.0 %null.0 V41.0 V39(0,0)
s2 :  dpas.s2.s2.8.8 (M1, 16) V49.0 %null.0 V41.0 V39(0,0)
```

The s4/s2 SOURCE operands carry native `:s4` / `:s2` register operand types (the
int8 control shows `:b`), destination `:d` (int32). This is a distinct machine
encoding that the BMG-G31 code generator accepts and emits -- not an int8 widen,
not a compile stub. This is the definitive answer to "is native int4 DPAS a real
encoding the ISA accepts": yes, at both vISA and final Gen-ISA levels.

--------------------------------------------------------------------------------
## 4. Correctness on real B70 silicon

`proto_int4/int4_dpas.cpp` computes one DPAS tile
Result[M=8 x N=16] (s32) = A[M x K] * B[K x N] (B VNNI-encoded), with host packing
of signed nibbles/crumbs, and compares against a CPU integer-matmul reference.
Run on card 1 (`proto_int4/run.sh` under `gpu-run --card 1`):

```
s8  precision=s8  K=32  mismatches: 0 / 128   PASS  (control)
s4  precision=s4  K=64  mismatches: 0 / 128   PASS
s2  precision=s2  K=64  mismatches: 0 / 128   PASS
```

The s8 control uses the SAME packing/layout code (only the precision enum + K
differ for s4/s2), so its pass validates the harness and makes the s4/s2 passes
trustworthy: native s4/s2 DPAS is a TRUE int4/int2 multiply-accumulate into int32.

--------------------------------------------------------------------------------
## 5. Throughput vs int8 (MEASURED)

`proto_int4/bench.cpp`: 8192 ESIMD threads, each a chain of DPAS ops accumulating
in registers; best of 5, card 1. Two variants run.

### 5a. Controlled apples-to-apples (single dependent chain) -- the trustworthy ratio

s4 and s8 have IDENTICAL register footprint (A=64, B=128, C=128 int32) here; only
the precision enum and K-depth differ.

| prec | K  | instr/s   | MAC/s     | note |
|------|----|-----------|-----------|------|
| s8   | 32 | 1.412e10  | 5.78e13   | control |
| s4   | 64 | 1.417e10  | 1.161e14  | SAME instr rate, 2x K -> 2.0x MAC/s |
| s2   | 64 | 3.037e10  | 2.488e14  | ~4.3x s8 (but smaller operands; see note) |

Key result: s4 issues DPAS at the SAME rate as s8 (1.41e10 instr/s) but each s4
instruction does 2x the MACs (K=64 vs 32). Native int4 DPAS = a clean, measured
**2.0x the int8 MAC rate** = ~4x bf16. This directly confirms the hardware
"int4 = 2x int8, int2 = 4x int8" scaling and rules out a silent int8 upcast (an
upcast would have given 1.0x, not 2.0x).

The s2 number (~4.3x) is inflated by a latency effect: s2 operands are half the
GRF width (A=32, B=64 int32), so the dependent chain has lower per-op latency.
Treat s2 as ">= 2x s4 in the compute-bound limit" rather than a precise 4.3x.

### 5b. Higher-ILP variant (4 independent chains) -- corroboration, noisier ratio

| prec | TOPS (2*MAC) | MAC/s    |
|------|--------------|----------|
| s8   | 156.5        | 7.82e13  |
| s4   | 507.6        | 2.538e14 |
| s2   | 390.4        | 1.952e14 |

The ILP variant reaches higher absolute throughput (s4 ~508 TOPS, ~69% of the
estimated ~734-TOPS int4 ceiling) and still shows s4 clearly faster than s8, but
the s4:s8 ratio here (~3.2x) is physically implausible for a pure compute ratio
and the s2<s4 ordering is inconsistent -- i.e. this variant is distorted by
precision-dependent occupancy / register-allocation effects and should NOT be used
for a precise multiplier. The controlled 5a number (2.0x) is the honest headline;
5b corroborates "native s4 is materially (>= 2x) faster than int8".

Absolute-TOPS caveat: neither variant saturates the ~367-TOPS int8 roofline
(these are register-resident latency/occupancy-bound microbenches, not tuned
GEMMs), so the ABSOLUTE TOPS understate peak; the RATIO between precisions in the
controlled 5a case is the load-bearing signal.

--------------------------------------------------------------------------------
## 6. So what -- where this helps (and where it does NOT)

- DECODE (M=1 GEMV) is weight-bandwidth bound (30-300x below the compute
  roofline, memory `b70-int8-xmx-roofline` / `zml-int8-decode-layout-bound`). A
  faster int4 COMPUTE unit does nothing when the DPAS is idle waiting on weight
  reads. Native int4 DPAS is IRRELEVANT to decode. The decode lever remains
  "read fewer weight bytes" (4-bit resident + register dequant), which needs no
  native int4 DPAS. (Unchanged from INT4_DPAS_RESEARCH.md section 5.)
- PREFILL / large-M (compute-bound region, crossover ~M=150-300 on B70) is exactly
  where the measured 2.0x-over-int8 native int4 DPAS pays off -- IF the operands
  are genuinely 4-bit x 4-bit. This matters for schemes with 4-bit ACTIVATIONS AND
  4-bit weights (W4A4). For our current NVFP4 and W4A16/W4A8 paths the activation
  is int8 or fp16, so the relevant instruction is the MIXED s8xs4 (W4A8) DPAS,
  which oneDNN already emits via weight-decompression -- native symmetric s4xs4
  only helps once activations are also 4-bit.
- NVFP4-specific: NVFP4 codes are E2M1 FLOATING, not two's-complement s4. A native
  s4 DPAS treats the nibble as a signed integer, so NVFP4 cannot be fed to s4 DPAS
  without a value remap; and the E2M1*2 -> int8 trick we use needs +/-12 which
  overflows s4's [-8,7]. So native s4 DPAS is a lever for INTEGER W4A4, not for
  NVFP4 as-is (NVFP4 stays on the int8-XMX repack / register-LUT dequant paths).

--------------------------------------------------------------------------------
## 7. How this would be upstreamed / used by the community

The reusable artifact is a standalone ESIMD int4-DPAS microkernel (this repo's
`proto_int4/`), plus the knowledge that the path exists. Concrete community value:

1. A minimal, validated `esimd::xmx::dpas<...,s4,s4>` / `<...,s2,s2>` example +
   the exact VNNI packing that makes it bit-exact -- this is currently
   undocumented (the SYCL joint_matrix combinations table and oneDNN both floor at
   int8; ESIMD is the only door and had no public correctness proof for s4/s2 on
   Xe2). This doc + `proto_int4/int4_dpas.cpp` fill that gap.
2. oneDNN feature request: expose a TRUE s4xs4->s32 (and s2xs2) GPU matmul
   primitive, not only s4/u4 weight-decompression. The silicon + ESIMD backend
   clearly support it (this proof); the blocker is purely the library abstraction.
3. A W4A4 GEMM building block: a rectangular-subgroup ESIMD kernel built on the
   `s4,s4` atom (mirror the int8 sycl-tla mainloop in `kernels/SYCLTLA_SCAFFOLD.md`,
   swap the `XE_DPAS_TT<...,s8,s8,...>` atom for the s4 ESIMD dpas). Only worth it
   once a W4A4 (4-bit activation) scheme is on the table, since decode stays
   bandwidth-bound (section 6).
4. joint_matrix gap note for intel/llvm: the portable joint_matrix surface still
   lists no 4-bit combination on Xe2 even though ESIMD + the ISA support it -- a
   candidate to add symmetric s4/s2 to the supported-combinations table.

--------------------------------------------------------------------------------
## 8. Deadends / caveats encountered (for honesty)

- No hard deadend hit on the primary ESIMD path -- it worked. The survey's
  pessimism ("UNCONFIRMED whether even ESIMD exposes s4/u4") is refuted.
- The ILP microbench (5b) is an unreliable throughput RATIO estimator
  (precision-dependent occupancy); do not cite its 3.2x. Use the controlled 5a
  2.0x. Absolute TOPS in both variants are microbench floors, below the tuned-GEMM
  roofline.
- Not tested here: mixed s8xs4 / s8xs2 DPAS numerics (the W4A8 / W2A8 case), and
  the unsigned u4/u2 variants. The header permits them; correctness + rate left as
  follow-up. u1/s1 are enum-reserved and marked unsupported -- do not attempt int1.
- joint_matrix and oneDNN remain int8-floored (unchanged); the win is ESIMD-only
  and thus hand-written / unsupported-by-portable-API today.

--------------------------------------------------------------------------------
## 9. Files

- `proto_int4/int4_dpas.cpp`  -- correctness kernel (PREC 8/4/2), CPU-ref validated
- `proto_int4/bench.cpp`      -- DPAS throughput microbench (PREC 8/4/2)
- `proto_int4/build.sh`       -- AOT build + IGC asm dump (compile-only)
- `proto_int4/run.sh`         -- correctness run (card 1)
- `proto_int4/build_bench.sh`, `proto_int4/run_bench.sh` -- bench build/run
- `proto_int4/disasm_evidence.txt` -- extracted vISA + Gen-ISA dpas lines
- build artifacts (git-ignored): `/mnt/vm_8tb/b70/int4_dpas_build/`
