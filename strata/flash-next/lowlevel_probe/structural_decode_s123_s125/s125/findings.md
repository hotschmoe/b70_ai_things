# S125: primary dense Q6_K pre-expansion / canonical-Q8 DPAS

CONFIG -> SOURCE ONLY, 2026-10-16. Read current source and retained S41 tensor
metadata/S46 capacity/S74-S75 results. No model/header reread, implementation,
build, GPU, model consultation or live source change. Exact pins/excerpts and
integer-only layout calculations are beside this report.

COMMAND -> Inspect native_mmvq.dp.cpp:96-105,1550-1645; NativeDense load/free;
current XMX helpers/CMake; retained q6-header-index.json. capacity-algebra.json
is derived from descriptor dimensions/counts, not model payload or allocation.

RESULT: algebra is possible, but a plain K32 matmul is not the native operation.

- Native SG32 lane l has g=l%8, sub=l/8; kbx=sub+4*j. Each iteration computes
  TWO independent dot16 values s0/s1. For g, their positions inside a block are
  a=128*(g/4)+16*(g%4), a+64. Their signed scales are scales[h],scales[h+4],
  h=8*(g/4)+g%4. Each Q8 packet scale is per32; neighboring weight half-scales
  cannot be merged by an unscaled K32 dot.
- Decode produces exact int8 q in [-32,31]; scales are int8. Baking scales into
  int8 can require magnitude4096. Keep scales and raw half-d separate. A dot16
  with arbitrary int8 activation has |s|<=65536, hence exact INT32 and FP32
  conversion. No FP16 activation substitution or conversion is proposed.
- Preserve dsc0=dw*float(sc0), dsc1=dw*float(sc1), then EXACT source expression:
  acc += dx0*dsc0*float(s0) + dx1*dsc1*float(s1).
  Accumulate kbx in that lane's original order; finally XOR16/8/4/2/1 and store
  lane0. Folding scales into integer dots, accumulating across differently
  scaled K segments, sequential K summation, or changing pair parenthesization
  does not establish this contract. Source order still needs compiled/full-byte
  checks; optimizer contraction/storage differences are not disproved here.

RESULT: ONE distinct, source-feasible concept (not a prepared kernel).

- First eligible shape: C6,N6144,M2560, original native wide A2/RPW1/PF0 path.
  Immutable compact tiles store the 256 signed codes +16 raw int8 scales +2 raw
  half-d bytes per original256 weights. Pack only integer fields; never create
  rounded FP16 weights or precompute scales via host sycl::half conversion.
- Use one SG16 group for (8 output rows, native lane l). Matrix A is8x32:
  six canonical activation rows concatenating the two original16-byte runs,
  two zero rows. B is32x16: virtual columns2r/2r+1 contain respectively
  [q0(16),zero16] and [zero16,q1(16)]. One signed INT8->INT32 matrix MAD gives
  both original dot16 values for eight real outputs. Reset its INT32 accumulator
  at EVERY kbx; shared zero staging is required, not persisted doubled weights.
- Store the full INT32 tile before a local barrier, then unique output owners
  update six private FP partials with the exact native expression/order. Reuse
  A/B/results/scales only after all consumers finish. Each group handles ONE
  native lane, so there is no S75-style serial32-lane FP ledger in SLM.
- Save32 partials/output to global scratch; a SECOND kernel performs the original
  SG32 XOR tree, stored lane0. Whole candidate includes both launches/dependency
  and scratch traffic. For M2560 this is10240 SG16 groups, scratch1966080B;
  N2560/M10240 would require40960 groups/scratch7864320B. No shape sweep selected.
- A/B/INT32 staging base is1280B/group plus scale storage/compiler overhead.
  Padding alone leaves37.5% useful MAC cells (6/8 token rows and half-zero B).
  That is logical geometry, not measured XMX utilization, traffic or occupancy.

RESULT: memory and ownership are material costs.

| Dense subset | Original bytes retained | EXTRA compact expanded bytes |
| --- | ---: | ---: |
| One6144x2560 | 12902400 | 16834560 |
| 29 out +14 qkv | 675225600 | 881008640 |
| S41 large52 | 842956800 | 1099857920 (1048.90625MiB) |
| All71 Q6 matrices | 864460800 | 1127915520 (1075.6640625MiB) |

Expanded/original=274/210; original+expanded=484/210. These are logical bytes,
not allocator/free-memory measurements. Historical1778MiB primary free minus
large52 extra leaves729.09375MiB, below the reference1024MiB reserve; peer2010
is not primary room. This neither predicts a new149/363 split nor guarantees
all24576 experts remain resident. Actual pre-cache pricing/free/reserve and
both-card residency would be mandatory before serving; ownership changes also
change the interpretation of an end-to-end comparison.

NativeDense owns originals, which prefill/fallback still need. A later pack
would be completed before cache pricing/capture (generate2939 vs4480), preferably
while original host bytes already exist: extra decode/write/upload has an
unmeasured startup cost. Publish only completed owner-managed immutable tiles;
allocate each captured user's scratch before capture. Free tiles/scratch only
after all graphs/users drain. Current owner has no Q6 expanded registry/lifetime
API; weight_bytes/allocated_bytes accounting must not silently omit new planes.

VERDICT -> CONDITIONAL SOURCE FEASIBILITY; HOLD implementation/build/GPU.

This is distinct from S44 compressed-plane DP4A packing and S75 on-the-fly IQ3
single-SG serial-K DPAS (strict PASS,7.7-7.9x slower): new mechanism is offline
Q6 decode plus multi-output DPAS with native K parallelism. Existing xmxg/xmx
helpers use FP16; xmx_int8_bench uses per32 epilogues/ei++ and skips C6 stores,
so none supplies this exact path. S74/S75 establish selected BMG INT8 capability,
not this new layout. No current public dense API avoids two half-dot outputs
and the FP-order problem automatically.

The smallest future discriminator, ONLY if root selects it, is one retained
S44 out fixture: exact decoded codes/scales/raw-d and INT32 dot16 first; full
public/native-versus-candidate initial/changed/restore bytes, guards/readonly,
own graphs; then balanced COMPLETED two-kernel timings with pack cost/memory
reported separately. No numerical tolerance, speed forecast, serving claim,
permanent VRAM commitment or authorization to execute follows from this audit.
