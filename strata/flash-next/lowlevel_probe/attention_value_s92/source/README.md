# S92 corrected binding for retained S13 value pipeline (source only)

CONFIG -> Private INT8 attention oracle, SG32/256 threads/CHUNK64/G12,
NC1; current public00bf1c45 source and bbae92dc fatobject. Candidate source
qsa-value-gather.cpp is BYTE-EXACT retained S13 source49ff3788, not S91's
reduce12 source. No production source, math policy, query batching, lane-cell
ownership, kernel geometry, root property or merge change is proposed.

COMMAND -> Root full source/model review before ANY build. Build ONLY the
candidate TU and existing strict fixture, prelink their private images,
ordinary host-link with the frozen current public object and EXACT existing
S91 public.images.o df7655d8. No public TU/common/engine rebuild or public AOT
regeneration. build.sh uses existing pinned e41, cpu-run, four CPU jobs,
no /dev/dri, --network none, disk TMP/cache, user UID/GID and fresh build/.
setvars precedes strict mode; target BMG-G31, precise math and correctly
rounded divide/sqrt. build/ DOES NOT EXIST during this preparation.

RESULT -> SOURCE PREPARATION ONLY. All eleven candidate chunk/merge class
tags are renamed via compile-D to s92_private_* IN ADDITION TO exported
value_batch/value_step/value_scratch names. The original source bytes remain
unchanged. kernel-class-renames.json and source-change.json inventory them.
The one lane-cell class and all four-mode batch/step/merge classes are
included, although this fixture exercises ONLY default INT8 batch+merge.
STRATA_ATTN_LANECELL must be ABSENT, including not0, in this fixture.

Why this is new evidence: root's S91 host audit found shared weak
DeviceKernelInfo/CompileTimeKernelInfo despite distinct device IDs; root's
independent historical S13 audit found the same defect. Historical S13 PASS
logs remain untouched, but their neutral candidate-attribution is invalid.
S91 private spill/scratch and bad binding do not justify a rename-only S91
retry. S92 asks the original S13 value question with genuinely separate host
kernel metadata. No new runtime or numerical outcome has been obtained.

Math/SLM audit of retained source:
- The query sq[G][HD] array holds3072 floats. After ALL query reads, the
  existing post-score workgroup barrier separates them from new probability
  writes. Probabilities occupy64*12=768 floats in this retired query region.
  They never alias still-live scores sp[G][CHUNK] read by the exp/max loop.
- Each (cell,head) has one writer: warp w visits heads w,w+8 and lane
  visits cell lane,lane+32. All heads0..11 are covered; no others are read.
  Existing following workgroup barrier completes probability publication.
- Cell-major probability stride12floats=48bytes; float4 groups start at
  offsets0/4/8. The original sq base already supports float4 query reads.
  The bounded768-float payload fits without new local allocation.
- Score/Kload/exp/max/sum/part_m/part_l are source-identical. Only probability
  destination/layout and the subsequent value loading/folding differ.
- Four values are loaded ahead, but fold(c),fold(c+1),fold(c+2),fold(c+3)
  update each head accumulator in EXACT original increasing-cell FMA order.
  On any negative page in a four-cell group the loop breaks before loading;
  scalar fallback resumes at that same c, skipping negative rows as before.
  n_here0 returns as before; n_here1..64 and ragged groups keep old bounds.
- The body retains generic KV modes, but this finite oracle only qualifies
  INT8. Other modes and production cap2051 are not independently qualified
  by synthetic cap2112. Compiler contraction/resource effects remain actual
  build/runtime questions; source arithmetic alone is not strict proof.

Fixture: attn_value_probe.cpp is EXACT S91 strict36 fixture except candidate
value_* names and required diagnostic comments/messages. Case matrix,
seeds/RNG, original FP64 baseline sanity2e-5, full output/scratch byte gates,
4096-byte guards,64-byte data alignment, readonly witnesses, own initial
repeat, sign-flipped query/changed repeat, restored input/output, and raw
4warm/20alternating timing rounds are unchanged. Width0 observability stays
vacuous. There are no graphs because this production prefill path is eager.
No epsilon applies to cross-candidate acceptance. ALL36 must pass before
any timing acceptance; raw host-completed samples are not device service.

VERDICT -> Ready for root source review ONLY, not build/GPU permission.
After an authorized build, inspect actual public/candidate host callsites,
weak DeviceKernelInfo/CompileTimeKernelInfo separation for ALL eleven tags,
plus per-arm registered-name pointers in the actual final executable and
map/wrapper retention. Distinct device IDs/nm exports alone are NOT the
binding gate. Eleven source renames are a proposed correction, not measured
host-binding success. Preserve the exact reused public object/AOT hashes.
Actual private AOT is newly generated; no same-GPU-binary claim. Kernel
resources/spills and any common helper overlap must be reported, without
assuming all overlap invalid or harmless. Numerical success, health/cleanup
and a meaningful timing comparison remain future conditional steps. No
serving, cache/fullcontext, old-neutral, speed or target claim follows here.
