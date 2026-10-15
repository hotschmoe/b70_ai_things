# S97 IQ2_XS routed multi-entry source ledger, 2026-10-15

Author: /root/ple_upstream. SOURCE ONLY, no build/GPU/model/payload read.
CONFIG -> Current iq_kernels.dp.cpp, native_gu_port<17,8,true>, SG32/WG256,
N2560/FF640. Source eligibility is not compiled hoisting or a speed result.

COMMAND -> Read Fmt17/vec_dot_iq2_xs_q8_1, Split17/SplitLs2, current generic
MultiW/NW8/row_dot_multi_iq/row_entries and GU port/phase1 ownership. Read
retained INTEL.md alias/split/grid failures, S42 numerical failure and
fixture descriptors only. Source ranges/hashes are separately pinned.

RESULT -> GO for minimal PRIVATE source preparation of Multi17 reuse.
Current Multi<TY> defaults has=false; no Multi17 or KqSplit17 exists.
The LN8 port therefore invokes Fmt17 separately for each entry. Existing
Split17 is used elsewhere and is not evidence the routed lane port hoists
weight decoding. Keep the SAME row/entry/launch mapping; do not switch to
the historically slower upstream EXPERT_SPLIT warp kernel.

## Exact scalar-to-NW8 ledger

1. Block_iq2_xs is74B: half d[0..1],32 uint16 qs[2..65], scales[66..73].
   Fmt17 qk256/ipb8/step2 yields iqs0,2,...14. Two original get_int_b2
   reads cover bytes2+4*iqs..9+4*iqs, last58..65; scale[iqs/2] is66..73.
   All reads are inside the final block, with only original halfword/code
   alignment. DO NOT restore uint16* aliasing into int2: INTEL.md597..601
   records that exact historical compiler failure. Keep uint32 shifts and
   four uint16 extraction expressions unchanged.
2. Code[j]&511 selects iq2xs_grid[0..511]; code[j]>>9 is0..127. Existing
   unpack_ksigns parity + masks + swar_ne4/sub4 yield the same eight signed
   packed int words. Natural order: code0 ->w0,w1; code1->w2,w3;
   code2->w4,w5; code3->w6,w7. Weight-only prep may share these across
   entries of THIS expert/row/K-part; no reuse across different rows/blobs.
3. Each entry reads its own Q8 block kbx*8+iqs/2: u0..u7 from32signed
   bytes and ds[0] only. Q8 bytes/sum/scale are unchanged; repeated ent_tok
   values legally alias READONLY activations. No distinct-xs assumption.
4. Original l0=0,2 make ordered DP4A(w0,u0)->...->(w3,u3) in sumi0;
   l0=4,6 make w4..w7 in sumi1. Generic NW8 splits at4 and is IDENTICAL,
   not a changed reduction of eight dot calls. Each signed byte product
   <=16384 magnitude; four-DP4A half has16 products, |s0|,|s1|<=262144.
   Scales ls0/ls1 in0..15; numerator magnitude<=31*262144=8126464 <INT32_MAX.
   Thus these integer operations do not overflow even with conservative
   signed-byte bounds. This is a domain derivation, not an executed test.
5. Preserve exactly `(s0*ls0+s1*ls1+(s0+s1)/2)/4`: both divisions signed,
   truncate toward0, including negative odd sums. Do not shift/reassociate
   divisions or substitute independent half rounding. Candidate finish:
   `const int sumi = (s0*m.a+s1*m.b+(s0+s1)/2)/4;`
   `const float d = m.d*ds; return d*sumi;`
   m.d is the SAME original half-to-float automatic conversion. KEEP this
   explicit FP statement split; do not borrow Multi22's chained return.
   Signed-byte DP4A comes through iq_kernels94/strata::dp4a/dpct::dp4a.
   Compiler contraction/resource/codegen equality remains a later gate.
6. LN8 keeps k=sub,sub+8,...<80: ten parts in ascending K-block order per
   lane. Each entry's s starts+0.f, adds identical dot values in the SAME
   order, then lanes_sum XOR4/2/1. Prep moves out of the entry loop only,
   not the K loop. No lane split, FMA, FP16 activation or scale policy change.

## Routing/tails/ownership

row_entries works inside one g and row, never across grp_start boundaries.
Natural E4 loop, optional E2 then scalar tail gives n1=1,n2=2,n3=2+1,
n4=4,n5=4+1,n6=4+2; zero writes none. Nonzero e0 and launch grid_groups
smaller than n_groups retain the same g+=gridY loop. Only sub0 writes
gate/up[(e+i)*FF+r]. Repeated/permuted ent_tok changes input selection,
not output ownership. Gate/up rows and separate planes are disjoint;
inactive entries/unused group rows must retain their initial bytes.

Public native_expert_grouped phase1 is a valid GU anchor: at5289 gate is
scratch+0,up+fa,h+2fa,hq+3fa, fa=align256(cap_entries*FF*4). Phase1 returns
at5431 after GU+SwiGLU/quantize but before DOWN. Later steps read, do not
rewrite gate/up. Finite inactive sentinel avoids poisoning its scratch
consumer; compare fullcapacity gate/up and all guards/read-only arrays.
Actual public/private Info/cache/guard/registered-name/device image binding
must be checked before GPU, not inferred from different kernel names.

## Retained fixture and rejection history

native_grouped_parity.cpp Setup/plan_sizes already tests random repeated
token routing, nonzero starts, grid stride and poisoned inactive outputs,
including GU17. It is synthetic and has no toggle for adding Multi17 to
the current port; its v1 control is not the requested new scalar/multi arm.
Reuse its layout contract, not a blind full parity/format sweep. The old
native_expert_bench reads real GGUF but its SYCL file retains CUDA calls;
it is not a ready no-build command for this leaf.

Already exported REAL model data is sufficient: S42 run-clean/exports/
fixtures.tsv row2 is layer0/expert41 QT17/down20, H2560/FF640,
blob_offset3072000, gate473600B/up473600B/down_offset947200,
fullblob1868800. Row12 is layer0/expert211 at20992000 with same geometry.
weights.bin28160000B SHA a9549c6a3c9c44c1c0fd374e62d603d4a2c82bea131f07a765ff6b45d5e13dc6
is already pinned by S80 fixture-pins.json. No payload was read/hashed here.
One existing slice plus retained S65 buffer/graph machinery can feed a
private public-anchor/original/multi leaf without a main build/model copy.
Root selects only inputsA/B and layouts active sizes1..6 plus zero.

Prior INTEL.md542..547: EXPERT_SPLIT=1 was11-16% slower and LN sweeps
changed rounding; do not repeat either. Source384..389 records slower
SLM grid staging; unchanged SWAR was parity-clean without decode gain
(INTEL.md289). S42 grouped oneMKL changed GU bytes at element0 (maxabs
7.45e-8); do not import that accumulation/backend or relax strict bytes.
S65 dense21 SWAR and S75 DPAS are different already-measured mechanisms.

VERDICT -> Minimal candidate is a private Multi17 specialization only,
using existing generic NW8 E4/E2/singleton dispatch and exact finish split.
One source recommendation, no new flags/global edits/staging/lane/format
sweep or predicted speed. Root is obtaining current routed ISA; repeated
decode may already be compiler-hoisted. Strict public-anchor/original
equality and actual binding must precede candidate evidence. No build/GPU
or model consultation is authorized/performed by this source ledger.
