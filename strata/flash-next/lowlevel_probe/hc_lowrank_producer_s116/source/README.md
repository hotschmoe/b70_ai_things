# S116 private HC producer oracle SOURCE preparation (2026-10-16)

CONFIG -> ONE external private oracle, source only. Root full-read accepted
S115 selected original ISA and gave HC_SOURCE_PREP_GO. Actual Opus/Fable source
review, CPU_BUILD_GO, new compiled audit and GPU_GO are still REQUIRED. Nothing
here authorizes a build/run or production edit. Native source remains untouched.

Exactly SIX private namespace-scope classes: shared HCNormS116/HCDownS116,
HCControlReduceS116/HCControlUpS116, HCCandidateReduceS116/HCCandidateUpS116.
Each tag has ONE kernel launch/lambda definition. Helpers/functions/types and
macro use a private namespace/prefix. No public HC name/Info/cache collision,
whole fused_gr TU include, all-variant build, frozen archive prelink or main AOT.
Only four extracted control bodies, two candidate copies and needed helpers.
Template MORE/ILP arms remain source text but only false/C16/LO_ONCE instantiate.

Scope: norm-split ->80-slice down ->normal reduce ->C16 UP, all FOUR boundaries,
SG32/default GRF, MORE0/ILP0/V30/sliced1/nsplit1, T1/2/6 only. Private direct
launchers choose these exact paths, rather than environment-mediated public
selection. Header preserves public FusedGrArgs layout; own Multi matches source.
Other kernels/paths/T/aliases are outside this fixture and unchanged.

ONE mechanism, in kernel-delta.patch:
- Candidate reduce preserves80-slice raw sum and m.part raw publication, then
  NEW float+0; add m.part value; source /=float(HC); unary minus/native exp;
  denominator +1; final division; GR_PICK owner lo publication. Inject branch
  remains original. Do not multiply0.25 in source, use a sigmoid helper, skip
  extra +0, reassociate sums or invoke a universal negative-zero argument.
- Candidate UP stages the completed owner lo in the SAME SLM, then keeps all
  original dot/FMA/butterfly/gates/residual/mean arithmetic and BOTH barriers.
  Its original global lo store is removed. Norm/down and raw part2 are common.

Materialization tradeoff: candidate source reads m.part immediately after its
preserved FP32 store. No volatile is added. Compiler forwarding may remove the
physical reload. Later ISA must trace the exact 32-bit operand published by the
raw store into extra+0 and the complete epilogue, rejecting a wider accumulator,
changed rounding boundary/order or unsupported mode inference. Source equality
alone does not prove store/load rounding equivalence. A volatile reload would
change data movement and is not silently substituted on a mismatch.

COMMAND -> Preparation only; no script imported/executed, binary/archive/model
payload read, CLI/model call, build, GPU or network work. Source small reads,
metadata/stat of retained anchors, projections, lexical/static checks only.
Future root-approved finite order:

1. Full-read this packet and actual Opus/Fable responses. Bind e41 image,
   DPC2026.1/O3/precise/SG32/perkernel/BMG-G31/correct-rounded-divsqrt and helpers.
2. bin/cpu-run bash build.sh (ONLY after CPU_BUILD_GO). NoDRI/networknone/UID1000,
   jobs4/diskTMP; one private TU ->six images ->small SYCL-only host link. No
   original archive/main AOT or rehash. New build directory only, no retry.
3. CPU selected compiled audit BEFORE GPU: six distinct host Info/cache/guard/
   registered-name/image bindings. Compare control UP/reduce to S115 actual
   images (UP d8b21b... /reduce e3bc58...). First require exact old computational
   prefix through EOT82848/16272, normalized argument/execution metadata,
   options/constants/relocations and resources. Any unexpected difference HOLD
   for explicit root inspection; never silently weaken to source similarity.
4. Inspect candidate raw-part store and FP operand/operation order, exact
   immediates/div lowering/CR0, original unchanged UP downstream body and
   actual resources/spills. Raw image/text byte identity, computational body
   identity and ordered FP ledger are DIFFERENT claims. Register/scheduling
   differences must be explicit. CR0 OR0x400004C0 from S115 is raw observation,
   not denorm/rounding/exception interpretation. Missing metadata stays unknown.
5. bin/cpu-run python3 verify_anchors.py BEFORE GPU: verify36 actual retained
   S39 raw hashes against s39-anchors.json. This script has NOT been run during
   source prep. Their747776-byte sizes were stat-checked; hashes are historical.
6. Only separate root GPU_GO: one explicitly pinned physical card under
   bin/cpu-run --quiet OUTSIDE bin/gpu-run --card N, matching selector/PCI;
   oracle NEW_OUTPUT_DIR --pci PHYSICAL_PCI [--timing]. Root prepares lifecycle,
   image/executable identity, pre/post health, normal container exit/removal
   and analyzer; no runner/lease or health bypass is introduced here.

RESULT -> Source contract; no runtime PASS claimed. source-projection.json
records exact selected source spans/SHA and inverse identifier-only control
projection, plus exact inverse of both candidate seams. These are not compiled
identity/numerical proofs. Original control reference is rooted in S115 actual
selected images; fixture RNG/snapshot/lifetime idioms reuse pinned S39/S73/S112.

Expected strict coverage (source-derived counts, future runtime verifies):
-12 base cases T1/2/6 x apply0/1 x inject0/1. Each executes initial A, changed B,
  then A again, eager and captured, both routes:72 eager +72 captured WHOLE reads
  total, not144 independent base cases. Eager prefix snapshots preserve exact
  raw part2/raw part/inject/rs/xn/R; candidate pre-UP lo equals control post-UP
  lo. All final R/lo/rs/inject/mixed/xn/part/part2 bytes match, including inactive
  planes/tails, guards, untaken writes and read-only inputs/weights. Graphs keep
  stable addresses/immutable mode and replay actual changing data. Mixed must
  change on B and return byte-exact on A; captures must equal their own eager.
-36 UNIQUE retained S39 anchors checked inside the native oracle BEFORE timing,
  byte-for-byte in exact historical field order R/lo/rs/inject/mixed/xn. Four
  T1 direct-A anchors plus32 T2/T6 direct/captured A/B. No claim of48 anchors or
  a new reference generated from candidate. Raw anchor files are read-only.
-Private owner buffers are separate USM allocations with explicit span checks
  against other owner buffers and shared read-only inputs/weights. R_out==R
  remains the intentional S73 in-place alias. Every lo span is private and
  disjoint from part/part2/R/xn/other owners. No generic public alias claim.
-The selected six HC kernels use NO counters. No counter reset/recovery logic
  is introduced. Buffer data/output poison occurs only before a planned next
  successful fixture execution; no reset/retry after any failed result.

Special FP diagnostic, same SIX kernel classes:
-Synthetic part2 ->actual reduce ->UP, T2/apply0/inject1.14 raw patterns cover
  +/-0, +/-minimum/maximum subnormal, +/-minimum normal, +/-1, +/-Inf and two
  quiet-NaN payloads. Initial A, changed second-slice B, A again, eager/captured
  both routes:12 TWO-kernel route executions, no norm/down in this diagnostic.
  All raw parts/finals remain exact; actual reduced raw-part bits and candidate
  pre-UP lo bits are printed. No finite requirement on intentional nonfinites.
  Never assume a +/-0 input partial produces the same-signed reduced zero.
-ONE DIRECT CONTROL-UP-ONLY raw-part probe bypasses reduce, using the same
  existing control UP class and exact bit inputs. It prints source-input/lo
  bits and checks guards/read-only spans. This is NOT candidate cross-context
  proof. Reachability gaps remain explicit for root/actual model review; no
  seventh class or runtime branch is added to manufacture them.

Optional timing: only after ALL strict/anchor/special checks pass. Reuse the
already warmed T6/apply1/inject1 FULL four-kernel graph from correctness, two
ABBA rounds (8 total points,4 per arm), one completed whole read per point.
Uploads/poison/pre-drain occur outside window; Clock covers graph submission
through q.wait_and_throw. Check full bytes after EVERY point. No UP-only/device
cycle/model throughput claim, long soak, sweep, automatic rerun or setup repeat.
Four kernel boundaries and matched source work stay intact. Read-only full
weight checks occur at strict/diagnostic/final boundaries, avoiding a repeated
13MB transfer for every unchanged case. No new full-size snapshot files needed.

Capture cleanup balances a still-owned recording BEFORE draining on record/
end/finalize error, preserves/rethrows original exception, then reports/stops.
Normal destruction drains before USM free; cleanup errors block final verdict.
Any mismatch/nonfinite in ordinary cases, unexpected bit/guard/input/capture/
anchor/resource failure means HOLD. No tolerance, reinitialization as recovery,
recapture, redraft or retry. Optional timing never starts after strict failure.

VERDICT -> FROZEN SOURCE ONLY for root and actual Opus/Fable review. Candidate
correctness, selected compiled control reuse and net whole-read benefit remain
unproved. No serving prediction from S115 source160-fold geometry/static op
counts, older recorded HC service or S112 private host-chain percentages.
