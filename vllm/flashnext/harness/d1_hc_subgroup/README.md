# D1 C1 HC subgroup-local candidate, CPU preparation only

CONFIG: original no-spec O4, D1plan16 OFF, N1 OFF. Target short-decode
critical path. Candidate is scratch-only, no serving/fork/plugin changes.
CPU build now PASS, active36s after Q1 quiet lock release. No GPU run or
model load performed by this preparation stage; build-result.json pinsSO.

COMMAND: CPU source inspection plus analyze_ld.py/cpu_check.py, no Torch
imports. ld-census.json is derived from trusted saved Ld steps.pkl; input
SHA pinned. Exactly506 stable1938-kernel card0 steps, rather than including
five startup/boundary steps. ASCII/AST/bash checks and81920 row-lane/chunk
ownership cases pass. This does not test compiled numerical behavior.

RESULT: median traced device kernel sums per step:

- HCup97calls1.249625ms + collapse97calls0.158ms; HCdown1.2315ms.
- Router topk48calls0.60075ms + dispatch0.17275 + bump0.06725ms.
- Entire source-mapped PLEstage103kernels0.3095ms; gate+conv alone0.015ms
  is NOT the full PLE cost. Stage also includes metadata/hash, gather/DQ
  and two projections. No claimed percentage of this stage is eliminable.

HC cost is stable over early/late64tokens; it does not explain the FA KV
slope. Trace instrumentation affects absolute duration. Ld is ORIGINAL
unfused route/HC, confirmed ex-Ld.txt and kernel names. Existing Lc
route+Q8F combined already improved16.25->15.70ms mean, and old component
48mixes(no norm)1.639->1.216ms atM1. Those are existing gains, excluded from
any new estimate. Current matched canonical original61.64tok/s trails
Strata66.2; no arithmetic sum here proves an end-to-end rate.

Choose one small HCup candidate before broad four-projection streaming:
M1/K320/H2560 only, same original LPR8 chunk/FMA/XOR sums, selected lanes0/8,
BF16 gate conversion, sigmoid rounding, product rounding and stream0..3
sum. Each subgroup owns one h and computes streams0/1 then2/3 in separate
accumulators. Four subgroups/workgroup own four h:640groups rather than
1280. This removes cross-SG SLM and group barrier and reuses activation
loads for two rows. Unique namespace/kernel b70_d1_hc_sg_r1/UpCollapseM1.
Inputs are guarded exact shapes/device/dtype; output allocated fresh.
M2..8 must use original fallback; no integration override exists here.

VERDICT: plausible bounded component candidate only. Fewer independent
subgroups and more per-lane registers can LOSE occupancy/latency. No speed
estimate until GPU measurement, and no new serving plan at this stage.

Build only after root grants CPU quiet gap:

    B70_AGENT=D1 MAX_JOBS=4 bin/cpu-run bash /mnt/vm_8tb/b70/build/vllm-flashnext/d1/hc_subgroup_r1/build.sh

Original30-90s estimate; actual active build36s (excludes CPU lock wait).
Reuse existing oneAPI
2026.1/Torch/toolchain; no image/package setup. TMPDIR stays scratch/disk.
After CPU compile review and separate root GPU grant:

    TAG=probe-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/d1/hc_subgroup_r1/window.sh

Physicalcard1/logical0,120s bounded body plus nativepre/post, quietCPU
outsideGPUlease, unchanged source/SO hashes and explicit container cleanup.
Baseline O4 library default is current _C.so, SHA0377d20b..., equal preserved
b70_o4_C.h.so and containing original+existingfused ops. CandidateSO89704bytes,
SHA84b64b232579423e6b7e55f1eb0bd0a2611ffaa01ba410f003eb241b17dcb2d9.

Probe original w8_gemv(-8)+Triton collapse vs existing w8_up_collapse vs
candidate. Requires bit-exact random/zero/large-gate cases, every96distinct
weights, and changed-input graph replay. M2 rejected deliberately. Compare
one-weight warmcache and96-distinct-weight graph chains, three timing
samples each, no per-op events. Packed weight+scale bytes334233600 for96.
Actual model trace has97upcalls; this omits one, HCdown, norm, act readiness,
fullgraph kernel interleaving/other working set, EP and nearfullVRAM.
Ready activation/flat buffers are precomputed and distinct per weight;
component timing cannot be labeled a full decode/streaming gain.

Consultation: no new Claude call attempted because four earlier calls
failed before a model responded; no responding model ID or advice claimed.

Default-vs-research clarification after root review: current matched
merged-d1-control-r1 recipe-args explicitly includesQ8=1 but has NOQ8F or
EP_ROUTE. Both require exactenv1; runtimeimageENV has neither. P1pair
savedrecipeargs match this, and Q1 frozenwrapper supplies neither.
Thus currentcanonical61.64 uses original HC up+separatecollapse, not the
existing fused candidate. Lc explicitly enables route+Q8F (+HC_FUSE and
oldFA8overlay) in a separate research arm. Old -.54ms is measured research,
not an improvement already deployed in the current default. Retest the
existing flags on the unchanged merged recipe BEFORE considering new
subgroup kernel serving integration. Three-arm component probe still
correctly separates original/existingfused/newcandidate. See
default-vs-Lc-audit.json. No source/default changes authorized here.


## Measured R1/R2 closure and source port

CONFIG: same kernel source0561d9de/SO84b64b23 in both one-card windows.
R1original source bytes45fb9bb5 retained in probe-r1-source.py unchanged;
R2probe only repairs partial-report loss and remains a strict bit gate.
measured-source-sha256.json links every source path and both raw runs.
SO remains external under B/d1/hc_subgroup_r1; no binary copied into git.
Build/window paths point to that disk scratch even after sourceport.
CPU utilities use explicit --out to keep generated files outside repo.

COMMAND: TAG=probe-r1 then parent-reviewed TAG=probe-r2 on scratch window.
R1body17s, R2body16s; both normal container closure, nativepre/post HEALTHY,
source/SO checks pass. No rebuild after initial CPU36s build. R1post15s,
R2post14s; leases released18:52:17/18:55:55. No model serve or reset.

RESULT: R1failfast stops at largegate weight1 existingfused comparison;
no timing. R2persists ALL9case comparisons. Random/zero/all other cases
are bitexact across three arms. Only h386 of largegate weight1 differs:
originalBF16=0x0000, existingfused=newcandidate=0x8000. One bit mismatch,
orderedBF16ULP1, maxabs0, all finite. Small affectedCPU payload41723bytes.
Raw archive storage read with stdlib confirms EXACT+0/-0 (no FTZ/subnormal
ambiguity). Existingfused/newcandidate match each other in all9cases.

VERDICT: STRICT COMPONENT_FAIL, timingSKIPPED. No weakening/renamingpass,
new subgroup speed UNMEASURED. Keep source for reproducibility; no kernel
change/rebuild justified by signedzero alone. Parent may separately run
existing route+Q8F as diagnostic; no universalbitexactness/promotion claim.
Raw probe-r1/probe-r2 directories remain untouched. Path-only sourceport
CPU81920ownership/ASCII/AST/bash PASS; no new GPU work for sourceport.
