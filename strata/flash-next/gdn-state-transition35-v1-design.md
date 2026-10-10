# Source35 SYCL GDN transition leaf V1

CONFIG -> exact current source35 SDK receipt/recipe63/27/8/six and native
SYCL headers, implementation sources, static archives and eight rebuilt
ELFs. Actual locked header inventory gives state width128, key heads16,
value heads48, channels10240, conv4. Recurrence is F32 [row][head][column],
3145728 bytes; conv history is F32 [channel][3],122880 bytes. Inputs and
weights are deterministic synthetic F32 values; zero is the initial state.
The current verifier source uses EPS1e-6. This is not a model-payload test.

COMMAND -> python3 strata/flash-next/test_gdn_state_transition35_cpu_v1.py
and python3 strata/flash-next/test_collect_gdn_state_transition35_cpu_v1.py.
The tracked compile/run proposal records metadata/source hashes and exact
fresh leaf icpx/SYCL commands. Root must review and execute those commands
under owned leases/lifecycle, not reuse an old binary. This agent does not
compile or execute GPU work. Root owns strict per-card and compiled P2P0
health before/after, real logical allocation/free traces, clean owned
container exit/removal, no new kernel faults, and fresh all4 publisher
hashes after terminal/post-health with both known page guards bracketing.

RESULT -> CPU/source contracts passed; actual leaf compilation/native run
pending. No observed transition result or real model arithmetic claim yet.

VERDICT -> feasible fresh current-SYCL component oracle, no backend patch
required. The migrated current gdn_parity uses the same public APIs; this
leaf adds the composed T2 forward -> accepted-state commit -> T1 carry
comparison that is relevant to the still unresolved real row3 seam.

Each T2 forward must preserve both full zeroF32 buffers and have bitwise
normalized qkv/output equality to the matched T1 sequence. Accepted0/1/2
uses gdn_conv_commit and gdn_step_norm_multi with n_keep and t_out_begin2,
matching the verifier commit graph. The subsequent T1 call self-commits
conv and recurrence and must match T1-sequence fullstate/history/output.
There are27 raw comparisons. A deliberately modified persistent-state
float must change the next carried output. Raw artifacts and the read-only
collector bind the comparison log to full finite raw vectors; duplicate,
missing, changed, truncated and nonfinite evidence rejects. Actual USM
free/health/teardown are parent proof, not the leaf's printed assertions.

The scalar FP64 diagnostic derives its own conv, SiLU, q/k L2 normalization
and recurrence from independent deterministic synthetic inputs. It receives
no native activation/state. Its diagnostic NMSE has no numerical tolerance
qualification and cannot replace the campaign's independent original-GGUF
reference. The SYCL transition compares native paths to each other and can
localize component differences, not establish original model math.

This initial lane fixes STRATA_GDN_SPLIT absent/0 and requires eager absent
including0. The compiler follows the current SDK precise/subgroup32/per
kernel recipe with correctly rounded divide/sqrt link flags. It links the
currentSDK static kernel/core archives. Their bytes are newly pinned by
this proposal; the original SDK build receipt binds eight ELF targets but
not archive hashes. That historical association gap is explicit. Root may
require a fresh archive rebuild/provenance receipt to strengthen it before
calling the leaf current-build-qualified. No archived backend/ABI is used.

Real row3 normalized qkv, decay/beta and persistent states remain
unobserved. A synthetic PASS cannot dismiss that real divergence; capturing
its actual inputs or adding a deliberately reviewed observer may still be
needed. No fullmodel correctness, speed, cache, concurrency or shelf claim
follows from this fixture.
