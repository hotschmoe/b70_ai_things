# S93 staged-prefill value-pipeline proposal (source only)

CONFIG -> Current actual SYCL attention source00bf1c45 and current prefill
source (hashes in source-pins.json). S92's correctly bound retained S13 value
body passes strict36 cases/1440 raw timing samples, all health/cleanup0 per
root handoff. Full32/width2051 page4 and512 nested seed-paired medians were
-8.469%/-9.182%; masked32/1999 was+2.392%. These are primitive observations,
not current serving speed or permission to hide the masked regression.

COMMAND -> SOURCE PROPOSAL ONLY: experimental-attn-value-prefill-s93.patch
adds include/strata/kernels/qsa_prefill_attn.hpp, includes it ONLY in the
SYCL attention TU and prefill.cpp, and changes those two TUs. Existing broad
qsa_decode_attn.hpp, CUDA sources, verify/MTP callers and all other files are
untouched. No patch is applied, build, GPU or model call performed here.

RESULT -> The existing attention template gains VALUE_PIPELINE=false.
Only probability destination/value loop use if constexpr. Selecting false
and removing the declaration/include/wrapper additions reconstructs the
ENTIRE original attention source exactly. Selecting true reconstructs the
ENTIRE retained S13/S92 source exactly (after removing wrapper/include and
parameter additions). source-change.json records both full-file identities.
Score reduction, K loads, softmax arithmetic, FMA cell order, merge and all
barriers remain unchanged. No extra local-memory allocation is introduced.
Source byte recovery is not compiled instruction identity or a numeric gate.

New qsa_prefill_attn_value_batch is a separate entrypoint. Exact literal
STRATA_ATTN_VALUE_PIPELINE=1 opts in; absent/other strings delegate to the
unchanged original batch entrypoint. It preserves a requested LANECELL=1
via the original entrypoint too. Independent guards require32queries,
INT8 K/V+scale pointers/noQ4, cap2051,24heads/2KV/HD256/page4 and buffers.
After all guards, one thread-safe local-static stderr marker records
HOST selection: `strata: prefill attention value pipeline selected (nq32
cap2051 INT8 page4)`. OFF/ineligible calls print none; first eligible ON call
prints once. It is NOT independent GPU dispatch/count proof. A future first
4K diagnostic pays that print; later32K calls do not. Disclose that small
host logging cost rather than assuming no instrumentation.
It launches only attn_chunk_kernel<1,false,true> plus unchanged merge, with
NEW explicit class tags attn_chunk_value_prefill_s93 and
attn_merge_value_prefill_s93. The old eleven class tags remain unchanged.
Original qsa_decode_attn_batch and qsa_decode_attn_step host function bodies
are byte-exact; their default kernel instantiations select VALUE=false.
No signal is inferred from stream identity or runtime tensor contents.

Actual caller and domain:
- prefill.cpp1208 takes cap=qsa_selection_width(2048,s)=2048+4-1=2051.
  core/layout.hpp29 uses qsa_real_shapes page4. attn_batch is32 at708;
  scratch allocation1232 is32 single-query regions.
- qsa_step_fill fills pos=p0+t, n_kv=pos+1 and width=min(n_kv,2051).
  prefill.cpp2535 fills host steps and uploads them on m.cs. Width is
  monotone within a chunk. Caller3330 now uses the new entrypoint ONLY
  staged(st.kv_mode1) && st.kv_int8 && nb32 && cap2051 && firstwidth2051.
  Thus every one of32queries has width2051. Short-width batches, all ragged
  query tails, direct/non-staged pools and other KV formats stay original.
- init1125 builds table[i]=i and waits for its upload. staged pools3301
  use m.stage plus this identity table, not cache slots with negative pages.
  Existing kv_ready barrier/staging3083 and append3126 publish preceding
  history/current chunk before attention on the SAME m.cs. No new event,
  wait, queue, copy, GPU table scan or host readback is added. Full staged
  history validity remains the existing caller contract, not a new proof.
- qsa_prompt_attn_batch is still tried first; this candidate affects ONLY
  its existing eager fallback. Source/profile S90 confirms current fallback;
  future XMX success may make this branch inactive. Decode/verify/MTP never
  call the new entrypoint, even if a future decode batch reaches32.
- Fresh S90 main32721/12288+12288+8145 would select319+384+254 full batches
  per QSA layer:11484 calls over12layers. First65 batches (pos0..2079) and
  each17-query tail remain original. This is source count arithmetic, not
  independently measured S93 dispatch or an end-to-end saving estimate.

CAP pitch resolution: S92 used cap2112 with width2051. Both cap2112 and
production2051 launch33 chunks, use identical204400 scratch floats/query,
identical part offsets, and same n_here values (last partial3 for width2051).
Only per-query ids base q*cap differs; local reads stay c0+t<width, never
use padding. One additional correctly bound32-query/width2051/cap2051/page4
strict numerical case is justified before applying/building main to catch
that packed ids pitch. Root prepares it separately; NO result is claimed
here, no repeat of old36cases or new format/tolerance matrix is requested.

VERDICT -> Ready for root/actual model source review only; root owns the
additional CAP gate, application/build/runtime decisions. Guard preserves
original GPU operation sequence when OFF and all old entrypoints; it adds
CPU wrapper checks plus the once-only eligible ON marker. The caller's staged-table
contract, not width alone, excludes the masked-page regression workload.
No statement that masks are numerically invalid: original negative-page
logic is still present in the candidate and S92 strict masked equality passed.
Independent main host metadata/registration ownership and new class IDs must
be checked after a real build: S13/S91 showed distinct device IDs alone were
insufficient. Future serving must use the SAME new binary flag0/1 with all
other controls fixed, output/work/coherence/health/cleanup gates. No default,
shelf/daily/fullcontext qualification or2000tok/s target achievement.
