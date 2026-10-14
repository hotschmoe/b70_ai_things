# Source57 default-OFF final-chunk decode candidate

CONFIG -> Direct actual52 SDK: only sycl/src/program/generate.cpp changes.
Same74 sources/all headers/ggml/compiler image/ABI; original weights/math.
New STRATA_BATCH_FINAL_CHUNK_DECODE=1 acts only on the final q>=b0 chunk that
previously continued before servicing a live slot. Default0 preserves old path.
Scope: strict public native2/full-state chain, no MTP/pipeline, static no-borrow.
Existing decode_share0.5 budget is reused; SHARE0 still performs no service.
No chunk512, logical checkpoint/turn/pin/tail cut or prompt operation changes.
No new model workspace, device copy, event or device wait is introduced.

Safety source audit: Prefill::run drains its successor pipeline before returning;
PP callback is moved to final stage at generate7828, not stage0. It reports
progress/checkpoints but does not execute batch_step. Batch record_window uses
slot_ss(t)=owned slots_[brow[t]] for GDN/PLE/QSA; batch_rec_ resets after graph
recording. run_slot_rows/commit_slots operate on those banks, with OnDevice
scopes restoring the host context. batch_step updates bs[].ids/x/p, not incoming
cur/ss/checks. It returns after target run and slot commit. Thus servicing on the
owning read_part thread after a completed/drained prompt chunk follows the same
bank/descriptor path already used between nonfinal chunks. It does not run on
stage1's callback thread. Cache math still requires actual matched proof.

Final-boundary controls: queued BYIELD/BSTOP are not dequeued or reinterpreted.
If either is already pending, or stop_req is set, new final service is skipped
and the old control path remains. Controls arriving during a slice retain the
existing bounded share-slice semantics; full cancellation-family qualification
is not implied by this latency probe. Any batch_step failure is terminal as on
the existing nonfinal path. No budget/deadline relaxation.

Observed actual52 fairness run: a long gap9.740653s between A tokens342/343
contains incoming RID2 last-stage PP1536 at+4.129643s, PP2041 at+8.609675s,
PP2047 at+8.919024s, REUSED at+9.014590s, firstT/DONE at+9.05714s, BADM at
+9.402075s, first rows2 completed at+9.737889s and next A BT at+9.740653s.
This includes final prefill spans, checkpoint/refill/bookkeeping, first token,
slot admission and first two-row execution. It is not9.74s of pure prefill or
GPU idle. The skipped final512 before pin1536 followed by spans505 and6, each
<=chunk512, has no batch callback. Candidate can split this cadence hole, but
may increase incoming TTFT by spending the preserved share on the old stream.
One512 chunk can still take seconds; no subsecond/prefill-speed promise.

Serial actual logs512windows avgT1.00/draft0 and source11437 uses T1 withoutMTP.
Reserved spec2 is not throwaway rows in this serving path. BGEN uses its own
exec_bm slot graph family. The prior serial/native timing difference has no
assigned source cause.

COMMAND -> Build source57-incremental-from52-build-recipe-v1.json exact argv.
Isolated copies, no CMake reconfigure, same image, one generate.cpp object
compiled,123 objects/eight archives reused with hashes; delete and relink all8
ELFs. No archive update is needed because generate.cpp belongs to strata only.
Strict Ninja dryrun must show exactly1 compile+8 links, no other compile/archive.
Full builder fallback can use native57 engine proposal; no reused old stack.
Receipt states incremental provenance, not all74 freshly compiled objects.
Root alone runs selected card1 NoDri lease while compatible card0 work proceeds.

After actual57 receipt, append real --sdk/--sdk-receipt-sha256 to model argv in
source57-native2-boundary-root-run-recipe-v1.json. The parser rejects missing,
wrong or invented receipt; exact all74 sources/all8 ELFs are rechecked. Actual57
OFF and ON both run A512/C64 native2 with C arriving after actual256th A token,
defaultshare0.5/mirror0/no heavy capture. Entire histories match each other and
immutable accepted52 serial references; actual57 teardown/health/identity is
mandatory. Old52 timings never substitute for actual57 defaultOFF.

Public reader after actual terminal proof:

  PYTHONPATH=strata/flash-next python3 strata/flash-next/validate_source57_native2_boundary_v1.py --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source57-native2-boundary-v1-run-v1 --output NEW_EXTERNAL_OUTPUT.json

RESULT -> Direct source check applied reviewed patch to a temporary actual52
file, verified exact changed SHA and unchanged prefix/suffix outside final branch.
Exactly1 of74 source hashes differs. Pure accepted52 reference admission passes.
No GPU, compiler, Docker/model payload or native CPU model process by author.
Actual57 SDK build and genuine actual-SDK constructor remain pending; no fake
construction receipt or all8 qualification is provided.

VERDICT -> Root-buildable bounded defaultOFF scheduling candidate and exact
actor/reader seams. Actual hist/cache/health/latency proof pending; no broad
fairness, stability, full-math, API/shelf or speed promotion.
