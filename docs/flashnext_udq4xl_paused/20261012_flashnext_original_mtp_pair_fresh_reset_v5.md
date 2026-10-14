# Paired original MTP: explicit fresh target reset

CONFIG -> NEW paired compile proposal V5 over immutable V4. Same original
34 tensors, own recurrence, target24/24, source52 precision contracts and
P2048/GEN512 geometry. No live V4 binary/runner or current natural run edited.

COMMAND -> Source preparation only, run by root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/prepare_original_mtp53_two_stage_compile_v5.py --sdk /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww --sdk-receipt-sha256 8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94 --output /mnt/vm_8tb/b70/build/original-mtp53-two-stage-compile-v5

Root then uses the SAME NoDri V4 compile lifecycle with the V5 generated
compile-recipe.json and output tree. This proposal does not contain fabricated
compile/ELF or runtime pass. Fresh controls and natural runner intake must bind
that NEW receipt/ELF/source proposal; old controls do not transfer automatically.

RESULT -> Actual source52 generate.cpp3769 zeros primary on construction;
secondary session_init3792 only carves state. The stage constructor loop does
not call session_zero. The one-shot pair hook bypasses normal serve fresh
reset10040-10056, unlike the reference native serving request. V5 extracts
that exact actual public reset device body from the pinned source, and executes
it BEFORE bind_target/controls/natural. It validates exact24/24 session carves,
waits the root verifier's commit, validates both owned queue/device/context and
persistent QSA metadata, then session_zero clears GDN recurrence+conv, QSA
cache/indexer and PLE history/token window on BOTH owners. It also resets
index-block/step/attention-status/position staging and identity page tables,
matching normal public fresh semantics. Only the original tail clearing
live/live_imgs/checks is omitted: request bookkeeping has not begun at this
hook. Those host variables do not own device recurrent state.

Successful stderr contains one MTP53_PAIR_FRESH_RESET marker before target
binding. Reset failure retains failure via exit86 before any target token;
the original owned paired shutdown/cleanup obligations remain unchanged.
Prefill and verifier retain the original shared SessionState references, so
this reset updates the state both will actually consume, not a copied struct.

VERDICT -> Source correctness fix, not attribution of actual token103
mismatch. One tiny test executes transformation of actual pinned source and
removes exactly the new reset hook to recover V4 byte-for-byte. It checks
both-stage source reset and its order; no GPU/compiler/model reads. Separate
normal reference differences (cache/PIN cuts, auto vs6500 cache, batch/slot
path and residency) remain unresolved. No quant/method or tolerance changes.
