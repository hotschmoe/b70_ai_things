# Original MTP first-kept-row head omission

CONFIG: The faithful original publisher two-stage V6 target uses GPU0/1 24/24,
normal logical prompt cuts, context4096, prefill512, and own original drafter
on GPU1. This candidate preserves all34 original tensors, every front/QSA/FFN
operation, both per-row waits, bank settlement, target acceptance, and commit.
Default-OFF STRATA_ORIGINAL_MTP53_DECODE_FIRST_HEAD_SKIP accepts only0/1.
Teacher-head skip, HC reuse, expert reuse, record-settle chaining and phase
timing remain0. Both experiment arms enable original MTP.

The source seam is draft_step_teacher_head_v1.hpp:112. The authoritative
keep decision precedes the catchup loop: reject, first-output EOS and remaining1
use keep1. Only keep2,row0 may pass compute_head=false to existing record().
Its optional final HC/vocab head reads next_hidden and writes disjoint head
scratch/Q8/logits. It writes no recurrent state or next_hidden. There is no
proposal between the two rows. Row1 overwrites these outputs with its fresh
head; the existing proposal head_ready guard and final wait remain mandatory.
A fresh-head check now also closes each successful bounded window.

COMMAND: Root prepares source without devices, then compiles under card0 lease:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/prepare_original_mtp53_decode_head_skip_v1.py --sdk /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww --sdk-receipt-sha256 8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94 --output /mnt/vm_8tb/b70/build/original-mtp53-decode-head-compile-v1
    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/build_original_mtp53_decode_head_v1.py --compile-root /mnt/vm_8tb/b70/build/original-mtp53-decode-head-compile-v1 --recipe-sha256 ACTUAL_PREPARED_RECIPE_SHA
    bin/gpu-run env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/run_original_mtp53_decode_head_v1.py --compile-root /mnt/vm_8tb/b70/build/original-mtp53-decode-head-compile-v1 --compile-receipt-sha256 ACTUAL_NEW_RECEIPT_SHA --elf-sha256 ACTUAL_NEW_ELF_SHA --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-mtp53-decode-head-natural-v1 --phase natural --case-index 0 --repeat-count 1

RESULT: Source-only. Three targeted CPU checks exercise exact source delta,
actual aggregate parsing, negative stale-head/count evidence and flag isolation.
No compile, provider read or GPU execution by the author. The original timing
run observed282 windows,511 catchup rows,229 keep2 windows. With unchanged
acceptance the candidate would compute511 decode heads OFF and282 ON, skipping
229. Teacher head calls remain2048 BOTH. Normal-close aggregate counters must
join the actual window roster; no per-token logging/query/copy/wait is added.
The parser requires full512 IDs/EOS equal the faithful V6 historical reference
and matched OFF/ON histories/residency. Historical reference metadata grants no
current model/health authority. Actual source/compiler/config, fresh full5,
health, kernel evidence, original owned EOF and removal remain required.

VERDICT: Unqualified default-OFF source candidate. The existing8.085s record
host span includes compute and queued preparation; skipping229 heads does not
predict an8s saving. Actual same-ELF OFF/ON elapsed results and full512 fidelity
are required before benefit claims. This is first-case diagnostics, not full
three-role/repeat, API, cache/concurrent serving or model qualification.
