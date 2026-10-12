# Paired original MTP: isolated normal-prefix schedule

CONFIG -> NEW paired_greedy_v2 and preparer/proposal V6 over fresh-reset V5.
Same source52 target, original34 tensors, batch0/cache0/6500, materialized0,
NO_HOST/device plan, paired24/24, context4096/pref512, ownGPU1, STOP_EOS and
P2048/GEN512. No slot transfer, real cache/PIN, quantization or method change.

COMMAND -> Root source preparation after independent source review:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/prepare_original_mtp53_two_stage_compile_v6.py --sdk /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww --sdk-receipt-sha256 8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94 --output /mnt/vm_8tb/b70/build/original-mtp53-two-stage-compile-v6

Same NoDri compile lifecycle; fresh V6 recipe/ELF and runner intake needed.
No author compile/GPU/provider/model payload. Active V5 run stays unchanged.

RESULT -> Policy comes from exact accepted source52-native2-fairness-v1
serial recipe SHA7d4edcac: fresh/ckpt defaults, pin1536, cache-root1,
turn-token248045 and tail-role default-1; no reread/message boundary feature.
Positions are scanned from EACH actual prompt, not assigned24/2041 to every
case. First selected prose prompt yields logical24/1536/2041/2047 and physical
T24/512/512/488/505/6. A synthetic different prompt proves changed root/turn
positions are used. The policy is a diagnostic segmentation reference, not
an activated cache or PIN feature in this cache0 actor.

The same completed-LAST callback persists across all parts, checks actual
absolute p0==seen and own catch-up position, and consumes source-owned full
prompt next IDs. first_prefill.run(prompt+start,end-start,start) drains each
part before the next one, just as normal read_part. Whole seen must be2047;
final target token remains the original run1/commit/wait at2047. Original own
state recurrence, head, token selection, verification, accepted keep and both
stage commits are untouched. Each actual part emits MTP53_PAIR_PREFILL_CUT;
no head/residual equality or stage GPU timing is fabricated.

VERDICT -> Source candidate only. Two targeted CPU controls verify actual
reference-derived boundaries/physical shapes and exact undo to V1 after
removing ONLY the changed prefix call and added include. This tests the next
isolated hypothesis: chunk schedule after freshness fix. It does not fit a
tolerance or silently change cache/slot/residency to force output agreement.
V4/V5 actual reports and frozen headers/preparers remain intact.
