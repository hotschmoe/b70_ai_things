#!/bin/bash
cd /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/night
export PR_BUILD_IMAGE=strata-sycl-dev:020 PR_BUILD_JOBS=4 PR_BUILD_TAIL=3
while read ref tag tgt; do
  echo "== $(date +%T) $tag"; ./pr_build.sh $ref $tag $tgt 2>&1 | grep -E "exit|strata-|error" | tail -3
done <<'L'
pr/sycl-mmvq-kreg kreg020 strata
pr/sycl-mmvq-kreg kreg020-parity mmvq_kreg_parity
pr/sycl-iq-expert-decode iq020 strata
pr/sycl-iq-expert-decode iq020-parity native_grouped_parity
pr/sycl-decode-attn da020 strata
pr/sycl-decode-attn da020-parity qsa_verify_batch_parity
pr/sycl-prefill-attn-xmx pattn020 strata
pr/sycl-prefill-attn-xmx pattn020-parity qsa_prompt_attn_union_parity
pr/sycl-hc-q8 hcq8020 strata
pr/sycl-hc-q8 hcq8020-parity gr_q8_parity
pr/sycl-hc-fused-read hcf020 strata
pr/sycl-hc-fused-read hcf020-parity gr_fused_parity
batch-slot-cache-preservation-main-020 pr2526-020 strata
batch-mtp-restore-accounting-020 pr2511-020 strata
pr/sycl-ep-prefill-reduce ep020 strata
L
echo ALL_DONE
