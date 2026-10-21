# L1 -- localmaxxing submissions (2026-10-21)

Four non-speculative runs were submitted: both suites (reasoning-v1 and
code-v1) for each 35B quant, all APPROVED with verifiedRun=true. Flash-Next
was measured but NOT submitted. At 61.3 / 61.5 tok/s it trails our own
Strata no-spec runs (66.2 / 65.9), and the orchestrator decided to hold it.
No MTP runs were made or submitted.

| Model / quant | Cards | ctx | reasoning-v1 tok/s (TTFT ms) | code-v1 tok/s (TTFT ms) | Status |
| --- | --- | ---: | --- | --- | --- |
| Qwen3.6-35B-A3B UD-IQ3_XXS | 1 (card 1) | 131072 | 120.9 (719) | 122.1 (577) | submitted, Verified |
| Qwen3.6-35B-A3B UD-Q4_K_M | 1 (card 1) | 65536 | 97.0 (563) | 97.2 (449) | submitted, Verified |
| Flash-Next IQ3_XXS (ISTA GSQ-RCO), EP sidecar | 2 | 131072 | 61.3 (2315) | 61.5 (1455) | NOT submitted |
| Reference: Strata Flash-Next no-spec (2026-10-16) | 2 | 200000 | 66.2 (545) | 65.9 | submitted earlier |

Submission IDs (localmaxxing speed-test run IDs):

- 35B UD-IQ3_XXS reasoning-v1: cmvhi9x0k033rrf01tej20yt5
- 35B UD-IQ3_XXS code-v1: cmvhibbft033urf01uqycd7eu
- 35B UD-Q4_K_M reasoning-v1: cmvhix9xu033yrf01fma4faax
- 35B UD-Q4_K_M code-v1: cmvhiyocc0341rf01lm0dd1k9

Payloads, dry-run/validate output, submit responses and evidence are in
vllm/flashnext/localmaxxing/submitted-20261021/ (SUBMITTED.txt, one dir per
run, engine/). Raw windows: /mnt/vm_8tb/b70/build/vllm-flashnext/lmx-l1/runs/.

## Engine freeze (integrity a)

- The fork worktree is /mnt/vm_8tb/b70/build/vllm-flashnext/vllm-lmx-20261021 on
  branch b70/lmx-20261021. The engine commit is
  cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95, whose parent is
  b70/flashnext HEAD 0a6170da8c. Nothing was pushed.
- This commit is the exact dirty state of the shared tree at 02:10 UTC: 13
  modified and 6 untracked files, copied as `git diff HEAD --binary` plus the
  untracked files. vllm/_version.py was copied as well (ignored by git).
- Byte identity was checked with a temporary index on the shared tree
  (read-tree HEAD, add -A, write-tree). It gives tree 7f69ba45..., the same
  as the commit, both before and after the copy. The shared working tree and
  its index were not touched.
- Every window serves with PYTHONPATH set to the worktree. Each window
  checks that HEAD is the engine commit and that `git status` is empty,
  before and after (ENGINE_STILL_FROZEN in every window).
- The campaign harness, plugins, bin and models.yaml were copied to
  lmx-l1/frozen-repo, and the manifest frozen-repo.final.sha256 is checked
  before and after each window. Local edits to that copy point the paths
  into the frozen copy instead of the live repo:
  - xpu-entrypoint patches path, x5_serve_proxy/o4_serve REPO and fork path
  - d3 GDN plugin path, B70_GGUF_CPU_DEQUANT
  - read-only mounts of the frozen b70_memory/b70_lowbit editable plugins
  The diff is engine/frozen-repo-local-edits.diff.
- External .so sha256 values are in each evidence/so-before.sha256 and
  so-after.sha256 (unchanged in all windows):
  - 35B: _C.o3g.so 38f7176a, libb70_x5_xpu.so (q8-fast) 51f304b4,
    libb70_gdn_xpu.so (gdn-bf16) 2bb3661f, eventless stub 425788b2,
    helper b2668e59.
  - Flash-Next: _C.o4g.so 2eac42c3, b70_ep_C.o4c.so 10495833,
    b70_o4_xpu _C.so b9268747, plus the c7b-fa-overlay libraries.
- Live /version is 0.32.1rc1.dev94+g5fd62688f. Image vllm-flashnext-xpu:py312,
  host kernel 7.1.0-070100, UMD 26.22.
- engineRepository is https://github.com/vllm-project/vllm. The notes say
  that the commit sits on a local, unpushed fork branch.

## Recipes

- 35B: this is X5's selected no-spec recipe from reports/1X5_proxy35b.md,
  run through frozen x5_serve_proxy.sh by localmaxxing/l1/l1_window_35b.sh.
  - Card 1, TP1, BF16, FULL_DECODE_ONLY (1,2,3,4,6,8), FLASH_ATTN, no
    prefix cache, seqs 2, batched 1024, util 0.90.
  - Flags: GDN_BF16, EVENTLESS(+LAZY), PACK_LINEAR, Q8_LINEAR, CANONICAL,
    Q8_FAST, FUSE_SHARED. MIXED, RAGGED and BATCH_LINEAR are 0;
    B70_PREFILL=0.
  - X5_LMX=1 adds the read-only /server_info route.
  - max_model_len: IQ3 131072 (KV 569,740 tokens), Q4 65536 (KV 161,792
    tokens). Q4 could also have held 131072; 65536 is what was run and
    declared.
- Flash-Next: O4's s4 recipe (recipe-s4.env = harness/o4_recipe.env at f13,
  C1 HTTP ITL 15.25 ms), run by localmaxxing/l1/l1_window_flashnext.sh.
  - The EP sidecar runs on both cards: card 0 holds the model and 138 of 512
    experts per layer, card 1 holds the other 374. One vLLM worker, no oneCCL.
  - FULL_DECODE_ONLY (1,2,3,4,8), seqs 4, batched 2048, util 0.98,
    mamba-cache align, no prefix cache.
  - Hyper-connection weights are rounded to Q8_0, as in Strata's DD. The
    bf16-exact path is about 16.2 ms.
  - The entrypoint is lmx-l1/l1_api.py: stock vLLM CLI plus X5's /server_info
    route.
  - The research alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-vllm-sidecar-mtp0
    was registered in evals/configs/models.yaml.

## Promotion gate (integrity b)

X4's promo_gate.py was run on every live server under bin/cpu-run --quiet.

| Check | 35B IQ3 | 35B Q4 | Flash-Next |
| --- | --- | --- | --- |
| identity before/after, unchanged | PASS | PASS | PASS |
| greedy 17x19, 23x29 | PASS | PASS | PASS |
| C4 Paris golden (proxy only) | PASS | FAIL (see below) | n/a |
| exact repeat, both C1 passes | PASS (bitwise) | PASS (bitwise) | PASS (bitwise) |
| exact repeat, C2 passes | prompt 3 differs from C1 at token 80, identical in both C2 rounds | 2nd C2 round, prompt 0 differs at token 34 | one prompt differs from C1 at token 1-2, identical in both C2 rounds |
| 8K / 32K three-needle recall | PASS / PASS | PASS / PASS | PASS / PASS |

Notes on the two failing rows:

- **C4 Paris (Q4).** The golden 16-token continuation was recorded on a
  different quant. The Q4 output is a coherent but different continuation:
  "Paris, a city renowned for its iconic landmarks...".
- **C2 exact repeat.**
  - IQ3 and Flash-Next: the C2 divergence is batch-shape variance. The M=2
    and M=1 decode kernels give different results, but the same results
    every time, so it is not run-to-run noise.
  - Q4: the C2 divergence is run-to-run batch-2 variance. This is the GDN
    reduction race that X5 fixed after the snapshot (journal 02:31), and the
    frozen recipe does not contain that fix.

Neither failure disqualifies the runs. The submitted metric is batch-one C1,
where the repeats are bitwise exact, and the three timed samples spread by
0.1-1.4 tok/s. Both failures are explained in the submission notes field.

## Measurement (integrity c, d)

- Client: X4's lmx_vllm.py --measure-local under bin/cpu-run --quiet.
  - Canonical prompts with a fresh leading nonce, temperature 0, thinking
    at the server default, 512 max tokens, 1 warmup + 3 timed iterations.
    The payload uses the correlated median sample.
  - Each sample is isolated with Prometheus counters: one request, zero
    drafts, zero prefix hits. The run also requires speculative_config=null.
- Windows were announced in the JOURNAL. Other agents' leases interleaved
  between windows but not inside the measurement, because the server held
  the lease.
- l1_finalize.py adds contextLength, modelRevision (GGUF revision 5bc3e238
  for 35B), mtpEnabled=false, and notes. Each payload then went through
  validate-local and the API dry-run (valid=true, verified=true,
  issues=[]), and was submitted at most once per 65 s.
- Two early W1 attempts failed at load, before any measurement. The frozen
  copy still had hardcoded live-repo paths: the d3 GDN plugin and
  libdequant_cpu.so. Both were fixed, then the window was rerun as w1c.

## Flash-Next: why lmx trails the 64-token bench, and recipe gaps

- **Decode drifts with position.** Mean ITL per 64-token block across the
  512-token lmx stream:
  - reasoning: 15.92, 15.99, 16.09, 16.27, 16.35, 16.51, 16.61, 16.73 ms
  - code: 15.81 to 16.60 ms
  - Overall effective ITL is 16.3 ms, with no spikes (max chunk 20.8 ms, so
    graph reclaim is not visible).
  - O4's s4 bench is the p50 over 64 tokens after a short prompt (15.25 ms),
    so it never sees this context-length growth. The lmx prompt is 282-347
    tokens.
  - Strata holds 15.1 ms averaged over the same 512 tokens. Our per-token
    cost therefore grows faster with context than Strata's: about +0.8 ms
    over 512 tokens. Attention, or another context-dependent decode op, is
    the place to look.
- **max_model_len does not matter.** At 40960 (control W3b) and 131072 (W3)
  the results are 61.3 / 61.5 tok/s for both, and the per-block ITL matches
  within 0.05 ms. Using 131072 costs nothing.
- **Recipe gaps:**
  - s4 does not include F1's sidecar prefill path (B70_EP_PREFILL /
    B70_PREFILL native bucketed op, 2000+ tok/s). It prefills at about
    150 tok/s, so lmx TTFT is 1.45-2.3 s, against Strata's 545 ms.
  - The F1 and O4 recipes use different GGUF/EP .so builds and different PLE
    modes (F1 uses PLE_MLOCK/FAST; O4 uses PLE_DEVICE, which O4 says must not
    be combined with those). A combined recipe needs integration and its own
    qualification.
  - f14 (B70_DECODE_HC_FUSE, ep o4d) postdates this snapshot.
- **What it takes to submit Flash-Next:** a combined O4 + F1 recipe whose
  lmx dry-run beats 66.2 / 65.9. On these suites that means an average ITL
  of about 15.0 ms or less over 512 tokens, which needs the position drift
  fixed, plus F1 prefill for TTFT. Valid draft packets from W3 are kept in
  submitted-20261021/flashnext-iq3_xxs-NOT-SUBMITTED/.

## Comparison vs Strata submissions

- **35B UD-IQ3_XXS:** 120.9 / 122.1 tok/s on one card. It is a different
  and smaller model (3B active, 256 experts), so it is not comparable to
  Flash-Next.
- **35B Q4_K_M:** 97.0 / 97.2 tok/s on one card. X5's earlier canonical
  packet measured 96.1 / 96.3.
- **Flash-Next on vLLM:** 61.3 / 61.5 tok/s. That is 7% below Strata no-spec
  (66.2 / 65.9) on matched suites, method and hardware, and TTFT is 2.7-4.2x
  higher.

## Files

- Scripts: vllm/flashnext/localmaxxing/l1/
  - l1_window_35b.sh
  - l1_window_flashnext.sh
  - l1_finalize.py
  - l1_submit.sh
- Evidence: vllm/flashnext/localmaxxing/submitted-20261021/
- Registry: evals/configs/models.yaml, entry
  l1-flashnext-ista-iq3xxs-sidecar-lmx-mtp0.
- Frozen engine: /mnt/vm_8tb/b70/build/vllm-flashnext/vllm-lmx-20261021
  (b70/lmx-20261021 @ cfd2b6ff).
- Frozen repo copy and manifests: /mnt/vm_8tb/b70/build/vllm-flashnext/lmx-l1/
