# Strata 0.2.16 update journal

## S0 - inventory and build setup (2026-11-19 UTC)

CONFIG -> Host Linux 7.1.0-070100-generic; libze-intel-gpu1 and intel-opencl-icd 26.22.38646.4-0; Level Zero loader libze1 1.28.2-2. Daily driver inactive, frontdoor active. Old night HEAD e753e9cbd7d38dac1a7851c75ef6fc0ae016d5c0. Upstream origin/main and remote main 58ee3227720629ff2e41d640165ebe89c880dfef (v0.2.16-4); clean new worktree /mnt/vm_8tb/b70/build/strata-0216/src branch dd-0.2.16. Build image strata-sycl-dev:020 sha256:8ab2589c53f02eab0e6bf99cda66c5a4cd01115e89a5ebba74dd2a0446e53e5d, icpx 2026.1.0, ocloc 26.35.39758.10; old image strata-ab-build:ocloc sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2.

COMMAND -> Read DD config, compare its switches to engine/sycl sources; git ls-remote origin refs/heads/main; gh pr view for #2802 #2804 #2806 #2808 #2810 #2812 #2526 #2511 #3273. Consulted claude-fable-5-1 and claude-opus-5-5 on comparison risk; both identified the execution-layout mismatch. Build uses bin/cpu-run, docker --memory 90g, no GPU device, four jobs, bmg-g31 AOT; log /mnt/vm_8tb/b70/results/strata-0216/stock-build.log.

RESULT -> All nine PRs open, #2526 and #3273 drafts. Upstream has layer splitting, KV streaming and batch support but no expert-parallel-device; it explicitly disables batch MTP for multi-GPU split. Feature inventory in feature_map.md; equivalence and pack compatibility remain untested. PR #4185 upstream fast IQ helpers may supersede part of #2802. Build launched; result belongs to S1.

VERDICT -> S0 static inventory complete; stock two-card layer-split results are not configuration-matched to the old EP+batch-MTP DD and cannot isolate version effects. Reuse the existing build image and harnesses to limit setup; measure feature behavior before ports.

## S1 - stock and old baselines (in progress, 2026-11-19 UTC)

CONFIG -> Pre-health on both cards under whole-box lease; stock bmg-g31 AOT build at /mnt/vm_8tb/b70/build/strata-0216/build, image and limits as S0. Old binary /mnt/vm_8tb/b70/build/strata-ab/releases/strata-night-20261016, copied config /mnt/vm_8tb/b70/results/strata-0216/old-night.json, port 18134. Stock candidate config /mnt/vm_8tb/b70/results/strata-0216/stock.json changes EP to layer split 30 and requests batch MTP, which upstream will disable on split.

COMMAND -> bin/gpu-run bin/xpu-health; bin/cpu-run docker run ... cmake --build /work/build --target strata -j 4 (stock-build.log); nl_launch.sh strata-ab-serve old-night.json 18134; queued bench_serving.py with bin/cpu-run --quiet, one repeat after build releases CPU lease. Independent read-only consultations requested: claude-fable-5-1 identified dependency on upstream EP decode/prefill PRs but could not read external build trees due to its own tool permission layer; claude-opus-5-5 hit the same restriction and its turn limit. Do not treat either as a confirmed source comparison.

RESULT -> Initial per-card xpu-health passed (pre-health.log). Old engine startup and stock build are in progress; no benchmark result or claim yet. Build and server startup overlapped; the timed benchmark waits on the CPU lease. Fable analysis is saved as fable-ep-analysis.json, Opus attempt as opus-batch-analysis.json.

VERDICT -> S1 incomplete. Compare actual startup behavior and inspect identity, coherence and teardown before reporting any rate. Complete post-health after the last GPU work.

S1 continuation (scope update) -> Stock AOT build succeeded, SHA256 49aad6e5dc400411095595a616d13e97aa044d7dc388e783b9e8aed6c531b1e4. Old-night C1/PP, C2, cold C4 and cached C4 harnesses completed; six 131905-token conversations passed forced RAM restore and retained-history checks, with five native full-restore records. Old server teardown exited 0. Results under /mnt/vm_8tb/b70/results/strata-0216/old-{bench,c2,c4,c4-warm,context}. Under the updated scope, skip stock performance measurement; stock load check continues. Initial stock launcher failed before the native engine started because its inherited Python environment lacked jinja2; isolated 0216 Python dependencies were installed under /mnt/vm_8tb/b70/build/strata-0216/venv and startup retried. Pre-health passed; final post-health is pending.

S1 load check -> Stock SYCL AOT loaded the IQ3 model, vision and 200K/four-slot layer-split configuration after enabling the upstream host mirror and reducing the test reserve to 512 MiB; /v1/models and deterministic 17*19 response 323 are in stock-load-check.json. The server reported batch MTP disabled on split, and automatically changed spec 4 to 2. It shut down cleanly; stock timings deliberately omitted under the scope update. The two earlier starts refused startup cleanly due to missing mirror and inadequate stage cache; neither reported DEVICE_LOST.

## S2 - DD carries (in progress, 2026-11-19 UTC)

CONFIG -> Worktree dd-0.2.16 at /mnt/vm_8tb/b70/build/strata-0216/src. Preserve 200K/four-slot EP+batch-MTP, vision and DD-selected options; use upstream implementations where supported. Candidate source reviews are isolated in ep-candidate, batch-candidate and night-candidate; these do not touch the old night tree.

COMMAND -> Port #2810 Q8 HC, #2804 KREG MMVQ, #2808 prompt union and the score/top-k portions of #2806; add server EP placement handling. Compile the affected objects in the 020 image with bin/cpu-run, four jobs and 90 GiB limit; check EP server placement with the new Python environment. Parallel read/edit-only candidates requested from claude-fable-5-1 (EP and selected night flags) and claude-opus-5-5 (batch MTP), without GPU access. No per-patch benchmarks.

RESULT -> Stock upstream does have a Battlemage active attention merge and batched row handling; equivalence is under review. First object compile found a Q8 helper namespace conflict; corrected, then all requested feature objects compiled. Server placement smoke asserts explicit EP is not rewritten to a layer split. The candidate EP, batch MTP and selected night changes are not yet integrated or validated. Logs: port-objects.log, port-objects-retry.log.

VERDICT -> S2 remains incomplete; compile the assembled port and validate default/opt-in correctness before measuring the whole candidate once against the existing old baseline.

S2 continuation -> Base EP plus paired MTP assembled and loaded on :18134 with 200000 context, four slots, spec 4 and two paired draft heads; four concurrent arithmetic responses passed and EP_MIRROR/EP_VERIFY reported zero differing rows through 192 verify windows. Server teardown exited 0. The independent EP prefill PUSH correctness probe passed 26 cases (ep-probe-push.log). The direct-copy/moved-checkpoint snapshot test passed 5009 checks on card 0 (slot-test.log). Subsequent configured EP, slot, GDN, router, BF16 and IQ2_XS carries compile as objects; assembled runtime and the final once-only benchmark remain pending. The stock binary was preserved as /mnt/vm_8tb/b70/build/strata-0216/strata-stock-0216 before incremental linking.

S2 assembled candidate -> Final source HEAD cf1a2c35914d934213cbc13b57a008f3fa4bf54f, candidate binary /mnt/vm_8tb/b70/build/strata-0216/strata-dd-0216 SHA256 2e9a20cef43e14a17b9a3f9f29dbf8fe770e1f018fa22b9e8aed6c531b1e4. The final EP PUSH probe passed 26 cases (ep-probe-final.log). The first full build found a SYCL header declaration mismatch; moved the declaration to the SYCL override, and the combined build succeeded (final-build-retry.log). The final code was served with the configured EP and night flags. Note: the mirrored EP check disables PSUM and late-shared-expert optimizations, so it is not direct row parity evidence for those modes.

## S3 - candidate qualification (in progress, 2026-11-19 to 2026-11-20 UTC)

CONFIG -> strata/flash-next/dd/hotschmoe-dd-0216.json, port 18134, stable served name hotschmoe-dd first and research alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216. Two B70 EP, four slots, spec 4, batch MTP depth 1, int8 200000 context with 65536 resident cells, 32768 MiB conversation parking and CPU vision. Original DD config and service remain untouched.

COMMAND -> Same bench_serving, bench_concurrency and C4 cold/cached fixtures as the old arm under bin/cpu-run --quiet; bench_context_four --slots 4 --target 132000; dd_functional.py --base-url http://127.0.0.1:18134/v1 --no-api-key --image-fixture <prior blue-circle PNG>. Native server and all GPU work remained inside bin/gpu-run lease. Logs and responses under /mnt/vm_8tb/b70/results/strata-0216/.

RESULT -> Identity and arithmetic checks passed; C1, C2, cold and cached C4 data captured. Long context: six prompts of 131905 tokens, all markers correct; four forced returns each reused all 132243 held tokens with only 31 new prompt tokens read; native logs and cache counters recorded five full restores. The first harness summary falsely marked restore failure because upstream 0.2.16 omits restored_payload_bytes from its log. Fixed the parser and reanalyzed the saved raw phase logs without rerunning; original summary retained and corrected evidence is candidate-context/summary-reconciled.json. Payload byte counts are unknown. Functional checks passed for thinking, tool round trip, four concurrent requests, image, and 2048-token stream. Server teardown exited 0. Final GPU post-health pending.

VERDICT -> Candidate passes bounded correctness and no-recompute restore checks; no shelf promotion, production deployment, or multi-day soak has been performed. Performance claims await final post-health. Local PR refreshes are underway.

Correction -> The S2 assembled-candidate SHA256 above is a transcription error. The sha256sum result for strata-dd-0216 is 2e9a20cef43e14a17b9a3f9f29dbf8fe770e1f018fa22b9f7ed4c691f2ddba50.

## S4 - local PR refresh (2026-11-20 UTC)

CONFIG -> Local branches based on upstream 58ee3227720629ff2e41d640165ebe89c880dfef. No push or GitHub comments. The #2812 branch is stacked on #2810; #3273 includes its EP base.

COMMAND -> Compared each open PR to current SYCL/server source; prepared branches and ran SYCL source-level checks for #2802 and #2812. Consulted claude-fable-5-1 on #2802 coverage and claude-opus-5-5 on #2511; their reports are pr2802-refresh.json and pr2511-refresh.json in the results directory.

RESULT -> Eight local refresh branches for #2802, #2804, #2806, #2808, #2810, #2812, #2511 and #3273. #2526's slot-before-overwrite behavior is upstream and the candidate passed forced restores, so no refreshed branch. #2802 is partly superseded by #4185; its unmeasured IQ2 row-reuse remainder is on the local branch, not in the measured candidate. #2511 agent reports 14 targeted Python tests passing; individual standalone backend builds and GPU parity remain unrun. Exact refs, HEADs and limitations: PR_REFRESH.md.

VERDICT -> Local source refresh complete for review, not ready to publish without standalone validation and a reviewable EP stack. Run final GPU post-health, then stop.

S3 completion -> Both candidate and old test servers exited 0; post-health.log reports HEALTHY on both cards after all GPU work. Same model, GGUF/pack, EP layout, four slots, 200K context, 65536 resident cells, MTP depth 1, prompts and harnesses. One C1/PP repeat; two C2 and two measured cached C4 repeats. C2 is the median of the four client stream rates; C4 is median client stream decode and aggregate over the cached decode window. Prompt cases contain 4059 and 32727 tokens. Short-prompt TTFT is the coding case. Forced restore TTFT is the median of four returning conversations. Values are bounded measurements, not a stability or promotion claim.

| Metric | Old night | Candidate 0216 |
| --- | ---: | ---: |
| C1 decode tok/s | 138.6 | 134.6 |
| C2 tok/s per stream | 74.61 | 65.40 |
| Cached C4 tok/s per stream | 41.02 | 37.35 |
| Cached C4 aggregate tok/s | 151.81 | 141.18 |
| Prefill 4059 / 32727 tok/s | 1082.1 / 1181.6 | 888.6 / 1172.7 |
| Coding TTFT s | 0.624 | 1.341 |
| Forced RAM restore TTFT median s | 3.747 | 2.083 |
| Native full-restore median ms | 617.6 | 211.2 |
| Coherence and retained-history restore | pass | pass |

Candidate source and benchmark evidence have one qualification: upstream #4185 replaces much of old STRATA_NIGHT_IQ, but the later #2802 review found remaining IQ2 row reuse absent from the measured candidate. That unmeasured opt-in remainder is only in refresh/pr-2802-0216. See feature_map.md and PR_REFRESH.md. The current stable service remains stopped and the frontdoor remains active per user direction.
