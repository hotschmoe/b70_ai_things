# MTP_TODO.md — Multi-Token Prediction as the primary decode-speed lever

**Created:** 2026-06-20 · **Updated:** 2026-06-23 (ctx2048 spec sweep logged)
**Owner:** b70 team
**Status:** ✅ **CAMPAIGN COMPLETE 2026-06-22 (M0-M5 all done).** HEADLINE: **single-card dense 27B W4A16 + MTP spec=4
PIECEWISE = 55.28 t/s vs 30.84 = 1.79x** (beats Lorbus 45.2; refutes the stale -19%). Half-KV FREE (accept 3.29 vs 3.25).
FULL capture BLOCKED (gdn_attention spec op can't run in any captured graph on v0230 0.1.9). TP=2 MTP DEAD (spec-allgather
not graph-capturable; even MTP-off TP2 is 0.87x single-card) -> single-card DP-replica is the path. 35B-A3B MoE + MTP =
+3% FLAT (MTP is a DENSE lever, not a sparse-MoE lever; MoE headline is CAPTURE 66.8 t/s). **PRODUCTION ACTION: enable
MTP spec=4 on the daily-driver 27B int4 DP replicas for single interactive streams (+79% in the TTFT-cancelled probe),
but leave it OFF for C4+ batch/fan-out unless re-benchmarked. A ctx=2048 `vllm bench serve` follow-up (random 2048/128)
showed C1 `tg` improves 29.78 -> 46.69 tok/s, while C4 regresses: agg out 51.69 -> 40.56, `tg` 19.54 -> 16.09, TTFT
3.398s -> 4.444s. A later C1 spec sweep at the same ctx found spec=4 fp16 KV best for pure `tg` (57.64 tok/s),
spec=3 fp16 KV best for aggregate output (35.70 tok/s), and Half-KV slower at 2K ctx for every spec.** Full
per-experiment log in JOURNAL.

> ### [2026-06-23] compressed-tensors W4A16 NOW SERVES -- but it (and ALL our 27B quants) are MTP-DEAD
> `Qwen3.6-27B-W4A16` (compressed-tensors) is FIXED + on the shelf (`rdy_to_serve/qwen36-27b-w4a16`,
> docs/kernel/22). The old "won't serve / 4304-dim XPUwNa16 wall" was a RED HERRING -- the real bug was a
> text-only-checkpoint weight-name mismatch (all weights silently skipped -> random init). The
> `int4_gemm_w4a16` kernel itself is CORRECT (NT-format weight; matches a reference dequant, maxerr 0.016).
> **MTP gap (the parity blocker):** Lorbus int4 (AutoRound) ships the MTP module (29 tensors); our
> `{W4A16, W4A8-sqgptq, W8A8-sqgptq}` 27B quants ALL have **0 mtp tensors** -> they cannot do useful MTP as-is.
> FIX = GRAFT the 15 bf16 `mtp.*` from the bf16 base **and force the MTP drafter unquantized**. Raw tensor graft alone
> is insufficient for compressed-tensors W4A16 because vLLM otherwise instantiates the drafter as quantized/fused and
> skips the BF16 MTP linears. VALIDATED on W4A16: ctx2048 C1 no-MTP `tg` 21.73 -> MTP spec=4 `tg` 42.97,
> accept_len 3.49; C4 improves `tg` 16.67 -> 28.98 but aggregate output regresses 46.29 -> 38.58 and TTFT jumps.
> Parity stack at ctx2048 C1:
> **W4A16 no-MTP 20.97 -> AutoRound no-MTP 29.85 (kernel gap 1.42x) -> AutoRound+MTP ~46.7 (MTP ~1.57x).**
> So to make W4A16 the headline: (1) graft MTP (the bigger lever, currently MISSING), then (2) the int4
> decode-GEMV (1.42x) measured at the SPEC batch size (K+1~5, near-GEMM where int4_gemm_w4a16 already wins
> prefill -> MTP likely shrinks the kernel gap).

> ### [M0 RESULT 2026-06-22 -- PASS] (JOURNAL has the full log)
> Serve 27B (Lorbus W4A16 int4-AutoRound) on `vllm-xpu-env:v0230` + PIECEWISE + `--speculative-config '{"method":"mtp",
> "num_speculative_tokens":3}'` -> healthy, MTP head + drafter loaded (embedding+lm_head shared), PIECEWISE capture OK,
> coherent gen, rejection-sampler ran. NO crash. **Crucial M1 finding:** `splitting_ops` includes `gdn_attention_core_xpu`,
> so under PIECEWISE the GDN/attention verify runs EAGER -> this is the known -19% regime. **M1 is therefore NOT "re-measure
> PIECEWISE"; it is "capture attention+GDN" via `--attention-backend TRITON_ATTN` -> FULL (host script `ATTN` knob; PR #34482).**

> ### [M1-M3 EXECUTION RECIPE -- codex-validated 2026-06-22]
> - **M1-C (the frontier) must use `CGMODE=FULL_DECODE_ONLY`, NOT `FULL`.** The spec verify is a uniform decode of query
>   len `1+N`; only FULL/FULL_DECODE_ONLY captures that (no PIECEWISE knob captures attention/GDN -- doc 12). `FULL` also
>   tries to capture prefill -> more likely to hit the SYCL-Graph `work_group_scratch_memory` wall. So:
>   `TRITONSHIM=1 ATTN=TRITON_ATTN GRAPH=1 CGMODE=FULL_DECODE_ONLY` (RagingNoper recipe AS-IS, KEEP `gdn_attention_core_xpu`
>   in splitting_ops; removing it is a last-resort force-capture experiment). Confirm Triton: grep `Using Triton`, no
>   `Disabling Triton`, `(decode, FULL)`. Fallback ladder: FULL -> FULL_DECODE_ONLY -> PIECEWISE (the -19% floor).
>   [The currently-running M1 launched config C as plain `FULL`; if it fails, retry FULL_DECODE_ONLY -- already the plan.]
> - **Accept length from `/metrics`** (`vllm:spec_decode_num_{accepted,draft,emitted}_tokens_total`), not docker stdout
>   (`scripts/38_specdecode_bench.sh` pattern).
> - **M2 (spec {2,3,4,5,6}):** fix everything, sweep only `SPECTOK`; median >=3 warmed runs; WINNER = max tok/s (not max
>   accept). Bucket accept% by gen-token position (0-128 / 129-256 / 257-512) to catch the Lorbus 86->65% decay.
> - **M3 (Half-KV):** same winner; A = full KV (omit KVDTYPE/fp16), B = Half-KV (`KVDTYPE=fp8_e4m3`), same MAXLEN. Decider:
>   `delta_accept = acc(Half-KV) - acc(full-KV)`; drop > 0.2 tok or > 5% rel = Half-KV costs acceptance, else keep it.

> ### [FULL-capture frontier -- crash root cause + retry recipe, codex 2026-06-22]
> M1-C (plain `FULL` + `TRITON_ATTN`, spec=5, CAPSIZES=1,2,4,8) CRASHED at capture: `RuntimeError: spec_query_start_loc
> must have size [num_spec_decodes + 1]` in `_xpu_ops.py::_gdn_attention_core_xpu_impl`. **Root cause = capture
> dummy-metadata mismatch** (not bad MTP wiring): FULL warmup runs synthetic batches whose `num_spec_decodes` !=
> `spec_query_start_loc.shape[0]` -- a known vLLM-XPU GDN-spec-capture bug. PIECEWISE/eager build consistent metadata so
> they run fine (hence M1's 1.72x). **Retry recipe (NOT YET RUN):** `CGMODE=FULL_DECODE_ONLY` (decode-only uniform shapes,
> skips FULL's mixed prefill-decode capture) + `ATTN=TRITON_ATTN TRITONSHIM=1` + **`CAPSIZES` MUST include the spec-verify
> len `1+num_spec`** (spec=5 -> include 6: `CAPSIZES=1,2,4,6,8,16,32`) -- my M1-C's `1,2,4,8` omitted 6. Keep
> `gdn_attention_core_xpu` IN splitting_ops (vLLM default already does -> not force-captured -> dodges the bug). If it still
> crashes, escalate splitting_ops variants. **FULL is UPSIDE -- PIECEWISE already gives 1.72x, so this is not a blocker.**
>
> **[RESULT 2026-06-22] FULL_DECODE_ONLY ALSO CRASHES (same bug) -> FULL is BLOCKED on stock v0230.** Retried with
> `CGMODE=FULL_DECODE_ONLY spec=4 CAPS=1,2,4,5,8,16,32` (capture size 5 = 1+spec included): SAME crash
> `spec_query_start_loc must have size [num_spec_decodes+1]`, now inside the inductor-compiled `gdn_attention_core_xpu`
> call for `layers.0.linear_attn`. So the bug is mode-independent (FULL == FULL_DECODE_ONLY) and capture-size-independent --
> the v0230 baked `gdn_attention` (vllm_xpu_kernels 0.1.9) spec op simply cannot run inside ANY captured graph. **VERDICT:
> single-card MTP ceiling on stock v0230 = PIECEWISE 1.79x (spec=4). FULL needs a KERNEL/DISPATCHER fix.**
>
> **[FRONTIER UPDATE 2026-06-22 -- web research]:** vLLM 0.23.0 is the NEWEST vLLM (no v0.24.x); v0230 is the frontier.
> - **The 0.1.10-kernel candidate is DOWNGRADED:** vllm_xpu_kernels v0.1.10 (06-18) changes are ALL GDN *prefill*
>   (#378/#379) + a >=32K NaN-race fix (#411) -- **NOTHING touches the spec/cudagraph metadata path**, so it will NOT fix
>   FULL spec-capture. (Still worth a cheap KERNEL_SO mount for the MoE/long-context PREFILL win + the NaN fix; wants
>   compute-runtime 26.18, check ABI.) The `spec_query_start_loc must have size` assert is UNTRACKED upstream -> filing a
>   vllm-xpu-kernels issue with our repro is high-value.
> - **The real FULL-capture path = port vllm-ascend PR #7148** (merged 2026-03-12): same bug class -- FULL-mode spec
>   capture mis-treated as uniform-decode when `1+num_spec` is in capture sizes. Fix = a Python monkeypatch overriding
>   `CudagraphDispatcher._create_padded_batch_descriptor` (`patch_cudagraph.py` pattern), GPU-cheap to test. If the patch
>   alone doesn't clear it, the assert is KERNEL-side -> file the issue. (Upstream's clean refactor #23679 was CLOSED unmerged.)
> - **mtp.fc gotcha (Lorbus HF card, CONFIRMED):** plain AutoRound packs `mtp.fc` as int4 -> vLLM SKIPS the MTP head ->
>   0% accept. Keep `mtp.fc` BF16 (our `re:.*mtp.*` ignore + Q8 layer_config already do this -> safe). Check this on ANY
>   self-rolled MTP quant. Also: FP8 native is NOT coming to Xe2 (Xe3 cancelled) -- INT8 W8A8 stays the low-precision path.
>
> **[FULL-CAPTURE EXPERIMENT -- READY TO RUN 2026-06-22]:** inspected v0230's `vllm/v1/cudagraph_dispatcher.py`.
> `_create_padded_batch_descriptor` HARD-ASSERTS `num_tokens_padded % uniform_decode_query_len == 0` -- with MTP spec=4,
> `uniform_decode_query_len = 1+spec = 5`, and the capture sizes (1,2,4,8,...) aren't multiples of 5 -> mis-built batch
> descriptor feeds gdn_attention a wrong-sized `spec_query_start_loc`. **The #7148 port = `scripts/88_patch_cudagraph_xpu.py`**
> (gate the divisibility instead of asserting it; fall back to uniform_decode=False when not divisible). TEST recipe:
> mount scripts/88 as sitecustomize into v0230, serve 27B int4 with `cudagraph_mode=FULL_DECODE_ONLY` + `MTPTOK=4` +
> capture sizes INCLUDING `1+spec` (e.g. `1,2,4,5,8,16,32,64`). OUTCOME: (a) crash clears -> FULL_DECODE_ONLY MTP works
> (past 1.79x); (b) crash persists but moves -> dispatcher fixed, the residual is the XPU gdn_attention kernel op ->
> file a vllm-xpu-kernels issue with the repro (the `spec_query_start_loc must have size` assert is UNTRACKED upstream).
> Either way the bug is localized. This is the #1 post-Q8 GPU lever (RESEARCH_TODO Track 1d frontier).
>
> **[RESULT 2026-06-22 -- FULL is KERNEL-GATED, definitively. Experiment DONE, outcome (b).]** Ran it: Lorbus 27B int4,
> FULL_DECODE_ONLY + TRITON_ATTN + MTP spec=5 + caps [1,2,4,6,8,16,32], with the scripts/88 #7148 dispatcher patch
> appended to triton_shim/sitecustomize.py. The patch LOADED in all procs (`[b70 sitecustomize] patched cudagraph
> _create_padded_batch_descriptor`) and WORKED -- capture got PAST the dispatcher assert and reached
> `Capturing CUDA graphs (decode, FULL): 0/3`. It then crashed in the **baked XPU KERNEL**, not the dispatcher:
> `torch.ops.vllm.gdn_attention_core_xpu -> vllm/_xpu_ops.py:151 -> torch.ops._xpu_C.gdn_attention -> RuntimeError:
> spec_query_start_loc must have size [num_spec_decodes + 1]`. So the #7148 port correctly fixes the Python/dispatcher
> side, but **FULL-capture MTP on B70 is blocked by the `_xpu_C.gdn_attention` op (vllm_xpu_kernels 0.1.9) itself** --
> NOT fixable from the vLLM Python layer. NOTE: TRITON_ATTN did NOT dodge it -- the GDN *decode* core always routes
> through the baked `gdn_attention_core_xpu` op regardless of attention-backend. **VERDICT: PIECEWISE 1.79x is the
> CONFIRMED single-card ceiling on stock v0230; FULL needs an Intel vllm_xpu_kernels fix (the assert is untracked
> upstream -> issue draft: docs/kernel/21_gdn_spec_capture_issue.md). Shim restored; lease freed.**

> ### [M4 / M5 recipe -- community-research 2026-06-22]
> - **M4 (TP=2 MTP):** the capture-safe path is ALREADY ours -- `CCL_ENABLE_SYCL_KERNELS=1` makes TP=2 PIECEWISE capture
>   SUCCEED on our box (P2P_GPU H.5; the old Lorbus TP=2-MTP [NEG] was vLLM 0.20.1 WITHOUT that knob). Recipe: v0230 TP=2 +
>   `CCL_ENABLE_SYCL_KERNELS=1` + #41663 env + `SPEC spec=4` + PIECEWISE. BUT TP=2 is a CAPACITY play, not speed (allreduce
>   tax: TTFT 3.3x, c8 agg 2x worse -- P2P_GPU H.6); single-card DP-replica MTP beats it for models that fit one card. So M4
>   is mainly to CONFIRM the [NEG] is stale and to serve the 27B W8A8 (35GB, needs 2 cards) with MTP.
> - **M5 (35B-A3B MoE + MTP captured) -- DONE 2026-06-22. KEY FINDING: MTP is a DENSE lever, NOT a MoE lever.** Ran the
>   35B int4-AutoRound MoE on :v0230moe single-card, PIECEWISE + fp8 KV (NOVEL combo, works, no crash): MTP-OFF **66.82 t/s**
>   -> MTP-ON spec=4 **68.83 t/s, accept 2.68 = +3% (1.03x), FLAT**. Vs +79% on the dense 27B. Mechanism: the MoE activates
>   only ~3B of 35B params/token -> per-token decode is already compute-light (little weight-BW to amortize), and the verify
>   pass runs the MoE x(1+spec) with a WIDER expert union -> verify overhead ~cancels the draft savings. **Production: the
>   35B MoE headline is graph CAPTURE (66.8 t/s single-card), NOT MTP. Don't plumb MTP onto the MoE.** (The int8 35B MoE
>   capture is separately blocked by the dequant-linear inference-tensor issue forcing enforce-eager -- scripts/76.)
**Related:** [`docs/literature/07_w8a8_int8_recovery.md`](docs/literature/07_w8a8_int8_recovery.md) · [`JOURNAL.md`](JOURNAL.md) (MTP entries) · [`docs/COMMUNITY_CONFIGS.md`](docs/COMMUNITY_CONFIGS.md) · [`data/localmaxxing/`](data/localmaxxing/) + [`scripts/75_localmaxxing.py`](scripts/75_localmaxxing.py) · `contrib/vllm_int8_xpu/`

---

## Why this is the priority (the reframe)

**[UPDATED 2026-06-22] The 4xB70 TP=4 MTP headline is REAL and PUBLIC -- it is a community submission on localmaxxing.com** (user `ytnszmy`, 2026-06-10; cached in [`data/localmaxxing/`](data/localmaxxing/), pulled by [`scripts/75_localmaxxing.py`](scripts/75_localmaxxing.py)). The earlier "private source / not publicly documented" framing was WRONG: a web search misses it only because localmaxxing is not crawled, not because the data is secret. The row: Qwen3.6-27B **BF16** (fp16 runtime), TP=4 on 4x B70, **decode 54.2 t/s, prefill ~2100, accept 4.04 @ spec=5** (88.9% accept @ spec=3), 256K native ctx -> ~3x vs its raw column. **Most of the old "HIGH-VALUE ACTION: get the exact repro" is now ANSWERED by that row's own notes:** image `intel/llm-scaler-vllm:0.14.0-b8.3`, `vllm_xpu_kernels v0.1.9` wheel, `qwen3_5.py` spec-wiring (**vLLM #43565**), Half-KV, `num_speculative_tokens=5`. **The ONE ingredient the row does NOT state is `cudagraph_mode` (PIECEWISE vs FULL)** -- that single unknown is now the highest-value question (see the localmaxxing-evidence section below, which has a working FULL recipe from another submitter).

**The tension to resolve (this is why the recipe matters):** the 4xB70 result CONTRADICTS our own single-card B70 MTP, which is currently NET-NEGATIVE -- **25.5 t/s = -19% vs 31.4 MTP-off** (PIECEWISE, 86.9% first-token accept; the verify runs attention EAGER x(K+1)). So that config achieves what ours does not -- the exact recipe is the unlock. Most likely difference to probe: **FULL graph capture** (vs our PIECEWISE) -- and we now have a concrete FULL recipe from localmaxxing (RagingNoper's `cudagraph_mode=FULL_DECODE_ONLY`, see below) that hits ~102 t/s single-stream with NO spec at all, proving FULL capture is the dominant single-stream lever. Note the headline ran the DEPRECATED `intel/llm-scaler-vllm:0.14.0-b8.3` + manual #43565 patch + `vllm_xpu_kernels v0.1.9`; **per CLAUDE.md we do NOT reproduce on 0.14.x -- on our `vllm-xpu-env:v0230` image the GDN-MTP fix (#43565) is NATIVE (no patch, no 0.14.x kernels wheel)**, which is our cleanest path. **For OUR 2-card rig specifically:** MTP does NOT need TP, and TP>1 HURTS it on a no-P2P link (our TP=2 = 0.53x; **independently confirmed** by the Lorbus TP=2 MTP row below at 0.79x -- slower than both TP2-no-MTP AND single-card MTP, cause named by the submitter: "vLLM disables XPU graph capture for TP2 communication ops"), so our payoff path is **MTP per data-parallel replica** once FULL capture makes MTP net-positive on a single card.

The strategic consequence:

- **MTP is a ~3-4x multiplier on 4 cards -- but graph capture is the BIGGER single-stream lever AND a prerequisite for MTP.** localmaxxing shows FULL XPU graph capture alone takes single-stream 11 -> 102 t/s (~9.5x, no spec); MTP then stacks on top. On ONE card the only public true-MTP datapoint (Lorbus 27B-int4, below) is just ~41-45 t/s -- NOT yet a clean 3-4x, because nobody has measured MTP-on/off on one identical config WITH FULL capture. That gap is exactly what Phase A/B exists to close. Weight-format choice (W4A8 vs W8A8) is only ~1.2-1.5x.
- **MTP stacks orthogonally with format.** At spec=5 single-stream the verify pass is still *bandwidth-bound* (M≈6, intensity ~6 ops/byte ≪ B70's ~800 ops/byte int8 ridge), so MTP just cuts the number of bandwidth-bound passes by ~L×. It does NOT make decode compute-bound, so the weight-byte advantage of smaller formats *persists* under MTP.

**Therefore: get MTP working first, on the simplest format, then layer quant on top.** The format question is second-order and can be measured *after* the MTP pipeline is proven. Do not let the W4A8-vs-W8A8 decision block the MTP prize.

**The key per-format unknown is MTP acceptance.** The MTP head stays BF16 (required) and was trained against a full-precision body; quantizing the body may lower accept length. Expected ordering: **BF16 (4.04, proven) ≥ W8A8 (near-lossless → likely holds ~3.7–4.0) ≥ W4A8 (int4 drift → may drop to ~3.0–3.5)**. All hypotheses — *measure per format, that's the point of this plan.*

---

## Reference "known-good" config to reproduce (localmaxxing `ytnszmy` 4xB70 row)

Source: localmaxxing.com community submission, user `ytnszmy`, 2026-06-10 (cached `data/localmaxxing/b70_benchmarks_raw.json`). Ingredients are from that row's own notes.

| Ingredient | Value (as-submitted) | Notes |
|---|---|---|
| Image (as-submitted) | `intel/llm-scaler-vllm:0.14.0-b8.3` | DEPRECATED for us -- do NOT reproduce here; see "image (OURS)" |
| **Image (OURS -- use this)** | **`vllm-xpu-env:v0230`** | vLLM 0.23.0; **#43565 GDN-MTP + `gdn_attention` are NATIVE, no patch, no 0.14.x** (CLAUDE.md rule) |
| XPU kernels | `vllm_xpu_kernels v0.1.9` wheel | spec-decode enablement; subsumed by v0230 |
| Spec wiring | `qwen3_5.py` patch -- **vLLM #43565** | landed upstream in v0.23.0 -> native on v0230 |
| KV | **Half-KV** | needed to fit 256K context |
| Spec config | `num_speculative_tokens=5` | got accept length 4.04, 88.9% @ spec=3 |
| Graph capture | **UNSTATED in the row** | the missing ingredient -- PIECEWISE or FULL? assume WE must add FULL (RagingNoper recipe below) |
| Engine | vLLM-XPU, TP=4 (their setup) | ours: TP=1 for 14B, TP=1 for 27B-int4, TP=2 for 27B-W8A8 |

**Integration note (our hard part):** our W8A8 fastpath is the *custom* `contrib/vllm_int8_xpu` oneDNN s8s8s32 kernel, NOT stock. With v0230 the `gdn_attention` GDN kernel and #43565 spec-wiring are now NATIVE, so the real engineering shrinks to **one image carrying: (a) our W8A8 int8 GEMM registration, (b) FULL graph capture (`cudagraph_mode=FULL_DECODE_ONLY`, RagingNoper recipe), (c) Half-KV, and -- for TP -- (d) capture-safe collectives.** That integration -- not the GEMM-format choice -- is where the time goes.

---

## localmaxxing community evidence -- single-card MTP + the FULL-capture recipe

Pulled via `scripts/75_localmaxxing.py` (cache in `data/localmaxxing/`). After stripping schema-noise (the `mtpEnabled`/`specDecoding` fields are present-but-null on every row), **9 rows actually enable MTP/spec**. The three that matter for us:

### [STAR] Single-card true-MTP 27B (Lorbus) -- our Phase B target, already demonstrated by a 3rd party
- **Model:** `Lorbus/Qwen3.6-27B-int4-AutoRound` (INT4 AutoRound **W4A16**), vLLM `0.20.1` XPU, **1x B70, TP=1**, flash_attn, ctx 4096, `mtpEnabled=true` `specDecoding=true`. By `steveseguin`, 2026-05-03.
- **Result:** **45.2 t/s** out (latency 5.67s, **MTP accept 86.0%**) on the short run; a longer `OUTPUT_LEN=512` run dropped to **41.3 t/s, accept 65.4%** (accept decays with generated content).
- **Mechanism note (important):** "Local vLLM XPU patches route Qwen3.6 MTP speculative **Gated DeltaNet through a generic fallback when speculative masks are present**." => MTP+DeltaNet is a *workaround path*, not native -- expect this fragility when we wire MTP on our stack.
- **No graph capture, no MTP-off baseline submitted.** So 45.2 t/s is MTP-on, eager-ish, no capture; it is NOT a
  hidden no-MTP speed lever. Our v0230 PIECEWISE ctx2048 C1 sweep already exceeds it on computed `tg`
  (57.64 tok/s fp16 KV) and roughly matches it on realistic aggregate output only after paying 2048-token TTFT
  (35-36 tok/s).
- **Read for our plan:** this is the closest public analog to **Phase B1/B2 (W4A16/W4A8 27B + MTP, single-card)** -- same single-card, int4-weight, MTP-on regime. It proves single-card 27B MTP *runs*. **Our value-add: reproduce it on v0230 WITH FULL capture AND the MTP-off baseline, to get the real multiplier (theirs cannot give one).**

### [NEG] TP=2 MTP (Lorbus, dual-B70) -- the TP>1-hurts-MTP confirmation, directly relevant to Phase C
- Same model, **2x B70 TP=2**, `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`: **35.6 t/s** (latency 7.20s, warmup accept 62.6%).
- Submitter's own verdict: **"negative result: slower than TP2 non-MTP AND single-B70 MTP. vLLM disables XPU graph capture for TP2 communication ops."**
- **Read for our Phase C (TP=2 W8A8 + MTP):** this is the headwind, spelled out. On a no-P2P 2-card link, TP collectives run UNCAPTURED, eating the draft savings. Two consequences: (1) Phase C will NOT see MTP gains unless we solve **capture-safe TP collectives** -- exactly the problem RagingNoper solved below; (2) for our 2-card rig, **DP replicas (MTP per card) beat TP=2** until capture-safe all-reduce exists. Also note their TP2 used `num_speculative_tokens=1` (a weak MTP setting); a fairer Phase C retry uses **spec=5 AND capture-safe collectives**.

### [KEY] The FULL-capture recipe that makes any of this fast (RagingNoper, no spec at all)
- `Qwen3.6-35B-A3B` BF16, 4x TP4: **102.5 t/s single-stream**; `Qwen3-Coder-Next` 71.7 t/s. **No MTP, no spec** -- pure graph capture. By `RagingNoper`, 2026-06-19.
- "Stock vLLM disables graph capture on XPU -> eager ~11 t/s. Un-gating XPUGraph -> ~102 t/s (**9.5x**). torch.compile/Inductor alone adds ~nothing." Ceiling ~104-108, PCIe-bound (all-reduce latency).
- **The recipe (this is what Playbook A item 1 wants, now concrete):**
  ```
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",
    "splitting_ops":["vllm::unified_attention_with_output","vllm::unified_kv_cache_update",
      "vllm::gdn_attention_core_xpu","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"],
    "cudagraph_capture_sizes":[1,2,4,8,16,32,48,100]}'
  ```
  plus env `VLLM_XPU_ENABLE_XPU_GRAPH=1 VLLM_XPU_CUSTOM_AR=1 CCL_ENABLE_SYCL_KERNELS=1` and `DISABLE_ESIMD_*=1`.
- **Capture-safe TP collective (the Phase-C unlock):** "stock oneCCL all-reduce **corrupts inside a captured graph**, and PCIe device-to-device atomics are unreliable on Battlemage -> local-write/remote-read + uncached loads + system_acquire fences." This is a *custom* `xpu_communicator.py`, NOT stock vLLM. **This is precisely the capability our TP=2 MTP (Phase C) needs**, and it is why the Lorbus TP2 MTP row above was negative.
- Note `vllm::gdn_attention_core_xpu` is in `splitting_ops` -> the DeltaNet op is graph-compatible in this recipe; a good sign for capturing the MTP verify on Qwen3.6.

---

## Playbook A — How to get GOOD MTP (the knobs that move accept length × speedup)

The goal is **max end-to-end tok/s**, which is `base_decode × (mean_accept_length / verify_overhead)`. Maximize accept length, minimize verify overhead. In priority order:

1. **Kill eager-attention verify overhead — this is what murdered our earlier MTP.** Our prior run was net-negative because attention ran eager during verify (the −7% even with PIECEWISE). MTP's verify is a fixed-shape repeated forward → it *loves* graph capture.
   - Get **PIECEWISE graph capture** working with MTP first (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, `cudagraph_mode=PIECEWISE`, the `:int8g`-style image with fake-kernel registrations). PIECEWISE alone gave us +16.7% decode (23.3→27.2).
   - **FULL capture is no longer hypothetical** -- localmaxxing RagingNoper shipped a working `cudagraph_mode=FULL_DECODE_ONLY` recipe (see the localmaxxing-evidence section: capture ladder `[1,2,4,8,16,32,48,100]`, `gdn_attention_core_xpu` in `splitting_ops`, a custom capture-safe all-reduce) hitting ~102 t/s single-stream with NO spec. FULL is the dominant single-stream lever and is what should flip spec-decode from -7% to strongly positive. **Reproduce that FULL recipe on our v0230 image FIRST, then layer MTP on top.** (`torch.xpu.XPUGraph` exists upstream in PyTorch 2.11, per doc 06.)
2. **Sweep `num_speculative_tokens`.** The bench used 5 (→ accept 4.04). Higher spec = more drafted tokens/pass but lower per-position acceptance and higher verify cost. Sweep spec ∈ {2,3,4,5,6,8}; **pick the value that maximizes tok/s, not accept length** (they peak at different points). Log both per spec value.
3. **Measure at greedy (temp=0) for the headline, then map the temperature→acceptance curve.** Acceptance is maximal when distributions are peaked; higher temperature lowers accept length. Report max speedup at greedy; note the production-temp number separately.
4. **Keep the MTP head in BF16** (required — doc 04). It must be in the ignore-list (`re:.*mtp.*`) for every quant scheme; quantizing it kills drafting.
5. **Isolate Half-KV's effect on acceptance.** Half-KV / quantized KV perturbs the verify distribution and *can* depress accept length. Measure accept with full-precision KV vs Half-KV to confirm the context-fitting trick isn't silently costing speedup.
6. **The body's quant is the acceptance risk, not the head's.** A BF16 head drafting for a quantized body drifts: BF16 (4.04) ≥ W8A8 (likely holds) ≥ W4A8 (may drop). This is the single number Phase A/B exists to measure.
7. **Measurement hygiene:** discard warmup tokens (compile/graph capture), fixed prompt set + fixed context, batch-1, report median accept length and median tok/s over ≥3 runs.
8. **MTP is a low-concurrency/latency lever.** At batch-1 it shines (bandwidth-bound); as batch grows the verify goes compute-bound and MTP's relative win shrinks. Don't expect the 3–4× to survive into high-concurrency batched serving — there, KV capacity (favoring smaller formats) matters more.

---

## Playbook B — Quant procedures for RECOVERABLE quants (W8A8 / W4A8 / W4A16)

Produced via `scripts/49` (SCHEME-parametrized) in the Intel llmcompressor container. "Recoverable" = go hard on the activation+weight recovery levers from [`docs/literature/07`](docs/literature/07_w8a8_int8_recovery.md) so the quant clears the accuracy gate *before* it earns MTP effort.

### Common to all schemes
- **Ignore-list (name-robust):** `lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*` — DeltaNet, vision tower, MTP head, lm_head stay BF16. Keep the **parent-module regex** (`re:.*linear_attn.*`); never enumerate leaf names (`in_proj_qkvz`/`in_proj_ba`) → avoids the vLLM #40252 silent-zeroing trap.
- **Calibration:** 512 samples final (128 was within noise + 3× faster → use 128 to iterate), `SEQLEN=2048`, **data matched to the model** (chat/instruction data for instruct models, not random tokens).
- **GPTQ flag:** `actorder=None` (act-reordering triggers an XPU gather device-lost on our stack).
- **Validate before MTP:** eval top-1 agreement / ppl / gsm8k vs the BF16 baseline. A quant that misses the accuracy gate is out regardless of MTP speed.

### W8A8 — int8 w × int8 a (best accuracy; 27B needs 2 cards)
1. **Selective SmoothQuant (THE fix for the hybrid).** Build SmoothQuant mappings only where pairing is clean — the **16 full-attn layers** (`input_layernorm → {q,k,v}`, `v_proj → o_proj`) + the **64 MLP layers** (`post_attention_layernorm → {gate,up}`). **Skip DeltaNet `linear_attn`.** This sidesteps the `ValueError: got [all 64 input_layernorm]` that currently forces `SMOOTHQUANT=0`. (smoothing_strength ~0.8; tune per-node alpha if needed.)
2. **GPTQ weights, symmetric per-channel int8** (NOT group-128 — that's an int4 tool; per-channel is standard + cheaper at int8).
3. **Per-token dynamic int8 activations** at runtime (our kernel's path).
4. *Optional rescue:* hold early+late `down_proj` at W8A16 (GLU spike site).
- Expected: ~0.908 top-1 agreement (GPTQ), essentially W8A16-class quality.

### W4A8 — int4 w × int8 a (single-card 27B; hits int8 systolic) ← the recovery-critical one
1. **Selective SmoothQuant** — same mapping as W8A8 (recovers the int8-activation half).
2. **GPTQ weights, int4, group size 128** (int4 weights need finer granularity than per-channel; this is where group-128 earns its keep).
3. **Per-token dynamic int8 activations.**
4. **`down_proj` higher-precision carve-out matters MORE here** — int4 weights amplify the GLU/Super-Weight sensitivity. Try early+late `down_proj` at W4A16 or W8A16.
- Expected: RTN ~0.822 → GPTQ + SmoothQuant lifts toward ~0.86+. **This is the scheme where the recovery work decides viability** — if it can't clear the gate near W8A8, W4A8 is out.

### W4A16 — int4 w × fp16 a (single-card 27B; simplest accuracy, no int systolic)
1. **No SmoothQuant** — weight-only, so it's a no-op (`scripts/49` auto-sets `SQD=0` for `*A16`).
2. **GPTQ (or AWQ) weights, int4, group size 128.**
3. **fp16 activations** (no activation quant → no activation-fidelity loss; the only error is int4 weights).
- Expected: ~0.883 agreement (GPTQ); GPTQ lift +4.2 over RTN here — int4 weight error is real and GPTQ recovers it.
- **Caveat:** the 4304-dim `XPUwNa16` (multiple-of-32) kernel-coverage gap blocked our W4A16 27B once — verify the layer dims actually serve before counting on this scheme.

### Recovery levers, ranked (apply in this order until the gate clears)
1. **Selective SmoothQuant** on the 16 full-attn layers — the activation-fidelity lever (recovers the ~−10pt W8A16→W8A8 drop).
2. **GPTQ weights** — recovers int4/int8 weight error; lift scales with quant error (measured: W8A8 +2.7, W4A16 +4.2).
3. **`down_proj` at higher precision** (early+late layers) — the GLU/Super-Weight spike site.
4. **Better calibration** — 512 samples, chat-domain data.
5. **KL-sensitivity layer ranking** (arXiv 2604.13440, forward-only) — pick *which* layers to hold high-precision by measurement instead of guessing.

---

## Success criteria

- **Primary:** ≥ **3× decode speedup** from MTP (target 3–4×) on each qualified config, measured as `decode_tok/s(MTP on) / decode_tok/s(MTP off)` at batch-1 single-stream.
- **Secondary:** mean accept length logged per (model, scheme); accuracy within gate (see Phase B) for the quantized 27B configs.
- Every run logged in the table below — no run counts unless it's logged.

---

## Phase 0 — Prerequisites (do before anything)

- [ ] **Confirm Qwen3-14B has an MTP head** (`mtp.*` weights in the checkpoint). The proven bench was the **27B**; MTP shipped with the Qwen3-Next/3.6 family, and our 14B test vehicle is the *dense* Qwen3-14B which **may not have MTP**. Resolve one of:
  - (a) 14B checkpoint *does* have `mtp.*` → proceed with Phase A as written.
  - (b) 14B lacks MTP → qualify the *pipeline plumbing* on 14B via **ngram / draft-model spec-decode** (proves the spec loop, image, kernels), and accept that true MTP-acceptance numbers only come from the 27B (Phase B).
  - (c) substitute a 14B-class model that *does* ship MTP as the test vehicle.
- [ ] Stand up **`vllm-xpu-env:v0230`** (NOT the 0.14.x reference image -- #43565 + `gdn_attention` are native on v0230) + Half-KV + the RagingNoper FULL-capture `--compilation-config`; confirm it boots and serves *something* on one B70 with graph capture active.
- [ ] Pin the logging template (below) and a repeatable bench harness (reuse `perf_probe` / existing eval harness; fixed prompt set, fixed context, batch-1).

---

## Phase A — Qualify the MTP pipeline on Qwen3-14B (fast iteration, fits one card)

Rationale: the 14B is the clean, fast test vehicle (dense, fits one card, no 2-card TP complexity). Walk BF16 → W8A8 → W4A8 so each step adds exactly one variable.

- [ ] **A1 — BF16 14B + MTP.** Qualify the wiring with *zero* quant complexity. This isolates "does our MTP spec loop work" from "does quant break it." Log decode (MTP on/off), accept length, prefill, VRAM.
  - Gate to pass: ≥3× MTP speedup, accept length in a sane range (≥3). If this fails, the spec wiring/image is wrong — fix before touching quant.
- [ ] **A2 — W8A8 14B + MTP.** Our custom int8 kernel + spec wiring (the integration). Question: **does accept length hold vs BF16?** (expected: ~holds, int8 near-lossless). Log everything + top-1 agreement / gsm8k vs BF16.
- [ ] **A3 — W4A8 14B + MTP.** Adds int4 weights. Question: **does accept length hold at int4, or drop?** Log everything + accuracy. This is the first real datapoint on the W4A8-acceptance risk.

**Phase A exit:** we know (i) the MTP pipeline works on our stack, (ii) how much acceptance each format costs, (iii) the per-format MTP'd decode rate — i.e. enough to make the 27B format decision on evidence, not speculation.

---

## Phase B — Good 27B quant for single-card MTP: W4A16 + W4A8 (fit on ONE card)

Rationale: W4A16 (~25 GB) and W4A8 (~17 GB) both **fit one B70** → we can do this *now*, before a second card arrives. "Good quant" = go hard on recovery first (per doc 07): GPTQ weights + **selective SmoothQuant on the 16 full-attn layers**, DeltaNet `linear_attn` kept BF16, lm_head/vision/mtp ignored.

**Head start (localmaxxing):** a single-card W4A16 27B MTP run already exists -- `Lorbus/Qwen3.6-27B-int4-AutoRound`, 1x B70, **45.2 t/s -> 41.3 t/s, accept 86% -> 65%** (see the localmaxxing-evidence section). That is a direct precedent for **B1**, but it had NO graph capture and NO MTP-off baseline, so it cannot report a multiplier. **Start B1 by reproducing the Lorbus row on v0230, then add (i) FULL graph capture, (ii) the MTP-OFF baseline on the same config, (iii) GPTQ+SmoothQuant recovery vs their plain AutoRound.** That turns a bare 45 t/s into a real, multiplier-bearing single-card datapoint.

- [ ] **B0 — Produce the good quants** (no MTP yet): `scripts/49` SCHEME=W4A16 and SCHEME=W4A8, with GPTQ + selective-SmoothQuant. Eval accuracy vs the BF16 and W8A8 27B baselines (top-1 agreement, ppl, gsm8k).
  - **Accuracy gate:** W4A16 and W4A8 must land "close to W8A8/BF16" (within our noise band on gsm8k; agreement gap acceptable). If a scheme misses the gate, it's out for the 27B headline regardless of speed.
- [ ] **B1 — W4A16 27B + MTP.** Fits one card; simplest accuracy story. Caveat: fp16 verify (no int8 systolic), so it's the *weakest* MTP-compute fit — but the cheapest to stand up. Log MTP metrics + accuracy.
- [ ] **B2 — W4A8 27B + MTP.** Fits one card; int8 systolic verify. The interesting one — does accept length hold at int4 weights on the *27B* (which is more quant-robust than the 14B)? **Datapoint to beat:** the Lorbus W4A16 single-card row decayed accept 86% -> 65% over a longer decode -- watch for the same decay at W4A8 and **log accept vs token position, not just a single number**. Log MTP metrics + accuracy.

**Phase B exit:** a *serving-ready, MTP-accelerated 27B that runs on a single B70*, with W4A16 vs W4A8 decided on measured accuracy + accept length + decode rate.

---

## Phase C — Headline 27B W8A8 + MTP (DEFERRED — needs the 2nd card)

Rationale: 27B W8A8 ≈ 33 GB weights → **does not fit one 32 GB card**; needs TP=2 for VRAM headroom (and leaves ~37 GB KV across two cards — generous for long context). This is the production headline target, but it's blocked on hardware.

- [ ] **C1 — (when 2nd card lands) W8A8 27B + MTP, TP=2.** Best accuracy of the int-fastpath schemes + best expected accept length + native int8 prefill. Expected to be the production default. **HARD DEPENDENCY (from localmaxxing):** TP=2 MTP is net-NEGATIVE unless TP collectives are graph-capture-safe -- stock oneCCL all-reduce corrupts inside a captured graph, and the Lorbus TP=2 MTP row (35.6 t/s) was slower than single-card. **Port RagingNoper's capture-safe all-reduce** (local-write/remote-read + uncached loads + system_acquire fences; custom `xpu_communicator.py`) BEFORE expecting MTP gains at TP=2; otherwise prefer **DP replicas (MTP per card)**. Use **spec=5** (not the spec=1 the negative Lorbus row used). Log MTP metrics + accuracy + TP=2 PCIe overhead.

---

## Logging template (fill one row per run; newest at bottom)

Capture in this table (and mirror notable runs into `JOURNAL.md`):

| Date | Model | Scheme | Image / kernels / patch | TP | Ctx / KV | spec_toks | **accept len** | accept% @3 | **dec MTP-on** | dec MTP-off | **MTP ×** | prefill | TTFT | VRAM | acc (agree/gsm8k) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-10 | Qwen3.6-27B | BF16 | llm-scaler 0.14.0-b8.3 / v0.1.9 / #43565 | 4 | 256K/Half | 5 | 4.04 | 88.9% | 54.2 | ~17 | ~3x | ~2100 | — | — | — | localmaxxing ytnszmy [REF]; cudagraph_mode unstated |
| 2026-05-03 | Qwen3.6-27B | W4A16 | vLLM 0.20.1 AutoRound, NO capture | 1 | 4096 | mtp | — | 86.0% acc | 45.2 | — | — | — | — | — | — | localmaxxing Lorbus single-card [REF] |
| 2026-05-03 | Qwen3.6-27B | W4A16 | vLLM 0.20.1 AutoRound, NO capture | 1 | 4096 | mtp | — | 65.4% acc | 41.3 | — | — | — | — | — | — | Lorbus long run, accept decay [REF] |
| 2026-05-03 | Qwen3.6-27B | W4A16 | vLLM 0.20.1 AutoRound, TP2 | 2 | 4096 | 1 | — | 62.6% acc | 35.6 | — | — | — | — | — | — | Lorbus TP2 MTP [NEG][REF]; no capture under TP comm |
| — | Qwen3-14B | BF16 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | A1 |
| — | Qwen3-14B | W8A8 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | A2 |
| — | Qwen3-14B | W4A8 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | A3 |
| 2026-06-22 | Qwen3.6-27B | W4A16 | v0230 PIECEWISE (Lorbus int4-AR) | 1 | 8192/fp16 | 5 | 3.46 | — | 52.60 | 30.84 | 1.71x | — | — | ~17GiB | — | B1 spec=5 (M2 sweep) |
| 2026-06-22 | Qwen3.6-27B | W4A16 | v0230 PIECEWISE (Lorbus int4-AR) | 1 | 8192/fp16 | **4** | 3.25 | — | **55.28** | 30.84 | **1.79x** | — | — | ~17GiB | — | **B1 WINNER** single-card MTP, beats Lorbus 45.2; FULL crashes spec_query_start_loc |
| — | Qwen3.6-27B | W4A8 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | B2 |
| 2026-06-22 | Qwen3.6-35B-A3B | int4-AR MoE | v0230moe PIECEWISE+fp8KV | 1 | 8192/fp8 | 4 | 2.68 | — | **68.83** | 66.82 | **1.03x** | — | — | — | — | **M5** MoE+MTP NOVEL; MTP FLAT on MoE (sparse 3B-active); capture is the MoE headline |
| 2026-06-22 | Qwen3.6-27B | W4A16 | v0230 TP=2 PIECEWISE +SYCLKERNELS | 2 | 8192/fp16 | off/4 | — | — | CRASH | 26.96 | — | — | — | — | — | **M4** TP2 MTP-off 0.87x single-card; MTP-on CRASHES (spec-allgather not graph-capturable) -> TP=2 MTP DEAD |
| — | Qwen3.6-27B | W8A8 | — | 2 | — | 5 | — | — | — | — | — | — | — | — | — | C1 (2-card) -- needs RagingNoper capture-safe collectives (see M4) |

**Reference (REAL + PUBLIC -- localmaxxing.com, user `ytnszmy`, 2026-06-10; cached `data/localmaxxing/`):** Qwen3.6-27B BF16, **4x B70 TP=4**, spec=5 -> accept 4.04, dec 54.2, prefill 2100. Repro from the row's notes: `intel/llm-scaler-vllm:0.14.0-b8.3` + `vllm_xpu_kernels v0.1.9` + `qwen3_5.py` (#43565) + Half-KV; **on our v0230 image #43565 is native -- reproduce THERE, not on 0.14.x**. The ONE missing ingredient is `cudagraph_mode` (PIECEWISE vs FULL) -- add FULL ourselves (RagingNoper recipe). It beats our single-card MTP (currently **-19%: 25.5 vs 31.4, PIECEWISE**), so FULL capture is the unlock. (Other localmaxxing datapoints: RagingNoper 35B-A3B FULL-capture **102.5 t/s no-spec**; Lorbus 27B-int4 single-card MTP **45.2/41.3 t/s**, accept 86/65%; Lorbus TP2 MTP **35.6 t/s [NEG]**. Public sanity: Puget 4xB70 TP=4 27B-dense 13.1 t/s/1u no-MTP; vLLM PR #43565 MTP on B60/Qwen3-Next-80B/spec=2.)

---

## Open risks / watch-items

- **14B MTP-head existence** (Phase 0) — the whole 14B plan assumes it; resolve first.
- **Acceptance decay with quant** — the W4A8 thesis lives or dies on whether accept length holds; A3 + B2 are the deciding measurements.
- **Image integration** — our custom W8A8 kernel + FULL graph capture in one **v0230** image is the real engineering (`gdn_attention` + #43565 spec-wiring are now NATIVE on v0230, so the lift is mainly our int8 GEMM registration + the FULL-capture `--compilation-config` + capture-safe collectives); budget for it.
- **TP=2 MTP needs capture-safe collectives** — confirmed [NEG] on localmaxxing (Lorbus TP2): stock oneCCL all-reduce corrupts inside a captured graph, so TP comm runs UNCAPTURED and eats the draft savings. Phase C must port RagingNoper's capture-safe all-reduce, or prefer DP replicas (MTP per card).
- **The headline's `cudagraph_mode` is unknown** — the `ytnszmy` 54.2 row does not state PIECEWISE vs FULL. Treat FULL capture (RagingNoper recipe) as something WE must add; do not assume the headline already had it.
- **27B int4 kernel-coverage gaps** — CORRECTED 2026-06-23: the W4A16 "4304-dim `XPUwNa16` wall" was a RED
  HERRING (it was the weightless vision tower of a text-only checkpoint, not a serve-blocking kernel limit).
  W4A16 compressed-tensors serves (kernel/22); `int4_gemm_w4a16` is correct (NT-format weight). Still verify
  dims/load (`not found in params_dict` greps) when bringing up a re-homed checkpoint.
- **ALL our 27B quants are MTP-DEAD** (0 mtp tensors): W4A16, W4A8-sqgptq, W8A8-sqgptq. To get MTP, graft the
  bf16 `mtp.*` from the base (QUANTS_TODO QM, overnight) -- NOT a re-quant (mtp must stay bf16). Lorbus has it.
- **Half-KV / KV-quant interaction with MTP** — confirm Half-KV doesn't depress accept length.
- **DeltaNet ignore-list correctness** — keep `re:.*linear_attn.*` (parent-module regex, name-robust); don't regress to enumerating leaf names (`in_proj_qkvz`/`in_proj_ba`) or you risk silent layer-zeroing (vLLM #40252).
