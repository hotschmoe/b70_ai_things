# O5 -- Flash-Next MTP on the EP sidecar (vLLM)

Owner: O5 (Claude Opus 5.5). Layout: world size 1 vLLM rank on card 0, EP
sidecar owns card 1's experts (b70_ep_xpu), FULL_DECODE_ONLY for target
and draft. Model: ISTA GSQ-RCO IQ3_XXS + X1's MTP drafter (Strata rt
dense Q8_0/BF16 + mtp-q2_0.gguf experts, all on card 0). No commits, no
DD start, health checked around every window. Raw evidence:
`/mnt/vm_8tb/b70/build/vllm-flashnext/o5/` (run-<tag>/ = tokens, steps,
recipe copy, run log, health).

STATUS: see "Final numbers" and "Parity" below (updated at the end of the
session).

## Recipe

Base = O4's no-MTP decode recipe f9 (JOURNAL 2026-10-20 23:35), plus:

| piece | where | what it does |
| --- | --- | --- |
| hooks guard | plugins/b70_ep_xpu/b70_ep_xpu/hooks.py | `SpeculatorCudaGraphManager` replays do not submit a card-1 graph (draft graphs hold no card-1 work; their token-count keys would collide with the target's card-1 graphs) |
| speculative_config | harness/o5_mtp_profile.py | `{"method": "mtp", "num_speculative_tokens": k}`; optional `num_speculative_tokens_per_batch_size` (dynamic k) and `use_local_argmax_reduction` with the draft vocab |
| draft vocab head | fork vllm/models/qwen4_exp/xpu/mtp.py `get_top_tokens` | `B70_MTP_DRAFT_VOCAB=draft_vocab_en.bin` (40525 rows, Strata's file): head rows gathered once (packed), GEMV over the subset, argmax mapped back. Drafts only; verify uses the full head |
| GDN spec decode | fork vllm/_xpu_ops.py | `B70_GDN_SPEC_DECODE=1`: a pure 1-token decode batch runs the speculative GDN/conv kernels with one token per sequence, so MTP off and MTP verify run the same compiled per-token code |
| FA fixed KV splits | private overlay o5/fa-overlay/vllm_xpu_kernels/flash_attn_interface.py | `B70_FA_KV_SPLITS=16`: fixed paged-decode split count for both q_len=1 decode and the spec fast path; the kernel then splits each row's own KV range, independent of batch and max_seqlen_k |
| kreg multi-column | o5/gguf-src (repo csrc + patch) -> _C.o5b.so | KREG at NCOLS >= 3 issues all R row loads and decodes each block once, then R x NCOLS dots (Strata launch_night_kreg order). Bitwise identical outputs, M <= 2 unchanged |
| Q8_0 KREG | same, _C.o5c.so, `B70_KREG_Q8=1` | Q8_0 through KREG (drafter rt dense only; the target has no Q8_0 tensors) |

Recipe files: harness/o5_recipe.env (f9 + MTP experts), o5_recipe_b.env
(+ o5b + GDN flag), o5_recipe_c.env (+ o5c + B70_KREG_Q8 + B70_FA_KV_SPLITS,
needs `O5_PYPATH_PRE=/mnt/vm_8tb/b70/build/vllm-flashnext/o5/fa-overlay`).
Window: harness/o5_window.sh (health -> xpu-run both cards -> in-process
LLM probe -> health). Patches: harness/patches/o5_kreg_multicol.patch,
o5_kreg_multicol_q8.patch, o5_fa_kv_splits.patch, o5_gdn_spec_decode.patch,
o5_x1_plus_draft_vocab_mtp.patch (X1's mtp.py work plus get_top_tokens).

## Measurement

harness/o5_mtp_profile.py: in-process vLLM (no HTTP), gates 17*19=323 and
23*29=667, then 4 prompts x 128 forced greedy tokens at C1 and as two C2
pairs (each row with a different neighbour). Effective ms/token = decode
window (first decode step start to last sample end) / 127, per stream.
Per step: target verify graph device time (event pair), propose device
time, step wall. Acceptance from vLLM spec-decode metrics. A same-load
repeat of C1 (c1r) checks determinism. This is a latency probe, not the
promotion gate (X4 owns that).

## Results (see final table below)

## Parity

Gate: harness/o5_compare.py, MTP on vs off with the same recipe, every
C1 and C2 stream token-identical over 128 tokens.

## Strata comparison

Strata (DD): `--spec 4` = 3 drafts, fixed width (min-p gate off), one
4-row verify, drafter all on card 0 and serial (no EP, no overlap with
card 1), own dense KV, no PLE term (PLE table only prefetched), 40525-row
draft head; C1 window ~22.4 ms at ~3.12 tokens/window. Batch MTP depth 2
at C2 (PAIR_HEAD/PAIR_GRAPH). Our vLLM port uses the same depth/vocab
choices; dynamic k (3 at C1, 2 at C2) maps onto vLLM's
num_speculative_tokens_per_batch_size.
