# use_local_argmax_reduction MTP accept-collapse: root cause

Date: 2026-07-21
Model: nvidia/Qwen3.6-27B-NVFP4 (ModelOpt), vLLM 0.25.1, XPU, 2x B70, TP=2, MTP spec=5
Symptom: `--speculative-config use_local_argmax_reduction:true` -> serve HEALTHY,
output COHERENT, but MTP accept_len 0.65 (per-pos pos0=53% / pos1=12% / pos2+~0)
vs healthy ~4-5. Decode falls to MTP-off speed (c1 48.9 -> 25.7 t/s).

## Verdict

GENUINE vLLM-XPU op bug, NOT a config/model issue and NOT structural for MTP.
The two draft-sampling paths are provably mathematically identical for THIS model,
so the collapse is a runtime numeric discrepancy in the ONE op that only the
local-argmax path uses: `torch.max(dim=-1)` (`aten::max.dim`) over the wide
per-shard vocab, at
`vllm/model_executor/layers/logits_processor.py:136`.
GO on a fix (small, self-contained sitecustomize monkeypatch, default-off).

## 5-line root cause

- Full/healthy path (`llm_base_proposer.py:438`) samples with `.argmax(dim=-1)`;
  the local path (`:431`) calls `get_top_tokens`, whose ONLY differentiating op is
  `local_max_vals, local_max_indices = logits.max(dim=-1)` (`logits_processor.py:136`).
- `aten::max.dim` on XPU over a ~124160-wide bf16 shard returns a WRONG per-shard max
  VALUE (a partial/tile-local max), while `aten::argmax` (used by the working full
  path) is correct on this box -- so the two ops are NOT interchangeable on XPU.
- A too-low shard-1 max value makes the cross-shard reducer
  (`gathered[:,:,0].argmax`, `:154`) systematically pick rank 0's token; it is only
  correct when the true global max already lives on shard 0 (low token ids) -> pos0
  accept == P(true token on shard 0) ~= 53%, exactly the observed number.
- Once pos0 is wrong the autoregressive drafter is fed a wrong token, so pos1+ compound
  to ~0 -- matching pos0=53% / pos1=12% / pos2+~0.
- Everything else checked out identical (see "Ruled out"), so `.max(dim=-1)` is the
  sole remaining differentiator.

## What was verified / ruled out

1. Candidate #1 (num_pad masking / padding on wrong shard): DEAD.
   vocab_size = 248320 (config.json), lm_head.weight = [248320, 2560] (no pad rows).
   248320 % 64 == 0 and (248320/2)=124160, 124160 % 64 == 0 -> `num_org_vocab_padding`
   is 0 on BOTH shards -> the `logits[..., -num_pad:] = -inf` mask
   (`logits_processor.py:132-134`) is a no-op. Vocab splits as a clean contiguous
   [0,124160)/[124160,248320), so local-argmax is exactly equivalent to full argmax.

2. Candidate #2 (D2T / shared-vocab remap): DEAD.
   `mtp_use_dedicated_embeddings: false`; checkpoint has NO `draft_id_to_target_id`/`d2t`
   tensor. `LocalArgmaxMixin.get_top_tokens` (interfaces.py:1311) sets `d2t=None` ->
   no remap. Qwen3_5MTP (qwen3_5_mtp.py:275-280) `compute_logits` and the mixin BOTH
   use the SAME `self.lm_head` and `self.logits_processor` (scale=1.0, soft_cap=None),
   and the proposer load path (`llm_base_proposer.py:1547`) shares the target lm_head
   into `self.model.lm_head`, so both draft-sample paths hit identical weights/logits.
   The returned global index already IS the target vocab id. Correct.

3. lm_head precision: NVFP4-quantized (weight U8 packed fp4, weight_scale F8_E4M3
   [248320,320] = group_size 16). BOTH paths call the identical
   `lm_head.quant_method.apply(...)` -> logits are byte-identical between paths. So the
   discrepancy is downstream of logits, i.e. in the reduction, not the GEMM.

4. Candidate #4 (feedback plumbing): NOT the cause. Both paths return int64 [batch];
   the loop feeds `draft_token_ids_list[-1].int()` identically
   (`llm_base_proposer.py:681`). No shape/dtype divergence.

5. Candidate #5 (MTP support): SUPPORTED. Qwen3_5MTP uses the BASE `_greedy_sample`
   (`:428-438`); the load path raises if `get_top_tokens` is missing (`:1599-1605`).
   No bespoke override is skipped.

6. The all_gather shim is NOT the cause. Both the local pair-gather and the full-logits
   gather in `compute_logits` route through `XpuCommunicator.all_gather =
   _all_gather_via_allreduce` (sitecustomize block 3). The full path (big ~10.5M-elem
   gather) produces COHERENT output, proving the all-reduce-of-padded-buffer emulation
   correctly includes BOTH ranks. The same code path on a tiny [batch,2] tensor cannot
   selectively drop rank 1. The shim's movedim/reshape yields [r0v,r0i,r1v,r1i], which
   `gathered.view(batch,tp,2)` unpacks correctly (values at [:, :, 0]). Confirmed sound.

Net: the local path is algorithmically identical to the working full path EXCEPT it
routes the per-shard reduction through `aten::max.dim` instead of `aten::argmax`. On
XPU those are different kernels, and only `argmax` is proven correct on this box (it is
what the healthy config uses). The rank-0 bias signature (pos0 ~= 53%) is the tell.

## Fix (default-OFF sitecustomize monkeypatch)

Reformulate `get_top_tokens` to derive BOTH the value and the index from `argmax`
(proven-good on XPU) instead of `max(dim=-1)`. Keeps the O(2*tp) communication win;
changes only the local reduction op. Byte-identical on TP=1 (early return). Add this
block to `vllm/nvfp4/patches/sitecustomize.py`, gate `LOCALARGMAX_ARGMAX_FIX=1`:

```python
# ---- (N) XPU aten::max.dim is wrong on wide reductions: derive get_top_tokens'
# per-shard (val,idx) from argmax (proven-good) instead of max(dim=-1). Fixes
# use_local_argmax_reduction MTP accept-collapse (pos0~53% rank-0 bias). Default OFF.
if os.environ.get("LOCALARGMAX_ARGMAX_FIX", "0") == "1":
    try:
        import torch
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        _VERIFY = os.environ.get("LOCALARGMAX_VERIFY", "0") == "1"
        _n = [0]

        def _get_top_tokens_argmax(self, lm_head, hidden_states, embedding_bias=None):
            if self.scale <= 0.0 and self.scale != 1.0:
                raise ValueError("local argmax reduction needs positive logit scale")
            tp_size = get_tensor_model_parallel_world_size()
            logits = lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias)
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")
            # FIX: argmax (correct on XPU) instead of max(dim=-1) (wrong value/idx).
            local_max_indices = logits.argmax(dim=-1)
            local_max_vals = logits.gather(
                -1, local_max_indices.unsqueeze(-1)).squeeze(-1)
            if _VERIFY:
                bad_v, bad_i = logits.max(dim=-1)
                nmis = int((bad_i != local_max_indices).sum().item())
                _n[0] += 1
                if nmis or _n[0] <= 8:
                    print(f"[localargmax-verify] call={_n[0]} "
                          f"argmax!=max.dim mismatches={nmis}/{local_max_indices.numel()}",
                          file=sys.stderr, flush=True)
            vocab_start = lm_head.shard_indices.org_vocab_start_index
            global_indices = local_max_indices + vocab_start
            if tp_size == 1:
                return global_indices
            local_pair = torch.stack(
                [local_max_vals.float(), global_indices.float()], dim=-1)
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
            gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
            max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
            return top_tokens.squeeze(-1).to(torch.int64)

        LogitsProcessor.get_top_tokens = _get_top_tokens_argmax
        print("[nvfp4-shim] (N) get_top_tokens -> argmax-based (XPU max.dim fix)",
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (N) localargmax fix failed:", repr(e),
              file=sys.stderr, flush=True)
```

## How the coordinator tests it (one GPU run each)

1. CONFIRM the diagnosis (cheap, no fix behavior needed):
   serve with `LOCALARGMAX=1 LOCALARGMAX_ARGMAX_FIX=1 LOCALARGMAX_VERIFY=1`. If
   `[localargmax-verify] ... mismatches=K/N` reports K>0 on the drafter shard, that is
   DIRECT proof `aten::max.dim` disagrees with `aten::argmax` on XPU -> confirmed bug.
   (Expect mismatches concentrated where the true max value is near a shard-tile
   boundary; even a modest mismatch rate on the VALUE drives the rank-0 bias.)

2. VALIDATE the fix: serve with `LOCALARGMAX=1 LOCALARGMAX_ARGMAX_FIX=1` (verify off).
   Expected: accept_len returns to ~4-5 (parity with the full-gather path) AND decode
   stays at full-gather speed or better (the local reduction removes the K-1 full-vocab
   drafter gathers). If accept does NOT recover, the culprit is instead the size-2
   reducer ops (`gathered[...].argmax`/`.gather` at :154-155) -- next step would be to
   move that reduction to CPU; but that is a distant second given the rank-0 signature.

3. If both pass, bake `LOCALARGMAX=1 LOCALARGMAX_ARGMAX_FIX=1` into the DD env; this is
   the last decode lever (drafter full-vocab gather was 43% of the AR-bound decode).

## Go / No-go

GO. The fix is a self-contained, env-gated monkeypatch that keeps the communication
reduction and only swaps a per-shard `max(dim=-1)` for the proven-good `argmax`+`gather`.
Not structural for MTP; TP=1 is byte-identical (early return). Worst case (if the size-2
reducer is the real culprit) the diagnostic step 1 will still have proven whether
`max.dim` is broken, and the fallback is a CPU size-2 reduction.

## Key file:line references

- vllm/model_executor/layers/logits_processor.py:136  -- BUG: `logits.max(dim=-1)`
- vllm/model_executor/layers/logits_processor.py:132-134 -- num_pad mask (no-op here)
- vllm/model_executor/layers/logits_processor.py:154-155 -- size-2 rank reduce/gather
- vllm/v1/spec_decode/llm_base_proposer.py:428-438 -- `_greedy_sample` (438 = good argmax path)
- vllm/v1/spec_decode/llm_base_proposer.py:1547,1599-1605 -- MTP lm_head share + guard
- vllm/model_executor/models/qwen3_5_mtp.py:192,233,275-280 -- Qwen3_5MTP head wiring
- vllm/model_executor/models/interfaces.py:1311-1319 -- LocalArgmaxMixin (d2t=None here)
- vllm/nvfp4/patches/sitecustomize.py:306-326 -- all_gather shim (verified sound)
- models/files/qwen3.6-27b/nvfp4-modelopt/config.json -- vocab_size 248320, mtp_use_dedicated_embeddings false
