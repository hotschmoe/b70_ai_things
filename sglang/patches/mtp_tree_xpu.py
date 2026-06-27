# mtp_tree_xpu.py -- pure-torch XPU fallback for sglang's two unregistered EAGLE/NEXTN tree kernels,
# LINEAR-CHAIN case (topk==1): build_tree_kernel_efficient + verify_tree_greedy_func.
# Semantics ported verbatim from sgl-kernel/csrc/speculative/eagle_utils.cu (VerifyTreeGreedy +
# build_tree_efficient). Chain = D=num_verify_tokens nodes in a line: node 0 = root/bonus, node i child
# of i-1. With topk==1 every retrieve_next_sibling is -1, so the tree walk collapses to a prefix scan.
# Gated opt-in via B70_XPU_MTP=1 (installed from woq_shim). No-op unless sglang+xpu present.
#
# Key off-by-one (from eagle_utils.cu + the python wrapper): the kernel's accept_token_num counts
# DRAFTS ONLY (excludes the always-accepted root) = accept_len-1; eagle_sample adds the +1 downstream.
import os


def install():
    import torch
    from sglang.srt.speculative import eagle_utils as eu
    from sglang.srt.speculative.eagle_utils import TreeMaskMode

    def _build_tree(
        bonus_tokens,
        parent_list,
        top_scores_index,
        draft_tokens,
        seq_lens,
        seq_lens_sum,
        topk,
        spec_steps,
        num_verify_tokens,
        tree_mask_mode=TreeMaskMode.FULL_MASK,
        tree_mask_buf=None,
        position_buf=None,
    ):
        assert topk == 1, f"XPU MTP tree fallback supports only topk==1 (chain); got topk={topk}"
        if os.environ.get("B70_MTP_DEBUG") == "1":
            print(f"[mtp-dbg] build_tree ENTER bs={seq_lens.numel()} nv={num_verify_tokens} mode={int(tree_mask_mode)}", flush=True)
        draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()
        bs = seq_lens.numel()
        device = seq_lens.device
        D = num_verify_tokens
        sl = seq_lens.to(torch.long)
        depth = torch.arange(D, device=device, dtype=torch.long)

        # positions [bs*D] = past_seq_len + depth (chain depth == node index)
        positions = (
            position_buf
            if position_buf is not None
            else torch.empty((bs * D,), device=device, dtype=torch.long)
        )
        positions.copy_((sl.unsqueeze(1) + depth.unsqueeze(0)).flatten())

        # retrieve graph (chain): identity index, next=i+1 (leaf -1), no siblings
        retrieve_index = torch.arange(bs, device=device, dtype=torch.long).unsqueeze(1) * D + depth.unsqueeze(0)
        rnt = torch.where(depth < D - 1, depth + 1, torch.full_like(depth, -1))
        retrieve_next_token = rnt.unsqueeze(0).expand(bs, D).contiguous()
        retrieve_next_sibling = torch.full((bs, D), -1, device=device, dtype=torch.long)

        # tree_mask: causal chain -- node i (query) attends nodes 0..i (key). tri[i] = ancestor-or-self row.
        tri = torch.tril(torch.ones(D, D, dtype=torch.bool, device=device))
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            # layout [bs][query][key], flat bs*D*D
            tm = tri.unsqueeze(0).expand(bs, D, D).reshape(-1)
            if tree_mask_buf is not None:
                tree_mask_buf.copy_(tm)
                tree_mask = tree_mask_buf
            else:
                tree_mask = tm.clone()
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            # per batch b: D query rows, key width seq_len[b]+D; first seq_len keys are past (True),
            # last D are the causal tree cols. Region for b starts after b's predecessors.
            if tree_mask_buf is not None:
                tree_mask = tree_mask_buf
                tree_mask.fill_(True)
            else:
                tree_mask = torch.ones(
                    (seq_lens_sum * D + D * D * bs,), dtype=torch.bool, device=device
                )
            sl_cpu = sl.tolist()
            off = 0
            for b in range(bs):
                s = sl_cpu[b]
                width = s + D
                for tid in range(D):
                    row = off + width * tid
                    tree_mask[row + s : row + s + D] = tri[tid]
                off += width * D
        else:
            raise NotImplementedError(f"XPU MTP: unsupported tree_mask_mode {tree_mask_mode}")

        if os.environ.get("B70_MTP_DEBUG") == "1":
            print(f"[mtp-dbg] build_tree EXIT mask_numel={tree_mask.numel()}", flush=True)
        return (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        )

    def _verify(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        topk=-1,
    ):
        # Chain greedy verify. predicts: flat [bs*D] (init 0). accept_index: [bs,S] (init -1).
        # accept_token_num: [bs] = DRAFTS ONLY. candidates/target_predict: [bs,D].
        bs, D = candidates.shape
        device = candidates.device
        S = accept_index.shape[1]
        if os.environ.get("B70_MTP_DEBUG") == "1":
            print(f"[mtp-dbg] verify ENTER bs={bs} D={D} S={S}", flush=True)
        # draft j (1..D-1) accepted iff candidates[:,j]==target_predict[:,j-1] AND the chain prefix accepted.
        match = candidates[:, 1:] == target_predict[:, :-1]  # [bs, D-1]
        if match.numel() == 0:
            n_acc = torch.zeros(bs, device=device, dtype=torch.long)
        else:
            n_acc = torch.cumprod(match.to(torch.int32), dim=1).sum(dim=1).to(torch.long)  # leading run
        accept_token_num.copy_(n_acc.to(accept_token_num.dtype))

        # accept_index: front-filled flat node indices b*D+0 .. b*D+n_acc; rest -1.
        ar = torch.arange(S, device=device).unsqueeze(0)  # [1,S]
        valid = ar <= n_acc.unsqueeze(1)  # [bs,S] slots 0..n_acc
        base = (torch.arange(bs, device=device, dtype=torch.long) * D).unsqueeze(1)
        node_flat = base + ar  # [bs,S] flat node index (== slot for a chain)
        accept_index.copy_(
            torch.where(valid, node_flat.to(accept_index.dtype), torch.full_like(accept_index, -1))
        )

        # predicts[node] = target_predict[node] for each accepted node (0..n_acc).
        tp_flat = target_predict.reshape(-1)
        sel_flat = node_flat[valid]  # accepted flat node indices
        predicts[sel_flat] = tp_flat[sel_flat].to(predicts.dtype)
        if os.environ.get("B70_MTP_DEBUG") == "1":
            print(f"[mtp-dbg] verify EXIT n_acc={n_acc.tolist()}", flush=True)
        return predicts, accept_index, accept_token_num

    eu.build_tree_kernel_efficient = _build_tree
    eu.verify_tree_greedy_func = _verify

    # DEBUG (B70_MTP_DEBUG=1): trace the MHA KV-write shapes to locate the spec-decode 2-vs-75840 mismatch.
    if os.environ.get("B70_MTP_DEBUG") == "1":
        try:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
            _orig_skb = MHATokenToKVPool.set_kv_buffer
            _n = [0]

            def _traced_skb(self, layer, loc_info, cache_k, cache_v, *a, **k):
                if _n[0] < 10:
                    try:
                        loc, _ = unwrap_write_loc(loc_info)
                        lshape = tuple(loc.shape)
                    except Exception:
                        lshape = "?"
                    try:
                        bshape = tuple(self.k_buffer[0].shape)
                    except Exception:
                        bshape = "?"
                    print(f"[mtp-dbg] KVwrite layer={getattr(layer,'layer_id','?')} cache_k={tuple(cache_k.shape)} loc={lshape} buf={bshape}", flush=True)
                    _n[0] += 1
                return _orig_skb(self, layer, loc_info, cache_k, cache_v, *a, **k)

            MHATokenToKVPool.set_kv_buffer = _traced_skb
            print("[mtp-dbg] KV-write shape trace installed", flush=True)
        except Exception as e:
            print(f"[mtp-dbg] KV trace install failed: {e}", flush=True)

    print("[mtp-tree-xpu] installed chain (topk=1) build_tree + verify_tree_greedy torch fallbacks", flush=True)
