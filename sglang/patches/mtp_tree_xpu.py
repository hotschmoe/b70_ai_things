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

    # --- THE MTP UNBLOCK (2026-06-27): the spec-decode VERIFY out_cache_loc is None on XPU, not a bad index.
    # triton_ops/cache_locs.py:assign_extend_cache_locs_func gates on _is_cuda/_is_hip/_is_musa/_is_npu with
    # NO _is_xpu branch -> on a torch+xpu build it falls through and returns None -> batch.out_cache_loc=None
    # -> the verify KV write does k_cache[None]=k -> "[1,75904,4,256] vs [2,4,256]" crash. The underlying
    # triton kernel ALREADY runs on XPU (move_accept_tokens_to_target_kvcache calls it ungated); only the
    # wrapper's hardware gate is spurious. eagle_prepare_for_verify does a function-LOCAL import of this symbol,
    # so patching the module attribute here is picked up at verify time. Pure-torch gather of the new draft
    # slots (decode-prep already wrote them into req_to_token[req, seq_lens : seq_lens+draft_token_num]).
    try:
        from sglang.srt.speculative.triton_ops import cache_locs

        def _xpu_assign_extend(req_pool_indices, req_to_token, start_offset, end_offset,
                               batch_size, draft_token_num, device):
            rows = req_to_token[req_pool_indices.long()]                                  # [bs, pool_len]
            ar = torch.arange(draft_token_num, device=device, dtype=torch.long).view(1, -1)
            idx = start_offset.to(torch.long).view(-1, 1) + ar                            # [bs, draft_token_num]
            return torch.gather(rows.to(torch.long), 1, idx).reshape(-1).contiguous()  # int64 [bs*draft_token_num]

        cache_locs.assign_extend_cache_locs_func = _xpu_assign_extend
        print("[mtp-tree-xpu] patched assign_extend_cache_locs_func (XPU None-index -> draft-slot gather)", flush=True)
    except Exception as e:
        print(f"[mtp-tree-xpu] assign_extend patch FAILED: {e}", flush=True)

    # --- THE NEXT DOMINO: the post-verify GDN-state-commit triton fns (mamba_state_scatter_triton.py)
    # raise "only supports CUDA tensors" via a spurious `if not dst.is_cuda...` guard -- but the inner
    # @triton.jit kernels run fine on XPU (like every other GDN triton kernel). Strip the is_cuda guard by
    # re-exec'ing each function's source minus the guard block, and patch BOTH the source module AND
    # hybrid_linear_attn_backend (which imported the names by value at module level). Generic stripper also
    # future-proofs against further identical gates.
    try:
        import inspect
        from sglang.srt.layers.attention.mamba import mamba_state_scatter_triton as _mss
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as _hlab

        def _strip_is_cuda_guard(src):
            lines = src.split("\n"); out = []; i = 0
            while i < len(lines):
                ln = lines[i]
                if "is_cuda" in ln and ln.lstrip().startswith("if "):
                    indent = len(ln) - len(ln.lstrip()); i += 1
                    while i < len(lines) and (not lines[i].strip()
                                              or (len(lines[i]) - len(lines[i].lstrip())) > indent):
                        i += 1
                    continue
                out.append(ln); i += 1
            return "\n".join(out)

        for _fn in ("fused_mamba_state_scatter_with_mask", "fused_conv_window_scatter_with_mask"):
            _ns = dict(_mss.__dict__)  # module globals: triton, tl, torch, the inner @triton.jit kernels
            exec(_strip_is_cuda_guard(inspect.getsource(getattr(_mss, _fn))), _ns)
            setattr(_mss, _fn, _ns[_fn])
            if hasattr(_hlab, _fn):
                setattr(_hlab, _fn, _ns[_fn])
        print("[mtp-tree-xpu] stripped is_cuda guards on fused mamba/conv state-scatter (XPU)", flush=True)
    except Exception as e:
        print(f"[mtp-tree-xpu] mamba-scatter guard strip FAILED: {e}", flush=True)

    # --- DOMINO 3: top_p_renorm_probs is UNREGISTERED on XPU (top_k_renorm IS). eagle_sample's verify calls
    # top_p_renorm_prob for any top-p (non-greedy) request -> torch.ops.sgl_kernel.top_p_renorm_probs
    # AttributeError -> scheduler crash. (Greedy/temperature=0 skips it -- why single-stream coherence passed
    # but sampling load crashed.) Standard nucleus renorm is trivial in torch; patch eagle_utils' bound ref.
    try:
        def _xpu_top_p_renorm(probs, top_p):
            probs = probs.float()
            p = top_p.float().view(-1, 1) if isinstance(top_p, torch.Tensor) else float(top_p)
            sp, si = torch.sort(probs, dim=-1, descending=True)
            keep = (sp.cumsum(dim=-1) - sp) < p          # nucleus: include the token that crosses top_p
            sp = sp * keep
            sp = sp / sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            return torch.zeros_like(probs).scatter_(-1, si, sp)
        # eagle_sample does a FUNCTION-LOCAL `from sgl_kernel import top_p_renorm_prob`, so patch the SOURCE
        # package (not just eagle_utils) -- the local import re-binds the name from sgl_kernel each call.
        import sgl_kernel as _sk
        from sgl_kernel import sampling as _sks
        for _m in (_sk, _sks, eu):
            for _nm in ("top_p_renorm_prob", "top_p_renorm_probs"):
                if hasattr(_m, _nm):
                    setattr(_m, _nm, _xpu_top_p_renorm)
        print("[mtp-tree-xpu] patched top_p_renorm_prob in sgl_kernel + eagle_utils (XPU nucleus-renorm)", flush=True)
    except Exception as e:
        print(f"[mtp-tree-xpu] top_p_renorm patch FAILED: {e}", flush=True)

    # --- DOMINO 4: put XPU in the greedy-verify branch (with NPU/HIP). eagle_sample's NON-greedy path calls
    # tree_speculative_sampling_target_only, which is UNREGISTERED on XPU (and signature-skewed). The greedy
    # branch (verify_tree_greedy, our working fallback) is gated `if is_all_greedy or _is_npu or _is_hip:` --
    # XPU is missing. Real API requests get the model-default top_k>1 (is_all_greedy=False) -> crash. Re-exec
    # eagle_sample with XPU added to that branch -> ALL verify uses verify_tree_greedy. COST (documented):
    # sampling (temperature/top_p/top_k) is verified GREEDILY on XPU+MTP, exactly like NPU/HIP. The correct
    # long-term fix is a torch chain rejection-sampler (task #14); this unblocks concurrent MTP today.
    try:
        import inspect
        _cond = "if sampling_info.is_all_greedy or _is_npu or _is_hip:"
        _src = inspect.getsource(eu.eagle_sample)
        if _cond in _src:
            _ns = dict(eu.__dict__)
            # eagle_utils uses `from __future__ import annotations` (string annotations); replicate it so the
            # re-exec does not eval signature annotations (EagleVerifyInput etc.) -> NameError.
            _patched = "from __future__ import annotations\n" + _src.replace(
                _cond, _cond[:-1] + " or True:  # XPU greedy-verify (tree_speculative unregistered)")
            exec(_patched, _ns)
            eu.eagle_sample = _ns["eagle_sample"]
            try:
                from sglang.srt.speculative import eagle_worker_v2 as _ewv2
                if hasattr(_ewv2, "eagle_sample"):
                    _ewv2.eagle_sample = _ns["eagle_sample"]
            except Exception:
                pass
            print("[mtp-tree-xpu] forced greedy-verify branch in eagle_sample (XPU, with NPU/HIP)", flush=True)
        else:
            print("[mtp-tree-xpu] eagle_sample greedy-force SKIPPED (condition not found -- upstream changed)", flush=True)
    except Exception as e:
        print(f"[mtp-tree-xpu] eagle_sample greedy-force FAILED: {e}", flush=True)

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

    # --- DOMINO 5 (opt-in B70_XPU_MAMBA_EXTRA_BUFFER=1): un-gate the mamba extra_buffer radix strategy on XPU.
    # server_args._validate_mamba_extra_buffer asserts (is_cuda() or is_musa() or is_npu()) "needs CUDA/MUSA/NPU
    # (FLA)" -- but "FLA" here is sglang's VENDORED Triton (layers/attention/fla/), and the whole extra_buffer
    # runtime (track/restore, checkpoint copy, int8 quant) is Triton or pure-torch, all proven on XPU (the
    # MTP-path scatter guards are already stripped in DOMINO 2). Dropping this one assert lets extra_buffer run
    # with the intel_xpu XMX attention backend at page_size 64/128 -- prefix caching WITHOUT the
    # no_buffer+page_size=1 long-context decode collapse. Inert unless --mamba-radix-cache-strategy extra_buffer.
    if os.environ.get("B70_XPU_MAMBA_EXTRA_BUFFER") == "1":
        try:
            import inspect, textwrap
            import sglang.srt.server_args as _sa
            _src = textwrap.dedent(inspect.getsource(_sa.ServerArgs._validate_mamba_extra_buffer))
            _needle = "is_cuda() or is_musa() or is_npu()"
            if _needle in _src:
                _ns = dict(_sa.__dict__)
                exec(_src.replace(_needle, "True  # XPU: extra_buffer path is Triton/pure-torch"), _ns)
                _sa.ServerArgs._validate_mamba_extra_buffer = _ns["_validate_mamba_extra_buffer"]
                print("[mtp-tree-xpu] un-gated extra_buffer for XPU (dropped CUDA/MUSA/NPU assert)", flush=True)
            else:
                print("[mtp-tree-xpu] extra_buffer un-gate SKIPPED (assert text changed upstream)", flush=True)
        except Exception as e:
            print(f"[mtp-tree-xpu] extra_buffer un-gate FAILED: {e}", flush=True)

    print("[mtp-tree-xpu] installed chain (topk=1) build_tree + verify_tree_greedy torch fallbacks", flush=True)
