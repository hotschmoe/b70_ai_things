#!/usr/bin/env python3
"""Prepare an isolated FA Python overlay; do not alter installed packages."""

from pathlib import Path


def main():
    build = Path("/mnt/vm_8tb/b70/build/vllm-flashnext")
    source = build / "c7b-fa-overlay/vllm_xpu_kernels"
    root = build / "d1/fa-overlay"
    target = root / "vllm_xpu_kernels"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in ("flash_attn_interface.py", "__pycache__"):
            continue
        dest = target / item.name
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(item.resolve())
    body = (source / "flash_attn_interface.py").read_text()
    marker = "    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)\n"
    assert body.count(marker) == 1
    patch = '''    # D1: graph-safe compact plans for short-context no-spec decode.
    # Default off; other shapes and prefill retain the original interface.
    _d1_count = int(os.environ.get("B70_D1_FA_PLAN", "0"))
    if (_d1_count > 0 and fa_version == 2 and max_seqlen_q == 1
            and block_table is not None and seqused_k is not None
            and q.shape[0] == seqused_k.numel()
            and q.shape[1:] == (24, 256) and k.shape[1:] == (1600, 2, 256)
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16 and causal
            and not return_softmax_lse and not return_attn_probs
            and real_window_size == (-1, -1) and softcap == 0.0
            and dropout_p == 0.0 and alibi_slopes is None
            and q_v is None and q_descale is None and k_descale is None
            and v_descale is None and scheduler_metadata is None):
        from d1_fa_plan import paged_attention
        return paged_attention(q, k, v, out, cu_seqlens_q, seqused_k,
                               block_table, max_seqlen_k, _d1_count,
                               softmax_scale, s_aux)

'''
    body = body.replace(marker, patch + marker)
    compile(body, str(target / "flash_attn_interface.py"), "exec")
    (target / "flash_attn_interface.py").write_text(body)
    helper = root / "d1_fa_plan.py"
    if not helper.exists():
        helper.symlink_to(Path(__file__).with_name("d1_fa_plan.py").resolve())
    print(root)


if __name__ == "__main__":
    main()
