# xpu_cudagraph.py -- wire torch.xpu.XPUGraph capture into sglang's decode path on Intel B70.
# Stock sglang stays eager on XPU ("cuda graph: False") because (a) model_runner.init_cuda_graphs gates
# init_decode_cuda_graph() behind a hardcoded device allow-list ("cuda","musa","cpu","npu") that omits "xpu",
# and (b) XPUAttentionBackend (full-attn) has NO cuda-graph hooks (init_cuda_graph_state -> base
# NotImplementedError; its eager init_forward_metadata builds page_table from FRESH advanced-indexed tensors).
# The torch.cuda.{CUDAGraph,graph,graph_pool_handle}->torch.xpu redirects are already done in woq_shim under
# B70_XPU_CUDAGRAPH=1; torch.xpu.XPUGraph is proven STABLE on B70 (scripts/137, 4000 replays no degradation).
#
# This patch adds (1) "xpu" to the init_cuda_graphs device gate, and (2) the missing graph hooks on
# XPUAttentionBackend for the NORMAL DECODE (no-spec, topk<=1) case: a fixed-width TOKEN-LEVEL static
# page_table [max_bs, max_context_len] + static cache_seqlens/cu_seqlens, refilled IN-PLACE each replay
# (cache_seqlens bounds the kernel read, so the zero-padded tail is ignored). The GDN/mamba sub-backend is
# already capture-safe (static MambaPool + in-place writes). Spec-decode (MTP) graph is NOT handled here.
# Gated opt-in via B70_XPU_CUDAGRAPH=1 (installed from woq_shim). No-op unless sglang+xpu present.
import os


def install():
    import torch
    try:
        from sglang.srt.utils import is_xpu
        if not is_xpu():
            return
    except Exception:
        return
    if os.environ.get("B70_XPU_CUDAGRAPH") != "1":
        return

    dbg = os.environ.get("B70_XPU_CUDAGRAPH_DEBUG") == "1"

    # ---- 1. Device gate: add "xpu" to model_runner.init_cuda_graphs allow-list ----
    try:
        import inspect
        import sglang.srt.model_executor.model_runner as mr
        src = inspect.getsource(mr.ModelRunner.init_cuda_graphs)
        needle = '("cuda", "musa", "cpu", "npu")'
        if needle in src:
            import textwrap
            src = textwrap.dedent(src).replace(needle, '("cuda", "musa", "cpu", "npu", "xpu")')
            ns = dict(mr.__dict__)
            exec(src, ns)
            mr.ModelRunner.init_cuda_graphs = ns["init_cuda_graphs"]
            print("[xpu-cudagraph] init_cuda_graphs gate: added 'xpu'", flush=True)
        else:
            print("[xpu-cudagraph] WARN: device gate needle not found (upstream changed) -- decode graph stays off", flush=True)
    except Exception as e:
        print(f"[xpu-cudagraph] device-gate patch FAILED: {e}", flush=True)
        return

    # ---- 2. XPUAttentionBackend graph hooks (normal decode, no-spec, topk<=1) ----
    try:
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
        from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
    except Exception as e:
        print(f"[xpu-cudagraph] import XPUAttentionBackend FAILED: {e}", flush=True)
        return

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        dev = self.device
        self._b70_cg = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=dev),
            "cu_seqlens_q": torch.arange(0, max_bs + 1, dtype=torch.int32, device=dev),
            "cu_seqlens_k": torch.zeros(max_bs + 1, dtype=torch.int32, device=dev),
            # TOKEN-level page_table (matches the eager XPU path: req_to_token[req, :len]), fixed width.
            "page_table": torch.zeros(max_bs, self.max_context_len, dtype=torch.int32, device=dev),
        }
        self.decode_cuda_graph_metadata = {}
        if dbg:
            print(f"[xpu-cudagraph] init_cuda_graph_state max_bs={max_bs} max_ctx={self.max_context_len}", flush=True)

    def _b70_bind(self, bs):
        b = self._b70_cg
        m = FlashAttentionMetadata()
        m.cache_seqlens_int32 = b["cache_seqlens"][:bs]
        m.cu_seqlens_q = b["cu_seqlens_q"][: bs + 1]
        m.cu_seqlens_k = b["cu_seqlens_k"][: bs + 1]
        m.page_table = b["page_table"][:bs]
        m.max_seq_len_q = 1
        self.decode_cuda_graph_metadata[bs] = m
        return m

    def _b70_fill(self, bs, req_pool_indices, seq_lens):
        m = self.decode_cuda_graph_metadata[bs]
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        max_len = int(seq_lens.max().item())
        m.max_seq_len_k = max_len
        m.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
        # cu_seqlens_k = [0, cumsum(seq_lens)]; index 0 stays 0 (static zero buffer).
        torch.cumsum(seq_lens, dim=0, dtype=torch.int32, out=m.cu_seqlens_k[1:])
        # page_table[:, :max_len] <- req_to_token[req, :max_len] (in-place); tail untouched (cache_seqlens bounds).
        m.page_table[:, :max_len].copy_(self.req_to_token[req_pool_indices, :max_len])

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        # Called by the hybrid wrapper for the full-attn sub-backend: at capture (in_capture=True) and
        # before each replay (in_capture=False, line ~950 of decode_cuda_graph_runner).
        fm = forward_batch.forward_mode
        if not (fm.is_decode_or_idle() and forward_batch.spec_info is None):
            # spec/verify/extend not handled by this graph patch -- leave eager metadata (won't be captured here).
            return XPUAttentionBackend.init_forward_metadata(self, forward_batch)
        bs = forward_batch.batch_size
        if in_capture or bs not in self.decode_cuda_graph_metadata:
            _b70_bind(self, bs)
        _b70_fill(self, bs, forward_batch.req_pool_indices, forward_batch.seq_lens)
        self.forward_metadata = self.decode_cuda_graph_metadata[bs]

    def init_forward_metadata_in_graph(self, forward_batch):
        # Metadata is fully bound by out_graph; nothing to do inside the captured region.
        return None

    def on_after_cuda_graph_warmup(self):
        return None

    XPUAttentionBackend.init_cuda_graph_state = init_cuda_graph_state
    XPUAttentionBackend.init_forward_metadata_out_graph = init_forward_metadata_out_graph
    XPUAttentionBackend.init_forward_metadata_in_graph = init_forward_metadata_in_graph
    if not hasattr(XPUAttentionBackend, "on_after_cuda_graph_warmup") or \
       XPUAttentionBackend.on_after_cuda_graph_warmup.__qualname__.startswith("AttentionBackend"):
        XPUAttentionBackend.on_after_cuda_graph_warmup = on_after_cuda_graph_warmup
    print("[xpu-cudagraph] XPUAttentionBackend decode graph hooks installed (no-spec, token-level static page_table)", flush=True)
