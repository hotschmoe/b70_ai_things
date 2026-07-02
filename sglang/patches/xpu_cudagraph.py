"""XPU cudagraph hooks for sglang on Intel B70.

Spec TARGET_VERIFY capture is supported for topk<=1. Spec DRAFT capture stays
separately gated in section 3; the intended MTP serve shape is target-verify
captured while draft remains eager via this fork's per-phase cuda_graph_config
decode backend gate in eagle_worker_v2._capture_cuda_graphs.
"""
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
# already capture-safe (static MambaPool + in-place writes). MTP target-verify graph is handled for topk<=1.
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
        draft = int(getattr(self, "speculative_num_draft_tokens", 0) or 0)
        self._b70_cg = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=dev),
            "cu_seqlens_q": torch.arange(0, max_bs + 1, dtype=torch.int32, device=dev),
            "cu_seqlens_k": torch.zeros(max_bs + 1, dtype=torch.int32, device=dev),
            # TOKEN-level page_table (matches the eager XPU path: req_to_token[req, :len]), fixed width.
            "page_table": torch.zeros(max_bs, self.max_context_len, dtype=torch.int32, device=dev),
        }
        self._b70_target_verify_cg = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=dev),
            "cu_seqlens_q": (
                torch.arange(0, max_bs * draft + 1, step=draft, dtype=torch.int32, device=dev)
                if draft > 0
                else torch.zeros(max_bs + 1, dtype=torch.int32, device=dev)
            ),
            "cu_seqlens_k": torch.zeros(max_bs + 1, dtype=torch.int32, device=dev),
            # TOKEN-level page_table for target-verify: req_to_token[req, :max_seq_len_k].
            "page_table": torch.zeros(max_bs, self.max_context_len, dtype=torch.int32, device=dev),
        }
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        if dbg:
            if draft > 0:
                print(f"[xpu-cudagraph] init_cuda_graph_state max_bs={max_bs} max_ctx={self.max_context_len} draft={draft}", flush=True)
            else:
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

    def _b70_bind_target_verify(self, bs):
        b = self._b70_target_verify_cg
        m = FlashAttentionMetadata()
        m.cache_seqlens_int32 = b["cache_seqlens"][:bs]
        m.cu_seqlens_q = b["cu_seqlens_q"][: bs + 1]
        m.cu_seqlens_k = b["cu_seqlens_k"][: bs + 1]
        m.page_table = b["page_table"][:bs]
        m.max_seq_len_q = int(getattr(self, "speculative_num_draft_tokens", 0) or 0)
        self.target_verify_metadata[bs] = m
        return m

    def _b70_fill_target_verify(self, bs, req_pool_indices, seq_lens):
        m = self.target_verify_metadata[bs]
        draft = int(getattr(self, "speculative_num_draft_tokens", 0) or 0)
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        max_len = int(seq_lens.max().item()) + draft
        m.max_seq_len_k = max_len
        m.cache_seqlens_int32.copy_((seq_lens + draft).to(torch.int32))
        # cu_seqlens_k = [0, cumsum(seq_lens + draft)]; index 0 stays 0 (static zero buffer).
        torch.cumsum(m.cache_seqlens_int32, dim=0, dtype=torch.int32, out=m.cu_seqlens_k[1:])
        # page_table[:, :max_len] <- req_to_token[req, :max_len] (in-place); tail untouched (cache_seqlens bounds).
        m.page_table[:, :max_len].copy_(self.req_to_token[req_pool_indices, :max_len])

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        # Called by the hybrid wrapper for the full-attn sub-backend: at capture (in_capture=True) and
        # before each replay (in_capture=False, line ~950 of decode_cuda_graph_runner).
        fm = forward_batch.forward_mode
        if (
            fm.is_target_verify()
            and getattr(self, "topk", 1) <= 1
            and forward_batch.spec_info is not None
            and int(getattr(self, "speculative_num_draft_tokens", 0) or 0) > 0
        ):
            bs = forward_batch.batch_size
            if in_capture or bs not in self.target_verify_metadata:
                _b70_bind_target_verify(self, bs)
            _b70_fill_target_verify(self, bs, forward_batch.req_pool_indices, forward_batch.seq_lens)
            self.forward_metadata = self.target_verify_metadata[bs]
            return None
        if not (fm.is_decode_or_idle() and forward_batch.spec_info is None):
            # topk>1 spec, draft-extend, and other modes stay on the eager metadata path.
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

    # ---- 3. (graph+MTP STACK) EAGLE/NEXTN spec-decode DRAFT cuda-graph handling on xpu ----
    # When B70_XPU_MTP=1 and graphs are globally ON, EagleDraftWorker._capture_cuda_graphs runs; its
    # Device2Draft/Device2Extend CudaGraphRunner dicts are keyed {npu,cuda,musa} -> KeyError on xpu
    # (num_steps>1). Two modes:
    #   B70_XPU_DRAFT_GRAPH=1  -> add "xpu" to the dicts so the DRAFT forward captures too (the 2026-06-28
    #                             attempt HUNG on replay -- keep opt-in until root-caused).
    #   default                -> patch _capture_cuda_graphs to SKIP cleanly (runners=None) so the draft
    #                             chain stays EAGER while the main DecodeCudaGraphRunner still captures the
    #                             TARGET_VERIFY forward (metadata hooks in section 2). This is the intended
    #                             "target captured, draft eager" Step-2 config.
    # ---- 4. BREAKABLE backend: run TP collectives EAGER between captured segments ----
    # With --cuda-graph-backend-decode breakable, attention/mamba already run eager between segment
    # graphs (eager_on_graph markers). oneCCL collectives RECORDED into a SYCL graph deadlock at replay
    # (host-staged half never re-executes -- RUN-4 watchdog stack, JOURNAL 2026-07-02), so mark the two
    # TP collectives the forward uses (row-parallel all_reduce, logits all_gather) as graph breaks too:
    # they end the current segment, run eagerly at capture AND at every replay (replay_fn re-invokes on
    # the weak-ref'd static tensors). No-op under the 'full' backend (capture ctx var is None).
    if os.environ.get("B70_XPU_EAGER_COLLECTIVES", "1") == "1":
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph
            from sglang.srt.distributed.device_communicators.xpu_communicator import XpuCommunicator as _XC
            import sglang.srt.distributed.parallel_state as _ps
            _XC.all_reduce = eager_on_graph(True)(_XC.all_reduce)
            _ps.GroupCoordinator.all_gather = eager_on_graph(True)(_ps.GroupCoordinator.all_gather)
            print("[xpu-cudagraph] TP collectives wrapped eager_on_graph (breakable: all_reduce/all_gather run between segments)", flush=True)
        except Exception as e:
            print(f"[xpu-cudagraph] eager_on_graph collective wrap FAILED: {e}", flush=True)

    # ---- 4b. BREAKABLE: run the per-layer ATTENTION/GDN entry eager too (B70_XPU_EAGER_ATTN=1) ----
    # The fork only breaks attention on the extend/tc_piecewise path; under breakable-DECODE the attention
    # and mamba kernels are captured into segments. Wrapping the OUTER hybrid backend .forward (what
    # RadixAttention delegates to for decode/verify) makes every attention + GDN layer a graph break, so
    # segments contain only the launch-heavy linear/norm chains (the part capture actually helps).
    if os.environ.get("B70_XPU_EAGER_ATTN", "0") == "1":
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
            HybridLinearAttnBackend.forward = eager_on_graph(True)(HybridLinearAttnBackend.forward)
            print("[xpu-cudagraph] HybridLinearAttnBackend.forward wrapped eager_on_graph (attn+GDN eager between segments)", flush=True)
        except Exception as e:
            print(f"[xpu-cudagraph] eager_on_graph attn wrap FAILED: {e}", flush=True)

    # ---- 4d. weak_ref_tensor XPU fallback (THE run-7/9/13 root cause) ----
    # sglang.srt.compilation.weak_ref_tensor hard-raises NotImplementedError at import on XPU (only
    # CUDA/NPU have the kernel; sgl-kernel-xpu PR #251, merged 2026-06-29, adds the real from_blob
    # implementation -- newer than our image). The breakable backend lazily imports it at the FIRST
    # eager break; the raise then unwinds into BreakableCUDAGraphCapture.__exit__, which double-ends
    # the already-ended segment -> capture_end on a dead sycl graph -> the masking SEGFAULT.
    # Fallback: IDENTITY refs (strong). Only cost: segment-pool memory cannot be aliased across breaks.
    try:
        import sys as _sys
        import types as _types
        if "sglang.srt.compilation.weak_ref_tensor" not in _sys.modules:
            _m = _types.ModuleType("sglang.srt.compilation.weak_ref_tensor")

            def _wrt(t):
                return t

            def _wrts(tensors):
                if isinstance(tensors, torch.Tensor):
                    return _wrt(tensors)
                if isinstance(tensors, list):
                    return [_wrt(t) for t in tensors]
                if isinstance(tensors, tuple):
                    return tuple(_wrt(t) for t in tensors)
                raise ValueError("Invalid type for tensors")

            _m.weak_ref_tensor = _wrt
            _m.weak_ref_tensors = _wrts
            _sys.modules["sglang.srt.compilation.weak_ref_tensor"] = _m
            print("[xpu-cudagraph] weak_ref_tensor XPU fallback installed (identity refs)", flush=True)
    except Exception as e:
        print(f"[xpu-cudagraph] weak_ref_tensor fallback FAILED: {e}", flush=True)

    # ---- 4e. BCG output handling for LogitsProcessorOutput (spec/TARGET_VERIFY forward output) ----
    # BreakableCudaGraphBackend._slice_output/_copy_output_to_buffer only handle Tensor/tuple/list/
    # PPProxyTensors; the MTP verify forward returns a LogitsProcessorOutput dataclass -> TypeError
    # (run 15). Recurse over its dataclass fields; pass non-tensor scalars through unchanged.
    try:
        import dataclasses as _dc
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput as _LPO
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend as _BCGB,
        )
        _oslice = _BCGB._slice_output
        _ocopy = _BCGB._copy_output_to_buffer

        def _slice2(self, output, num_tokens):
            if isinstance(output, _LPO):
                return _LPO(**{
                    f.name: _slice2(self, getattr(output, f.name), num_tokens)
                    for f in _dc.fields(output)
                })
            if output is None or torch.is_tensor(output) or isinstance(output, (tuple, list)):
                return _oslice(self, output, num_tokens)
            return output  # scalars/None-likes pass through

        def _copy2(self, output, output_buffer, num_tokens):
            if isinstance(output, _LPO) and isinstance(output_buffer, _LPO):
                for f in _dc.fields(output):
                    _copy2(self, getattr(output, f.name), getattr(output_buffer, f.name), num_tokens)
                return
            if output is None and not torch.is_tensor(output_buffer) and output_buffer is not None:
                return
            if not torch.is_tensor(output) and not isinstance(output, (tuple, list)) and output is not None:
                return  # non-tensor scalar field: nothing to copy
            return _ocopy(self, output, output_buffer, num_tokens)

        _BCGB._slice_output = _slice2
        _BCGB._copy_output_to_buffer = _copy2

        # BCG passes shape_key.size (= BS for the decode runner) as the slice length, but SPEC
        # forwards produce bs * num_tokens_per_bs (=11 for NEXTN draft=11) output rows -> the stored
        # verify logits get sliced 11x too small ("shape [1, 11] invalid for input of size 2", run 18).
        # Stash the runner's num_tokens_per_bs on the backend and scale the slice/copy lengths.
        _oinit = _BCGB.__init__

        def _init2(self, cuda_graph_runner, **kw):
            _oinit(self, cuda_graph_runner, **kw)
            self._b70_tpb = int(getattr(cuda_graph_runner, "num_tokens_per_bs", 1) or 1)
            if self._b70_tpb > 1:
                print(f"[xpu-cudagraph] BCG spec token scaling: num_tokens_per_bs={self._b70_tpb}", flush=True)

        _BCGB.__init__ = _init2
        _slice_base = _BCGB._slice_output
        _copy_base = _BCGB._copy_output_to_buffer

        def _slice3(self, output, num_tokens):
            return _slice_base(self, output, num_tokens * getattr(self, "_b70_tpb", 1))

        def _copy3(self, output, output_buffer, num_tokens):
            return _copy_base(self, output, output_buffer, num_tokens * getattr(self, "_b70_tpb", 1))

        _BCGB._slice_output = _slice3
        _BCGB._copy_output_to_buffer = _copy3
        print("[xpu-cudagraph] BCG LogitsProcessorOutput slice/copy support installed", flush=True)
    except Exception as e:
        print(f"[xpu-cudagraph] BCG LPO output patch FAILED: {e}", flush=True)

    # ---- 4c. BCG segment-lifecycle trace (B70_XPU_BCG_TRACE=1): find which segment begin/end dies ----
    if os.environ.get("B70_XPU_BCG_TRACE") == "1":
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                breakable_cuda_graph as _bcg,
            )
            _obeg = _bcg.BreakableCUDAGraphCapture._begin_new_segment
            _oend = _bcg.BreakableCUDAGraphCapture._end_current_segment

            def _tbeg(self):
                n = len(self.cuda_graph._segments)
                try:
                    _obeg(self)
                    print(f"[bcg-trace] begin seg {n} OK", flush=True)
                except Exception as e:
                    print(f"[bcg-trace] begin seg {n} RAISED {type(e).__name__}: {e}", flush=True)
                    raise

            def _tend(self):
                n = len(self.cuda_graph._segments)
                print(f"[bcg-trace] end seg {n-1} ...", flush=True)
                _oend(self)
                print(f"[bcg-trace] end seg {n-1} OK", flush=True)

            _bcg.BreakableCUDAGraphCapture._begin_new_segment = _tbeg
            _bcg.BreakableCUDAGraphCapture._end_current_segment = _tend

            # The observed SEGFAULT is __exit__ double-ending an already-ended segment while an
            # exception from an eager-break fn unwinds -- capture_end on a dead sycl graph = null
            # deref, MASKING the real error. On the exception path: print it and skip cleanup.
            _oexit = _bcg.BreakableCUDAGraphCapture.__exit__

            def _texit(self, et, ev, tb):
                if et is not None:
                    import traceback
                    print(f"[bcg-trace] __exit__ unwinding {et.__name__}: {ev}", flush=True)
                    traceback.print_tb(tb)
                    return False  # skip the fatal double capture_end; let the real error surface
                return _oexit(self, et, ev, tb)

            _bcg.BreakableCUDAGraphCapture.__exit__ = _texit
            print("[xpu-cudagraph] BCG segment trace installed", flush=True)
        except Exception as e:
            print(f"[xpu-cudagraph] BCG trace install FAILED: {e}", flush=True)

    if os.environ.get("B70_XPU_MTP") == "1":
        try:
            import inspect as _insp, textwrap as _tw
            import sglang.srt.speculative.eagle_worker_v2 as _ew
            if os.environ.get("B70_XPU_DRAFT_GRAPH") == "1":
                _src = _tw.dedent(_insp.getsource(_ew.EagleDraftWorker._capture_cuda_graphs))
                _d = '"cuda": EAGLEDraftCudaGraphRunner,'
                _e = '"cuda": EAGLEDraftExtendCudaGraphRunner,'
                if _d in _src and _e in _src:
                    _src = _src.replace(_d, _d + '\n            "xpu": EAGLEDraftCudaGraphRunner,')
                    _src = _src.replace(_e, _e + '\n            "xpu": EAGLEDraftExtendCudaGraphRunner,')
                    _ns = dict(_ew.__dict__)
                    exec(_src, _ns)
                    _ew.EagleDraftWorker._capture_cuda_graphs = _ns["_capture_cuda_graphs"]
                    print("[xpu-cudagraph] EAGLE draft cuda-graph gate: added 'xpu' (graph+MTP stack)", flush=True)
                else:
                    print("[xpu-cudagraph] WARN: EAGLE draft gate needles not found (upstream changed) -- draft stays eager", flush=True)
            else:
                def _skip_draft_capture(self):
                    self.cuda_graph_runner = None
                    self.cuda_graph_runner_for_draft_extend = None
                    print("[xpu-cudagraph] draft cuda-graphs SKIPPED (B70_XPU_DRAFT_GRAPH!=1) -- draft eager, target-verify captured", flush=True)
                _ew.EagleDraftWorker._capture_cuda_graphs = _skip_draft_capture
        except Exception as e:
            print(f"[xpu-cudagraph] EAGLE draft gate patch FAILED: {e}", flush=True)
