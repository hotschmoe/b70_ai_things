"""Narrow compressed-tensors W8A8 INT8 enablement for SGLang on XPU."""

import importlib
import inspect
import os
import sys
import textwrap
from dataclasses import fields


def _replace_method_source(method, replacements, namespace):
    source = textwrap.dedent(inspect.getsource(method))
    for old, new, expected in replacements:
        count = source.count(old)
        if count != expected:
            raise RuntimeError(
                f"upstream method changed while applying XPU MTP port: "
                f"expected {expected} occurrences of {old!r}, found {count}"
            )
        source = source.replace(old, new)
    exec(source, namespace)
    return namespace[method.__name__]


def _install_mtp_xpu() -> None:
    """Port current speculative allocation and sharing helpers to XPU."""
    import sglang.srt.mem_cache.memory_pool as memory_pool
    import sglang.srt.models.qwen3_5_mtp as qwen3_5_mtp
    import sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent as xpu_gdn
    import sglang.kernels.ops.mamba.mamba_state_scatter_triton as state_scatter

    mamba_pool = memory_pool.MambaPool
    mamba_pool.__init__ = _replace_method_source(
        mamba_pool.__init__,
        [("device=\"cuda\"", "device=device", 2)],
        dict(memory_pool.__dict__),
    )
    mamba_pool._allocate_deduplicated_conv_window = _replace_method_source(
        mamba_pool._allocate_deduplicated_conv_window,
        [
            (
                "device=\"cuda\"",
                'device=f"xpu:{torch.xpu.current_device()}"',
                1,
            )
        ],
        dict(memory_pool.__dict__),
    )

    mtp_cls = qwen3_5_mtp.Qwen3_5ForCausalLMMTP
    mtp_cls.set_embed_and_head = _replace_method_source(
        mtp_cls.set_embed_and_head,
        [("torch.cuda.", "torch.xpu.", 2)],
        dict(qwen3_5_mtp.__dict__),
    )

    xpu_gdn.fused_sigmoid_gating_delta_rule_update = _replace_method_source(
        xpu_gdn.fused_sigmoid_gating_delta_rule_update,
        [
            (
                "h0_indices=initial_state_indices,",
                "h0_indices=initial_state_indices,\n"
                "        stride_h0_source=(\n"
                "            initial_state_source.stride(0)\n"
                "            if initial_state_source is not None\n"
                "            else 0\n"
                "        ),",
                1,
            )
        ],
        dict(xpu_gdn.__dict__),
    )
    for module_name in (
        "sglang.srt.layers.attention.linear.kernels.gdn_triton",
        "sglang.srt.layers.attention.linear.kernels.kda_triton",
    ):
        module = importlib.import_module(module_name)
        module.fused_sigmoid_gating_delta_rule_update = (
            xpu_gdn.fused_sigmoid_gating_delta_rule_update
        )

    cuda_only_state_guard = (
        "    if not dst.is_cuda or not src.is_cuda:\n"
        "        raise ValueError(\n"
        '            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."\n'
        "        )\n"
    )
    state_scatter.fused_mamba_state_scatter_with_mask = _replace_method_source(
        state_scatter.fused_mamba_state_scatter_with_mask,
        [(cuda_only_state_guard, "", 1)],
        dict(state_scatter.__dict__),
    )
    cuda_only_conv_guard = (
        "    if not (dst.is_cuda and src.is_cuda and dst.device == src.device):\n"
        "        raise ValueError(\n"
        '            "fused_conv_window_scatter_with_mask requires dst and src to be CUDA "\n'
        '            f"tensors on the same device ({dst.device=}, {src.device=})."\n'
        "        )\n"
    )
    state_scatter.fused_conv_window_scatter_with_mask = _replace_method_source(
        state_scatter.fused_conv_window_scatter_with_mask,
        [
            (
                cuda_only_conv_guard,
                "    if dst.device != src.device:\n"
                "        raise ValueError(\n"
                '            f"dst and src must be on the same device: "\n'
                '            f"{dst.device=} {src.device=}"\n'
                "        )\n",
                1,
            )
        ],
        dict(state_scatter.__dict__),
    )
    if not hasattr(state_scatter, "_conv_multi_eligible"):
        raise RuntimeError("upstream XPU MTP commit helper no longer has multi eligibility")
    state_scatter._conv_multi_eligible = lambda pairs: False
    print(
        "[b70-xpu-mtp] enabled XPU speculative state, sharing, GDN, and safe commit",
        flush=True,
    )


def _install_breakable_graph() -> None:
    """Enable SGLang's existing segmented graph backend on XPU.

    The current backend already implements XPU segment capture, but its resolver
    conservatively rejects the breakable decode mode. Keep TP collectives eager
    between segments so oneCCL is never recorded into an XPUGraph.
    """
    import torch

    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )
    import sglang.srt.distributed.parallel_state as parallel_state
    import sglang.srt.model_executor.runner_backend.utils as backend_utils
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.server_args import ServerArgs

    reclaim_n = int(os.environ.get("B70_XPU_CG_RECLAIM", "0") or "0")
    if reclaim_n > 0:
        import torch.xpu.graphs as xpu_graphs

        base_xpu_graph = torch.xpu.XPUGraph
        original_replay = base_xpu_graph.replay
        replay_counts = {}
        reclaim_prints = [0]

        class ReclaimingXPUGraph(base_xpu_graph):
            def __new__(cls, *args, **kwargs):
                return base_xpu_graph.__new__(cls, keep_graph=True)

            def __init__(self, *args, **kwargs):
                super().__init__(keep_graph=True)

            def replay(self):
                graph_id = id(self)
                count = replay_counts.get(graph_id, 0) + 1
                replay_counts[graph_id] = count
                if count % reclaim_n == 0:
                    self.instantiate()
                    if reclaim_prints[0] < 8:
                        reclaim_prints[0] += 1
                        print(
                            "[b70-xpu-graph] executable re-instantiated "
                            f"graph={graph_id} replay={count}",
                            flush=True,
                        )
                return original_replay(self)

        torch.xpu.XPUGraph = ReclaimingXPUGraph
        xpu_graphs.XPUGraph = ReclaimingXPUGraph
        print(
            "[b70-xpu-graph] reclaim enabled: "
            f"re-instantiate every {reclaim_n} replays per graph",
            flush=True,
        )

    original_output_rows = BreakableCudaGraphBackend._output_rows
    original_alloc_full_buffer = BreakableCudaGraphBackend._alloc_full_buffer
    original_slice_output = BreakableCudaGraphBackend._slice_output
    original_copy_output = BreakableCudaGraphBackend._copy_output_to_buffer

    def output_rows(self, output, cap):
        if isinstance(output, LogitsProcessorOutput):
            if torch.is_tensor(output.next_token_logits):
                return output.next_token_logits.shape[0]
            rows = [
                output_rows(self, getattr(output, field.name), sys.maxsize)
                for field in fields(output)
                if getattr(output, field.name) is not None
            ]
            return min(rows) if rows else cap
        return original_output_rows(self, output, cap)

    def alloc_full_buffer(self, output, size):
        if isinstance(output, LogitsProcessorOutput):
            output_size = output_rows(self, output, sys.maxsize)
            return LogitsProcessorOutput(
                **{
                    field.name: alloc_full_buffer(
                        self, getattr(output, field.name), output_size
                    )
                    for field in fields(output)
                }
            )
        if output is None or torch.is_tensor(output) or isinstance(
            output, (tuple, list)
        ):
            return original_alloc_full_buffer(self, output, size)
        return output

    def slice_output(self, output, num_tokens):
        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                **{
                    field.name: slice_output(
                        self, getattr(output, field.name), num_tokens
                    )
                    for field in fields(output)
                }
            )
        if output is None or torch.is_tensor(output) or isinstance(
            output, (tuple, list)
        ):
            return original_slice_output(self, output, num_tokens)
        return output

    def copy_output(self, output, output_buffer, num_tokens):
        if isinstance(output, LogitsProcessorOutput) and isinstance(
            output_buffer, LogitsProcessorOutput
        ):
            for field in fields(output):
                source = getattr(output, field.name)
                destination = getattr(output_buffer, field.name)
                if source is None and destination is None:
                    continue
                if torch.is_tensor(source) and torch.is_tensor(destination):
                    destination[:num_tokens].copy_(source[:num_tokens])
                elif isinstance(source, LogitsProcessorOutput) and isinstance(
                    destination, LogitsProcessorOutput
                ):
                    copy_output(self, source, destination, num_tokens)
                else:
                    setattr(output_buffer, field.name, source)
            return
        return original_copy_output(self, output, output_buffer, num_tokens)

    BreakableCudaGraphBackend._output_rows = output_rows
    BreakableCudaGraphBackend._alloc_full_buffer = alloc_full_buffer
    BreakableCudaGraphBackend._slice_output = slice_output
    BreakableCudaGraphBackend._copy_output_to_buffer = copy_output

    original_handle_xpu_backends = ServerArgs._handle_xpu_backends

    def handle_xpu_backends(self):
        if (
            self.device == "xpu"
            and self.cuda_graph_config.decode.backend == Backend.BREAKABLE
        ):
            return
        return original_handle_xpu_backends(self)

    ServerArgs._handle_xpu_backends = handle_xpu_backends

    original_resolve_decode_backend = backend_utils.resolve_decode_backend

    def resolve_decode_backend(cuda_graph_runner):
        model_runner = cuda_graph_runner.model_runner
        config = backend_utils.get_exec().graph.cuda_graph_config
        backend_name = config.decode.backend if config is not None else Backend.FULL
        if model_runner.device == "xpu" and backend_name == Backend.BREAKABLE:
            return BreakableCudaGraphBackend(
                cuda_graph_runner,
                enable_memory_saver=model_runner.server_args.enable_memory_saver,
                debug_eager=backend_utils.get_exec().graph.debug_cuda_graph,
            )
        return original_resolve_decode_backend(cuda_graph_runner)

    backend_utils.resolve_decode_backend = resolve_decode_backend
    runner_module = sys.modules.get(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
    )
    if runner_module is not None:
        runner_module.resolve_decode_backend = resolve_decode_backend

    parallel_state.GroupCoordinator._all_reduce_in_place = eager_on_graph(True)(
        parallel_state.GroupCoordinator._all_reduce_in_place
    )
    parallel_state.GroupCoordinator.all_gather = eager_on_graph(True)(
        parallel_state.GroupCoordinator.all_gather
    )
    if os.environ.get("B70_XPU_MTP") == "1":
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

        original_xpu_metadata = XPUAttentionBackend.init_forward_metadata_out_graph

        def xpu_metadata_out_graph(self, forward_batch, in_capture=False):
            if forward_batch.spec_info is not None:
                return self.init_forward_metadata(forward_batch)
            return original_xpu_metadata(
                self, forward_batch, in_capture=in_capture
            )

        XPUAttentionBackend.init_forward_metadata_out_graph = xpu_metadata_out_graph
        XPUAttentionBackend.forward_extend = eager_on_graph(True)(
            XPUAttentionBackend.forward_extend
        )
        print(
            "[b70-xpu-graph] keeping speculative XPU full attention eager",
            flush=True,
        )
    print(
        "[b70-xpu-graph] enabled breakable decode with eager TP collectives",
        flush=True,
    )


def install() -> None:
    import torch

    import sglang.srt.layers.quantization.compressed_tensors.compressed_tensors as ct
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    config_cls = ct.CompressedTensorsConfig
    original_get_linear_scheme = config_cls.get_linear_scheme
    original_process = CompressedTensorsW8A8Int8.process_weights_after_loading
    use_native = os.environ.get("B70_XPU_W8A8_NATIVE") == "1"

    if use_native:
        importlib.import_module("vllm_xpu_kernels._xpu_C")
        expected_schemas = {
            "int8_gemm_w8a8": (
                "_xpu_C::int8_gemm_w8a8(Tensor A, Tensor A_scale, Tensor B, "
                "Tensor B_scale, ScalarType? out_dtype, Tensor? bias) -> Tensor"
            ),
            "per_token_quant_int8_xpu": (
                "_xpu_C::per_token_quant_int8_xpu(Tensor x) -> (Tensor, Tensor)"
            ),
        }
        for op, expected in expected_schemas.items():
            actual = str(
                torch._C._dispatch_find_schema_or_throw(
                    f"_xpu_C::{op}", ""
                ).schema()
            )
            if actual != expected:
                raise RuntimeError(
                    f"B70 native XPU W8A8 schema mismatch for {op}: {actual}"
                )

    def get_linear_scheme(self, *args, **kwargs):
        try:
            return original_get_linear_scheme(self, *args, **kwargs)
        except RuntimeError as error:
            expected = (
                "CompressedTensorsW8A8Int8 is not supported on XPU "
                "(no XPU kernel implementation)."
            )
            if str(error) != expected:
                raise

            # Re-enter only the already-selected W8A8 INT8 arm with the generic
            # capability gate disabled. Restore both globals before returning.
            original_is_xpu = ct._is_xpu
            original_check = config_cls._check_scheme_supported
            ct._is_xpu = False
            config_cls._check_scheme_supported = lambda self, capability, error=True: True
            try:
                scheme = original_get_linear_scheme(self, *args, **kwargs)
            finally:
                config_cls._check_scheme_supported = original_check
                ct._is_xpu = original_is_xpu

            if not isinstance(scheme, CompressedTensorsW8A8Int8):
                raise RuntimeError(
                    "B70 XPU W8A8 retry selected an unexpected quantization scheme: "
                    f"{type(scheme).__name__}"
                )
            return scheme

    def process_weights_after_loading(self, layer) -> None:
        original_process(self, layer)
        weight = layer.weight.data
        if use_native:
            if self.is_static_input_scheme or not self.input_symmetric:
                raise RuntimeError(
                    "B70 native XPU W8A8 requires dynamic symmetric activations"
                )
            weight_nk = weight.t().contiguous()
            layer.b70_weight_nk = weight_nk
            layer.b70_weight_nt = weight_nk.t()
            if layer.b70_weight_nt.stride(0) != 1:
                raise RuntimeError("B70 native XPU W8A8 weight is not NT-strided")
            layer.b70_weight_scale = (
                layer.weight_scale.data.reshape(-1).to(torch.float32).contiguous()
            )
        else:
            layer.b70_weight_t = weight.contiguous()
            layer.b70_weight_scale = layer.weight_scale.data.reshape(1, -1).to(
                torch.float32
            )
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=weight.dtype, device=weight.device),
            requires_grad=False,
        )

    def apply_generic_weights(self, layer, x, bias=None):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        absmax = x_2d.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5)
        x_scale = absmax * (1.0 / 127.0)
        x_int8 = torch.round(x_2d / x_scale).clamp_(-127, 127).to(torch.int8)
        accumulator = torch._int_mm(x_int8, layer.b70_weight_t)
        output = (
            accumulator.to(torch.float32)
            * x_scale.to(torch.float32)
            * layer.b70_weight_scale
        ).to(x.dtype)
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], -1)

    def apply_native_weights(self, layer, x, bias=None):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_int8, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x_2d)
        output = torch.ops._xpu_C.int8_gemm_w8a8(
            x_int8,
            x_scale,
            layer.b70_weight_nt,
            layer.b70_weight_scale,
            x.dtype,
            bias,
        )
        return output.reshape(*original_shape[:-1], -1)

    config_cls.get_linear_scheme = get_linear_scheme
    CompressedTensorsW8A8Int8.process_weights_after_loading = (
        process_weights_after_loading
    )
    CompressedTensorsW8A8Int8.apply_weights = (
        apply_native_weights if use_native else apply_generic_weights
    )
    route = "native per-token quant plus oneDNN GEMM" if use_native else "torch._int_mm"
    print(
        f"[b70-xpu-w8a8] installed compressed-tensors W8A8 via {route}",
        flush=True,
    )
    if os.environ.get("B70_XPU_MTP") == "1":
        _install_mtp_xpu()
    if os.environ.get("B70_XPU_BREAKABLE_GRAPH") == "1":
        _install_breakable_graph()


if os.environ.get("B70_XPU_W8A8") == "1":
    install()
