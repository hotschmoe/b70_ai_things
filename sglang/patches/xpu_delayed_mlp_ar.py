"""Contract-only delayed dense-MLP all-reduce route for Qwen3.5 on XPU.

The Qwen3.5 decoder already supports leaving a row-parallel MLP output local,
marking it, and letting the next layer's ``LayerCommunicator.prepare_attn``
finish the all-reduce plus input RMSNorm.  Upstream only enables that contract
for its fused CUDA/ROCm implementations.  This shim enables the same contract
for a narrow XPU TP=2 shape window while deliberately retaining prepare_attn's
generic all-reduce-plus-norm fallback.

This is experimental and default-off.  Set B70_XPU_DELAY_MLP_AR=1 to install.
"""

import hashlib
import inspect
import os
from functools import wraps


_ENV = "B70_XPU_DELAY_MLP_AR"
_MARKER = "_sglang_needs_allreduce_fusion"
_PENDING = "_b70_delayed_mlp_ar_pending"
_TAG = "_b70_delayed_mlp_ar_qwen35_dense"
_INSTALLED = False
_COUNTERS = {"eligible": 0, "consumed": 0, "generic": 0}
_FIRST_REPORT = 63
_REPORT_EVERY = 4096
_SHOULD_FUSE_SHA256 = "878c3a0cd695da2648107a6c3f010659290ccbc1d84e59231a309908f318b3e8"


def route_counters():
    """Return a copy suitable for low-cost runtime/debug inspection."""
    return dict(_COUNTERS)


def _batch_rows(forward_batch):
    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None or input_ids.ndim == 0:
        return 0
    return int(input_ids.shape[0])


def _request_batch_size(forward_batch):
    return int(getattr(forward_batch, "batch_size", 0) or 0)


def _tag_qwen35_dense_communicators(qwen35_module):
    def wrap_init(cls):
        original = cls.__init__
        if getattr(original, "_b70_delayed_mlp_ar_wrapped", False):
            return

        @wraps(original)
        def tagged_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            config = kwargs.get("config", args[0] if args else None)
            if getattr(config, "model_type", None) != "qwen3_5_text":
                return
            layer_id = kwargs.get("layer_id", args[1] if len(args) > 1 else None)
            communicator = self.layer_communicator
            setattr(communicator, _TAG, True)
            communicator._b70_delayed_mlp_ar_layer_id = int(layer_id)

        tagged_init._b70_delayed_mlp_ar_wrapped = True
        cls.__init__ = tagged_init

    wrap_init(qwen35_module.Qwen3_5LinearDecoderLayer)
    wrap_init(qwen35_module.Qwen3_5AttentionDecoderLayer)


def _upstream_rejection_reason(communicator, forward_batch, comm_module):
    """Mirror the rejection clauses ahead of upstream's backend capability gate."""
    if comm_module.is_enable_moe_cp_allgather():
        return "moe_cp_allgather"
    if (
        comm_module.is_dp_attention_enabled()
        and communicator._speculative_algo is not None
        and communicator._speculative_algo.is_eagle()
    ):
        return "dp_attention_eagle"
    if comm_module.get_attn_tp_context().input_scattered:
        return "input_scattered"
    if communicator.layer_scatter_modes.mlp_mode == comm_module.ScatterMode.SCATTERED:
        return "mlp_scattered"
    if communicator.is_last_layer:
        return "last_layer"
    if communicator._context.tp_size <= 1:
        return "tp_one"
    return None


def _is_eager_xpu_route(forward_batch, comm_module, qwen35_module):
    import torch

    parallel = comm_module.get_parallel()
    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None or input_ids.device.type != "xpu":
        return False
    if not torch.xpu.is_available():
        return False
    if parallel.tp_size != 2 or parallel.pp_size != 1:
        return False
    if parallel.moe_tp_size != parallel.tp_size or parallel.moe_ep_size != 1:
        return False
    if parallel.attn_dp_size != 1 or parallel.moe_dp_size != 1:
        return False
    if comm_module.is_dp_attention_enabled():
        return False
    server_args = comm_module.get_global_server_args()
    if server_args.enable_quant_communications:
        return False
    if not server_args.disable_cuda_graph:
        return False
    if qwen35_module.get_is_capture_mode():
        return False
    try:
        if torch.xpu.is_current_stream_capturing():
            return False
    except Exception:
        # The contract requires a positive eager-mode determination.
        return False
    return True


def _report_if_due(rank):
    consumed = _COUNTERS["consumed"]
    if consumed == _FIRST_REPORT or (
        consumed > _FIRST_REPORT and consumed % _REPORT_EVERY == 0
    ):
        print(
            f"[c3b-delayed-mlp] ROUTES rank={rank} "
            f"eligible={_COUNTERS['eligible']} consumed={consumed} "
            f"generic={_COUNTERS['generic']}",
            flush=True,
        )


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get(_ENV) != "1":
        return False

    import torch

    from sglang.srt.layers import communicator as comm_module
    from sglang.srt.models import qwen3_5 as qwen35_module
    from sglang.srt.utils import is_xpu

    if not is_xpu():
        return False

    layer_communicator = comm_module.LayerCommunicator
    original_should_fuse = layer_communicator.should_fuse_mlp_allreduce_with_next_layer
    original_prepare_attn = layer_communicator.prepare_attn
    should_fuse_source_sha = hashlib.sha256(
        inspect.getsource(original_should_fuse).encode()
    ).hexdigest()
    if should_fuse_source_sha != _SHOULD_FUSE_SHA256:
        raise RuntimeError(
            "C3b refusing an unknown LayerCommunicator.should_fuse contract: "
            f"sha256={should_fuse_source_sha}"
        )

    @wraps(original_should_fuse)
    def should_delay_mlp_allreduce(self, forward_batch):
        upstream_result = original_should_fuse(self, forward_batch)
        if upstream_result or not getattr(self, _TAG, False):
            return upstream_result

        # Do not override any rejection that upstream applies before checking
        # whether a fused backend is available.
        if _upstream_rejection_reason(self, forward_batch, comm_module) is not None:
            return False
        if not _is_eager_xpu_route(forward_batch, comm_module, qwen35_module):
            return False

        rows = _batch_rows(forward_batch)
        request_batch = _request_batch_size(forward_batch)
        if self._context.tp_size != 2:
            return False
        if not (1 <= rows <= 128 and 1 <= request_batch <= 128):
            return False

        assert not hasattr(forward_batch, _PENDING), (
            "C3b found an unconsumed delayed-MLP marker before producing the next one"
        )
        producer_layer = self._b70_delayed_mlp_ar_layer_id
        setattr(
            forward_batch,
            _PENDING,
            (producer_layer, rows, request_batch),
        )
        _COUNTERS["eligible"] += 1
        return True

    @wraps(original_prepare_attn)
    def prepare_attn_with_contract_checks(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format="",
        post_residual_addition=None,
    ):
        pending = getattr(forward_batch, _PENDING, None)
        if pending is None:
            return original_prepare_attn(
                self,
                hidden_states,
                residual,
                forward_batch,
                quant_format=quant_format,
                post_residual_addition=post_residual_addition,
            )

        producer_layer, expected_rows, expected_batch = pending
        consumer_layer = getattr(self, "_b70_delayed_mlp_ar_layer_id", None)
        assert getattr(self, _TAG, False), "C3b marker reached an untagged communicator"
        assert consumer_layer == producer_layer + 1, (
            f"C3b marker crossed non-adjacent layers: {producer_layer} -> {consumer_layer}"
        )
        assert getattr(hidden_states, _MARKER, None) is True, (
            "C3b pending route reached prepare_attn without the upstream tensor marker"
        )
        assert self._context.tp_size == 2
        assert not comm_module.get_attn_tp_context().input_scattered
        assert 1 <= expected_rows <= 128 and 1 <= expected_batch <= 128
        assert _batch_rows(forward_batch) == expected_rows
        assert _request_batch_size(forward_batch) == expected_batch
        assert hidden_states.device.type == "xpu"
        assert residual is not None
        assert post_residual_addition is None
        assert hidden_states.dtype == torch.bfloat16
        assert residual.dtype == torch.bfloat16
        assert hidden_states.ndim == 2 and hidden_states.shape == (expected_rows, 5120)
        assert hidden_states.shape == residual.shape
        assert hidden_states.device == residual.device
        assert hidden_states.is_contiguous() and residual.is_contiguous()

        # This route is intentionally contract-only: if either fused policy
        # becomes active, fail closed instead of calling
        # input_layernorm.forward_with_allreduce_fusion.
        assert not comm_module.apply_aiter_all_reduce_fusion(hidden_states)
        assert not comm_module.apply_flashinfer_allreduce_fusion(expected_rows)

        result = original_prepare_attn(
            self,
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )
        output, output_residual = result
        assert output.shape == hidden_states.shape
        assert output_residual is not None and output_residual.shape == residual.shape

        # Do not alter the returned tensors or marker. The upstream
        # prepare_attn implementation remains the sole AR+norm consumer; this
        # wrapper is observational apart from its own ForwardBatch bookkeeping.
        delattr(forward_batch, _PENDING)
        _COUNTERS["consumed"] += 1
        _COUNTERS["generic"] += 1
        _report_if_due(self._context.tp_rank)
        return output, output_residual

    _tag_qwen35_dense_communicators(qwen35_module)
    layer_communicator.should_fuse_mlp_allreduce_with_next_layer = (
        should_delay_mlp_allreduce
    )
    layer_communicator.prepare_attn = prepare_attn_with_contract_checks
    _INSTALLED = True
    print(
        "[c3b-delayed-mlp] installed contract-only route "
        "(Qwen3.5 dense, TP=2, rows/batch=1..128)",
        flush=True,
    )
    return True
