"""Restore Steve's June XPU INT8 linear registration in the later S2B image.

The pinned August image contains the native operators, but its vLLM snapshot
no longer lists an XPU candidate in ``_POSSIBLE_INT8_KERNELS``. This adapter is
limited to the registry class from Steve's first surviving June source
checkpoint (e190923b3). It intentionally carries none of the Ornith ABI or PP
patches.
"""

import os
import sys
import inspect

import torch
import torch.nn.functional as F


def _patch_shared_expert_abi() -> None:
    """Bridge the partial shared-expert runner merge in the August snapshot."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkW8A8Int8MoEMethod,
    )

    original_init = MoERunner.__init__
    if "shared_expert_gate" not in inspect.signature(original_init).parameters:

        def compatible_init(self, *args, shared_expert_gate=None, **kwargs):
            if shared_expert_gate is not None:
                raise RuntimeError(
                    "S2B MoERunner cannot consume a non-None shared_expert_gate"
                )
            return original_init(self, *args, **kwargs)

        MoERunner.__init__ = compatible_init

    for method_class in (QuarkW8A8Int8MoEMethod, UnquantizedFusedMoEMethod):
        original_apply = method_class.apply
        parameter = inspect.signature(original_apply).parameters.get("shared_experts")
        if parameter is None or parameter.default is not inspect.Parameter.empty:
            continue

        def compatible_apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts_input=None,
            shared_experts=None,
            _original=original_apply,
        ):
            return _original(
                self,
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        method_class.apply = compatible_apply

    print(
        "[qwen36-s2b] bridged August shared-expert runner ABI",
        file=sys.stderr,
        flush=True,
    )


def _patch_quark_int8_moe_native_route() -> None:
    """Restore the June XPU grouped-W8A8 route for Quark routed experts.

    The pinned S2B image contains the XPU INT8 MoE oracle and experts class,
    but its Quark method still unconditionally calls generic Triton
    ``fused_experts``. Mounting the June kernel package therefore proves only
    operator availability unless the Quark dispatcher is repaired as well.

    Keep the image's RoutedExperts/SharedExperts ABI and use its native INT8
    oracle, while restoring June's weight transpose and scale normalization
    before constructing the XPU modular kernel.
    """
    from vllm.model_executor.layers.fused_moe.oracle.int8 import (
        Int8MoeBackend,
        make_int8_moe_kernel,
        make_int8_moe_quant_config,
        select_int8_moe_backend,
    )
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkW8A8Int8MoEMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )
    from vllm.model_executor.utils import replace_parameter

    original_init = QuarkW8A8Int8MoEMethod.__init__
    original_process = QuarkW8A8Int8MoEMethod.process_weights_after_loading
    original_apply = QuarkW8A8Int8MoEMethod.apply

    def native_init(self, weight_config, input_config, moe):
        original_init(self, weight_config, input_config, moe)
        activation_key = (
            None if self.static_input_scales else kInt8DynamicTokenSym
        )
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=activation_key,
        )
        self.moe_kernel = None

    def native_process(self, layer):
        if getattr(layer, "_qwen36_native_int8_moe_ready", False):
            return
        original_process(self, layer)
        if self.int8_backend != Int8MoeBackend.XPU:
            return

        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            prepare_int8_moe_layer_for_xpu,
        )

        w13, w2, w13_scale, w2_scale = prepare_int8_moe_layer_for_xpu(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        self.moe_quant_config = make_int8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=not self.static_input_scales,
        )
        self.moe_kernel = make_int8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        layer._qwen36_native_int8_moe_ready = True

    def native_apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        if self.int8_backend == Int8MoeBackend.XPU:
            assert self.moe_kernel is not None
            return self.moe_kernel.apply(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=(
                    layer.apply_router_weight_on_input
                ),
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
        return original_apply(
            self,
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    native_init._qwen36_native_int8_moe_route = True
    native_process._qwen36_native_int8_moe_route = True
    native_apply._qwen36_native_int8_moe_route = True
    QuarkW8A8Int8MoEMethod.__init__ = native_init
    QuarkW8A8Int8MoEMethod.process_weights_after_loading = native_process
    QuarkW8A8Int8MoEMethod.apply = native_apply
    print(
        "[qwen36-s2b] restored June native XPU INT8 MoE route",
        file=sys.stderr,
        flush=True,
    )


def _patch_custom_allreduce_clone_contract() -> None:
    """Restore the June inner-clone custom-op contract under a local op name.

    The August source still honors a graph-side clone in XpuCommunicator's
    alternate path, but the accepted outer custom-op route bypasses that path.
    August removed the one active inner clone from parallel_state.all_reduce
    while leaving its environment setting inert. Re-registering the existing
    operator is not supported by torch.library, so install an equivalent
    attributed op.

    XPU uses GroupCoordinator's outer custom-op route when
    VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1. A custom-op implementation executes
    with torch.compiler.is_compiling() false, so patching only
    XpuCommunicator.all_reduce would never select this replacement. Route the
    GroupCoordinator call itself through the local op; retain the communicator
    patch for the alternate non-outer-custom-op compile route.
    """
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators.xpu_communicator import (
        XpuCommunicator,
    )
    from vllm.utils.torch_utils import direct_register_custom_op

    def s2b_all_reduce_clone(
        tensor: torch.Tensor, group_name: str
    ) -> torch.Tensor:
        group_ref = parallel_state._groups.get(group_name)
        group = group_ref() if group_ref is not None else None
        if group is None:
            raise ValueError(f"Group {group_name} is not found or was destroyed")
        if os.environ.get("VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT", "0") == "1":
            tensor = tensor.clone()
        return group._all_reduce_out_place(tensor)

    def s2b_all_reduce_clone_fake(
        tensor: torch.Tensor, group_name: str
    ) -> torch.Tensor:
        del group_name
        return torch.empty_like(tensor)

    direct_register_custom_op(
        op_name="s2b_all_reduce_clone",
        op_func=s2b_all_reduce_clone,
        fake_impl=s2b_all_reduce_clone_fake,
    )

    original_group_all_reduce = parallel_state.GroupCoordinator.all_reduce

    def compatible_group_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if (
            self.use_custom_op_call
            and os.environ.get("VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP", "0") == "1"
        ):
            return torch.ops.vllm.s2b_all_reduce_clone(
                input_, group_name=self.unique_name
            )
        return original_group_all_reduce(self, input_)

    parallel_state.GroupCoordinator.all_reduce = compatible_group_all_reduce

    original_all_reduce = XpuCommunicator.all_reduce

    def compatible_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if (
            torch.compiler.is_compiling()
            and os.environ.get("VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP", "0") == "1"
        ):
            if (
                os.environ.get(
                    "VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT", "0"
                )
                == "1"
            ):
                input_ = input_.clone()
            return torch.ops.vllm.s2b_all_reduce_clone(
                input_, group_name=self.unique_name
            )
        return original_all_reduce(self, input_)

    XpuCommunicator.all_reduce = compatible_all_reduce
    print(
        "[qwen36-s2b] restored June clone-safe custom all-reduce contract",
        file=sys.stderr,
        flush=True,
    )


def _patch_june_piecewise_capture_contract() -> None:
    """Keep June's general PIECEWISE graphs when prefill replay is disabled.

    June used VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY only at runtime
    dispatch: non-uniform prefill ran eagerly, while the relaxed general
    PIECEWISE descriptors were still captured for ordinary decode. The later
    August snapshot also filters every non-uniform descriptor during capture.
    With June's dispatcher keying, that removes the graphs decode needs.

    Temporarily hide only the prefill setting while the August capture filter
    runs. Its independent speculative-decode and decode filters remain active.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_filter = GPUModelRunner._xpu_filter_cudagraph_capture_descs
    setting = "VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY"

    def june_compatible_filter(self, capture_descs):
        value = os.environ.pop(setting, None)
        try:
            return original_filter(self, capture_descs)
        finally:
            if value is not None:
                os.environ[setting] = value

    june_compatible_filter._qwen36_june_contract = True
    GPUModelRunner._xpu_filter_cudagraph_capture_descs = june_compatible_filter
    print(
        "[qwen36-s2b] restored June prefill-replay capture contract",
        file=sys.stderr,
        flush=True,
    )


def _install() -> None:
    import vllm_xpu_kernels._xpu_C as xpu_kernel_module
    from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearKernel,
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms import PlatformEnum, current_platform

    grouped_name = "_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface"
    try:
        grouped_schema = str(
            torch._C._dispatch_find_schema_or_throw(grouped_name, "").schema()
        )
    except RuntimeError:
        grouped_schema = "ABSENT"
    print(
        "[qwen36-s2b] kernel package="
        f"{xpu_kernel_module.__file__} grouped_w8a8={grouped_schema}",
        file=sys.stderr,
        flush=True,
    )

    def register_fake_once(op_name, implementation) -> None:
        try:
            torch.library.register_fake(op_name)(implementation)
        except RuntimeError as exc:
            message = str(exc)
            if "already has" not in message and "already registered" not in message:
                raise

    def fake_int8_gemm(a, a_scale, b, b_scale, out_dtype, bias):
        del a_scale, b_scale, bias
        return torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=out_dtype or torch.bfloat16,
        )

    def fake_per_token_quant(x):
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(
            (*x.shape[:-1], 1),
            device=x.device,
            dtype=torch.float32,
        )
        return q, scales

    class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
        """XPU W8A8 INT8 dense GEMM for dynamic symmetric activations."""

        @staticmethod
        def _has_op(name: str) -> bool:
            try:
                torch._C._dispatch_find_schema_or_throw(f"_xpu_C::{name}", "")
                return True
            except RuntimeError:
                return False

        @classmethod
        def is_supported(cls, compute_capability=None):
            del compute_capability
            if not current_platform.is_xpu():
                return False, "XPUInt8ScaledMM only supports XPU"
            if not cls._has_op("int8_gemm_w8a8"):
                return False, "vllm-xpu-kernels lacks int8_gemm_w8a8"
            register_fake_once("_xpu_C::int8_gemm_w8a8", fake_int8_gemm)
            return True, None

        @classmethod
        def can_implement(cls, config: Int8ScaledMMLinearLayerConfig):
            if config.is_static_input_scheme:
                return False, "only dynamic activation scales are supported"
            if not config.input_symmetric:
                return False, "only symmetric activation quantization is supported"
            if not config.is_channelwise:
                return False, "only per-channel weight scales are supported"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            weight_name, scale_name, input_scale_name, input_zp_name, azp_name = (
                self.layer_param_names
            )
            weight = getattr(layer, weight_name)
            if os.getenv("VLLM_XPU_INT8_LINEAR_BF16_FALLBACK", "0") == "1":
                scale = getattr(layer, scale_name).data.to(torch.bfloat16)
                while scale.ndim < weight.ndim:
                    scale = scale.unsqueeze(-1)
                replace_parameter(
                    layer,
                    weight_name,
                    (weight.data.to(torch.bfloat16) * scale).contiguous(),
                )
                setattr(layer, input_scale_name, None)
                setattr(layer, input_zp_name, None)
                setattr(layer, azp_name, None)
                layer.xpu_int8_linear_bf16_fallback = True
                layer.xpu_native_int8_activation_quant = False
                return

            replace_parameter(layer, weight_name, weight.t().contiguous())
            scale = getattr(layer, scale_name)
            replace_parameter(layer, scale_name, scale.flatten().contiguous())
            setattr(layer, input_scale_name, None)
            setattr(layer, input_zp_name, None)
            setattr(layer, azp_name, None)
            layer.xpu_native_int8_activation_quant = self._has_op(
                "per_token_quant_int8_xpu"
            ) and os.getenv(
                "VLLM_XPU_DISABLE_NATIVE_INT8_ACTIVATION_QUANT", "0"
            ) != "1"
            if layer.xpu_native_int8_activation_quant:
                register_fake_once(
                    "_xpu_C::per_token_quant_int8_xpu", fake_per_token_quant
                )

        def apply_weights(self, layer, x, bias=None):
            weight, weight_scale, _, _, _ = self._get_layer_params(layer)
            if getattr(layer, "xpu_int8_linear_bf16_fallback", False):
                return F.linear(x, weight, bias)
            x_2d = x.view(-1, x.shape[-1]).contiguous()
            if getattr(layer, "xpu_native_int8_activation_quant", False):
                x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x_2d)
            else:
                x_q, x_scale = per_token_quant_int8(x_2d)
            out = torch.ops._xpu_C.int8_gemm_w8a8(
                x_q,
                x_scale,
                weight,
                weight_scale,
                x.dtype,
                bias,
            )
            return out.view(*x.shape[:-1], weight.shape[1])

    kernels = _POSSIBLE_INT8_KERNELS.setdefault(PlatformEnum.XPU, [])
    kernels[:] = [
        kernel
        for kernel in kernels
        if kernel.__name__ != "XPUInt8ScaledMMLinearKernel"
    ]
    kernels.insert(0, XPUInt8ScaledMMLinearKernel)
    print(
        "[qwen36-s2b] restored June XPU INT8 linear registry class",
        file=sys.stderr,
        flush=True,
    )


_patch_shared_expert_abi()
_patch_quark_int8_moe_native_route()
_patch_custom_allreduce_clone_contract()
_patch_june_piecewise_capture_contract()
_install()
