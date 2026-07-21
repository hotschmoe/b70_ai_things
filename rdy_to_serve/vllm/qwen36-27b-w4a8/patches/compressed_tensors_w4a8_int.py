# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# b70 OVERLAY of .../compressed_tensors/schemes/compressed_tensors_w4a8_int.py for
# vllm-xpu-env:int8g-v0251 (vLLM 0.25.1). Verbatim upstream 0.25.1 content (extracted
# from the image 2026-07-21) PLUS ONE b70 delta, marked "b70:" below:
#   VLLM_W4A8_PREPACKED: allocate the weight as int32 [out, in/8] (8 int4 per int32)
#   so the offline-prepacked checkpoint loads directly, instead of the upstream int8
#   [out, in] unpacked allocation (which would then need the ~28 GiB on-GPU repack
#   transient that hangs/OOMs the 32 GB B70 on the 27B). Pairs with the skip-pack
#   branch in the mounted mixed_precision/xpu.py.
# 0.25.1 API drift vs our old v0.23 patch (both verified in-image 2026-07-21):
#   - the group-size divisibility assert became
#     vllm.distributed.utils.verify_group_size_divides_partition (kept as upstream).
#   - imports MPLinearLayerConfig/choose_mp_linear_kernel from
#     vllm.model_executor.kernels.linear (unchanged, resolves in-image).
# If a future image drifts this upstream file, re-extract and re-apply the delta:
#   docker run --rm --entrypoint cat vllm-xpu-env:int8g-v0251 \
#     /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py

import os
from collections.abc import Callable

import torch

from vllm.distributed.utils import verify_group_size_divides_partition
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A8Int"]
W4A8_SUPPORTED_TYPES_MAP = {
    4: scalar_types.int4,
}
W4A8_SUPPORTED_BITS = list(W4A8_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsW4A8Int(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
    ):
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

        if num_bits not in W4A8_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}."
                f"Supported num_bits = {W4A8_SUPPORTED_TYPES_MAP.keys()}"
            )
        self.quant_type = W4A8_SUPPORTED_TYPES_MAP[num_bits]

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        row_parallel = input_size != input_size_per_partition

        # Compute effective group_size
        if self.group_size == -1:
            effective_group_size = (
                input_size_per_partition if row_parallel else input_size
            )
        else:
            effective_group_size = self.group_size

        # Ensure group_size divides input_size_per_partition
        verify_group_size_divides_partition(
            input_size_per_partition, effective_group_size
        )

        # Determine scale partitioning
        is_channelwise = self.group_size == -1
        repeat_scales = is_channelwise and row_parallel
        partition_scales = not repeat_scales

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,
            zero_points=False,
            has_g_idx=False,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsW4A8Int", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        scales_and_zp_size = input_size_per_partition // effective_group_size

        # b70: prepacked offline -> weights are int32 [out, in/8] (8 int4 per int32),
        # loaded directly; upstream default is unpacked int8 [out, in] (then packed
        # on-GPU at load, a ~28 GiB transient on the 27B that a 32 GB B70 cannot fit).
        if os.environ.get("VLLM_W4A8_PREPACKED"):
            assert input_size_per_partition % 8 == 0
            _wdata = torch.empty(
                output_size_per_partition,
                input_size_per_partition // 8,
                dtype=torch.int32,
            )
        else:
            _wdata = torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            )
        weight = ModelWeightParameter(
            data=_wdata,
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition, scales_and_zp_size, dtype=params_dtype
            ),
        }

        if partition_scales:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
        else:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name=None,
            w_gidx_param_name=None,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
