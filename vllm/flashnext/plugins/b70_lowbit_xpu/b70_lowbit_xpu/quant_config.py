"""vLLM quant config `b70_lowbit`. Registered from the general_plugins entry."""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from b70_lowbit_xpu.linear import B70LowbitLinearMethod
from b70_lowbit_xpu.moe import B70LowbitMoEMethod


def _component_hit(prefix: str, name: str) -> bool:
    """Match a module component, not a raw substring.

    "gate" matches "model.layers.0.mlp.gate" and does not match "gate_proj".
    """
    if not name:
        return False
    if prefix == name or prefix.endswith("." + name) or prefix.startswith(name + "."):
        return True
    return ("." + name + ".") in ("." + prefix + ".")


class _B70Linear(B70LowbitLinearMethod, LinearMethodBase):
    """LinearMethodBase subclass so ColumnParallelLinear accepts it."""


class _B70MoE(B70LowbitMoEMethod, FusedMoEMethodBase):
    """FusedMoEMethodBase subclass. The runner requires this type."""

    def __init__(self, moe, bits: int, group_size: int = 128) -> None:
        FusedMoEMethodBase.__init__(self, moe)
        B70LowbitMoEMethod.__init__(self, moe, bits, group_size)


def ensure_registered() -> None:
    # Class body of B70LowbitConfig performs the registration.
    assert B70LowbitConfig.get_name


@register_quantization_config("b70_lowbit")
class B70LowbitConfig(QuantizationConfig):
    def __init__(
        self,
        bits: int = 3,
        group_size: int = 128,
        sym: bool = True,
        pack: str = "b70lb1",
        modules_to_not_convert: list[str] | None = None,
        extra_bits: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if bits not in (2, 3, 4):
            raise ValueError(bits)
        if group_size != 128:
            raise ValueError("group_size must be 128")
        if pack != "b70lb1":
            raise ValueError(f"unsupported pack {pack}")
        if not sym:
            raise ValueError("b70_lowbit is symmetric only")
        self.bits = bits
        self.group_size = group_size
        self.sym = True
        self.pack = pack
        self.modules_to_not_convert = list(modules_to_not_convert or [])
        self.extra_bits = {str(k): int(v) for k, v in (extra_bits or {}).items()}
        for value in self.extra_bits.values():
            if value not in (2, 3, 4):
                raise ValueError(value)

    def get_name(self) -> str:
        return "b70_lowbit"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Not a CUDA capability gate. XPU selection is the platform check.
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "B70LowbitConfig":
        extra = cls.get_from_keys_or(config, ["extra_bits"], {}) or {}
        skipped = cls.get_from_keys_or(config, ["modules_to_not_convert"], []) or []
        return cls(
            bits=int(cls.get_from_keys_or(config, ["bits"], 3)),
            group_size=int(cls.get_from_keys_or(config, ["group_size"], 128)),
            sym=bool(cls.get_from_keys_or(config, ["sym"], True)),
            pack=str(cls.get_from_keys_or(config, ["pack"], "b70lb1")),
            modules_to_not_convert=list(skipped),
            extra_bits=dict(extra),
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
            self.modules_to_not_convert
        )
        if self.extra_bits:
            mapped = hf_to_vllm_mapper.apply_dict(self.extra_bits)
            self.extra_bits = {str(k): int(v) for k, v in mapped.items()}

    def _skipped(self, prefix: str) -> bool:
        return any(_component_hit(prefix, name) for name in self.modules_to_not_convert)

    def _bits_for(self, prefix: str) -> int:
        best = None
        best_len = -1
        for name, bits in self.extra_bits.items():
            if _component_hit(prefix, name) and len(name) > best_len:
                best = bits
                best_len = len(name)
        return self.bits if best is None else best

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if self._skipped(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return None
        bits = self._bits_for(prefix)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return _B70Linear(bits, self.group_size)
        if isinstance(layer, RoutedExperts):
            return _B70MoE(layer.moe_config, bits, self.group_size)
        return None
