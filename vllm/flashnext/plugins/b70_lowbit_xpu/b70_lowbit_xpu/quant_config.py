"""vLLM quant config `b70_lowbit`. Registered from the general_plugins entry."""

from __future__ import annotations

import re
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

# Checkpoint methods this config can repack when the user selects b70_lowbit.
_GPTQ_METHODS = {"gptq", "auto-round", "auto_gptq", "awq"}
_SKIP_BITS = 16


def _component_hit(prefix: str, name: str) -> bool:
    """Match a module component, not a raw substring.

    "gate" matches "model.layers.0.mlp.gate" and does not match "gate_proj".
    """
    if not name:
        return False
    if prefix == name or prefix.endswith("." + name) or prefix.startswith(name + "."):
        return True
    return ("." + name + ".") in ("." + prefix + ".")


def _pack_from_config(config: dict[str, Any]) -> str:
    explicit = config.get("pack")
    if explicit:
        return str(explicit)
    method = str(config.get("quant_method", ""))
    packing = str(config.get("packing_format", ""))
    fmt = str(config.get("checkpoint_format", config.get("format", "")))
    if method in _GPTQ_METHODS or "gptq" in fmt or "gptq" in packing:
        return "gptq"
    return "b70lb1"


def _rules_from_config(config: dict[str, Any]) -> list[tuple[re.Pattern[str], int]]:
    """Ordered first-match rules. bits 16 means leave the module unquantized.

    GPTQ `dynamic` keys that start with `-:` or map to {} are excludes.
    AutoRound `extra_config` values carry an explicit bits field. Both are
    regexes against the vLLM module prefix, not component names.
    """
    rules: list[tuple[re.Pattern[str], int]] = []

    def add(pattern: str, bits: int) -> None:
        rules.append((re.compile(pattern), int(bits)))

    dynamic = config.get("dynamic") or {}
    if isinstance(dynamic, dict):
        for key, val in dynamic.items():
            if not isinstance(key, str):
                continue
            if key.startswith("-:"):
                add(key[2:], _SKIP_BITS)
            elif isinstance(val, dict) and "bits" in val:
                add(key, int(val["bits"]))
            elif val in ({}, None):
                add(key, _SKIP_BITS)
    extra = config.get("extra_config") or {}
    if isinstance(extra, dict):
        for key, val in extra.items():
            if isinstance(key, str) and isinstance(val, dict) and "bits" in val:
                add(key, int(val["bits"]))
    return rules


class _B70Linear(B70LowbitLinearMethod, LinearMethodBase):
    """LinearMethodBase subclass so ColumnParallelLinear accepts it."""


class _B70MoE(B70LowbitMoEMethod, FusedMoEMethodBase):
    """FusedMoEMethodBase subclass. The runner requires this type."""

    def __init__(
        self, moe, bits: int, group_size: int = 128, pack: str = "b70lb1"
    ) -> None:
        FusedMoEMethodBase.__init__(self, moe)
        B70LowbitMoEMethod.__init__(self, moe, bits, group_size, pack)


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
        rules: list[tuple[re.Pattern[str], int]] | None = None,
    ) -> None:
        super().__init__()
        if bits not in (2, 3, 4):
            raise ValueError(bits)
        if group_size != 128:
            raise ValueError("group_size must be 128")
        if pack not in ("b70lb1", "gptq"):
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
        self._rules = list(rules or [])

    def get_name(self) -> str:
        return "b70_lowbit"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Not a CUDA capability gate. XPU selection is the platform check.
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        """Claim a checkpoint only when the user asked for b70_lowbit.

        auto-round and gptq stay on their own methods (INC for A3) unless
        `--quantization b70_lowbit` is set, or the file already says
        quant_method b70_lowbit.
        """
        del hf_config
        method = ""
        if isinstance(hf_quant_cfg, dict):
            method = str(hf_quant_cfg.get("quant_method", ""))
        if user_quant == "b70_lowbit" or method == "b70_lowbit":
            return "b70_lowbit"
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "B70LowbitConfig":
        extra = cls.get_from_keys_or(config, ["extra_bits"], {}) or {}
        skipped = cls.get_from_keys_or(config, ["modules_to_not_convert"], []) or []
        pack = _pack_from_config(config)
        if pack == "gptq" and bool(config.get("desc_act", False)):
            raise ValueError("b70_lowbit gptq path requires desc_act false")
        return cls(
            bits=int(cls.get_from_keys_or(config, ["bits"], 3)),
            group_size=int(cls.get_from_keys_or(config, ["group_size"], 128)),
            sym=bool(cls.get_from_keys_or(config, ["sym"], True)),
            pack=pack,
            modules_to_not_convert=list(skipped),
            extra_bits=dict(extra),
            rules=_rules_from_config(config),
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
            self.modules_to_not_convert
        )
        if self.extra_bits:
            mapped = hf_to_vllm_mapper.apply_dict(self.extra_bits)
            self.extra_bits = {str(k): int(v) for k, v in mapped.items()}
        # Regex rules are matched against the vLLM prefix as written in the
        # checkpoint. Do not rewrite them through the HF name mapper.

    def _first_rule(self, prefix: str) -> int | None:
        for pattern, bits in self._rules:
            if pattern.search(prefix):
                return bits
        return None

    def _skipped(self, prefix: str) -> bool:
        if any(_component_hit(prefix, name) for name in self.modules_to_not_convert):
            return True
        return self._first_rule(prefix) == _SKIP_BITS

    def _bits_for(self, prefix: str) -> int:
        best = None
        best_len = -1
        for name, bits in self.extra_bits.items():
            if _component_hit(prefix, name) and len(name) > best_len:
                best = bits
                best_len = len(name)
        if best is not None:
            return best
        decided = self._first_rule(prefix)
        if decided is not None and decided != _SKIP_BITS:
            return decided
        if prefix.endswith(".experts"):
            return self._expert_bits(prefix)
        if decided == _SKIP_BITS:
            return _SKIP_BITS
        return self.bits

    def _expert_bits(self, prefix: str) -> int:
        """One bits value for the fused RoutedExperts module.

        The checkpoint regexes name `experts.0.gate_proj`. The vLLM module
        prefix stops at `.experts`, so probe gate, up, and down. Mixed bits
        inside one fused module cannot share this kernel.
        """
        found: list[int] = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            decided = self._first_rule(prefix + ".0." + name)
            if decided is None:
                decided = self.bits
            found.append(decided)
        if any(bits == _SKIP_BITS for bits in found) and any(
            bits != _SKIP_BITS for bits in found
        ):
            raise ValueError(f"mixed expert skip/quant at {prefix}: {found}")
        if found[0] == _SKIP_BITS:
            return _SKIP_BITS
        if len(set(found)) != 1:
            raise ValueError(f"mixed expert bits at {prefix}: {found}")
        return found[0]

    def _unquantized(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if self._skipped(prefix):
            return self._unquantized(layer)
        bits = self._bits_for(prefix)
        if bits not in (2, 3, 4):
            return self._unquantized(layer)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return _B70Linear(bits, self.group_size, self.pack)
        if isinstance(layer, RoutedExperts):
            return _B70MoE(layer.moe_config, bits, self.group_size, self.pack)
        return None
