"""GGUF decode kernels for Intel XPU. vLLM loads this via the vllm.general_plugins group."""

from .ops import grouped_gemm, dequant_bf16, gemv, load, moe_gemv
from .types import BLOCK, NAMES, Q2_0, row_bytes

__all__ = [
    "BLOCK",
    "NAMES",
    "Q2_0",
    "dequant_bf16",
    "gemv",
    "grouped_gemm",
    "load",
    "moe_gemv",
    "register",
    "row_bytes",
]


def register() -> None:
    load()
