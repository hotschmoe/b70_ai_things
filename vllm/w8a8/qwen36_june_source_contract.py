#!/usr/bin/env python3
"""Verify the reconstructed June vLLM/kernel source seam off-device."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path


def _identity(module) -> dict[str, object]:
    path = Path(inspect.getsourcefile(module) or "").resolve()
    return {
        "module": module.__name__,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vllm import _xpu_ops
    from vllm.compilation import cuda_graph, piecewise_backend
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators import xpu_communicator
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe
    from vllm.model_executor.layers.mamba import gdn_linear_attn
    from vllm.model_executor.layers.quantization.quark import quark_moe
    from vllm.v1.core.sched import scheduler
    from vllm.v1.sample import sampler
    from vllm.v1.worker import gpu_model_runner
    from vllm_xpu_kernels import fused_moe_interface

    vllm_modules = (
        _xpu_ops,
        cuda_graph,
        piecewise_backend,
        parallel_state,
        xpu_communicator,
        xpu_moe,
        gdn_linear_attn,
        quark_moe,
        scheduler,
        sampler,
        gpu_model_runner,
    )
    identities = [_identity(module) for module in vllm_modules]
    identities.append(_identity(fused_moe_interface))

    required_tokens = {
        "vllm._xpu_ops": ("VLLM_XPU_GDN_NATIVE_FALLBACK",),
        "vllm.compilation.cuda_graph": ("class CUDAGraphWrapper",),
        "vllm.compilation.piecewise_backend": ("class PiecewiseBackend",),
        "vllm.distributed.parallel_state": (
            "VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT",
        ),
        "vllm.distributed.device_communicators.xpu_communicator": (
            "VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP",
        ),
        "vllm.model_executor.layers.fused_moe.experts.xpu_moe": (
            "VLLM_XPU_INT8_MOE_MIXED_WORKSPACE",
        ),
        "vllm.model_executor.layers.mamba.gdn_linear_attn": (
            "VLLM_XPU_GDN_PREFILL_RECURRENT_FALLBACK",
        ),
        "vllm.model_executor.layers.quantization.quark.quark_moe": (
            "class QuarkW8A8Int8MoEMethod",
        ),
        "vllm.v1.sample.sampler": ("VLLM_XPU_GREEDY_SAMPLE_TOPK_FALLBACK",),
        "vllm.v1.worker.gpu_model_runner": (
            "VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY",
        ),
    }
    missing_tokens: list[str] = []
    for module in vllm_modules:
        text = Path(inspect.getsourcefile(module) or "").read_text()
        for token in required_tokens.get(module.__name__, ()):
            if token not in text:
                missing_tokens.append(f"{module.__name__}:{token}")

    xpu_fused_moe_signature = str(inspect.signature(fused_moe_interface.xpu_fused_moe))
    vllm_paths_ok = all(
        identity["path"].startswith("/opt/forensic_vllm/")
        for identity in identities[:-1]
    )
    kernel_path_ok = identities[-1]["path"] == (
        "/opt/june-runtime/vllm_xpu_kernels/fused_moe_interface.py"
    )
    scratch_abi = "scratch" in inspect.signature(
        fused_moe_interface.xpu_fused_moe
    ).parameters
    failures = []
    if not vllm_paths_ok:
        failures.append("vllm source escaped /opt/forensic_vllm")
    if not kernel_path_ok:
        failures.append("fused MoE interface escaped /opt/june-runtime")
    if not scratch_abi:
        failures.append("fused MoE interface lacks scratch ABI")
    failures.extend(missing_tokens)

    document = {
        "protocol": "qwen36-june-source-contract-v1",
        "source_stack": "e190923b32e1b87fe33d08264bff9215fb7770fc",
        "components": identities,
        "xpu_fused_moe_signature": xpu_fused_moe_signature,
        "required_tokens": required_tokens,
        "missing_tokens": missing_tokens,
        "failures": failures,
        "verdict": "pass" if not failures else "fail",
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
