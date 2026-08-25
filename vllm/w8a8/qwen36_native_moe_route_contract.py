#!/usr/bin/env python3
"""Off-device guard for the Qwen3.6 Quark native XPU INT8 MoE route."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.layers.fused_moe.oracle.int8 import make_int8_moe_kernel
from vllm.model_executor.layers.quantization.quark import quark_moe


EXPECTED_IMAGE_QUARK_MOE_SHA256 = (
    "7e4c13d249298ed49378961faefac89bebe7716ced9414b5429980e910785c79"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cls = quark_moe.QuarkW8A8Int8MoEMethod
    methods = {
        "init": cls.__init__,
        "process_weights_after_loading": cls.process_weights_after_loading,
        "apply": cls.apply,
    }
    for name, method in methods.items():
        assert getattr(method, "_qwen36_native_int8_moe_route", False), (
            f"Quark native XPU INT8 MoE patch is absent from {name}"
        )

    image_source = Path(quark_moe.__file__).resolve()
    image_source_hash = sha256_file(image_source)
    assert image_source_hash == EXPECTED_IMAGE_QUARK_MOE_SHA256, (
        f"unexpected pinned-image Quark MoE source: {image_source_hash}"
    )

    apply_signature = inspect.signature(cls.apply)
    for name in ("shared_experts", "shared_experts_input"):
        parameter = apply_signature.parameters[name]
        assert parameter.default is None

    make_signature = inspect.signature(make_int8_moe_kernel)
    assert "routing_tables" in make_signature.parameters
    assert "shared_experts" not in make_signature.parameters

    kernel_apply_signature = inspect.signature(FusedMoEKernel.apply)
    for name in ("shared_experts", "shared_experts_input"):
        assert name in kernel_apply_signature.parameters

    call_abi = xpu_moe.xpu_fused_moe
    assert getattr(call_abi, "_qwen36_june_moe_call_abi", False)
    call_abi_signature = inspect.signature(call_abi)
    for name in ("scratch", "diagnostic_context"):
        parameter = call_abi_signature.parameters[name]
        assert parameter.default is None
    call_abi_source = inspect.getsource(call_abi)
    assert "if scratch is not None" in call_abi_source
    assert "return june_xpu_fused_moe(*args, **kwargs)" in call_abi_source

    process_source = inspect.getsource(cls.process_weights_after_loading)
    apply_source = inspect.getsource(cls.apply)
    for required in (
        "prepare_int8_moe_layer_for_xpu",
        "make_int8_moe_quant_config",
        "make_int8_moe_kernel",
        "_expert_routing_tables",
    ):
        assert required in process_source
    assert "self.moe_kernel.apply" in apply_source
    assert "shared_experts=shared_experts" in apply_source

    record = {
        "protocol": "qwen36-june-native-int8-moe-route-contract-v1",
        "image_quark_moe_source": str(image_source),
        "image_quark_moe_sha256": image_source_hash,
        "patched_methods": sorted(methods),
        "quark_apply_signature": str(apply_signature),
        "make_int8_moe_kernel_signature": str(make_signature),
        "fused_moe_kernel_apply_signature": str(kernel_apply_signature),
        "june_moe_call_abi_signature": str(call_abi_signature),
        "mixed_workspace_contract": "disabled; non-None scratch rejected",
        "weight_layout_repair": "E,N,K-to-E,K,N",
        "scale_layout_repair": "E,N,1-to-E,N",
        "verdict": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
