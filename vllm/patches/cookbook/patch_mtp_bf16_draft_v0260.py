#!/usr/bin/env python3
"""Force BF16/unquantized MTP draft build on vLLM 0.26.0 (our vllm-xpu-env:v0260).

Our baked 0.26.0 qwen3_5_mtp.py does NOT have the nightly original_quant null
gate. It only special-cases mtp.fc for modelopt_fp4. GPTQ/AutoRound bodies still
build MTP decoder layers under quant_config, so BF16 mtp.* weights either fail
to load or accept collapses.

This patch, gated by B70_MTP_BF16_DRAFT=1:
  1. nulls vllm_config.quant_config while constructing Qwen3_5MultiTokenPredictor
     layers (so draft MoE/linear layers are unquantized);
  2. restores quant_config afterward so the target body is unaffected;
  3. forces mtp.fc to build unquantized regardless of quant scheme.

Safe no-op when B70_MTP_BF16_DRAFT is unset (still applies the source edit;
runtime behavior only changes when the env is set).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MARKER = "B70_MTP_BF16_DRAFT_V0260"


def patch_text(t: str) -> str:
    if MARKER in t:
        return t

    # Inject import os if missing.
    if "\nimport os\n" not in t and not t.startswith("import os\n"):
        if "import torch\n" in t:
            t = t.replace("import torch\n", "import os\nimport torch\n", 1)
        else:
            t = "import os\n" + t

    # Replace the modelopt_fp4-only fc_quant special case with a broader gate,
    # and null quant_config for the whole MultiTokenPredictor construction.
    old_fc = '''        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
        # missing from hf_quant_config.json exclude_modules. Force unquantized.
        # Ref: https://github.com/vllm-project/vllm/pull/38650
        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
        fc_quant = (
            None
            if (quant_config and quant_config.get_name() == "modelopt_fp4")
            else quant_config
        )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
'''

    new_fc = '''        # B70_MTP_BF16_DRAFT_V0260: force unquantized MTP draft when env set.
        # Also keep the modelopt_fp4 mtp.fc workaround for NVFP4 checkpoints.
        _force_bf16_draft = os.environ.get("B70_MTP_BF16_DRAFT") == "1"
        _orig_quant = vllm_config.quant_config
        if _force_bf16_draft:
            print(
                "[B70] MTP draft: forcing unquantized build "
                "(env B70_MTP_BF16_DRAFT=1)",
                flush=True,
            )
            vllm_config.quant_config = None
            quant_config = None
        fc_quant = (
            None
            if (
                _force_bf16_draft
                or (quant_config and quant_config.get_name() == "modelopt_fp4")
            )
            else quant_config
        )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        if _force_bf16_draft:
            vllm_config.quant_config = _orig_quant
'''

    if old_fc not in t:
        raise RuntimeError(
            "v0260 qwen3_5_mtp.py fc/layers anchor not found; refusing to patch"
        )
    return t.replace(old_fc, new_fc, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor"
        / "models"
        / "qwen3_5_mtp.py"
    )
    original = path.read_text()
    try:
        patched = patch_text(original)
    except RuntimeError as e:
        # If nightly anchors exist, leave a clear message.
        if "original_quant = vllm_config.quant_config" in original:
            raise SystemExit(
                f"{e}; this looks like a nightly -- use patch_mtp_nightly.py"
            ) from e
        raise SystemExit(str(e)) from e
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
