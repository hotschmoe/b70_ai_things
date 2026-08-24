#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_ornith15_quark_w8a8 import convert


class QuantizeOrnithTest(unittest.TestCase):
    def test_layout_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            tensors = {
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 8),
                "model.language_model.layers.0.linear_attn.out_proj.weight": torch.randn(4, 8),
                "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 6, 8),
                "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, 8, 3),
                "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, 8),
                "mtp.fc.weight": torch.randn(4, 8),
            }
            save_file(tensors, source / "model-00001.safetensors")
            index = {
                "metadata": {},
                "weight_map": {name: "model-00001.safetensors" for name in tensors},
            }
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            (source / "config.json").write_text(json.dumps({"text_config": {"mtp_num_hidden_layers": 1}}))
            (source / "tokenizer.json").write_text("{}")

            contract = convert(source, output, "cpu", 4, expected_mtp_sha256=None)
            out_index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
            expected = {
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                "model.language_model.layers.0.mlp.experts.1.down_proj.weight",
                "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale",
            }
            self.assertTrue(expected.issubset(out_index))
            with safe_open(output / "model-00001.safetensors", framework="pt", device="cpu") as handle:
                self.assertEqual(handle.get_tensor("model.language_model.layers.0.self_attn.q_proj.weight").dtype, torch.int8)
                self.assertEqual(handle.get_tensor("model.language_model.layers.0.linear_attn.out_proj.weight").dtype, torch.float32)
                self.assertEqual(handle.get_tensor("model.language_model.layers.0.mlp.gate.weight").dtype, torch.float32)
                self.assertEqual(handle.get_tensor("mtp.fc.weight").dtype, torch.float32)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["quantization_config"]["quant_method"], "quark")
            self.assertEqual(contract["counts"]["expert_weights"], 6)
            self.assertEqual(contract["counts"]["dense"], 1)


if __name__ == "__main__":
    unittest.main()
