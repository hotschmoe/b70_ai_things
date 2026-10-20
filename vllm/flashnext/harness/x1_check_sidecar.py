"""CPU-only full-size Flash-Next MTP payload audit and wide-fc reference."""

import os

os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"
os.environ["ZE_AFFINITY_MASK"] = ""
os.environ["B70_MTP_RT"] = "/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt"
os.environ["B70_MTP_GGUF"] = "/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/mtp-q2_0.gguf"
os.environ["B70_GGUF_PATH"] = (
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)

import runpy
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from vllm.config import set_current_vllm_config
from vllm.models.qwen4_exp.gguf.config import B70GGUFConfig
from vllm.models.qwen4_exp.gguf.mtp_loader import load_mtp_weights
from vllm.models.qwen4_exp.xpu.mtp import Qwen4ExpMTP

fork = Path("/mnt/vm_8tb/b70/build/vllm-flashnext/vllm")
helpers = runpy.run_path(str(fork / "tests/models/qwen4_exp/test_b70_gguf_mapping.py"))
text = AutoConfig.from_pretrained(
    "/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext",
    trust_remote_code=True,
).get_text_config()
cfg = helpers["_tiny_vllm_config"](text, B70GGUFConfig())
with set_current_vllm_config(cfg):
    helpers["_init_cpu_parallel"]()
    with torch.device("meta"):
        draft = Qwen4ExpMTP(vllm_config=cfg)
    load_mtp_weights(draft, cfg.model_config)
print("X1_FLASHNEXT_PAYLOAD_AUDIT_OK", flush=True)
assert not hasattr(draft.model, "eh_proj")

# Compare the real packed Q8 fc matrices with fp32 dequant matmuls, keeping
# all four streams. This checks the daily-driver graph, not the old eh path.
from vllm.models.qwen4_exp.gguf.ops_interface import _ref_matrix

generator = torch.Generator().manual_seed(1)
hidden = text.hidden_size
wide = torch.randn(2, text.hc_count, hidden, generator=generator)
emb = torch.randn(2, hidden, generator=generator)
model = draft.model
refs = []
for layer, value in (
    (model.fc_embedding, model.pre_fc_norm_embedding(emb)),
    (model.fc_hidden, model.pre_fc_norm_hidden.forward_wide(wide)),
):
    shard = layer.b70_shards["full"]
    w = _ref_matrix(shard.packed, shard.qtype, hidden, hidden)
    # Independent RMS normalization uses the folded rt gamma.
    name = (
        "pre_fc_norm_embedding.weight"
        if value.ndim == 2
        else "pre_fc_norm_hidden.weight"
    )
    index = [
        line.split()
        for line in Path(os.environ["B70_MTP_RT"], "dense.txt").read_text().splitlines()
    ]
    row = next(row for row in index if row[0] == name)
    gamma = np.fromfile(
        Path(os.environ["B70_MTP_RT"], "dense.bin"),
        dtype=np.float32,
        count=int(row[3]),
        offset=int(row[4]),
    )
    source = emb if value.ndim == 2 else wide.flatten(1)
    expected = source * torch.rsqrt(
        source.square().mean(-1, keepdim=True) + text.rms_norm_eps
    )
    expected = (expected * torch.from_numpy(gamma)).reshape_as(value)
    torch.testing.assert_close(value.float(), expected, atol=2e-3, rtol=2e-3)
    refs.append(expected @ w.T)
reference = refs[0].unsqueeze(1) + refs[1]
os.environ["B70_GGUF_REF"] = "1"
actual = model._fc_stage(emb, wide)
torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
assert (reference[:, 0] - reference[:, 1]).abs().max() > 0.01
print("X1_WIDE_FC_REFERENCE_OK", reference.shape, flush=True)
torch.distributed.destroy_process_group()
