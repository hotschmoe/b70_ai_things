"""CPU behavior gates: rejected PLE drafts and reordered graph slots."""

import os

os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"
os.environ["ZE_AFFINITY_MASK"] = ""

from types import SimpleNamespace

import torch
from vllm.models.qwen4_exp.gguf import ple as packed_ple
from vllm.models.qwen4_exp.gguf.mtp_ple import PLEHistory
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpPLE

cfg = SimpleNamespace(
    hidden_size=4,
    hc_count=4,
    ple_embed_dim=4,
    ngram_size=3,
    heads_per_ngram=1,
    ple_conv_kernel_size=4,
    eos_token_id=1,
    rms_norm_eps=1e-6,
)
p = Qwen4ExpPLE(cfg)
p.ple_embedding.layer_multipliers.data.copy_(torch.tensor([2, 3, 5]))
p.ple_embedding.ngram_heads_offsets.data.copy_(torch.tensor([0, 97]))
p.ple_embedding.ngram_heads_vocab_sizes.data.copy_(torch.tensor([97, 101]))
p.conv1d.weight.data.copy_(torch.arange(64).reshape(16, 1, 4).float() / 64)
p.host_table = object()
history = PLEHistory(p, 3)
original_gather = packed_ple.gather_host_rows
packed_ple.gather_host_rows = lambda table, rows, device: (
    rows.to(device).float()[:, None].repeat(1, 2)
)
reference_values = {0: {}, 1: {}}
reference_tokens = {0: {}, 1: {}}
generator = torch.Generator().manual_seed(7)


def run(groups, spec):
    ids, positions, values = [], [], []
    qsl, slots = [0], []
    for slot, first, count in groups:
        for mapping in (reference_values[slot], reference_tokens[slot]):
            for key in list(mapping):
                if key >= first:
                    del mapping[key]
        ids.extend(torch.randint(2, 90, (count,), generator=generator).tolist())
        positions.extend(range(first, first + count))
        values.append(torch.randn(count, 16, generator=generator))
        slots.append(slot)
        qsl.append(qsl[-1] + count)
    meta = SimpleNamespace(
        spec_query_start_loc=torch.tensor(qsl) if spec else None,
        non_spec_query_start_loc=None if spec else torch.tensor(qsl),
        spec_state_indices_tensor=torch.tensor(slots)[:, None].repeat(1, 3)
        if spec
        else None,
        non_spec_state_indices_tensor=None if spec else torch.tensor(slots),
    )
    history.stage(
        torch.tensor(ids, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int32),
        meta,
        2,
    )
    inputs = torch.cat(values).bfloat16()
    expected = []
    offset = 0
    for slot, first, count in groups:
        for j in range(count):
            position = first + j
            reference_values[slot][position] = inputs[offset + j]
            reference_tokens[slot][position] = ids[offset + j]
            acc = torch.zeros(16)
            for k in range(4):
                value = reference_values[slot].get(
                    position - (3 - k) * 3, torch.zeros(16)
                )
                acc += value * p.conv1d.weight[:, 0, k]
            expected.append(torch.nn.functional.silu(acc))
        # Independent full-sequence hash from the now committed prefix.
        prefix = torch.tensor([reference_tokens[slot][i] for i in range(first + count)])
        p._token_hist[slot].fill_(p.eos)
        rows = p._compute_rows(
            prefix,
            torch.tensor([0, len(prefix)]),
            torch.tensor([slot]),
            torch.tensor([False]),
        )
        actual = p._graph_emb[offset : offset + count].view(count, 2, 2)[:, :, 0]
        torch.testing.assert_close(actual, rows[first:].float(), rtol=0, atol=0)
        offset += count
    actual = history.conv(inputs)
    torch.testing.assert_close(actual, torch.stack(expected).bfloat16(), atol=0, rtol=0)


try:
    run([(0, 0, 24), (1, 0, 19)], False)  # Long prefill, ring wraps.
    run([(0, 24, 3), (1, 19, 3)], True)
    run([(1, 20, 3), (0, 26, 3)], True)  # Different rejection lengths; reorder.
    run([(0, 27, 3), (1, 22, 3)], True)
    run([(0, 0, 4), (1, 0, 2)], False)  # Reuse physical slots for fresh requests.
    run([(1, 2, 3), (0, 4, 3)], True)
    print("X1_PLE_REJECTION_AND_SLOT_REORDER_OK")
finally:
    packed_ple.gather_host_rows = original_gather
