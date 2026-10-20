#!/usr/bin/env python3
"""O4: ple_fast._hash_rows vs Qwen4ExpPLE._compute_rows (CPU only)."""
import types, sys
import numpy as np
import torch
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpPLE
from vllm.models.qwen4_exp.gguf.ple_fast import _hash_rows

cfg = types.SimpleNamespace(hc_count=4, hidden_size=2560, rms_norm_eps=1e-6, ngram_size=3,
                            heads_per_ngram=8, ple_embed_dim=2560, ple_conv_kernel_size=4, eos_token_id=248046)
p = Qwen4ExpPLE(cfg)
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    p.ple_embedding.layer_multipliers.data = torch.randint(-2**62, 2**62, (3,), generator=g)
    p.ple_embedding.ngram_heads_offsets.data = torch.arange(16) * 20000003
    p.ple_embedding.ngram_heads_vocab_sizes.data = torch.randint(10**7, 3 * 10**7, (16,), generator=g)
p._ensure_state(4, torch.device("cpu"))
bad = 0
for trial in range(300):
    n = int(torch.randint(1, 9, (1,), generator=g))
    vals = torch.randint(0, 248320, (n,), generator=g)
    if trial % 5 == 0:
        vals[torch.randint(0, n, (1,), generator=g)] = cfg.eos_token_id
    prev1 = int(torch.randint(0, 248320, (1,), generator=g)) if trial % 3 else cfg.eos_token_id
    prev2 = int(torch.randint(0, 248320, (1,), generator=g))
    p._token_hist[1, 0] = prev1
    p._token_hist[1, 1] = prev2
    ref = p._compute_rows(vals, torch.tensor([0, n]), torch.tensor([1]), torch.tensor([True]))
    r2, n1, n2 = _hash_rows(p, vals.numpy(), prev1, prev2)
    ok = np.array_equal(ref.numpy(), r2) and n1 == int(p._token_hist[1, 0]) and n2 == int(p._token_hist[1, 1])
    bad += 0 if ok else 1
print("O4_PLE_HASH", "OK" if bad == 0 else f"FAIL {bad}/300")
sys.exit(1 if bad else 0)
