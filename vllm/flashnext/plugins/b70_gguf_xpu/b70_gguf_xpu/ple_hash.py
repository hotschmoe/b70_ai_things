"""Vectorized host n-gram hash with uint64 wrap and request history."""
import numpy as np
import torch


def compute_rows(ple, input_ids, qsl, slots, has_initial):
    tokens = input_ids.numpy().astype(np.uint64, copy=False)
    multipliers = np.array(
        [int(v) & ((1 << 64) - 1) for v in ple.ple_embedding.layer_multipliers.tolist()],
        dtype=np.uint64,
    )
    offsets = np.array(ple.ple_embedding.ngram_heads_offsets.tolist(), dtype=np.uint64)
    vocabs = np.array(ple.ple_embedding.ngram_heads_vocab_sizes.tolist(), dtype=np.uint64)
    rows = np.empty((len(tokens), ple.n_heads), dtype=np.int64)
    p = ple.per_gram
    for s, slot in enumerate(slots.tolist()):
        a, b = int(qsl[s]), int(qsl[s + 1])
        old = ple._token_hist[slot].tolist() if has_initial is not None and bool(has_initial[s]) else [ple.eos, ple.eos]
        if a == b:
            ple._token_hist[slot] = torch.tensor(old)
            continue
        x = tokens[a:b]
        prev = np.concatenate((np.array([old[1], old[0]], dtype=np.uint64), x))
        c1 = prev[1:-1]
        c2 = np.where(c1 == ple.eos, np.uint64(ple.eos), prev[:-2])
        mix2 = x * multipliers[0] ^ c1 * multipliers[1]
        mix3 = mix2 ^ c2 * multipliers[2]
        rows[a:b, :p] = mix2[:, None] % vocabs[:p] + offsets[:p]
        rows[a:b, p:] = mix3[:, None] % vocabs[p:] + offsets[p:]
        ple._token_hist[slot, 0] = int(x[-1])
        ple._token_hist[slot, 1] = int(c1[-1])
    return torch.from_numpy(rows)
