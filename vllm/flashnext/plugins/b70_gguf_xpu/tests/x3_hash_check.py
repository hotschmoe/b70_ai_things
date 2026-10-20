"""Check uint64 overflow, EOS reset, ragged histories and slot reorder on CPU."""
from types import SimpleNamespace
import numpy as np
import torch
from b70_gguf_xpu.ple_hash import compute_rows


def scalar(p, tokens, qsl, slots, has):
    rows = torch.zeros(len(tokens), 16, dtype=torch.int64)
    m = p.ple_embedding.layer_multipliers.tolist()
    voc = p.ple_embedding.ngram_heads_vocab_sizes.tolist()
    off = p.ple_embedding.ngram_heads_offsets.tolist()
    for s, slot in enumerate(slots.tolist()):
        prev1, prev2 = p._token_hist[slot].tolist() if has is not None and bool(has[s]) else [p.eos, p.eos]
        for i in range(qsl[s], qsl[s+1]):
            c = int(tokens[i]); c2 = p.eos if prev1 == p.eos else prev2
            m2 = (c*m[0] ^ prev1*m[1]) & ((1<<64)-1)
            m3 = (m2 ^ c2*m[2]) & ((1<<64)-1)
            for h in range(16):
                rows[i,h] = ((m2 if h < 8 else m3) % voc[h]) + off[h]
            prev2, prev1 = prev1, c
        p._token_hist[slot] = torch.tensor([prev1, prev2])
    return rows


def main():
    rng = np.random.default_rng(3)
    for trial in range(100):
        lens = rng.integers(0, 20, size=4)
        cu = torch.tensor(np.r_[0, lens.cumsum()])
        tokens = torch.tensor(rng.integers(0, 248321, size=int(cu[-1])))
        tokens[::3] = 248044
        slots = torch.tensor(rng.permutation(4))
        has = None if trial % 3 == 0 else torch.tensor(rng.integers(0,2,size=4), dtype=torch.bool)
        hist = torch.tensor(rng.integers(0,248321,size=(4,2)))
        emb = SimpleNamespace(layer_multipliers=torch.tensor([23703573157769,20109073645365,8052911324071]),
            ngram_heads_offsets=torch.arange(16)*20000000,
            ngram_heads_vocab_sizes=torch.arange(16)*2+20000003)
        p = SimpleNamespace(eos=248044, per_gram=8, n_heads=16, ple_embedding=emb, _token_hist=hist.clone())
        ref = scalar(p, tokens, cu, slots, has); expected_hist = p._token_hist.clone()
        p._token_hist = hist.clone()
        got = compute_rows(p, tokens, cu, slots, has)
        assert torch.equal(got, ref), trial
        assert torch.equal(p._token_hist, expected_hist), trial
    print('X3_HASH_OK 100 ragged trials')

if __name__ == '__main__':
    main()
