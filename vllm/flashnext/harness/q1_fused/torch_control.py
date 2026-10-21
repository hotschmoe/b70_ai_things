"""Exact parallel packed-key selector, no custom kernel or GPU import."""
import torch

class PackedTopk:
    def __init__(self, scores_n, cap, device):
        assert scores_n >= cap and scores_n < 2**32
        self.scores_n = scores_n
        self.cap = cap
        self.col = torch.arange(scores_n, dtype=torch.int64, device=device)[None, :]
        self.rank = torch.arange(cap, dtype=torch.int64, device=device)[None, :]

    def __call__(self, scores, ids, step):
        assert scores.shape[1] == self.scores_n and ids.shape[1] == self.cap
        # All score comparisons happen after raw-bit reinterpretation. NaN
        # payloads and signed-zero distinctions retain native key semantics.
        bits = scores.view(torch.int32).to(torch.int64) & 0xffffffff
        order = torch.where((bits & 0x80000000) != 0,
                            bits ^ 0xffffffff, bits ^ 0x80000000)
        unsigned_bits = (order << 32) | (0xffffffff ^ self.col)
        signed_keys = unsigned_bits ^ (-2**63)
        n = step[:, 1:2].to(torch.int64).clamp(min=0, max=self.scores_n)
        width = torch.minimum(step[:, 3:4].to(torch.int64).clamp(min=0, max=self.cap), n)
        # Packed valid keys are strictly above unsigned zero because the
        # inverted index is nonzero for this bounded score-column count.
        keys = torch.where(self.col < n, signed_keys, -2**63)
        _, selected = torch.topk(keys, self.cap, dim=-1, largest=True, sorted=True)
        # Fixed k allows device-varying widths, including inactive width0.
        # Exclude ranks beyond width before stable ascending-ID ordering.
        selected = torch.where(self.rank < width, selected, self.scores_n)
        ordered, _ = torch.sort(selected, dim=-1, stable=True)
        ids.copy_(torch.where(self.rank < width, ordered.to(torch.int32), ids))
        return ids
