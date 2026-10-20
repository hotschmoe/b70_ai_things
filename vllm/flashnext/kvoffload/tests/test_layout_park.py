"""CPU checks for the hybrid offload contract. No GPU."""

from __future__ import annotations

import unittest

from kvoffload.layout import (
    OrnithTextLayout,
    checkpoint_boundary,
    full_attn_bytes,
    scaled_attention_block_size,
    upstream_pool_bytes,
)
from kvoffload.park import Park, missing_kinds, prefill_tokens, resume_tokens


class LayoutTest(unittest.TestCase):
    def test_ornith_block_matches_serve_log(self) -> None:
        # A3 eager log: attention block size 1088 so the page covers GDN.
        layout = OrnithTextLayout()
        self.assertEqual(layout.gdn_state_bytes_per_layer(), 2_146_304)
        self.assertEqual(layout.scaled_block_size(), 1088)
        self.assertEqual(layout.attn_bytes_per_token(), 20_480)

    def test_scale_formula(self) -> None:
        # 16 * 131072 = 2097152, page is 49152 bytes over that, so 17 * 64.
        self.assertEqual(
            scaled_attention_block_size(2_146_304, 2048, 64),
            1088,
        )

    def test_checkpoint_drops_one_block_without_eagle(self) -> None:
        block = 1088
        tokens = 28 * block
        boundary = checkpoint_boundary(tokens, block)
        self.assertEqual(boundary, 27 * block)
        self.assertEqual(tokens - boundary, block)

    def test_eagle_drops_one_block_from_the_aligned_proof(self) -> None:
        # Matches get_mamba_prefill_checkpoint_position. An exact multiple
        # lands on the same boundary with or without the EAGLE drop.
        self.assertEqual(checkpoint_boundary(28 * 64, 64, drop_eagle=True), 27 * 64)
        tokens = 28 * 64 + 10
        self.assertEqual(checkpoint_boundary(tokens, 64, drop_eagle=False), 28 * 64)
        self.assertEqual(checkpoint_boundary(tokens, 64, drop_eagle=True), 27 * 64)

    def test_pool_covers_kv_and_per_block_gdn(self) -> None:
        layout = OrnithTextLayout()
        block = layout.scaled_block_size()
        tokens = 28 * block
        one = upstream_pool_bytes(layout, tokens, block, conversations=1)
        kv = full_attn_bytes(layout, checkpoint_boundary(tokens, block))
        self.assertGreater(one, kv)
        self.assertGreater(
            upstream_pool_bytes(layout, tokens, block, conversations=3),
            one,
        )


class ResumeTest(unittest.TestCase):
    def test_both_pieces_skip_recompute(self) -> None:
        self.assertEqual(resume_tokens(True, True, 29376, 30464), 29376)
        self.assertEqual(prefill_tokens(True, True, 29376, 30464), 1088)

    def test_either_piece_missing_recomputes(self) -> None:
        self.assertEqual(resume_tokens(True, False, 29376, 30464), 0)
        self.assertEqual(resume_tokens(False, True, 29376, 30464), 0)
        self.assertEqual(resume_tokens(True, True, 0, 30464), 0)
        self.assertEqual(prefill_tokens(True, False, 29376, 30464), 30464)

    def test_required_kinds(self) -> None:
        self.assertEqual(missing_kinds({"full_attention"}), ("gdn",))
        self.assertEqual(missing_kinds({"gdn"}), ("full_attention",))
        self.assertEqual(missing_kinds({"full_attention", "gdn", "mamba"}), ())


class ParkTest(unittest.TestCase):
    def test_lookup_requires_both_pieces(self) -> None:
        park = Park(slots=4, budget_bytes=10_000_000)
        with self.assertRaises(ValueError):
            park.store("A", 1000, kv_bytes=100, gdn_bytes=0)
        park.store("A", 1000, kv_bytes=400, gdn_bytes=50)
        hit = park.lookup("A", 1200)
        assert hit is not None
        self.assertEqual(hit.resume, 1000)
        self.assertEqual(hit.prefill, 200)
        self.assertIsNone(park.lookup("missing", 1200))

    def test_four_slots_evict_oldest(self) -> None:
        park = Park(slots=4, budget_bytes=10_000_000)
        for name in ("A", "B", "C", "D", "E"):
            park.store(name, 100, kv_bytes=10, gdn_bytes=1)
        self.assertIsNone(park.lookup("A", 100))
        hit = park.lookup("B", 100)
        assert hit is not None
        self.assertEqual(hit.resume, 100)
        self.assertEqual(len(park), 4)

    def test_byte_budget_evicts_until_fit(self) -> None:
        park = Park(slots=4, budget_bytes=150)
        park.store("A", 10, kv_bytes=80, gdn_bytes=20)
        park.store("B", 10, kv_bytes=80, gdn_bytes=20)
        self.assertIsNone(park.lookup("A", 10))
        self.assertIsNotNone(park.lookup("B", 10))

    def test_restoring_refreshes_lru(self) -> None:
        park = Park(slots=2, budget_bytes=10_000)
        park.store("A", 10, kv_bytes=10, gdn_bytes=1)
        park.store("B", 10, kv_bytes=10, gdn_bytes=1)
        self.assertIsNotNone(park.lookup("A", 10))
        park.store("C", 10, kv_bytes=10, gdn_bytes=1)
        self.assertIsNone(park.lookup("B", 10))
        self.assertIsNotNone(park.lookup("A", 10))

    def test_snapshot_larger_than_park_is_refused(self) -> None:
        park = Park(slots=4, budget_bytes=100)
        with self.assertRaises(ValueError):
            park.store("A", 10, kv_bytes=90, gdn_bytes=20)


if __name__ == "__main__":
    unittest.main()
