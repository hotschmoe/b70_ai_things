"""T0: exact-tail index, conversation LRU, and the Flash-Next byte model."""

from __future__ import annotations

import unittest

from kvoffload.index import (
    HOST_RESERVE_BYTES,
    ParkIndex,
    host_has_reserve,
    replay_tokens,
    turn2_accounting,
)
from kvoffload.layout import (
    FlashNextTextLayout,
    OrnithTextLayout,
)


def _ids(n: int, salt: int = 0) -> tuple[int, ...]:
    return tuple(range(salt, salt + n))


class ByteModelTest(unittest.TestCase):
    def test_flashnext_matches_spec_table(self) -> None:
        layout = FlashNextTextLayout()
        self.assertEqual(layout.full_attention_layers, 12)
        self.assertEqual(layout.linear_attention_layers, 36)
        self.assertEqual(layout.attn_bytes_per_token(), 24_576)
        self.assertEqual(layout.mtp_bytes_per_token(), 2_048)
        # 36 * (conv 61440 + ssm 3_145_728) = 115_458_048, spec "115.5 MB".
        self.assertEqual(layout.gdn_snapshot_bytes(), 115_458_048)
        self.assertEqual(layout.ple_state_bytes(), 184_320)
        conv = layout.conversation_bytes(200_000)
        self.assertAlmostEqual(conv / 1e9, 5.44, delta=0.01)
        parked = layout.parked_with_one_checkpoint_each(4, 200_000)
        self.assertAlmostEqual(parked / 1e9, 22.2, delta=0.05)
        int8 = layout.int8_conversation_bytes(200_000)
        self.assertAlmostEqual(int8 / 1e9, 2.82, delta=0.01)
        parked_int8 = layout.parked_with_one_checkpoint_each(4, 200_000, int8=True)
        self.assertAlmostEqual(parked_int8 / 1e9, 11.7, delta=0.05)

    def test_ornith_dtypes(self) -> None:
        fp16 = OrnithTextLayout()
        bf16 = OrnithTextLayout(kv_dtype="bfloat16", conv_dtype="bfloat16")
        int8 = OrnithTextLayout(kv_dtype="int8")
        self.assertEqual(fp16.attn_bytes_per_token(), 20_480)
        self.assertEqual(bf16.attn_bytes_per_token(), fp16.attn_bytes_per_token())
        self.assertEqual(int8.attn_bytes_per_token(), 10_240)
        self.assertEqual(int8.gdn_snapshot_bytes(), fp16.gdn_snapshot_bytes())
        self.assertEqual(fp16.gdn_state_bytes_per_layer(), 2_146_304)


class AccountTest(unittest.TestCase):
    def test_turn2_replays_zero_and_computes_the_unfed_token(self) -> None:
        row = turn2_accounting(p1=1000, r1=64, u2=20)
        self.assertEqual(row["end"], 1063)
        self.assertEqual(row["host"], 1063)
        self.assertEqual(row["computed"], 21)
        self.assertEqual(row["replay"], 0)
        self.assertEqual(replay_tokens(1064, 1063), 0)

    def test_host_reserve_is_two_and_a_half_gib(self) -> None:
        self.assertEqual(HOST_RESERVE_BYTES, 2_684_354_560)
        self.assertFalse(host_has_reserve(HOST_RESERVE_BYTES - 1))
        self.assertTrue(host_has_reserve(HOST_RESERVE_BYTES))


class ParkIndexTest(unittest.TestCase):
    def _store(self, index: ParkIndex, name: bytes, end: int, nbytes: int = 100):
        block = index.block_size
        nfull = end // block
        tail_len = end - nfull * block
        parent = None if nfull == 0 else f"{name.decode()}-h{nfull - 1}".encode()
        full = tuple(f"{name.decode()}-h{i}".encode() for i in range(nfull))
        tail_key = f"{name.decode()}-tail-{end}".encode()
        return index.store(
            conv_id=name,
            parent_hash=parent,
            tail_ids=_ids(tail_len, salt=end),
            end=end,
            full_keys=full,
            tail_only_keys=(tail_key,),
            nbytes=nbytes,
            tail_slots={0: 1, 1: 2},
        )

    def test_exact_hit_and_full_prompt_refusal(self) -> None:
        index = ParkIndex(slots=4, budget_bytes=10_000, block_size=16)
        stored = self._store(index, b"A", end=40)
        tail = stored.record.tail_ids
        parent = stored.record.parent_hash
        self.assertIsNotNone(index.lookup_key(parent, tail, prompt_len=41))
        self.assertIsNone(index.lookup_key(parent, tail, prompt_len=40))
        self.assertIsNone(index.lookup_key(parent, tail, prompt_len=39))

    def test_find_longest_tail_on_a_longer_prompt(self) -> None:
        index = ParkIndex(slots=4, budget_bytes=10_000, block_size=16)
        stored = self._store(index, b"A", end=40)
        ids = [0] * 32 + list(stored.record.tail_ids) + [9, 9, 9]
        hashes = [f"A-h{i}".encode() for i in range(2)]
        hit = index.find(ids, hashes, prompt_len=len(ids))
        assert hit is not None
        self.assertEqual(hit.end, 40)
        ids[32] = 999
        self.assertIsNone(index.find(ids, hashes, prompt_len=len(ids)))

    def test_supersede_keeps_the_prefix_and_drops_the_old_tail(self) -> None:
        index = ParkIndex(slots=4, budget_bytes=10_000, block_size=16)
        first = self._store(index, b"A", end=40, nbytes=100)
        second = self._store(index, b"A", end=56, nbytes=140)
        self.assertIsNone(
            index.lookup_key(first.record.parent_hash, first.record.tail_ids, 80)
        )
        self.assertIsNotNone(
            index.lookup_key(second.record.parent_hash, second.record.tail_ids, 80)
        )
        self.assertEqual(len(index), 1)
        self.assertIn(b"A-h0", index.keys_of(b"A"))
        self.assertIn(b"A-h1", index.keys_of(b"A"))
        self.assertIn(b"A-h2", index.keys_of(b"A"))
        self.assertIn(b"A-tail-40", second.released_keys)
        self.assertNotIn(b"A-tail-40", index.keys_of(b"A"))
        self.assertIn(b"A-tail-56", index.keys_of(b"A"))
        self.assertEqual(index.used_bytes(), 140)

    def test_conversation_lru_evicts_the_oldest_whole_conv(self) -> None:
        index = ParkIndex(slots=4, budget_bytes=10_000, block_size=16)
        stored = {}
        for name in (b"A", b"B", b"C", b"D"):
            stored[name] = self._store(index, name, end=20, nbytes=10)
        fifth = self._store(index, b"E", end=20, nbytes=10)
        self.assertEqual(fifth.evicted, (b"A",))
        self.assertIsNone(
            index.lookup_key(
                stored[b"A"].record.parent_hash,
                stored[b"A"].record.tail_ids,
                30,
            )
        )
        self.assertIn(b"A-h0", fifth.released_keys)
        self.assertEqual(len(index), 4)
        self.assertIsNotNone(
            index.lookup_key(
                stored[b"B"].record.parent_hash,
                stored[b"B"].record.tail_ids,
                30,
            )
        )

    def test_lookup_refreshes_lru(self) -> None:
        index = ParkIndex(slots=2, budget_bytes=10_000, block_size=16)
        a = self._store(index, b"A", end=20)
        b = self._store(index, b"B", end=20)
        self.assertIsNotNone(
            index.lookup_key(a.record.parent_hash, a.record.tail_ids, 30)
        )
        evicted = self._store(index, b"C", end=20)
        self.assertEqual(evicted.evicted, (b"B",))
        self.assertIsNone(
            index.lookup_key(b.record.parent_hash, b.record.tail_ids, 30)
        )

    def test_pin_and_running_are_not_evicted(self) -> None:
        index = ParkIndex(slots=2, budget_bytes=10_000, block_size=16)
        a = self._store(index, b"A", end=20)
        self._store(index, b"B", end=20)
        index.pin(a.record)
        index.set_running(b"B", True)
        with self.assertRaises(ValueError):
            self._store(index, b"C", end=20)
        index.unpin(a.record)
        moved = self._store(index, b"C", end=20)
        self.assertEqual(moved.evicted, (b"A",))
        self.assertIsNotNone(
            index.lookup_key(
                index.by_conv[b"B"].records[0].parent_hash,
                index.by_conv[b"B"].records[0].tail_ids,
                30,
            )
        )

    def test_byte_budget_drops_a_whole_conversation(self) -> None:
        index = ParkIndex(slots=4, budget_bytes=250, block_size=16)
        self._store(index, b"A", end=20, nbytes=100)
        self._store(index, b"B", end=20, nbytes=100)
        third = self._store(index, b"C", end=20, nbytes=100)
        self.assertEqual(third.evicted, (b"A",))
        self.assertEqual(index.used_bytes(), 200)

    def test_snapshot_larger_than_the_park_is_refused(self) -> None:
        index = ParkIndex(slots=4, budget_bytes=100, block_size=16)
        with self.assertRaises(ValueError):
            self._store(index, b"A", end=20, nbytes=101)


if __name__ == "__main__":
    unittest.main()
