"""Exact-tail conversation index. No vLLM import, so T0 runs on host Python.

A parked conversation is the full-block chain plus one tail record at the
exact end position E. A hit requires the tail token ids to match and the
next prompt to extend past E (E < prompt length). That is Strata's
``n < prompt.size()`` rule.

A hit that covers the whole new prompt is refused. vLLM recomputes the
last token after a full remote load (scheduler backoff), and feeding that
token into a GDN state that already includes it double-applies it.

Eviction drops a whole conversation. Pinned records and running
conversations stay. Superseding a conversation keeps full-block keys that
are still in the new chain and releases the rest.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field


# 2.5 GiB, the Strata MemAvailable floor (spec 5.4).
HOST_RESERVE_BYTES = 5 << 29

TailKey = tuple[bytes | None, tuple[int, ...]]


def host_has_reserve(mem_available: int, reserve: int = HOST_RESERVE_BYTES) -> bool:
    """True when the host can admit another park without crossing the floor."""
    if mem_available < 0 or reserve < 0:
        raise ValueError("memory counts must be non-negative")
    return mem_available >= reserve


def replay_tokens(prev_total: int, cache_n: int) -> int:
    """Strata replay: max(0, prev_total - 1 - cache_n).

    prev_total is the previous turn's prompt plus sampled reply. cache_n is
    the parked end E. A correct park has E = prev_total - 1, so replay is 0.
    """
    if prev_total < 0 or cache_n < 0:
        raise ValueError("token counts must be non-negative")
    return max(0, prev_total - 1 - cache_n)


def turn2_accounting(p1: int, r1: int, u2: int) -> dict[str, int]:
    """Token accounting for a real second turn.

    E = len(P1) + len(R1) - 1, because the last sampled token is never fed.
    The turn-2 prompt is P1 + R1 + U2, which is E + 1 + len(U2) long. The
    engine restores E and computes len(U2) + 1. The spec's wording
    ``prompt - cached == len(U2)`` is one token low; replay is still 0.
    """
    if min(p1, r1, u2) < 0:
        raise ValueError("token counts must be non-negative")
    if r1 < 1:
        raise ValueError("a reply has at least one sampled token")
    end = p1 + r1 - 1
    prompt = p1 + r1 + u2
    return {
        "end": end,
        "prompt": prompt,
        "host": end,
        "computed": prompt - end,
        "replay": replay_tokens(p1 + r1, end),
        "u2": u2,
    }


@dataclass
class TailRecord:
    key: TailKey
    conv_id: bytes
    end: int
    full_keys: tuple[bytes, ...]
    tail_only_keys: tuple[bytes, ...]
    tail_slots: dict[int, int]
    nbytes: int
    pinned: int = 0

    @property
    def parent_hash(self) -> bytes | None:
        return self.key[0]

    @property
    def tail_ids(self) -> tuple[int, ...]:
        return self.key[1]


@dataclass
class ConversationEntry:
    conv_id: bytes
    records: list[TailRecord] = field(default_factory=list)
    full_keys: set[bytes] = field(default_factory=set)
    nbytes: int = 0

    @property
    def pinned(self) -> bool:
        return any(record.pinned > 0 for record in self.records)


@dataclass(frozen=True)
class StoreResult:
    record: TailRecord
    evicted: tuple[bytes, ...]
    released_keys: tuple[bytes, ...]


class ParkIndex:
    """LRU of at most ``slots`` conversations under a byte budget."""

    def __init__(
        self,
        slots: int = 4,
        budget_bytes: int = 32 << 30,
        block_size: int = 1088,
    ) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.slots = slots
        self.budget_bytes = budget_bytes
        self.block_size = block_size
        self.by_tail: dict[TailKey, TailRecord] = {}
        self.by_parent: dict[bytes | None, list[TailRecord]] = {}
        self.by_conv: OrderedDict[bytes, ConversationEntry] = OrderedDict()
        self._running: set[bytes] = set()

    def __len__(self) -> int:
        return len(self.by_conv)

    def used_bytes(self) -> int:
        return sum(entry.nbytes for entry in self.by_conv.values())

    def set_running(self, conv_id: bytes, running: bool) -> None:
        if running:
            self._running.add(conv_id)
        else:
            self._running.discard(conv_id)

    def pin(self, record: TailRecord) -> None:
        record.pinned += 1

    def unpin(self, record: TailRecord) -> None:
        if record.pinned <= 0:
            raise ValueError("tail record is not pinned")
        record.pinned -= 1

    def keys_of(self, conv_id: bytes) -> frozenset[bytes]:
        entry = self.by_conv.get(conv_id)
        if entry is None:
            return frozenset()
        return frozenset(entry.full_keys)

    def store(
        self,
        *,
        conv_id: bytes,
        parent_hash: bytes | None,
        tail_ids: tuple[int, ...],
        end: int,
        full_keys: tuple[bytes, ...],
        nbytes: int,
        tail_only_keys: tuple[bytes, ...] = (),
        tail_slots: dict[int, int] | None = None,
    ) -> StoreResult:
        """Insert the exact tail. Supersedes an older tail of the same conv.

        ``nbytes`` is the conversation's total after this store (full chain
        plus this tail). ``full_keys`` is the chain that must stay. Keys the
        old record held and this chain does not are released.
        """
        self._check_tail(parent_hash, tail_ids, end, full_keys, nbytes)
        key: TailKey = (parent_hash, tail_ids)
        existing = self.by_tail.get(key)
        if existing is not None and existing.conv_id != conv_id:
            raise ValueError("tail key already belongs to another conversation")

        is_new = conv_id not in self.by_conv
        evicted, released = self._evict_until(conv_id, nbytes, is_new)
        entry = self.by_conv.get(conv_id)
        if entry is None:
            entry = ConversationEntry(conv_id=conv_id)
            self.by_conv[conv_id] = entry
        else:
            self.by_conv.move_to_end(conv_id)

        new_keys = set(full_keys) | set(tail_only_keys)
        superseded = [key for key in entry.full_keys if key not in new_keys]
        released.extend(superseded)
        for old in list(entry.records):
            self._unlink_record(old)

        record = TailRecord(
            key=key,
            conv_id=conv_id,
            end=end,
            full_keys=full_keys,
            tail_only_keys=tail_only_keys,
            tail_slots=dict(tail_slots or {}),
            nbytes=nbytes,
        )
        entry.records.insert(0, record)
        entry.full_keys = new_keys
        entry.nbytes = nbytes
        self.by_tail[key] = record
        self.by_parent.setdefault(parent_hash, []).append(record)
        return StoreResult(record, tuple(evicted), tuple(released))

    def lookup_key(
        self,
        parent_hash: bytes | None,
        tail_ids: tuple[int, ...],
        prompt_len: int,
    ) -> TailRecord | None:
        """Exact key hit. Refuses a tail that does not sit strictly inside."""
        record = self.by_tail.get((parent_hash, tail_ids))
        if record is None or record.end <= 0 or record.end >= prompt_len:
            return None
        self.by_conv.move_to_end(record.conv_id)
        return record

    def find(
        self,
        token_ids: Sequence[int],
        block_hashes: Sequence[bytes],
        prompt_len: int,
    ) -> TailRecord | None:
        """Longest exact tail that is a strict prefix of ``token_ids``.

        ``block_hashes[i]`` covers tokens ``[i * B, (i + 1) * B)``. The tail
        after k full blocks uses ``block_hashes[k - 1]`` (or None when k is 0).
        """
        if prompt_len < 0:
            raise ValueError("prompt_len must be non-negative")
        if prompt_len > len(token_ids):
            raise ValueError("prompt_len cannot pass token_ids")
        best: TailRecord | None = None
        last_k = min(len(block_hashes), prompt_len // self.block_size)
        for k in range(last_k, -1, -1):
            parent = None if k == 0 else block_hashes[k - 1]
            prefix = k * self.block_size
            for record in self.by_parent.get(parent, ()):
                tail = record.tail_ids
                end = prefix + len(tail)
                if record.end != end or end >= prompt_len or end <= 0:
                    continue
                if tuple(token_ids[prefix:end]) != tail:
                    continue
                if best is None or record.end > best.end:
                    best = record
            if best is not None and best.end >= prefix:
                break
        if best is None:
            return None
        self.by_conv.move_to_end(best.conv_id)
        return best

    def _check_tail(
        self,
        parent_hash: bytes | None,
        tail_ids: tuple[int, ...],
        end: int,
        full_keys: tuple[bytes, ...],
        nbytes: int,
    ) -> None:
        if not isinstance(tail_ids, tuple):
            raise ValueError("tail_ids must be a tuple")
        if len(tail_ids) >= self.block_size:
            raise ValueError("tail_ids must be shorter than a block")
        if end <= 0 or nbytes <= 0:
            raise ValueError("end and nbytes must be positive")
        if nbytes > self.budget_bytes:
            raise ValueError(
                f"conversation {nbytes} bytes exceeds park budget {self.budget_bytes}"
            )
        prefix = end - len(tail_ids)
        if prefix < 0 or prefix % self.block_size != 0:
            raise ValueError("tail must start on a block boundary")
        if parent_hash is None and prefix != 0:
            raise ValueError("a tail with no parent hash starts at token 0")
        if parent_hash is not None and prefix == 0:
            raise ValueError("a block-aligned parent requires a parent hash")
        if prefix and not full_keys:
            raise ValueError("a non-empty prefix needs its full-block keys")

    def _evict_until(
        self,
        incoming: bytes,
        incoming_bytes: int,
        is_new: bool,
    ) -> tuple[list[bytes], list[bytes]]:
        """Plan every victim first, then drop. A refusal leaves the index."""
        victims = self._plan_victims(incoming, incoming_bytes, is_new)
        released: list[bytes] = []
        for conv_id, _nbytes in victims:
            released.extend(self._drop_conv(conv_id))
        return [conv_id for conv_id, _nbytes in victims], released

    def _plan_victims(
        self,
        incoming: bytes,
        incoming_bytes: int,
        is_new: bool,
    ) -> list[tuple[bytes, int]]:
        conv_count = len(self.by_conv) + (1 if is_new else 0)
        others = self.used_bytes()
        if not is_new:
            current = self.by_conv.get(incoming)
            if current is not None:
                others -= current.nbytes
        chosen: list[tuple[bytes, int]] = []
        chosen_ids: set[bytes] = set()
        while conv_count > self.slots or others + incoming_bytes > self.budget_bytes:
            victim: tuple[bytes, int] | None = None
            for conv_id, entry in self.by_conv.items():
                if (
                    conv_id == incoming
                    or conv_id in chosen_ids
                    or conv_id in self._running
                    or entry.pinned
                ):
                    continue
                victim = (conv_id, entry.nbytes)
                break
            if victim is None:
                raise ValueError(
                    "park cannot fit this conversation without evicting "
                    "a pinned or running one"
                )
            chosen.append(victim)
            chosen_ids.add(victim[0])
            conv_count -= 1
            others -= victim[1]
        return chosen

    def _drop_conv(self, conv_id: bytes) -> list[bytes]:
        entry = self.by_conv.pop(conv_id)
        for record in entry.records:
            self._unlink_record(record)
        self._running.discard(conv_id)
        return list(entry.full_keys)

    def _unlink_record(self, record: TailRecord) -> None:
        self.by_tail.pop(record.key, None)
        bucket = self.by_parent.get(record.parent_hash)
        if not bucket:
            return
        self.by_parent[record.parent_hash] = [
            item for item in bucket if item is not record
        ]
        if not self.by_parent[record.parent_hash]:
            del self.by_parent[record.parent_hash]
