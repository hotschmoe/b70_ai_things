"""Whole-conversation eviction for the CPU offload pool.

The upstream policy frees individual blocks. A hybrid conversation is
invalid if its GDN page is gone and its attention blocks remain, so
eviction takes every evictable chunk of a conversation at once.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class ConversationCachePolicy(LRUCachePolicy):
    """LRU of conversations, with per-chunk LRU for keys no conv owns."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self._conv_of: dict[OffloadKey, bytes] = {}
        self._conv_keys: dict[bytes, set[OffloadKey]] = {}
        self._conv_lru: OrderedDict[bytes, None] = OrderedDict()

    def unbind_conv(self, conv_id: bytes) -> None:
        """Forget a conversation whose tail store did not allocate."""
        for key in list(self._conv_keys.get(conv_id, ())):
            self._unbind_key(key)

    def bind(self, conv_id: bytes, keys: Iterable[OffloadKey]) -> None:
        """Remember which conversation a key belongs to. Does not insert it."""
        owned = self._conv_keys.setdefault(conv_id, set())
        for key in keys:
            previous = self._conv_of.get(key)
            if previous is not None and previous != conv_id:
                self._conv_keys.get(previous, set()).discard(key)
            self._conv_of[key] = conv_id
            owned.add(key)
        self._conv_lru[conv_id] = None
        self._conv_lru.move_to_end(conv_id)

    def take_evictable(
        self, keys: Iterable[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]]:
        """Remove idle chunks. Leave pinned, pending, and unknown keys."""
        taken: list[tuple[OffloadKey, ChunkStatus]] = []
        for key in keys:
            chunk = self.chunks.get(key)
            if chunk is None or chunk.ref_cnt != 0 or key not in self._evictable:
                continue
            self._evictable.discard(key)
            del self.chunks[key]
            self._ranks.pop(key, None)
            self._unbind_key(key)
            taken.append((key, chunk))
        return taken

    @override
    def remove(self, key: OffloadKey) -> None:
        super().remove(key)
        self._unbind_key(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        super().touch(keys, req_context)
        seen: set[bytes] = set()
        for key in keys:
            conv_id = self._conv_of.get(key)
            if conv_id is None or conv_id in seen:
                continue
            seen.add(conv_id)
            if conv_id in self._conv_lru:
                self._conv_lru.move_to_end(conv_id)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        selected: list[tuple[OffloadKey, ChunkStatus]] = []
        chosen: set[OffloadKey] = set()
        for conv_id in list(self._conv_lru):
            if len(selected) >= n:
                break
            ready, blocked = self._conv_ready(conv_id, protected, chosen)
            if blocked or not ready:
                continue
            selected.extend(ready)
            chosen.update(key for key, _chunk in ready)
        if len(selected) < n:
            extra = self._take_unbound(n - len(selected), protected, chosen)
            if extra is None:
                return None
            selected.extend(extra)
        if len(selected) < n:
            return None
        for key, _chunk in selected:
            self._evictable.discard(key)
            self.chunks.pop(key, None)
            self._ranks.pop(key, None)
            self._unbind_key(key)
        self._maybe_compact_heap()
        return selected

    def _conv_ready(
        self,
        conv_id: bytes,
        protected: set[OffloadKey],
        chosen: set[OffloadKey],
    ) -> tuple[list[tuple[OffloadKey, ChunkStatus]], bool]:
        ready: list[tuple[OffloadKey, ChunkStatus]] = []
        for key in self._conv_keys.get(conv_id, ()):
            if key in chosen:
                continue
            if key in protected:
                return [], True
            chunk = self.chunks.get(key)
            if chunk is None:
                continue
            if chunk.ref_cnt != 0 or key not in self._evictable:
                return [], True
            ready.append((key, chunk))
        return ready, False

    def _take_unbound(
        self,
        n: int,
        protected: set[OffloadKey],
        chosen: set[OffloadKey],
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        """Idle chunks that are not part of a conversation.

        Heap entries that are not taken are pushed back. A shortfall pushes
        the taken entries back too, so a failed evict changes nothing.
        """
        picked: list[tuple[tuple[int, OffloadKey], ChunkStatus]] = []
        skipped: list[tuple[int, OffloadKey]] = []
        while self._heap and len(picked) < n:
            entry = heapq.heappop(self._heap)
            if not self._is_current(entry):
                continue
            key = entry[1]
            chunk = self.chunks[key]
            if (
                key in chosen
                or key in self._conv_of
                or key in protected
                or chunk.ref_cnt != 0
            ):
                skipped.append(entry)
                continue
            picked.append((entry, chunk))
            chosen.add(key)
        if len(picked) < n:
            for entry, _chunk in picked:
                skipped.append(entry)
                chosen.discard(entry[1])
            for entry in skipped:
                heapq.heappush(self._heap, entry)
            return None
        for entry in skipped:
            heapq.heappush(self._heap, entry)
        return [(entry[1], chunk) for entry, chunk in picked]

    def _unbind_key(self, key: OffloadKey) -> None:
        conv_id = self._conv_of.pop(key, None)
        if conv_id is None:
            return
        owned = self._conv_keys.get(conv_id)
        if owned is None:
            return
        owned.discard(key)
        if not owned:
            self._conv_keys.pop(conv_id, None)
            self._conv_lru.pop(conv_id, None)
