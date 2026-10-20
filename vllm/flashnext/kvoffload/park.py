"""Host-RAM park for one hybrid conversation prefix.

A slot is valid only when it holds both pieces:

- full-attention KV for tokens ``[0, boundary)``
- the GDN conv + recurrent snapshot at ``boundary``

Missing either piece is a miss. The caller recomputes the prompt.
That is the Strata conversation-cache rule (KV plus GDN state, replay 0)
and the reason a full-attention hit without the recurrent boundary
cannot be resumed (vLLM issue 52773).

The live server copies bytes with upstream OffloadingConnector. This
class is the contract those copies have to satisfy, and the 4-slot
budget the Flash-Next goal asks for (32 GiB / 4 conversations on the
Strata daily driver; the Ornith proof uses the same slot rule).
"""

from __future__ import annotations

from dataclasses import dataclass


REQUIRED_KINDS = ("full_attention", "gdn")


def missing_kinds(present: set[str]) -> tuple[str, ...]:
    return tuple(kind for kind in REQUIRED_KINDS if kind not in present)


def resume_tokens(
    has_full_attn: bool,
    has_gdn: bool,
    boundary: int,
    num_tokens: int,
) -> int:
    """Tokens that can be restored with no recompute.

    Returns 0 unless both pieces exist and the boundary sits inside the
    prompt. The tail ``num_tokens - boundary`` is still prefilled.
    """
    if num_tokens < 0 or boundary < 0:
        raise ValueError("token counts must be non-negative")
    if boundary > num_tokens:
        raise ValueError("boundary cannot pass the prompt")
    if boundary == 0 or not has_full_attn or not has_gdn:
        return 0
    return boundary


def prefill_tokens(
    has_full_attn: bool,
    has_gdn: bool,
    boundary: int,
    num_tokens: int,
) -> int:
    return num_tokens - resume_tokens(has_full_attn, has_gdn, boundary, num_tokens)


@dataclass(frozen=True)
class Slot:
    key: str
    boundary: int
    kv_bytes: int
    gdn_bytes: int

    @property
    def nbytes(self) -> int:
        return self.kv_bytes + self.gdn_bytes

    @property
    def complete(self) -> bool:
        return self.kv_bytes > 0 and self.gdn_bytes > 0


@dataclass(frozen=True)
class Hit:
    key: str
    boundary: int
    resume: int
    prefill: int
    nbytes: int


class Park:
    """LRU of at most ``slots`` complete snapshots, under a byte budget."""

    def __init__(self, slots: int = 4, budget_bytes: int = 32 << 30) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.slots = slots
        self.budget_bytes = budget_bytes
        self._order: list[str] = []
        self._entries: dict[str, Slot] = {}

    def __len__(self) -> int:
        return len(self._order)

    def used_bytes(self) -> int:
        return sum(slot.nbytes for slot in self._entries.values())

    def store(self, key: str, boundary: int, kv_bytes: int, gdn_bytes: int) -> Slot:
        """Insert a complete snapshot, evicting LRU slots until it fits.

        Refuses a snapshot that is missing either piece or that cannot
        fit even in an empty park. Re-storing a key refreshes LRU order.
        """
        if not key:
            raise ValueError("key must be non-empty")
        if boundary <= 0:
            raise ValueError("boundary must be positive")
        if kv_bytes <= 0 or gdn_bytes <= 0:
            raise ValueError(
                "a parked conversation needs full-attention KV and GDN state"
            )
        slot = Slot(key, boundary, kv_bytes, gdn_bytes)
        if slot.nbytes > self.budget_bytes:
            raise ValueError(
                f"snapshot {slot.nbytes} bytes exceeds park budget {self.budget_bytes}"
            )
        if key in self._entries:
            self._order.remove(key)
            del self._entries[key]
        while self._order and (
            len(self._order) >= self.slots
            or self.used_bytes() + slot.nbytes > self.budget_bytes
        ):
            self._evict_oldest()
        self._entries[key] = slot
        self._order.append(key)
        return slot

    def lookup(self, key: str, num_tokens: int) -> Hit | None:
        """Return a hit only when the slot is complete. Touches LRU."""
        slot = self._entries.get(key)
        if slot is None or not slot.complete:
            return None
        resume = resume_tokens(True, True, slot.boundary, num_tokens)
        if resume == 0:
            return None
        self._order.remove(key)
        self._order.append(key)
        return Hit(
            key=key,
            boundary=slot.boundary,
            resume=resume,
            prefill=num_tokens - resume,
            nbytes=slot.nbytes,
        )

    def _evict_oldest(self) -> None:
        oldest = self._order.pop(0)
        del self._entries[oldest]
