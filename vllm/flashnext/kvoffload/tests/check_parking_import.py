"""Import the parking connector and exercise whole-conversation eviction.

Runs inside the vLLM image. Host CPython does not have the fork.
"""

from __future__ import annotations

from kvoffload.parking import ParkingOffloadConnector, ParkingScheduler
from kvoffload.policy import ConversationCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus


def _ready(chunk_id: int) -> ChunkStatus:
    chunk = ChunkStatus(chunk_id)
    chunk.ref_cnt = 0
    return chunk


def _check_policy() -> None:
    policy = ConversationCachePolicy(16)
    for index, key in enumerate((b"a0", b"a1", b"a2")):
        policy.insert(key, _ready(index))
    policy.bind(b"A", (b"a0", b"a1", b"a2"))
    for index, key in enumerate((b"b0", b"b1"), start=3):
        policy.insert(key, _ready(index))
    policy.bind(b"B", (b"b0", b"b1"))
    policy.insert(b"u0", _ready(5))

    taken = policy.evict(1, set())
    if taken is None or {key for key, _chunk in taken} != {b"a0", b"a1", b"a2"}:
        raise SystemExit(f"conv eviction returned {taken}")

    blocked = policy.evict(4, {b"b0", b"b1", b"u0"})
    if blocked is not None:
        raise SystemExit(f"protected evict returned {blocked}")
    if policy.get(b"b0") is None or policy.get(b"u0") is None:
        raise SystemExit("failed evict removed a chunk")

    pending = _ready(9)
    pending.ref_cnt = -1
    policy.insert(b"p0", pending)
    policy.bind(b"P", (b"p0",))
    if policy.take_evictable((b"p0", b"missing")):
        raise SystemExit("take_evictable removed a pending or missing key")
    policy.unbind_conv(b"B")
    if b"b0" in policy._conv_of:
        raise SystemExit("unbind_conv left a key")


def main() -> None:
    _check_policy()
    if ParkingOffloadConnector.__name__ != "ParkingOffloadConnector":
        raise SystemExit("connector class missing")
    if not issubclass(ParkingScheduler, object):
        raise SystemExit("scheduler class missing")
    print("IMPORT_OK")


if __name__ == "__main__":
    main()
