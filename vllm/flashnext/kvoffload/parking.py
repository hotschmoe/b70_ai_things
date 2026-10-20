"""Exact-tail parking on top of OffloadingConnector.

Full blocks stay on the upstream hash path. One tail record per finished
request holds the GDN state and the partial attention page at
E = num_computed_tokens (prompt + reply - 1). The next turn hits only
when that tail is a strict prefix, so the engine does not recompute the
last token on top of a state that already includes it.

offload_prompt_only is forced off. Host eviction drops a whole
conversation. The GPU blocks of a tail are pinned until the store
finishes; the connector does not delay-free the request, because this
worker never emits finished_sending for stores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    make_offload_key,
)
from vllm.v1.request import Request

from kvoffload.index import HOST_RESERVE_BYTES, ParkIndex, host_has_reserve

logger = init_logger("vllm.kvoffload.parking")


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _kind(spec: object, eagle: bool) -> str:
    if isinstance(spec, FullAttentionSpec):
        return "eagle" if eagle else "full_attention"
    if isinstance(spec, MambaSpec):
        label = getattr(spec.mamba_type, "name", str(spec.mamba_type))
        if "GDN" in str(label):
            return "gdn"
        return "mamba"
    return type(spec).__name__


def _tail_hash(conv_id: bytes, end: int, group_idx: int, kind: str) -> bytes:
    return (
        b"G4T"
        + kind.encode("ascii")
        + end.to_bytes(8, "big")
        + group_idx.to_bytes(2, "big")
        + conv_id
    )


@dataclass
class _GroupPlan:
    group_idx: int
    kind: str
    keys: tuple[bytes, ...]


@dataclass
class _Pending:
    req_id: str
    conv_id: bytes
    parent_hash: bytes | None
    tail_ids: tuple[int, ...]
    end: int
    full_keys: tuple[bytes, ...]
    tail_keys: tuple[bytes, ...]
    plans: tuple[_GroupPlan, ...]
    src_block_ids: tuple[int, ...]
    group_sizes: tuple[int, ...]
    block_indices: tuple[int, ...]
    pin_blocks: tuple[object, ...]


def _force_extra(vllm_config: VllmConfig) -> dict:
    transfer = vllm_config.kv_transfer_config
    if transfer is None:
        raise RuntimeError("ParkingOffloadConnector requires kv_transfer_config")
    extra = dict(transfer.kv_connector_extra_config or {})
    extra["offload_prompt_only"] = False
    extra["eviction_policy"] = "ConversationCachePolicy"
    extra["cache_policy_module_path"] = "kvoffload.policy"
    extra.setdefault("slots", 4)
    # Default stays the upstream swap path. The measured T2 used that path.
    if os.environ.get("B70_KVPARK_COPY", "0") == "1":
        extra["spec_name"] = "ParkingOffloadingSpec"
        extra["spec_module_path"] = "kvoffload.copy_spec"
    transfer.kv_connector_extra_config = extra
    return extra


class ParkingScheduler(OffloadingConnectorScheduler):
    """OffloadingConnectorScheduler plus an exact-tail index."""

    def init_parking(self, slots: int, block_size: int) -> None:
        chunks = int(getattr(self.manager, "_num_chunks", 0) or 0)
        if chunks <= 0:
            raise RuntimeError(
                "CPU offload pool has no chunks. Set cpu_bytes_to_use "
                "above one KV block, and --shm-size above that."
            )
        self.park = ParkIndex(
            slots=slots,
            budget_bytes=chunks,
            block_size=block_size,
        )
        self.block_size = block_size
        self.exact_tail = all(
            group.hashes_per_chunk == 1
            and group.tokens_per_chunk == group.tokens_per_block
            for group in self.config.kv_group_configs
        )
        self.gpu_pool = None
        self._pending: list[_Pending] = []
        self._plans: dict[tuple, tuple[_GroupPlan, ...]] = {}
        self._pin_holds: dict[int, tuple[object, ...]] = {}
        self._running: dict[bytes, int] = {}
        self._req_conv: dict[str, bytes] = {}
        # RequestOffloadState is a slots dataclass, so the chosen tail
        # cannot be stored on it.
        self._tail_choice: dict[str, tuple[object, int]] = {}
        kinds = [
            (
                group.group_idx,
                _kind(group.kv_cache_spec, group.is_eagle_group),
                group.tokens_per_block,
            )
            for group in self.config.kv_group_configs
        ]
        logger.info(
            "G4 parking: slots=%s chunks=%s block=%s exact_tail=%s groups=%s "
            "offload_prompt_only=%s",
            slots,
            chunks,
            block_size,
            self.exact_tail,
            kinds,
            self.config.offload_prompt_only,
        )

    def _policy(self):
        return self.manager._policy

    def _conv_of(self, request: Request) -> bytes | None:
        hashes = request.block_hashes
        if not hashes:
            return None
        return bytes(hashes[0])

    def _note_running(self, conv_id: bytes, delta: int) -> None:
        count = self._running.get(conv_id, 0) + delta
        if count <= 0:
            self._running.pop(conv_id, None)
            self.park.set_running(conv_id, False)
        else:
            self._running[conv_id] = count
            self.park.set_running(conv_id, True)

    def on_new_request(self, request: Request) -> None:
        super().on_new_request(request)
        conv_id = self._conv_of(request)
        if conv_id is None:
            return
        self._req_conv[request.request_id] = conv_id
        self._note_running(conv_id, 1)

    def request_finished(self, request: Request):
        choice = self._tail_choice.pop(request.request_id, None)
        if choice is not None:
            self._release_tail_pin(choice[0])
        conv_id = self._req_conv.pop(request.request_id, None)
        if conv_id is not None:
            self._note_running(conv_id, -1)
        return super().request_finished(request)

    def _release_tail_pin(self, key: object) -> None:
        record = self.park.by_tail.get(key)
        if record is not None and record.pinned > 0:
            self.park.unpin(record)

    def has_pending_push_work(self) -> bool:
        if self._pending:
            return True
        return super().has_pending_push_work()

    def bind_pool(self, pool: object) -> None:
        self.gpu_pool = pool

    def register_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> None:
        if not self.exact_tail:
            return
        end = request.num_computed_tokens
        if end <= 0 or request.num_in_flight_tokens != 0:
            return
        available = _mem_available()
        if available is not None and not host_has_reserve(available):
            logger.warning(
                "G4 skip park: MemAvailable %s is under the %s floor",
                available,
                HOST_RESERVE_BYTES,
            )
            return
        pending = self._build_pending(request, block_ids, partial_tail_offloads, end)
        if pending is None:
            return
        pending.pin_blocks = tuple(self._pin(pending.src_block_ids))
        self._pending.append(pending)
        logger.info(
            "G4 queued tail req=%s end=%s tail=%s full_keys=%s",
            request.request_id,
            end,
            len(pending.tail_ids),
            len(pending.full_keys),
        )

    def _build_pending(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
        end: int,
    ) -> _Pending | None:
        block = self.block_size
        if any(group.tokens_per_block != block for group in self.config.kv_group_configs):
            logger.warning("G4 skip park: KV groups do not share block size %s", block)
            return None
        conv_id = self._conv_of(request)
        if conv_id is None:
            return None
        nfull = end // block
        if nfull > len(request.block_hashes):
            logger.warning(
                "G4 skip park: end %s needs %s block hashes, have %s",
                end,
                nfull,
                len(request.block_hashes),
            )
            return None
        prefix = nfull * block
        tail_ids = tuple(int(token) for token in request.all_token_ids[prefix:end])
        if len(tail_ids) != end - prefix:
            return None
        parent = None if nfull == 0 else bytes(request.block_hashes[nfull - 1])
        offered = {
            group_id: block_id
            for group_id, block_id, boundary in partial_tail_offloads
            if boundary == end and block_id > 0
        }
        full_keys: list[bytes] = []
        tail_keys: list[bytes] = []
        plans: list[_GroupPlan] = []
        src_ids: list[int] = []
        group_sizes: list[int] = []
        block_indices: list[int] = []
        saw_attn = False
        saw_gdn = False
        for group in self.config.kv_group_configs:
            kind = _kind(group.kv_cache_spec, group.is_eagle_group)
            if kind == "eagle":
                # The copy op wants one slot per KV group. MTP is a later
                # T2 arm: this tail does not park the draft group, and an
                # empty plan is not part of the hit check.
                logger.info("G4 eagle group %s is not part of the tail", group.group_idx)
                plans.append(_GroupPlan(group.group_idx, "attn", ()))
                group_sizes.append(0)
                block_indices.append(0)
                continue
            if kind == "full_attention":
                saw_attn = True
                keys = [
                    make_offload_key(bytes(request.block_hashes[i]), group.group_idx)
                    for i in range(nfull)
                ]
                full_keys.extend(keys)
                if end % block == 0:
                    plans.append(_GroupPlan(group.group_idx, "attn", tuple(keys)))
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                page = cdiv(end, block) - 1
                table = block_ids[group.group_idx] if group.group_idx < len(block_ids) else []
                if page >= len(table) or table[page] <= 0:
                    logger.warning(
                        "G4 skip park: attention group %s has no partial block",
                        group.group_idx,
                    )
                    return None
                tail_key = make_offload_key(
                    _tail_hash(conv_id, end, group.group_idx, "attn"),
                    group.group_idx,
                )
                tail_keys.append(tail_key)
                plans.append(
                    _GroupPlan(group.group_idx, "attn", tuple(keys) + (tail_key,))
                )
                src_ids.append(table[page])
                group_sizes.append(1)
                block_indices.append(page)
                continue
            if kind in ("gdn", "mamba"):
                saw_gdn = saw_gdn or kind == "gdn"
                block_id = offered.get(group.group_idx, 0)
                if block_id <= 0:
                    logger.warning(
                        "G4 skip park: no running state block for group %s",
                        group.group_idx,
                    )
                    return None
                tail_key = make_offload_key(
                    _tail_hash(conv_id, end, group.group_idx, kind),
                    group.group_idx,
                )
                tail_keys.append(tail_key)
                plans.append(_GroupPlan(group.group_idx, "state", (tail_key,)))
                src_ids.append(block_id)
                group_sizes.append(1)
                block_indices.append(max(0, (end - 1) // block))
                continue
            logger.warning("G4 skip park: unsupported group kind %s", kind)
            return None
        if not saw_attn or not saw_gdn or not tail_keys:
            logger.warning(
                "G4 skip park: attn=%s gdn=%s tail_keys=%s",
                saw_attn,
                saw_gdn,
                len(tail_keys),
            )
            return None
        return _Pending(
            req_id=request.request_id,
            conv_id=conv_id,
            parent_hash=parent,
            tail_ids=tail_ids,
            end=end,
            full_keys=tuple(full_keys),
            tail_keys=tuple(tail_keys),
            plans=tuple(plans),
            src_block_ids=tuple(src_ids),
            group_sizes=tuple(group_sizes),
            block_indices=tuple(block_indices),
            pin_blocks=(),
        )

    def _pin(self, block_ids: tuple[int, ...]) -> list:
        pool = self.gpu_pool
        if pool is None:
            logger.warning("G4 tail has no GPU block pool; store may race a free")
            return []
        blocks = []
        seen: set[int] = set()
        for block_id in block_ids:
            if block_id <= 0 or block_id in seen:
                continue
            seen.add(block_id)
            block = pool.blocks[block_id]
            if block.is_null:
                continue
            blocks.append(block)
        if blocks:
            pool.touch(blocks)
        return blocks

    def _release_pin(self, blocks: tuple[object, ...] | list) -> None:
        pool = self.gpu_pool
        if pool is not None and blocks:
            pool.free_blocks(blocks)

    def _pin_resident(self, keys: tuple[bytes, ...]) -> list[bytes]:
        """Hold already-stored full blocks so this tail's eviction skips them."""
        policy = self._policy()
        evictable = getattr(policy, "_evictable", None)
        if evictable is None:
            return []
        manager = self.manager
        pinned: list[bytes] = []
        for key in keys:
            chunk = policy.get(key)
            if chunk is None or chunk.ref_cnt != 0 or key not in evictable:
                continue
            policy.mark_non_evictable(key)
            manager._num_evictable_cache_chunks -= 1
            chunk.ref_cnt += 1
            pinned.append(key)
        return pinned

    def _unpin_resident(self, keys: list[bytes]) -> None:
        if not keys:
            return
        policy = self._policy()
        manager = self.manager
        for key in keys:
            chunk = policy.get(key)
            if chunk is None:
                continue
            chunk.ref_cnt -= 1
            if chunk.ref_cnt == 0:
                manager._num_evictable_cache_chunks += 1
                policy.mark_evictable(key)

    def _abort_tail(self, record) -> None:
        self._plans.pop(record.key, None)
        try:
            self.park.discard(record.key)
        except ValueError as exc:
            logger.warning("G4 could not discard tail: %s", exc)
        policy = self._policy()
        if hasattr(policy, "unbind_conv"):
            policy.unbind_conv(record.conv_id)

    def _release(self, keys: tuple[bytes, ...] | list[bytes]) -> None:
        policy = self._policy()
        if not hasattr(policy, "take_evictable"):
            return
        freed = policy.take_evictable(keys)
        if not freed:
            return
        manager = self.manager
        manager._num_evictable_cache_chunks -= len(freed)
        for _key, chunk in freed:
            manager._free_chunk(chunk)

    def _prepare_pending_tails(self) -> dict[int, TransferJob]:
        if not self._pending:
            return {}
        pending = self._pending
        self._pending = []
        jobs: dict[int, TransferJob] = {}
        for item in pending:
            job = self._prepare_one(item)
            if job is None:
                self._release_pin(item.pin_blocks)
                continue
            job_id, transfer = job
            jobs[job_id] = transfer
            self._current_batch_jobs_to_flush.add(job_id)
            self._pin_holds[job_id] = item.pin_blocks
        return jobs

    def _prepare_one(self, item: _Pending) -> tuple[int, TransferJob] | None:
        req_status = self._req_status.get(item.req_id)
        if req_status is None:
            logger.warning("G4 tail %s has no request status", item.req_id)
            return None
        if req_status.transfer_jobs:
            any_id = next(iter(req_status.transfer_jobs))
            if not self._jobs[any_id].is_store:
                logger.warning("G4 tail %s still has a load in flight", item.req_id)
                return None
        try:
            stored = self.park.store(
                conv_id=item.conv_id,
                parent_hash=item.parent_hash,
                tail_ids=item.tail_ids,
                end=item.end,
                full_keys=item.full_keys,
                tail_only_keys=item.tail_keys,
                nbytes=max(1, len(item.full_keys) + len(item.tail_keys)),
            )
        except ValueError as exc:
            logger.warning("G4 tail %s does not fit the park: %s", item.req_id, exc)
            return None
        if stored.evicted:
            logger.info(
                "G4 park evict req=%s end=%s n=%d",
                item.req_id,
                item.end,
                len(stored.evicted),
            )
        self._release(stored.released_keys)
        self._plans = {
            key: plan for key, plan in self._plans.items() if key in self.park.by_tail
        }
        self._plans[stored.record.key] = item.plans
        policy = self._policy()
        if hasattr(policy, "bind"):
            policy.bind(item.conv_id, item.full_keys + item.tail_keys)
        cpu_pins = self._pin_resident(item.full_keys)
        try:
            store_output = self.manager.prepare_store(
                item.tail_keys, req_status.req_context
            )
        except Exception:
            self._unpin_resident(cpu_pins)
            logger.exception("G4 tail %s prepare_store raised", item.req_id)
            self._abort_tail(stored.record)
            return None
        self._unpin_resident(cpu_pins)
        if store_output is None:
            logger.warning("G4 tail %s prepare_store failed", item.req_id)
            self._abort_tail(stored.record)
            return None
        if not store_output.keys_to_store:
            logger.info("G4 tail %s chunks already resident", item.req_id)
            return None
        src = GPULoadStoreSpec(
            list(item.src_block_ids),
            group_sizes=list(item.group_sizes),
            block_indices=list(item.block_indices),
        )
        job_id = self._generate_job_id()
        req_status.transfer_jobs.add(job_id)
        self._jobs[job_id] = TransferJobStatus(
            req_id=item.req_id,
            pending_count=self.config.num_workers,
            keys=set(store_output.keys_to_store),
            is_store=True,
            deferred_fence_block_ids=list(item.src_block_ids),
        )
        return job_id, TransferJob(
            req_id=item.req_id,
            src_spec=src,
            dst_spec=store_output.store_spec,
        )

    def _build_connector_meta(self, scheduler_output: SchedulerOutput):
        extra = self._prepare_pending_tails()
        meta = super()._build_connector_meta(scheduler_output)
        if extra:
            meta.store_jobs.update(extra)
        return meta

    def _update_connector_output(self, connector_output) -> None:
        super()._update_connector_output(connector_output)
        done = [job_id for job_id in self._pin_holds if job_id not in self._jobs]
        for job_id in done:
            self._unpin_job(job_id)

    def _unpin_job(self, job_id: int) -> None:
        blocks = self._pin_holds.pop(job_id, ())
        self._release_pin(blocks)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ):
        hit, async_load = super().get_num_new_matched_tokens(
            request, num_computed_tokens, max_num_new_tokens
        )
        if hit is None or not self.exact_tail:
            return hit, async_load
        prompt_len = request.num_prompt_tokens
        if prompt_len <= num_computed_tokens:
            return hit, async_load
        ids = [int(token) for token in request.all_token_ids[:prompt_len]]
        if self._waiting_on_pending(ids, prompt_len):
            return None, False
        record = self.park.find(ids, [bytes(h) for h in request.block_hashes], prompt_len)
        if record is None:
            return hit, async_load
        base = num_computed_tokens + hit
        if record.end <= base:
            return hit, async_load
        if max_num_new_tokens is not None and record.end - num_computed_tokens > max_num_new_tokens:
            return hit, async_load
        status = self._tail_status(record, request)
        if status == "pending":
            return None, False
        if status != "hit":
            return hit, async_load
        req_status = self._req_status[request.request_id]
        req_status.partial_tail_boundary = None
        req_status.update_num_hit_chunks(record.end)
        prev = self._tail_choice.get(request.request_id)
        if prev is None or prev[0] != record.key:
            if prev is not None:
                self._release_tail_pin(prev[0])
            self.park.pin(record)
        self._tail_choice[request.request_id] = (record.key, record.end)
        return record.end - num_computed_tokens, True

    def _waiting_on_pending(self, ids: list[int], prompt_len: int) -> bool:
        block = self.block_size
        for item in self._pending:
            if item.end <= 0 or item.end >= prompt_len:
                continue
            prefix = item.end - len(item.tail_ids)
            if prefix % block != 0:
                continue
            if tuple(ids[prefix:item.end]) != item.tail_ids:
                continue
            return True
        return False

    def _tail_status(self, record, request: Request) -> str:
        plans = self._plans.get(record.key)
        if not plans:
            return "miss"
        req_status = self._req_status[request.request_id]
        pending = False
        for plan in plans:
            for key in plan.keys:
                result = self.manager.lookup(key, req_status.req_context)
                if result == LookupResult.MISS:
                    return "miss"
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    pending = True
        return "pending" if pending else "hit"

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        for group_blocks in blocks.blocks:
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )
        choice = self._tail_choice.pop(request.request_id, None)
        if num_external_tokens == 0:
            if choice is not None:
                self._release_tail_pin(choice[0])
            return
        req_status = self._req_status[request.request_id]
        local = req_status.num_locally_computed_tokens
        if (
            choice is None
            or local + num_external_tokens != choice[1]
        ):
            if choice is not None:
                self._release_tail_pin(choice[0])
            return super().update_state_after_alloc(
                request, blocks, num_external_tokens
            )
        key, end = choice
        try:
            self._load_tail(request, blocks, req_status, end, key, local)
        finally:
            self._release_tail_pin(key)

    def _load_tail(self, request, blocks, req_status, end: int, key, local: int) -> None:
        plans = self._plans.get(key)
        if not plans:
            raise RuntimeError(f"G4 tail {request.request_id} lost its load plan")
        self._acquire_step()
        block = self.block_size
        if local % block != 0:
            raise RuntimeError(
                f"G4 tail load local hit {local} is not a multiple of {block}"
            )
        keys: list[bytes] = []
        dst_ids: list[int] = []
        group_sizes: list[int] = []
        block_indices: list[int] = []
        req_status.load_key_ranges.clear()
        start = local // block
        for plan in plans:
            table = blocks.blocks[plan.group_idx]
            if plan.kind == "attn":
                take = plan.keys[start:]
                if not take:
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                first = start
                group_blocks = table[first : first + len(take)]
                if len(group_blocks) != len(take) or any(b.is_null for b in group_blocks):
                    raise RuntimeError(
                        f"G4 attention group {plan.group_idx} missing dest blocks"
                    )
                keys.extend(take)
                dst_ids.extend(b.block_id for b in group_blocks)
                group_sizes.append(len(take))
                block_indices.append(first)
                for offset, off_key in enumerate(take):
                    chunk = start + offset
                    begin = max(local, chunk * block)
                    stop = min(end, (chunk + 1) * block)
                    if begin < stop:
                        req_status.load_key_ranges.append((begin, stop, off_key))
                continue
            # The worker reads the state at (num_computed - 1) // block
            # after the load, which is the block-end index. External
            # allocation fills earlier slots with real blocks too, so the
            # first unhashed block is not the state.
            expected = max(0, (end - 1) // block)
            state = None
            index = -1
            if expected < len(table) and not table[expected].is_null:
                state = table[expected]
                index = expected
            else:
                limit = min(cdiv(end, block), len(table))
                for cursor in range(start, limit):
                    candidate = table[cursor]
                    if candidate.is_null:
                        continue
                    state = candidate
                    index = cursor
                    break
                if index != expected:
                    logger.warning(
                        "G4 state group %s dest index %s, block-end index %s",
                        plan.group_idx,
                        index,
                        expected,
                    )
            if state is None or len(plan.keys) != 1:
                raise RuntimeError(
                    f"G4 state group {plan.group_idx} has no fresh block"
                )
            if state.block_hash is not None:
                logger.warning(
                    "G4 state group %s dest block %s is already hashed",
                    plan.group_idx,
                    state.block_id,
                )
            keys.append(plan.keys[0])
            dst_ids.append(state.block_id)
            group_sizes.append(1)
            block_indices.append(index)
            req_status.load_key_ranges.append((local, end, plan.keys[0]))
        src_spec = self.manager.prepare_load(keys, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_ids, group_sizes=group_sizes, block_indices=block_indices
        )
        load_id = self._generate_job_id()
        self._current_batch_load_jobs[load_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        if req_status.transfer_jobs:
            raise RuntimeError(
                f"G4 tail load for {request.request_id} overlaps another transfer"
            )
        req_status.transfer_jobs.add(load_id)
        self._jobs[load_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys),
            is_store=False,
        )
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys)
        logger.info(
            "G4 load tail req=%s end=%s local=%s keys=%s",
            request.request_id,
            end,
            local,
            len(keys),
        )


class ParkingOffloadConnector(OffloadingConnector):
    """Hybrid offload that restores a conversation at its exact end."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        extra = _force_extra(vllm_config)
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_scheduler is None:
            return
        scheduler = self.connector_scheduler
        scheduler.__class__ = ParkingScheduler
        block_size = _attention_block_size(scheduler)
        scheduler.init_parking(int(extra.get("slots", 4)), block_size)
        missing = [
            kind
            for _idx, kind, _block in _group_table(scheduler)
            if kind == "gdn"
        ]
        if not any(kind == "full_attention" for _i, kind, _b in _group_table(scheduler)):
            raise RuntimeError("ParkingOffloadConnector found no full-attention group")
        if not missing:
            raise RuntimeError(
                "ParkingOffloadConnector found no GDN group. "
                "Serve with --mamba-cache-mode align and --enable-prefix-caching."
            )

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        super().bind_gpu_block_pool(gpu_block_pool)
        if isinstance(self.connector_scheduler, ParkingScheduler):
            self.connector_scheduler.bind_pool(gpu_block_pool)

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        if isinstance(self.connector_scheduler, ParkingScheduler):
            self.connector_scheduler.register_tail(
                request, block_ids, partial_tail_offloads
            )
        # Stores report completion through worker metadata, not
        # finished_sending. Returning True would delay-free forever.
        return False


def _group_table(scheduler: OffloadingConnectorScheduler):
    return [
        (
            group.group_idx,
            _kind(group.kv_cache_spec, group.is_eagle_group),
            group.tokens_per_block,
        )
        for group in scheduler.config.kv_group_configs
    ]


def _attention_block_size(scheduler: OffloadingConnectorScheduler) -> int:
    sizes = [
        group.tokens_per_block
        for group in scheduler.config.kv_group_configs
        if isinstance(group.kv_cache_spec, FullAttentionSpec)
        and not group.is_eagle_group
    ]
    if not sizes:
        raise RuntimeError("ParkingOffloadConnector found no attention block size")
    if any(size != sizes[0] for size in sizes):
        raise RuntimeError(f"attention groups disagree on block size: {sizes}")
    return sizes[0]
