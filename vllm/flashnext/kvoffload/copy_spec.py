"""Direct KV copies on the b70_kvpark queue.

Off by default. Set B70_KVPARK_COPY=1 before the engine starts. The
worker still builds the same pointer batches. This replaces
swap_blocks_batch, which stages H2D through a CPU memcpy and waits
each pageable D2H.

The loaded library must export ordered()==1. That build makes the copy
queue wait for the current (transfer) stream, and that stream wait for
the copy, without a host wait. The handler then records its end event.
An older library stays on swap_blocks_batch. mmap stays the pool.
"""

from __future__ import annotations

import logging

import torch

from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

logger = logging.getLogger(__name__)


def _as_i64(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.int64:
        return t
    return t.view(torch.int64)


def _direct_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
) -> None:
    del is_src_access_order_any
    # Current stream is the handler's transfer stream, which has already
    # been told to wait for the compute stream. synchronize() makes that
    # wait visible before the copy queue reads or writes those blocks.
    # sync=False. The .so barriers the copy queue on this stream and this
    # stream on the copy, then returns. The handler records its end event
    # after this call, so the event means the bytes have landed. A host
    # wait here would hold the worker thread across the copy and stall
    # decode (T3).
    torch.ops.b70_kvpark.copy_batch(_as_i64(src), _as_i64(dst), _as_i64(sizes), False)


def install_direct_copy(worker) -> bool:
    from b70_kvpark.ops import load

    load()
    try:
        ordered = int(torch.ops.b70_kvpark.ordered())
    except Exception:
        ordered = 0
    if ordered != 1:
        logger.warning(
            "G4 copy library has no queue ordering; keeping swap_blocks_batch"
        )
        return False
    ordinal = int(torch.ops.b70_kvpark.engine_ordinal())
    if ordinal < 0:
        logger.warning("G4 copy engine missing; keeping swap_blocks_batch")
        return False
    worker._store_handler._swap_blocks_batch = _direct_copy
    worker._load_handler._swap_blocks_batch = _direct_copy
    logger.info("G4 direct copy queue ordinal=%s", ordinal)
    return True


class ParkingOffloadingSpec(CPUOffloadingSpec):
    """CPU offload spec whose XPU copies use b70_kvpark.copy_batch."""

    def create_worker(self, kv_caches):
        worker = super().create_worker(kv_caches)
        install_direct_copy(worker)
        return worker
