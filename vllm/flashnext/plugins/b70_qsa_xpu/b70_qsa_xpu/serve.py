"""Paged-serve helpers. No SYCL.

Side state (indexer tail, dead key, block_pos, token count) is not inside
the uint8 page. It is keyed by the physical id of logical block 0, so a
batch row can move between steps without dropping the open block. pos < 0
is an inactive slot and scatters to a sink row that nothing reads.
"""

from __future__ import annotations

import numpy as np
import torch


def as_pages(cache: torch.Tensor | None, page_bytes: int) -> torch.Tensor | None:
    """View a vLLM layer cache as uint8 [num_pages, page_bytes].

    Returns None when the buffer is missing, empty, not uint8, or not a
    whole number of pages. A non-contiguous view is refused so a later
    write cannot land in a copy.
    """
    if cache is None or not isinstance(cache, torch.Tensor):
        return None
    if cache.dtype != torch.uint8 or cache.numel() == 0 or page_bytes < 1:
        return None
    if int(cache.numel()) % int(page_bytes):
        return None
    last = int(cache.shape[-1])
    # A 1-D buffer is raw bytes. Any other shape must already end on the
    # page, so a kernel-block split cannot be viewed as one QSA page.
    if cache.ndim != 1 and last != int(page_bytes):
        return None
    pages = cache.reshape(-1, int(page_bytes))
    if not pages.is_contiguous():
        return None
    return pages


def side_index(pos: torch.Tensor, phys0: torch.Tensor, sink: int) -> torch.Tensor:
    """Physical block 0 for active rows, `sink` where pos < 0."""
    sink_t = torch.full_like(phys0, int(sink))
    return torch.where(pos >= 0, phys0, sink_t)


def reset_open_rows(
    tail: torch.Tensor,
    dead: torch.Tensor,
    block_pos: torch.Tensor,
    length: torch.Tensor,
    pos: torch.Tensor,
) -> None:
    """Drop gathered side state on rows that are starting at position 0.

    pos > 0 keeps the open block. pos <= 0 clears the batch row. Inactive
    rows are cleared here and must scatter to the sink, not to a live key.
    """
    keep = (pos > 0).to(dtype=tail.dtype).reshape(-1, 1, 1)
    tail.mul_(keep)
    dead.mul_(keep.reshape(-1, 1))
    keep_i = (pos > 0).to(dtype=block_pos.dtype)
    block_pos.mul_(keep_i)
    length.mul_(keep_i)


def export_request_state(
    pages: torch.Tensor,
    tail: torch.Tensor,
    dead: torch.Tensor,
    block_pos: torch.Tensor,
    length: torch.Tensor,
    phys: np.ndarray,
    block_size: int,
) -> dict:
    """CPU copy of one request. `phys` is physical pages in logical order.

    The side-state key is phys[0]. Keys match QsaPaged.export_seq.
    pos_base is 0: pooled rows were rotated at absolute positions.
    """
    ids = np.asarray(phys, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        raise ValueError("no pages")
    key = int(ids[0])
    n_kv = int(length[key].item())
    n_used = (n_kv + int(block_size) - 1) // int(block_size) if n_kv else 0
    if n_used > int(ids.size):
        raise ValueError("block list is shorter than n_kv")
    page_bytes = int(pages.shape[1])
    out = np.zeros((n_used, page_bytes), dtype=np.uint8)
    src = pages.detach().to("cpu").numpy()
    for i in range(n_used):
        out[i] = src[int(ids[i])]
    return {
        "pos": (n_kv - 1) if n_kv else -1,
        "n_kv": n_kv,
        "pos_base": 0,
        "tail": tail[key].detach().to("cpu").numpy().copy(),
        "dead": dead[key].detach().to("cpu").numpy().copy(),
        "block_pos": int(block_pos[key].item()),
        "have_dead": bool(n_kv > 0),
        "block_table": ids.copy(),
        "pages": out,
        "logical_blocks": np.arange(n_used, dtype=np.int32),
    }


def import_request_state(
    pages: torch.Tensor,
    tail: torch.Tensor,
    dead: torch.Tensor,
    block_pos: torch.Tensor,
    length: torch.Tensor,
    dest_phys: np.ndarray,
    blob: dict,
) -> None:
    """Write pages through `dest_phys` and restore side state at dest_phys[0].

    `dest_phys` is the live physical list in logical order. A restore into
    a different block table remaps. pos_base must be 0.
    """
    if int(blob.get("pos_base", 0)) != 0:
        raise ValueError("pos_base must be 0")
    dest = np.asarray(dest_phys, dtype=np.int32).reshape(-1)
    if dest.size == 0:
        raise ValueError("no destination pages")
    logical = np.asarray(blob["logical_blocks"], dtype=np.int32).reshape(-1)
    src = np.asarray(blob["pages"], dtype=np.uint8)
    if logical.size and int(logical.max()) >= int(dest.size):
        raise ValueError("destination block list is short")
    if int(src.shape[0]) != int(logical.size):
        raise ValueError("page rows do not match logical blocks")
    # Touch only the logical blocks named in the blob. A vLLM block table
    # is padded with physical id 0, and page 0 may belong to another request.
    for i, block in enumerate(logical.tolist()):
        slot = int(dest[int(block)])
        pages[slot].zero_()
        page = torch.from_numpy(np.ascontiguousarray(src[i]))
        pages[slot].copy_(page)
    key = int(dest[0])
    tail[key].copy_(torch.from_numpy(np.ascontiguousarray(np.asarray(blob["tail"]))))
    dead[key].copy_(torch.from_numpy(np.ascontiguousarray(np.asarray(blob["dead"]))))
    block_pos[key] = int(blob["block_pos"])
    length[key] = int(blob["n_kv"])
