"""Refuse a second copy of the Flash-Next PLE table.

The table is IQ4_NL, 320001536 rows by 90 bytes, 28800138240 bytes.
The GGUF loader already views the shard memmap. This guard runs when
that view is installed. A private anon copy is what OOM-killed the
DP=2 load. One data pointer is the whole process.
"""

from __future__ import annotations

import os

PLE_NBYTES = 28800138240
MEM_MIN_KB = 8 * 1024 * 1024

_INSTALLED = False
_SEEN_PTR: int | None = None


def mem_available_kb() -> int:
    with open("/proc/meminfo", "r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def memmap_backed(owner: object) -> bool:
    current = owner
    for _ in range(16):
        if type(current).__name__ == "memmap":
            return True
        base = getattr(current, "base", None)
        if base is None:
            break
        current = base
    return False


def assert_one_table(
    owner: object,
    nbytes: int,
    data_ptr: int,
    *,
    expect_nbytes: int | None = None,
    mem_min_kb: int = MEM_MIN_KB,
    available_kb: int | None = None,
) -> None:
    """Raise unless this is the one shared memmap and the host still has room."""
    global _SEEN_PTR
    avail = mem_available_kb() if available_kb is None else int(available_kb)
    if avail < mem_min_kb:
        raise RuntimeError(
            f"b70_ep: MemAvailable {avail} kB is under {mem_min_kb} kB; refusing to load PLE"
        )
    if not memmap_backed(owner):
        raise RuntimeError("b70_ep: PLE host table is not a memmap; refusing a private copy")
    if expect_nbytes is not None and int(nbytes) != int(expect_nbytes):
        raise RuntimeError(
            f"b70_ep: PLE table is {nbytes} bytes, expected {expect_nbytes}"
        )
    if _SEEN_PTR is None:
        _SEEN_PTR = int(data_ptr)
        return
    if int(data_ptr) != _SEEN_PTR:
        raise RuntimeError(
            f"b70_ep: second PLE table at {data_ptr:#x}, first was {_SEEN_PTR:#x}"
        )


def expected_nbytes() -> int | None:
    raw = os.environ.get("B70_PLE_NBYTES", "").strip()
    if not raw:
        return None
    return int(raw)


def mem_floor_kb() -> int:
    """Host floor. Default is 8 GiB. B70_EP_MEM_MIN_KB overrides it."""
    raw = os.environ.get("B70_EP_MEM_MIN_KB", "").strip()
    if not raw:
        return MEM_MIN_KB
    return int(raw)


def install() -> None:
    """Wrap the loader's install_host_table. No-op until vLLM imports."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from vllm.models.qwen4_exp.gguf import ple as ple_mod
    except Exception as exc:
        # Flash-Next sets B70_PLE_NBYTES. A silent miss would load a private copy.
        if expected_nbytes() is not None:
            raise RuntimeError(
                f"b70_ep: PLE guard required but the ple module did not import: {exc}"
            ) from exc
        print(f"b70_ep: PLE guard not installed ({type(exc).__name__}: {exc})", flush=True)
        return

    orig = ple_mod.install_host_table

    def wrapped(root, table):
        data = table.as_tensor()
        expect = expected_nbytes()
        assert_one_table(
            table.owner,
            int(data.numel()) * int(data.element_size()),
            int(data.data_ptr()),
            expect_nbytes=expect,
            mem_min_kb=mem_floor_kb(),
        )
        print(
            f"b70_ep: PLE memmap bytes={data.numel() * data.element_size()} "
            f"ptr={int(data.data_ptr()):#x} qtype={table.qtype}",
            flush=True,
        )
        return orig(root, table)

    ple_mod.install_host_table = wrapped
    _INSTALLED = True
    print("b70_ep: PLE guard installed", flush=True)
