"""CPU checks for the Flash-Next snake and the PLE memmap guard.

No torch, no GPU. The profile file is the STRP blob Strata ships.
A tiny numpy memmap stands in for the 26.8 GiB PLE table.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b70_ep_xpu import placement  # noqa: E402
from b70_ep_xpu import ple_guard  # noqa: E402
from b70_ep_xpu.placement import (  # noqa: E402
    flashnext_plans,
    linear_half,
    read_profile,
    snake_deal,
    split_mode,
)


def _row_bytes(k: int, block: int, type_size: int) -> int:
    if k % block != 0:
        raise AssertionError(f"{k} is not a multiple of {block}")
    return k // block * type_size


def test_row_bytes() -> None:
    # Gate/up are quantized on K=2560 (IQ block 256). Downs on I=640.
    assert _row_bytes(2560, 256, 98) == 980  # IQ3_XXS
    assert _row_bytes(2560, 256, 66) == 660  # IQ2_XXS
    assert _row_bytes(2560, 256, 74) == 740  # IQ2_XS
    assert _row_bytes(2560, 256, 82) == 820  # IQ2_S
    assert _row_bytes(2560, 256, 110) == 1100  # IQ3_S
    assert _row_bytes(640, 64, 18) == 180  # Q2_0
    assert _row_bytes(640, 32, 18) == 360  # IQ4_NL
    # Shorthand one-expert pack: IQ3_XXS gate, IQ3_XXS up, Q2_0 down.
    one = 640 * 980 + 640 * 980 + 2560 * 180
    card0 = one * 138 * 48
    card1 = one * 374 * 48
    print(f"shorthand_bytes card0={card0} card1={card1} one={one}")
    # Real gate/up types are mixed, so this is a size band, not the file total.
    assert 9 * (1 << 30) < card0 < 14 * (1 << 30)
    assert 24 * (1 << 30) < card1 < 34 * (1 << 30)


def test_snake_small() -> None:
    # No ranked pairs: index order. Pattern starts 0,1,1,0.
    # q0=1 redirects the last card-0 deal onto card 1.
    plans = snake_deal([], n_layers=1, n_expert=4, q0=1)
    assert plans[0].card0 == (0,)
    assert plans[0].card1 == (1, 2, 3)
    assert plans[0].res0 == (0, -1, -1, -1)
    assert plans[0].res1 == (-1, 0, 1, 2)
    plans = snake_deal([], n_layers=1, n_expert=4, q0=2)
    assert plans[0].card0 == (0, 3)
    assert plans[0].card1 == (1, 2)
    # share0 moves the hottest experts off the default snake.
    shared = snake_deal([], n_layers=1, n_expert=4, q0=2, share0=0.25)
    assert shared[0].card0 == (2, 3)
    assert shared[0].card1 == (0, 1)
    # A repeated pair is skipped, so the deal still covers every expert once.
    plans = snake_deal([(0, 1), (0, 1), (0, 0)], n_layers=1, n_expert=4, q0=2)
    assert plans[0].card0 == (1, 3)
    assert set(plans[0].card0).isdisjoint(plans[0].card1)
    assert set(plans[0].card0) | set(plans[0].card1) == set(range(4))


def test_linear_half() -> None:
    card0, card1, res0, res1 = linear_half(4)
    assert card0 == (0, 1)
    assert card1 == (2, 3)
    assert res0 == (0, 1, -1, -1)
    assert res1 == (-1, -1, 0, 1)
    try:
        linear_half(3)
    except ValueError:
        pass
    else:
        raise AssertionError("odd linear split should fail")


def test_split_mode() -> None:
    saved = os.environ.get("B70_EP_SPLIT")
    try:
        os.environ["B70_EP_SPLIT"] = "auto"
        assert split_mode(512) == "profile"
        assert split_mode(256) == "linear"
        os.environ["B70_EP_SPLIT"] = "linear"
        assert split_mode(512) == "linear"
        os.environ["B70_EP_SPLIT"] = "profile"
        assert split_mode(256) == "profile"
    finally:
        if saved is None:
            os.environ.pop("B70_EP_SPLIT", None)
        else:
            os.environ["B70_EP_SPLIT"] = saved


def _write_strp(path: Path, pairs: list[tuple[int, int]], n_layers: int, n_expert: int) -> None:
    slots = max(len(pairs), 1)
    blob = bytearray(b"STRP")
    blob += struct.pack("<5I", 1, n_layers, n_expert, slots, len(pairs))
    for layer, expert in pairs:
        blob += struct.pack("<2H", layer, expert)
    path.write_bytes(blob)


def test_reader_rejects() -> None:
    with tempfile.TemporaryDirectory(dir="/mnt/vm_8tb/b70/build/vllm-flashnext/tmp") as tmp:
        bad = Path(tmp) / "bad.bin"
        bad.write_bytes(b"XXXX" + b"\0" * 20)
        try:
            read_profile(str(bad), 1, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("bad magic should fail")
        short = Path(tmp) / "short.bin"
        _write_strp(short, [(0, 9)], 1, 4)
        try:
            read_profile(str(short), 1, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range expert should fail")


def test_flashnext_profile() -> None:
    path = placement.DEFAULT_PROFILE
    plans = flashnext_plans(n_layers=48, n_expert=512, q0=138, path=path, share0=-1.0)
    assert len(plans) == 48
    profile = read_profile(path, 48, 512)
    by_layer: dict[int, list[int]] = {layer: [] for layer in range(48)}
    seen: set[tuple[int, int]] = set()
    for layer, expert in profile.ranked:
        key = (layer, expert)
        if key in seen:
            continue
        seen.add(key)
        by_layer[layer].append(expert)
    for plan in plans:
        assert len(plan.card0) == 138
        assert len(plan.card1) == 374
        assert plan.card0 == tuple(sorted(plan.card0))
        assert plan.card1 == tuple(sorted(plan.card1))
        assert set(plan.card0).isdisjoint(plan.card1)
        assert set(plan.card0) | set(plan.card1) == set(range(512))
        assert [plan.res0[expert] for expert in plan.card0] == list(range(138))
        assert [plan.res1[expert] for expert in plan.card1] == list(range(374))
        assert all(plan.res0[expert] == -1 for expert in plan.card1)
        assert all(plan.res1[expert] == -1 for expert in plan.card0)
        # Hottest unique expert is card 0. The next is card 1. Snake start.
        order = by_layer[plan.layer]
        if len(order) >= 2:
            assert order[0] in plan.card0
            assert order[1] in plan.card1
    print(
        f"profile layers=48 q0=138 q1=374 ranked={len(profile.ranked)} "
        f"layer0_card0_head={plans[0].card0[:4]}"
    )


def test_ple_guard() -> None:
    ple_guard._SEEN_PTR = None
    scratch = Path("/mnt/vm_8tb/b70/build/vllm-flashnext/tmp")
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="h1d-ple-", suffix=".mmap", dir=str(scratch))
    os.close(fd)
    path = Path(name)
    try:
        mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(64,))
        mm[:] = 7
        view = mm[:32]
        assert ple_guard.memmap_backed(view)
        assert not ple_guard.memmap_backed(np.zeros(4, dtype=np.uint8))
        avail = ple_guard.mem_available_kb()
        assert avail > 0
        print(f"MemAvailable_kB={avail}")
        try:
            ple_guard.assert_one_table(view, 32, 11, available_kb=1024)
        except RuntimeError as exc:
            assert "MemAvailable" in str(exc)
        else:
            raise AssertionError("low memory should fail")
        try:
            ple_guard.assert_one_table(np.zeros(4, dtype=np.uint8), 4, 11, available_kb=9 << 20)
        except RuntimeError as exc:
            assert "memmap" in str(exc)
        else:
            raise AssertionError("a private array should fail")
        ple_guard.assert_one_table(
            view, 32, 11, expect_nbytes=32, available_kb=9 << 20
        )
        ple_guard.assert_one_table(
            view, 32, 11, expect_nbytes=32, available_kb=9 << 20
        )
        try:
            ple_guard.assert_one_table(view, 32, 99, available_kb=9 << 20)
        except RuntimeError as exc:
            assert "second PLE" in str(exc)
        else:
            raise AssertionError("a second pointer should fail")
        ple_guard._SEEN_PTR = None
        try:
            ple_guard.assert_one_table(
                view, 16, 11, expect_nbytes=32, available_kb=9 << 20
            )
        except RuntimeError as exc:
            assert "expected 32" in str(exc)
        else:
            raise AssertionError("a size mismatch should fail")
        saved = os.environ.get("B70_PLE_NBYTES")
        try:
            os.environ.pop("B70_PLE_NBYTES", None)
            assert ple_guard.expected_nbytes() is None
            os.environ["B70_PLE_NBYTES"] = "28800138240"
            assert ple_guard.expected_nbytes() == 28800138240
            os.environ["B70_EP_MEM_MIN_KB"] = "1048576"
            assert ple_guard.mem_floor_kb() == 1048576
        finally:
            if saved is None:
                os.environ.pop("B70_PLE_NBYTES", None)
            else:
                os.environ["B70_PLE_NBYTES"] = saved
            os.environ.pop("B70_EP_MEM_MIN_KB", None)
    finally:
        ple_guard._SEEN_PTR = None
        path.unlink(missing_ok=True)


def main() -> None:
    test_row_bytes()
    test_snake_small()
    test_linear_half()
    test_split_mode()
    test_reader_rejects()
    test_flashnext_profile()
    test_ple_guard()
    print("PLACEMENT_OK")
    print("PLE_GUARD_OK")


if __name__ == "__main__":
    main()
