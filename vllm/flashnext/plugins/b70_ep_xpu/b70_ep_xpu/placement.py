"""Strata traffic-snake expert placement.

Port of sycl/src/program/generate.cpp (the block that prints
"each layer's traffic dealt evenly between them"). The profile file is
the STRP v1 layout from include/strata/core/expert_cache.hpp: magic,
five uint32 header fields, then n_ranked (uint16 layer, uint16 expert)
pairs. The frequency table after the pairs is not read.

Card 0's quota is the count that fits beside attention, not half.
The daily driver log is 138 on card 0 and 374 on card 1 (512 experts).
Hot experts are dealt 0,1,1,0 until a quota fills. The cold remainder
goes to the card that still has room. Local slots are the ascending
global ids of that card's set, so the packed blob is a dense prefix.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

DEFAULT_PROFILE = (
    "/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin"
)
# ggml ids b70_gguf_xpu decode.hpp will run. Gate/up on the IQ3_XXS
# shard are IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, or IQ3_S. Downs are
# Q2_0 (30 layers) or IQ4_NL (18). reports/02_model_support.md.
ROUTED_TYPES = frozenset({8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 42})


@dataclass(frozen=True)
class Profile:
    version: int
    n_layers: int
    n_expert: int
    slots: int
    ranked: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LayerPlan:
    layer: int
    card0: tuple[int, ...]
    card1: tuple[int, ...]
    # Global expert id -> local slot, or -1. Length n_expert.
    res0: tuple[int, ...]
    res1: tuple[int, ...]


def read_profile(path: str, n_layers: int, n_expert: int) -> Profile:
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != b"STRP":
            raise ValueError(f"{path} does not start with STRP")
        header = handle.read(20)
        if len(header) != 20:
            raise ValueError(f"{path} is too short to hold a header")
        version, nl, ne, slots, n_ranked = struct.unpack("<5I", header)
        if version != 1:
            raise ValueError(f"{path} is profile version {version}; this reader wants 1")
        if nl != n_layers or ne != n_expert:
            raise ValueError(
                f"{path} is {nl}x{ne} but this model is {n_layers}x{n_expert}"
            )
        if n_ranked > slots:
            raise ValueError(f"{path} claims more ranked pairs than slots")
        raw = handle.read(n_ranked * 4)
        if len(raw) != n_ranked * 4:
            raise ValueError(f"{path} ranked list is truncated")
    pairs: list[tuple[int, int]] = []
    for index in range(n_ranked):
        layer, expert = struct.unpack_from("<2H", raw, index * 4)
        if layer >= n_layers or expert >= n_expert:
            raise ValueError(f"{path} pair {index} is ({layer}, {expert}), out of range")
        pairs.append((int(layer), int(expert)))
    return Profile(version, n_layers, n_expert, slots, tuple(pairs))


def snake_deal(
    ranked: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    n_layers: int,
    n_expert: int,
    q0: int,
    share0: float = -1.0,
) -> tuple[LayerPlan, ...]:
    """Deal one layer. share0 in (0, 1) is STRATA_NIGHT_EP_SHARE0. Off otherwise."""
    if q0 < 1 or q0 >= n_expert:
        raise ValueError(f"card 0 quota {q0} does not leave a card 1 set inside {n_expert}")
    q1 = n_expert - q0
    order: list[list[int]] = [[] for _ in range(n_layers)]
    seen: set[tuple[int, int]] = set()
    for layer, expert in ranked:
        key = (int(layer), int(expert))
        if key in seen:
            continue
        seen.add(key)
        order[key[0]].append(key[1])
    for layer in range(n_layers):
        have = set(order[layer])
        for expert in range(n_expert):
            if expert not in have:
                order[layer].append(expert)
    use_share = 0.0 < share0 < 1.0
    plans: list[LayerPlan] = []
    for layer in range(n_layers):
        got0: list[int] = []
        got1: list[int] = []
        n0 = 0
        n1 = 0
        for index, expert in enumerate(order[layer]):
            pair_odd = ((index // 2) & 1) != 0
            card = 1 if (((index & 1) != 0) != pair_odd) else 0
            if use_share:
                card = 0 if (n0 + 1) <= share0 * (index + 1) else 1
            if card == 0 and n0 >= q0:
                card = 1
            if card == 1 and n1 >= q1:
                card = 0
            if card == 0:
                got0.append(expert)
                n0 += 1
            else:
                got1.append(expert)
                n1 += 1
        if n0 != q0 or n1 != q1:
            raise RuntimeError(f"layer {layer} dealt {n0}/{n1}, wanted {q0}/{q1}")
        plans.append(_pack(layer, got0, got1, n_expert))
    return tuple(plans)


def _pack(layer: int, card0: list[int], card1: list[int], n_expert: int) -> LayerPlan:
    """Local slot is the rank of the global id inside that card's set."""
    c0 = tuple(sorted(card0))
    c1 = tuple(sorted(card1))
    if len(set(c0) & set(c1)) != 0:
        raise RuntimeError(f"layer {layer} cards overlap")
    if len(c0) + len(c1) != n_expert:
        raise RuntimeError(f"layer {layer} does not cover every expert")
    res0 = [-1] * n_expert
    res1 = [-1] * n_expert
    for slot, expert in enumerate(c0):
        res0[expert] = slot
    for slot, expert in enumerate(c1):
        res1[expert] = slot
    return LayerPlan(layer, c0, c1, tuple(res0), tuple(res1))


def linear_half(n_expert: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Contiguous 0..n0-1 / n0..E-1. The 35B path. No profile."""
    if n_expert < 2 or n_expert % 2 != 0:
        raise ValueError(f"linear split wants an even expert count, got {n_expert}")
    n0 = n_expert // 2
    card0 = tuple(range(n0))
    card1 = tuple(range(n0, n_expert))
    res0 = tuple(expert if expert < n0 else -1 for expert in range(n_expert))
    res1 = tuple(expert - n0 if expert >= n0 else -1 for expert in range(n_expert))
    return card0, card1, res0, res1


_CACHED: dict[tuple, tuple[LayerPlan, ...]] = {}


def flashnext_plans(
    n_layers: int = 48,
    n_expert: int = 512,
    q0: int | None = None,
    path: str | None = None,
    share0: float | None = None,
) -> tuple[LayerPlan, ...]:
    """Plans for the IQ3_XXS sidecar. Cached for the process."""
    profile_path = path or os.environ.get("B70_EP_PROFILE", DEFAULT_PROFILE)
    quota = int(os.environ.get("B70_EP_Q0", "138")) if q0 is None else int(q0)
    if share0 is None:
        raw = os.environ.get("B70_EP_SHARE0", "").strip()
        share0 = float(raw) if raw else -1.0
    key = (profile_path, n_layers, n_expert, quota, float(share0))
    hit = _CACHED.get(key)
    if hit is not None:
        return hit
    profile = read_profile(profile_path, n_layers, n_expert)
    plans = snake_deal(profile.ranked, n_layers, n_expert, quota, share0)
    _CACHED[key] = plans
    return plans


def split_mode(n_expert: int) -> str:
    """auto: snake when E is 512, linear half otherwise (the 35B)."""
    mode = os.environ.get("B70_EP_SPLIT", "auto").strip().lower()
    if mode in ("linear", "half"):
        return "linear"
    if mode in ("profile", "snake"):
        return "profile"
    if mode != "auto":
        raise ValueError(f"B70_EP_SPLIT={mode} is not auto, linear, or profile")
    return "profile" if n_expert == 512 else "linear"
