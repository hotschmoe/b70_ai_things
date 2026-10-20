"""C12 fast-path coverage. Eager moe_layer vs fp32 dequant.

Run under bin/gpu-run --card 0. Point B70_GGUF_SO at the candidate .so.
Set B70_MOE_PHASE=1 so a hit prints C12_PHASE. A case that should take the
schedule fails closed when that line is absent.

Shapes are cropped to I=K=256 from real GGUF blocks (whole quant blocks only).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import gguf  # noqa: E402
import torch  # noqa: E402

from b70_gguf_xpu import moe_layer  # noqa: E402
from b70_gguf_xpu.types import BLOCK, NAMES  # noqa: E402
from gpu_check import close_enough, cpu_dequant, load_cpu  # noqa: E402
from moe_layer_bench import ref_layer  # noqa: E402

SHARD = (
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)
DEV = (
    "/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/"
    "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
)
PHASE_LOG = os.environ.get(
    "C12_PHASE_LOG", "/mnt/vm_8tb/b70/build/vllm-flashnext/c12/types_phase.log"
)

# ggml ids
Q8_0, Q2_K, Q3_K, Q6_K = 8, 10, 11, 14
IQ2_XXS, IQ2_XS, IQ3_XXS = 16, 17, 18
IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0 = 20, 21, 22, 23, 42


def crop(tensor, n_rows: int, k_logical: int):
    tid = int(tensor.tensor_type)
    be, bb = BLOCK[tid]
    if k_logical % be != 0:
        return None
    k_full = int(tensor.shape[0])
    if k_full < k_logical or k_full % be != 0:
        return None
    data = np.asarray(tensor.data)
    flat = data.reshape(-1, data.shape[-1])
    rb_full = (k_full // be) * bb
    if flat.shape[1] != rb_full or flat.shape[0] < n_rows:
        return None
    rb = (k_logical // be) * bb
    return np.ascontiguousarray(flat[:n_rows, :rb])


def find_crop(readers, tid: int, n_rows: int, k: int):
    for reader in readers:
        for tensor in reader.tensors:
            if int(tensor.tensor_type) != tid:
                continue
            got = crop(tensor, n_rows, k)
            if got is not None:
                return got, tensor.name
    return None, None


def pack_experts(rows: np.ndarray, e: int, n: int):
    need = e * n
    if rows.shape[0] < need:
        return None
    return np.ascontiguousarray(rows[:need].reshape(e, n, rows.shape[1]))


def run_case(lib, readers, name, gty, uty, dty, sgty, suty, sdty, i_dim, k_dim, t_tok, topk, e, phase_path):
    gate_rows, gn = find_crop(readers, gty, e * i_dim, k_dim)
    up_rows, un = find_crop(readers, uty, e * i_dim, k_dim)
    down_rows, dn = find_crop(readers, dty, e * k_dim, i_dim)
    if gate_rows is None or up_rows is None or down_rows is None:
        print(f"SKIP {name} missing rows gate={gate_rows is not None} up={up_rows is not None} down={down_rows is not None}")
        return False, set()
    gate = pack_experts(gate_rows, e, i_dim)
    up = pack_experts(up_rows, e, i_dim)
    down = pack_experts(down_rows, e, k_dim)
    sg = su = sd = None
    if sgty is not None:
        sg, sgn = find_crop(readers, sgty, i_dim, k_dim)
        su, sun = find_crop(readers, suty, i_dim, k_dim)
        sd, sdn = find_crop(readers, sdty, k_dim, i_dim)
        if sg is None or su is None or sd is None:
            print(f"SKIP {name} missing shared")
            return False, set()
        print(f"CASE {name} gate={NAMES[gty]}:{gn} up={NAMES[uty]}:{un} down={NAMES[dty]}:{dn} shared={NAMES[sgty]}:{sgn}")
    else:
        print(f"CASE {name} gate={NAMES[gty]}:{gn} up={NAMES[uty]}:{un} down={NAMES[dty]}:{dn} shared=0")
    rng = np.random.default_rng(12)
    x_np = rng.standard_normal((t_tok, k_dim)).astype(np.float32)
    x_np = torch.from_numpy(x_np).to(dtype=torch.bfloat16).float().numpy()
    ids = rng.integers(0, e, size=(t_tok, topk), dtype=np.int32)
    router = rng.random((t_tok, topk)).astype(np.float32)
    ref = ref_layer(
        lib, gate, gty, up, uty, down, dty, x_np, ids, router,
        sg, 0 if sgty is None else sgty, su, 0 if suty is None else suty, sd, 0 if sdty is None else sdty,
    )
    before = phase_path.stat().st_size if phase_path.exists() else 0
    y = moe_layer(
        torch.from_numpy(x_np).to(device="xpu", dtype=torch.bfloat16),
        torch.from_numpy(gate.reshape(-1)).to(device="xpu"), gty,
        torch.from_numpy(up.reshape(-1)).to(device="xpu"), uty,
        torch.from_numpy(down.reshape(-1)).to(device="xpu"), dty,
        torch.from_numpy(ids).to(device="xpu"),
        torch.from_numpy(router).to(device="xpu"),
        i_dim, k_dim,
        None if sg is None else torch.from_numpy(sg).to(device="xpu"),
        0 if sgty is None else sgty,
        None if su is None else torch.from_numpy(su).to(device="xpu"),
        0 if suty is None else suty,
        None if sd is None else torch.from_numpy(sd).to(device="xpu"),
        0 if sdty is None else sdty,
    )
    torch.xpu.synchronize()
    # The .so writes C12_PHASE to stderr, which this process dup'd onto phase_path.
    sys.stderr.flush()
    text = phase_path.read_text(errors="replace") if phase_path.exists() else ""
    fresh = text[before:]
    hit = "C12_PHASE" in fresh
    ok, mx, mr = close_enough(y.cpu().numpy(), ref, atol=8e-2, rtol=8e-2)
    status = "OK" if ok and hit else "FAIL"
    print(f"{status} {name} T={t_tok} topk={topk} E={e} I={i_dim} K={k_dim} sched={int(hit)} max_abs={mx:.3e} max_rel={mr:.3e}")
    for line in fresh.splitlines():
        if "C12_PHASE" in line:
            print(line)
    used = {gty, uty, dty}
    if sgty is not None:
        used |= {sgty, suty, sdty}
    return ok and hit, used


def main() -> int:
    os.environ["B70_MOE_PHASE"] = "1"
    assert torch.xpu.device_count() == 1
    phase_path = Path(PHASE_LOG)
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    phase_path.write_text("")
    saved = os.dup(2)
    log_fd = os.open(str(phase_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 2)
    failures = 0
    covered = set()
    try:
        lib = load_cpu()
        readers = []
        for path in (SHARD, DEV):
            if not Path(path).is_file():
                print(f"MISSING {path}")
                failures += 1
                continue
            print(f"OPEN {path}", flush=True)
            readers.append(gguf.GGUFReader(path))
        cases = [
            ("q6k_shared", IQ3_S, IQ2_S, Q2_0, Q6_K, Q6_K, Q6_K, 256, 256, 2, 2, 2),
            ("iq4xs_q2k", IQ4_XS, IQ4_XS, Q2_K, None, None, None, 256, 256, 1, 1, 1),
            ("q3k", IQ2_XS, IQ3_XXS, Q3_K, None, None, None, 256, 256, 1, 1, 1),
            ("q8_shared", IQ2_XXS, IQ3_XXS, IQ4_NL, Q8_0, Q8_0, Q8_0, 256, 256, 1, 1, 1),
            ("iq4nl_down", IQ2_S, IQ3_S, IQ4_NL, None, None, None, 256, 256, 1, 4, 2),
        ]
        for c in cases:
            ok, used = run_case(lib, readers, *c, phase_path)
            if not ok:
                failures += 1
            covered |= used
        need = {Q8_0, Q2_K, Q3_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0}
        missing = sorted(NAMES[t] for t in (need - covered))
        if missing:
            print("MISSING_COVERAGE " + ",".join(missing))
            failures += len(missing)
        # Q8_0 dequant bit-exact against gguf-py, two rows.
        from gguf import GGMLQuantizationType  # noqa: E402
        from gguf.quants import dequantize  # noqa: E402

        q8_rows, q8_name = find_crop(readers, Q8_0, 2, 256)
        if q8_rows is None:
            print("FAIL Q8_0 dequant rows missing")
            failures += 1
        else:
            k = 256
            out = cpu_dequant(lib, Q8_0, q8_rows, k)
            ref = np.asarray(dequantize(q8_rows, GGMLQuantizationType(Q8_0)), dtype=np.float32)
            mismatch = int(np.count_nonzero(out.view(np.uint32) != ref.view(np.uint32)))
            status = "OK" if mismatch == 0 else "FAIL"
            if mismatch:
                failures += 1
            print(f"{status} Q8_0 dequant vs gguf-py tensor={q8_name} mismatches={mismatch}")
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(log_fd)
        os.close(saved)
    print(f"VERDICT failures={failures} covered={len(covered)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
