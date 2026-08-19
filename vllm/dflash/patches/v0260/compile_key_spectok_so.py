# Include SPECTOK + mounted _xpu_C SO identity in the vLLM 0.26 compile cache
# key. Stock SpeculativeConfig.compute_hash() omits num_speculative_tokens
# (D2/D3 shared hash b3f7e9e010). compiler_hash is inductor-only, so a
# GDN_SO / fusedq remount reuses graphs. Patch both.
# Loaded via compile_key_sitecustomize.py (PYTHONPATH first). Idempotent.
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any

_INSTALLED = False

_XPU_C = (
    "/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so"
)
_GDN = (
    "/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/"
    "libgdn_attn_kernels_xe_2.so"
)
_SO_CACHE: dict[str, str] = {}


def so_id(path: str) -> str:
    hit = _SO_CACHE.get(path)
    if hit is not None:
        return hit
    if not path or not os.path.isfile(path):
        ident = "missing"
    else:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ident = h.hexdigest()
    _SO_CACHE[path] = ident
    return ident


def extra_so_paths() -> list[str]:
    paths = [_XPU_C, _GDN]
    extra = os.environ.get("B70_COMPILE_KEY_SO", "")
    if extra:
        paths.extend(p for p in extra.split(":") if p)
    return paths


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return False
    from vllm.config.speculative import SpeculativeConfig
    from vllm.config.utils import hash_factors
    import vllm.envs as envs

    _orig_spec = SpeculativeConfig.compute_hash
    _orig_cf = envs.compile_factors

    def _spec_hash(self) -> str:
        return hash_factors(
            {
                "orig": _orig_spec(self),
                "num_speculative_tokens": getattr(
                    self, "num_speculative_tokens", None
                ),
                "method": getattr(self, "method", None),
            }
        )

    def _compile_factors() -> dict[str, object]:
        d = dict(_orig_cf())
        for i, p in enumerate(extra_so_paths()):
            d[f"b70_so_{i}"] = f"{p}={so_id(p)}"
        return d

    SpeculativeConfig.compute_hash = _spec_hash  # type: ignore[method-assign]
    envs.compile_factors = _compile_factors  # type: ignore[assignment]
    _INSTALLED = True
    print(
        "[compile-key] SPECTOK+SO in cache key "
        f"(xpu_C={so_id(_XPU_C)[:12]})",
        file=sys.stderr,
        flush=True,
    )
    return True


def selftest() -> dict[str, Any]:
    """No-GPU checks. Call after install()."""
    from vllm.config.speculative import SpeculativeConfig
    import vllm.envs as envs

    class _T:
        num_speculative_tokens = 3
        method = "dspark"
        draft_model_config = None

    t = _T()
    h3 = SpeculativeConfig.compute_hash(t)  # type: ignore[arg-type]
    t.num_speculative_tokens = 4
    h4 = SpeculativeConfig.compute_hash(t)  # type: ignore[arg-type]
    t.num_speculative_tokens = 3
    t.method = "mtp"
    h3m = SpeculativeConfig.compute_hash(t)  # type: ignore[arg-type]
    cf = envs.compile_factors()
    so_keys = [k for k in cf if str(k).startswith("b70_so_")]
    xpu = so_id(_XPU_C)
    return {
        "installed": _INSTALLED,
        "spectok_3_ne_4": h3 != h4,
        "method_dspark_ne_mtp": h3 != h3m,
        "h3": h3,
        "h4": h4,
        "so_keys": so_keys,
        "xpu_c_sha256": xpu,
        "xpu_c_len64": len(xpu) == 64,
        "xpu_c_present": os.path.isfile(_XPU_C),
    }
