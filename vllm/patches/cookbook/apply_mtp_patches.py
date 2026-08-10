#!/usr/bin/env python3
"""Apply the best-matching MTP cookbook patches for the installed vLLM.

Order:
  1. BF16 draft gate (nightly OR v0260 adaptive)
  2. GDN partial-final-group boundary patch

Idempotent. Intended to run inside the container before `vllm serve`.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _try(script: str) -> bool:
    path = HERE / script
    if not path.exists():
        print(f"[apply] missing {path}", file=sys.stderr)
        return False
    print(f"[apply] running {script}")
    try:
        runpy.run_path(str(path), run_name="__main__")
        return True
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if code == 0:
            return True
        print(f"[apply] {script} exited {e.code}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[apply] {script} failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    # Try nightly anchors first (public image), then our v0.26.0 bake.
    # Silence expected nightly miss on v0260 by checking source markers.
    import importlib.util
    from pathlib import Path

    draft_ok = False
    try:
        spec = importlib.util.find_spec("vllm")
        root = Path(next(iter(spec.submodule_search_locations)))  # type: ignore[arg-type]
        mtp = (root / "model_executor/models/qwen3_5_mtp.py").read_text()
        if "original_quant = vllm_config.quant_config" in mtp:
            draft_ok = _try("patch_mtp_nightly.py")
        else:
            draft_ok = _try("patch_mtp_bf16_draft_v0260.py")
    except Exception as e:
        print(f"[apply] probe failed ({e}); trying both draft patches", file=sys.stderr)
        draft_ok = _try("patch_mtp_nightly.py") or _try("patch_mtp_bf16_draft_v0260.py")
    if not draft_ok:
        print("[apply] ERROR: no BF16-draft patch applied", file=sys.stderr)
        return 2
    if not _try("patch_mtp_boundary.py"):
        print("[apply] ERROR: boundary patch failed", file=sys.stderr)
        return 3
    print("[apply] OK: MTP cookbook patches applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
