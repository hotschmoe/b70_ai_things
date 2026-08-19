# First PYTHONPATH sitecustomize: SPECTOK+SO compile-key, then chain
# the previous first-on-path shim (push-AR, which chains mtp_shim).
from __future__ import annotations

import importlib.util
import os
import sys

try:
    import compile_key_spectok_so as _ck

    _ck.install()
except Exception as _e:
    print("[compile-key] install failed:", repr(_e), file=sys.stderr, flush=True)

for _p in (
    "/opt/push_ar/sitecustomize.py",
    "/opt/mtp_shim/sitecustomize.py",
):
    if os.path.isfile(_p):
        try:
            _spec = importlib.util.spec_from_file_location(
                "_chained_sitecustomize", _p
            )
            assert _spec is not None and _spec.loader is not None
            _m = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            print("[compile-key] chained", _p, file=sys.stderr, flush=True)
        except Exception as _e:
            print(
                "[compile-key] failed to chain",
                _p,
                _e,
                file=sys.stderr,
                flush=True,
            )
        break
