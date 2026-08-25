# sitecustomize.py -- robust combined loader. Python auto-imports the FIRST `sitecustomize` on sys.path;
# put THIS dir first on PYTHONPATH. If another sitecustomize must also run (e.g. the rdy_to_serve MTP-graft
# shim), point PUSH_AR_CHAIN_SITECUSTOMIZE at its file and we exec it by path BEFORE applying our patch --
# so both take effect regardless of venv ENABLE_USER_SITE (which gates usercustomize auto-import).
import importlib.util
import os
import sys

_chain = os.environ.get("PUSH_AR_CHAIN_SITECUSTOMIZE", "")
if _chain and os.path.exists(_chain):
    try:
        _spec = importlib.util.spec_from_file_location("_chained_sitecustomize", _chain)
        _m = importlib.util.module_from_spec(_spec)
        # Dynamo imports the defining module again while tracing classes and
        # functions installed by the chained adapter. Keep the synthetic module
        # importable instead of only executing an otherwise anonymous object.
        sys.modules[_spec.name] = _m
        _spec.loader.exec_module(_m)
        print(f"[push_ar] chained sitecustomize: {_chain}", flush=True)
    except Exception as _e:
        print(f"[push_ar] failed to chain {_chain}: {_e}", flush=True)

import _push_ar_patch  # noqa: F401  (runs the monkeypatch installer at import)
