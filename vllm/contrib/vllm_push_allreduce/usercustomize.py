# usercustomize.py -- Python auto-imports this AFTER sitecustomize.py, so the push-allreduce patch
# coexists with the rdy_to_serve MTP-graft sitecustomize.py (no PYTHONPATH collision). Put THIS dir on
# PYTHONPATH alongside the MTP shim dir. See _push_ar_patch.py for the actual monkeypatch.
import _push_ar_patch  # noqa: F401
