"""vLLM general plugin: QSA indexer attention on XPU.

Registration does not load the SYCL library. The first kernel call does.
The KV-cache spec is registered here when vLLM is already imported.
"""

import sys

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    try:
        from b70_qsa_xpu.kv_spec import register_specs

        register_specs()
    except Exception as exc:
        print(
            f"b70_qsa_xpu: KV spec not registered ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
