"""vLLM general plugin: QSA indexer attention on XPU.

Registration does not load the SYCL library. The first kernel call does.
B70_QSA=1 on the fork model imports model_hook separately; this entry point
stays a no-op so a missing .so does not break process start.
"""

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
