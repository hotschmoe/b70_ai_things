"""vLLM general plugin: symmetric int2/int3/int4 WNA16 on XPU.

The entry point is re-entrant. Quant-method registration does not load
the SYCL library; the first kernel call does.
"""

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from b70_lowbit_xpu.quant_config import ensure_registered

    ensure_registered()
    _REGISTERED = True
