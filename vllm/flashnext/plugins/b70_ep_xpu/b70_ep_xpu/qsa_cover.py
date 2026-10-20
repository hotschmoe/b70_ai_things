"""Turn on the QSA plugin when B70_QSA=1.

The fork attaches a pooled runner from Qwen4ExpDecoderLayer when that
variable is set. KV specs are registered here, before the model is
built. An MTP decoder (prefix contains "mtp") is retargeted to the
draft runner, which does not keep pooled indexer rows.
"""

from __future__ import annotations

import os

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("B70_QSA", "") != "1":
        return
    import b70_qsa_xpu
    from b70_qsa_xpu.model_hook import mark_draft
    from vllm.models.qwen4_exp.xpu.model import Qwen4ExpDecoderLayer

    b70_qsa_xpu.register()
    orig = Qwen4ExpDecoderLayer.__init__
    if getattr(orig, "_b70_ep_qsa", False):
        _INSTALLED = True
        return

    def wrapped(self, vllm_config, layer_type, prefix: str = ""):
        orig(self, vllm_config, layer_type, prefix)
        if "mtp" in str(prefix).lower() and getattr(self, "qsa_runner", None) is not None:
            mark_draft(self)
            print(f"b70_ep: QSA draft {prefix}", flush=True)

    wrapped._b70_ep_qsa = True
    Qwen4ExpDecoderLayer.__init__ = wrapped
    _INSTALLED = True
    print("b70_ep: B70_QSA=1 registered; MTP layers will be marked draft", flush=True)
