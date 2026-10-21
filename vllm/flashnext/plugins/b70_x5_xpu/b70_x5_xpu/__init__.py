"""X5 aligned-Q8 linear kernel; opt-in and independent of O3g."""
import os
from pathlib import Path
import torch

_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    so = os.environ.get('B70_X5_SO', '/mnt/vm_8tb/b70/build/vllm-flashnext/x5/q8-fast/libb70_x5_xpu.so')
    torch.ops.load_library(so)

    @torch.library.register_fake('b70_x5::q8_linear')
    def fake(x, codes, scales, fp32=False):
        return x.new_empty((x.shape[0], codes.shape[0]), dtype=torch.float32 if fp32 else torch.bfloat16)
    _loaded = True
