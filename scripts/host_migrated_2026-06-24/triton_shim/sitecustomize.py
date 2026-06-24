import torch
try:
    torch.xpu.device_count()
except Exception:
    pass
