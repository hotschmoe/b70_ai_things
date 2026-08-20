#!/usr/bin/env python3
"""Print XPU count and ze/torch peer access. Exit 0 even if peer is False."""
import torch

n = torch.xpu.device_count()
print("xpu_count", n)
p01 = torch.xpu.can_device_access_peer(0, 1) if n > 1 else None
p10 = torch.xpu.can_device_access_peer(1, 0) if n > 1 else None
print("peer_0_1", p01)
print("peer_1_0", p10)
