"""Short one-card probe of reserved-segment/profile headroom accounting."""
import gc
import json
import os
import time
import torch

torch.xpu.set_device(0)  # Physical card selected by xpu-run --card N.

def snapshot():
    torch.xpu.synchronize()
    time.sleep(1.1)
    stats = torch.xpu.memory_stats()
    free, total = torch.xpu.mem_get_info()
    return dict(allocated=stats["allocated_bytes.all.current"],
                reserved=stats["reserved_bytes.all.current"],
                peak=stats["allocated_bytes.all.peak"], free=free, total=total)

getter = torch._C._accelerator_getAllocatorSettings
setter = torch._C._accelerator_setAllocatorSettings
original = getter()
rows = []
warm = torch.empty(1024, dtype=torch.uint8, device="xpu")
warm.fill_(1)
torch.xpu.synchronize()
del warm
torch.xpu.empty_cache()
try:
    for config in json.loads(os.environ.get("B70_PROBE_POLICIES", '["", "max_split_size_mb:20"]')):
        setter(config)
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        before = snapshot()
        temporary = torch.empty(64*1024*1024, dtype=torch.uint8, device="xpu")
        temporary.fill_(1)
        torch.xpu.synchronize()
        del temporary
        pin = torch.empty(2*1024*1024, dtype=torch.uint8, device="xpu")
        pin.fill_(2)
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        after = snapshot()
        consumed = before["free"] - after["free"]
        rows.append(dict(config=getter(), before=before, after=after,
                         current_profiler=consumed + after["peak"]-after["allocated"],
                         reserve_adjusted=consumed+max(0,after["peak"]-after["reserved"])))
        del pin
        torch.xpu.empty_cache()
finally:
    setter(original)
print(json.dumps(dict(rows=rows, restored=getter()==original), indent=2))
