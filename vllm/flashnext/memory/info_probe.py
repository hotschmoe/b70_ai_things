"""Compare both XPU memory-info APIs around an allocator segment lifecycle."""
import gc
import json
import time
import torch
import vllm.platforms.xpu

torch.xpu.set_device(0)

def snap(stage):
    torch.xpu.synchronize()
    time.sleep(1.1)
    return dict(stage=stage, native=torch.xpu.mem_get_info(),
                vllm=torch.accelerator.get_memory_info(),
                allocated=torch.xpu.memory_allocated(), reserved=torch.xpu.memory_reserved())

warm = torch.empty(1024, dtype=torch.uint8, device="xpu")
warm.fill_(1)
torch.xpu.synchronize()
del warm
torch.xpu.empty_cache()
rows = [snap("baseline")]
temp = torch.empty(64*1024*1024, dtype=torch.uint8, device="xpu")
temp.fill_(1)
rows.append(snap("temporary"))
del temp
pin = torch.empty(2*1024*1024, dtype=torch.uint8, device="xpu")
pin.fill_(2)
torch.xpu.synchronize()
gc.collect()
torch.xpu.empty_cache()
rows.append(snap("pinned"))
del pin
gc.collect()
torch.xpu.empty_cache()
rows.append(snap("released"))
print(json.dumps(rows, indent=2))
