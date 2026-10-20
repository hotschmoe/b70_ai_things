import sys, torch
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_o4_xpu")
import b70_o4_xpu as o4
o4.load()
t = torch.ops.b70_o4.host_table(100000, 90)
c = torch.ops.b70_o4.host_cpu_view(t)
src = torch.randint(0, 255, (100000, 90), dtype=torch.uint8)
c.copy_(src)
ids = torch.randint(0, 100000, (32,), device="xpu")
g = t.index_select(0, ids)
ok = torch.equal(g.cpu(), src[ids.cpu()])
gr = torch.xpu.XPUGraph()
out = torch.empty(32, 90, dtype=torch.uint8, device="xpu")
with torch.xpu.graph(gr):
    out.copy_(t.index_select(0, ids))
ids.copy_(torch.randint(0, 100000, (32,), device="xpu"))
gr.replay(); torch.xpu.synchronize()
ok2 = torch.equal(out.cpu(), src[ids.cpu()])
print("O4_HOSTTAB", ok, ok2, t.device)
