# 50_verbose_dispatch.py -- capture the exact oneDNN impl each op dispatches to on B70.
# Run under ONEDNN_VERBOSE=dispatch,profile_exec and grep 'onednn_verbose'. One int8 and
# one int4 call on a 27B gate/up shape (K5120 N17408) at M=1.
import torch
import vllm_xpu_kernels._xpu_C  # noqa
DEV = "xpu"
def line(*a): print(*a, flush=True)

K, N, M = 5120, 17408, 1
dt = torch.float16
x = torch.rand(M, K, device=DEV, dtype=dt)

line(">>> INT8 int8_gemm_w8a16 call")
w8 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=DEV)
wt = w8.t().contiguous(); sc8 = torch.ones(N, device=DEV, dtype=dt)
torch.ops._xpu_C.int8_gemm_w8a16(x, wt, sc8, None)
torch.xpu.synchronize()

line(">>> INT4 int4_gemm_w4a16 call")
rand = torch.randint(-128, 128, [(K * N) // 2], device=DEV).to(torch.int8)
wq = rand.view(dtype=torch.int32).reshape(K // 8, N)
weight_ba = wq.transpose(0, 1).contiguous().transpose(0, 1)
gs = 128; gnum = K // gs
scale4 = torch.rand(gnum, N, device=DEV, dtype=dt)
zp = torch.tensor([8], dtype=torch.int8, device=DEV)
torch.ops._xpu_C.int4_gemm_w4a16(x, weight_ba, torch.Tensor().to(DEV), scale4, zp, gs, None)
torch.xpu.synchronize()
line(">>> DONE")
