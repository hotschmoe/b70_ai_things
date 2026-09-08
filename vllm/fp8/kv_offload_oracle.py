#!/usr/bin/env python3
"""Small per-card byte round trip through the native offloader's XPU DMA op."""
import json
import torch
from vllm import _custom_ops as ops

for card in range(torch.xpu.device_count()):
    torch.xpu.set_device(card)
    device = 'xpu:' + str(card)
    source = torch.arange(4 * 1024 * 1024, dtype=torch.int64).remainder(251).to(torch.uint8).to(device)
    host = torch.empty(source.shape, dtype=torch.uint8, pin_memory=True)
    dest = torch.zeros_like(source)
    sizes = torch.tensor([source.numel()], dtype=torch.uint64)
    copy_stream = torch.xpu.Stream(device=device)
    copy_stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(copy_stream):
        ops.swap_blocks_batch(torch.tensor([source.data_ptr()], dtype=torch.uint64), torch.tensor([host.data_ptr()], dtype=torch.uint64), sizes)
    copy_stream.synchronize()
    # Clear the GPU source to prove the reload uses retained host bytes.
    expected = source.cpu()
    source.zero_()
    copy_stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(copy_stream):
        ops.swap_blocks_batch(torch.tensor([host.data_ptr()], dtype=torch.uint64), torch.tensor([dest.data_ptr()], dtype=torch.uint64), sizes)
    copy_stream.synchronize()
    assert torch.equal(dest.cpu(), expected)
    print(json.dumps({'card': card, 'bytes': sizes.item(), 'roundtrip_exact': True}), flush=True)
