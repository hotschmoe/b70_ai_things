"""Real-weight bucket integration gate. Requires an authorized card1 lease."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import os
import signal
import sys
import time

import numpy as np
import torch
from gguf import GGUFReader

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
signal.alarm(180)
torch.set_grad_enabled(False)
torch.manual_seed(3312)
B = Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
R = Path('/mnt/vm_8tb/github/b70_ai_things')
# Serving uses this same F1 control binary by default, without profiling waits.
control = Path(os.environ.get('P1_GATE_CONTROL_SO', str(B/'f1/_C.f1bucket.so')))
torch.ops.load_library(str(control))
torch.ops.load_library(str(B/'p1/_C.p1selected.so'))
torch.ops.load_library(str(B/'p1/_C.p1bucket.so'))
rows = []

def save(row):
    rows.append(row)
    print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2) + '\n')

spec = importlib.util.spec_from_file_location('p1_bucket_placement', R/'vllm/flashnext/plugins/b70_ep_xpu/b70_ep_xpu/placement.py')
placement = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = placement
spec.loader.exec_module(placement)
plans = placement.flashnext_plans(q0=138)
readers = [GGUFReader(str(x)) for x in sorted(Path('/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS').glob('*.gguf'))]
tensors = {t.name: t for reader in readers for t in reader.tensors}

def pack(layer, kind):
    tt = tensors[f'blk.{layer}.ffn_{kind}_exps.weight']
    ar = np.asarray(tt.data).reshape(512, -1)
    return torch.from_numpy(np.array(ar[list(plans[layer].card1)], copy=True)).to('xpu'), int(tt.tensor_type)

for layer in (0, 1, 8, 12, 35):
    gate, gt = pack(layer, 'gate')
    up, ut = pack(layer, 'up')
    down, dt = pack(layer, 'down')
    assert gt == ut
    H, F, E, K = 2560, 640, 374, 10
    for T in ((33, 129, 12283) if layer == 8 else (33, 129)):
        rng = np.random.default_rng(3312 + layer * 100 + T)
        probs = np.arange(1, E + 1, dtype=np.float64)**(-.7)
        probs /= probs.sum()
        iid = rng.choice(E, (T, K), p=probs).astype(np.int32)
        iid[rng.random((T, K)) < .27] = -1
        ids = torch.from_numpy(iid).to('xpu')
        w = torch.rand(T, K, device='xpu', dtype=torch.float32)
        w /= w.sum(1, keepdim=True)
        x = torch.randn(T, H, device='xpu', dtype=torch.bfloat16) * .03
        args = (x, ids, w, gate.flatten(), up.flatten(), down.flatten(), gt, dt, H, F, 32, 32768)
        outputs = []
        for label, selected in (('original', None), ('original_repeat', None), ('overlay_control', False), ('overlay_selected', True)):
            start = time.perf_counter()
            if selected is None:
                y = torch.ops.b70_gguf.prefill_moe_bucketed(*args)
            else:
                y = torch.ops.b70_p1.prefill_moe_bucketed(*args, selected)
            torch.xpu.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000
            assert torch.isfinite(y).all().item(), (layer, T, label, 'nonfinite')
            ref = outputs[0] if outputs else y
            exact = torch.equal(ref.view(torch.int32), y.view(torch.int32))
            save(dict(layer=layer, T=T, H=H, F=F, E=E, topk=K, tile=32, maxpad=32768,
                      gate_type=gt, down_type=dt, label=label, exact_original=exact,
                      max_abs=float((ref-y).abs().max()), wall_ms=wall_ms,
                      packed_weight_bytes=gate.numel()+up.numel()+down.numel(),
                      output_bytes=y.numel()*y.element_size()))
            outputs.append(y)
            assert exact, 'Exact original-repeat/overlay gate failed; no serving control qualification'
        del outputs, x, ids, w
    del gate, up, down
    gc.collect()
    torch.xpu.empty_cache()
print('P1_BUCKET_GATE_OK', flush=True)
