"""Execute exact pinned ordinary sampler helpers using CPU PyTorch only."""
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Optional
import torch

source_root = Path(sys.argv[1])
helper_path = Path(sys.argv[2])
source = source_root / 'python/sglang/srt/layers/sampler.py'
tree = ast.parse(source.read_text())
names = {'sampling_from_probs_torch', 'top_k_top_p_min_p_sampling_from_probs_torch'}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(functions) == 2
module = types.ModuleType('sglang.srt.layers.sampler')
module.__dict__.update(torch=torch, Optional=Optional,
    envs=types.SimpleNamespace(SGLANG_OPT_USE_GUMBEL_SAMPLE=types.SimpleNamespace(get=lambda:True)))
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), module.__dict__)
sys.modules[module.__name__] = module
spec = importlib.util.spec_from_file_location('candidate', helper_path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

def info(temperatures, top_ks, top_ps):
    return types.SimpleNamespace(
        temperatures=torch.tensor(temperatures, dtype=torch.float32, device='cpu').reshape(-1,1),
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device='cpu'),
        top_ps=torch.tensor(top_ps, dtype=torch.float32, device='cpu'),
        min_ps=torch.zeros(len(temperatures), device='cpu'),
        need_top_k_sampling=any(k != 2 for k in top_ks),
        need_top_p_sampling=any(p != 1 for p in top_ps))

torch.set_num_threads(2)
torch.manual_seed(790)
# Interleaved request metadata: first request top-k1 selects 0, second top-p.1 selects 1.
logits=torch.tensor([[2.,0.]]*3+[[0.,2.]]*3, device='cpu')
out=candidate.sample_target_ids(logits,info([.7,.4],[1,2],[1.,.1]),batch_size=2,verify_width=3)
assert out.tolist()==[[0,0,0],[1,1,1]]
# Actual helper temperature distribution, independently sampled rows.
n=40000
logits=torch.tensor([[0.,1.]], device='cpu').repeat(n,1)
expected=torch.softmax(torch.tensor([0.,1.])/.7,dim=0)[1].tolist()
sampled=candidate.sample_target_ids(logits,info([.7],[2],[1.]),batch_size=1,verify_width=n)
observed=(sampled==1).float().mean().tolist()
assert abs(observed-expected)<.015,(observed,expected)
# Top-p excludes the low-probability token exactly.
masked=candidate.sample_target_ids(logits[:100],info([.7],[2],[.7]),batch_size=1,verify_width=100)
assert masked.eq(1).all().tolist()
try:
    candidate.sample_target_ids(logits[:2],info([.7],[2],[1.]),batch_size=1,verify_width=3)
except ValueError:
    pass
else:
    raise AssertionError('shape mismatch accepted')
print(json.dumps({'torch':torch.__version__,'device':'cpu','rows':n,'expected_p_token1':expected,'observed_p_token1':observed,'request_row_expansion':'pass','top_p_exclusion':'pass','shape_gate':'pass','scope':'actual helper source; no SGLang imports or Triton execution'},indent=2))
