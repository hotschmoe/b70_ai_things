"""CPU reference and optional leased XPU integer-hash/FP32-noise probe.

No sampler activation. No model or image changes. Default device is CPU.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import torch

MASK=(1<<32)-1
SOURCE_COMMIT="2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed"
EXPECTED_SOURCES={'python/sglang/kernels/ops/sampling/murmur_hash.py': 'e932e4a7c5ae39eadc6400cc74ef4e646eaedc8370061e52b97a2544bbb5919b', 'python/sglang/srt/layers/sampler.py': '87e8c1bef75a8ad9c26a0b5dca827e8dd8724954b0edccf49bb9898d08cb70b4'}


def hash_reference(seed,position,column):
    def rotate(x,r):
        return ((x<<r)|(x>>(32-r)))&MASK
    h=0
    for k in (seed&MASK,(seed>>32)&MASK,position&MASK,column&MASK):
        k=(k*0xcc9e2d51)&MASK;k=rotate(k,15);k=(k*0x1b873593)&MASK
        h^=k;h=rotate(h,13);h=(h*5+0xe6546b64)&MASK
    h^=16;h^=h>>16;h=(h*0x85ebca6b)&MASK
    h^=h>>13;h=(h*0xc2b2ae35)&MASK;h^=h>>16
    return h


def gumbel_fp32(hashed):
    # Subtract/complement as integers BEFORE float conversion. Reflect both
    # branches into (0,.5], keeping every intermediate finite, then use log1p
    # for the upper tail instead of forming an FP32 value rounded to one.
    h=hashed.to(torch.int64)
    lower=h < (1<<31)
    distance=torch.where(lower,h,MASK-h)
    tail=(distance.to(torch.float32)+.5)*(2.0**-32)
    lower_noise=-torch.log(-torch.log(tail))
    upper_noise=-torch.log(-torch.log1p(-tail))
    return torch.where(lower,lower_noise,upper_noise)


def midpoint_reference(h):
    return -torch.log(-torch.log((h.to(torch.float64)+.5)*(2.0**-32)))


def legacy_reference(h):
    # Exact current sampler's FP64 mapping/clamps, before adding logprobs.
    x=h.to(torch.float64)/MASK
    x.log_().clamp_(min=torch.finfo(torch.float64).min,max=-(2.0**-32)).neg_()
    return x.log_().neg_()


def actual_legacy_source_reference(source, hashed):
    # Execute actual current FP64 arithmetic on CPU. Substitute only the hash
    # input and expose the pre-argmax score tensor for numeric comparison.
    path=source/'python/sglang/srt/layers/sampler.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='multinomial_with_seed')
    node.decorator_list=[]  # Inspect arithmetic without compiling a CPU graph.
    assert ast.unparse(node.body[-1].value)=='torch.argmax(x, dim=1, keepdim=True)'
    node.body[-1]=ast.Return(value=ast.Name(id='x',ctx=ast.Load()))
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    scope={'torch':torch,'murmur_hash32':lambda *args:hashed.reshape(1,-1)}
    exec(compile(module,str(path),'exec'),scope)
    return scope['multinomial_with_seed'](
        torch.zeros((1,hashed.numel()),dtype=torch.float64),
        torch.tensor([42]),torch.tensor([7])).reshape(-1)


def run(args):
    source_hashes={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest()
                   for name in EXPECTED_SOURCES}
    assert source_hashes==EXPECTED_SOURCES, 'source differs from reviewed pin'
    torch.set_num_threads(2)
    device=args.device
    generator=torch.Generator(device='cpu').manual_seed(9031)
    random_hashes=torch.randint(0,1<<32,(100000,),dtype=torch.int64,generator=generator)
    adversarial=sorted(set([0,1,2,3,127,128,129,(1<<31)-1,1<<31,(1<<31)+1,
                           MASK-256,MASK-129,MASK-128,MASK-127,MASK-3,MASK-2,MASK-1,MASK]))
    h=torch.cat([torch.tensor(adversarial,dtype=torch.int64),random_hashes])
    actual=gumbel_fp32(h.to(device)).cpu()
    ideal=midpoint_reference(h)
    legacy=actual_legacy_source_reference(args.source,h)
    assert torch.equal(legacy,legacy_reference(h)), 'legacy source arithmetic changed'
    assert torch.isfinite(actual).all()
    max_error=float((actual.double()-ideal).abs().max())
    assert max_error<4e-6,max_error
    assert torch.equal(actual,gumbel_fp32(h.to(device)).cpu()),'repeatability failed'
    # Ordinary prefill/verify INPUT positions: prompt 7 -> prefill6, root7,
    # next chain node8, next9. Large positions exercise integer wrap semantics.
    seeds=[42,843240016,1160881401,2120442760,(1<<63)-1]
    positions=[6,7,8,9,(1<<32)-1]
    cols=list(range(257))
    expected=torch.tensor([[hash_reference(s,p,c) for c in cols]
                           for s,p in zip(seeds,positions)],dtype=torch.int64)
    if device=='xpu':
        path=args.source/'python/sglang/kernels/ops/sampling/murmur_hash.py'
        spec=importlib.util.spec_from_file_location('pinned_murmur_hash',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        actual_hash=mod.murmur_hash32(
            torch.tensor(seeds,dtype=torch.int64,device=device).to(torch.uint64),
            torch.tensor(positions,dtype=torch.int64,device=device),
            torch.tensor(cols,dtype=torch.int64,device=device)).to(torch.int64).cpu()
        assert torch.equal(actual_hash,expected),'Triton hash differs from integer reference'
    else:
        actual_hash=expected
    hashed_noise=gumbel_fp32(actual_hash.to(device)).cpu()
    assert float((hashed_noise.double()-midpoint_reference(expected)).abs().max())<4e-6
    # A finite logit and a forbidden token remain well-defined at both endpoints.
    endpoint_noise=gumbel_fp32(torch.tensor([[0,MASK],[MASK,0]],device=device))
    logits=torch.tensor([[0.,float('-inf')],[float('-inf'),0.]],device=device)
    endpoint_scores=endpoint_noise+logits
    assert not torch.isnan(endpoint_scores).any().cpu()
    assert endpoint_scores.argmax(-1).cpu().tolist()==[0,1]
    # Empirical categorical check over hash-keyed rows and a binary distribution.
    # Fixed independent request seeds; columns0/1, same ordinary input position7.
    sample_hash=torch.tensor([[hash_reference(i,7,c) for c in (0,1)]
                              for i in range(30000)],dtype=torch.int64)
    noise=gumbel_fp32(sample_hash.to(device))
    probabilities=torch.tensor([.2,.8],device=device)
    choices=(noise+probabilities.log()).argmax(-1).cpu()
    observed=float((choices==1).float().mean())
    assert abs(observed-.8)<.012,observed
    return {'device':device,'torch':torch.__version__,'hash_execution':
            'actual pinned Triton' if device=='xpu' else 'Python integer reference only',
            'hash_rows':len(seeds),'hash_columns':len(cols),
            'max_abs_error_vs_midpoint_fp64':max_error,
            'max_abs_error_vs_legacy_fp64':float((actual.double()-legacy).abs().max()),
            'finite':'pass','repeatability':'pass','masked_endpoint_scores':'pass',
            'categorical_p1_expected':.8,'categorical_p1_observed':observed,
            'categorical_trials':len(choices),
            'adversarial':[{'hash':v,'fp32':float(actual[i]),'midpoint_fp64':float(ideal[i]),
                            'legacy_fp64':float(legacy[i])} for i,v in enumerate(adversarial)],
            'source_commit':SOURCE_COMMIT,'source_hashes':source_hashes}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','xpu'],default='cpu')
    parser.add_argument('--source',type=Path,required=True)
    print(json.dumps(run(parser.parse_args()),indent=2))
