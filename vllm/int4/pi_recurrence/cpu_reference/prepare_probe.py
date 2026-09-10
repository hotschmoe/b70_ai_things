"""CPU-only bounded source/packing/tokenization fixtures; never loads a model.

Run in a no-device container with <=8 CPUs and <=12 GiB memory. This is
preparation evidence, not a model output oracle. GPTQ dequant rounding remains
explicit until matched against the pinned native source.
"""
import argparse
import ast
import hashlib
import json
import resource
import time
from pathlib import Path


def unpack_symmetric(qweight, scales, torch, group_size=128, round_half=True):
    """GPTQ input-axis nibble packing; explicit symmetric zero point eight."""
    shifts = torch.arange(8, dtype=torch.int64) * 4
    q = ((qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & 15)
    q = q.reshape(qweight.shape[0] * 8, qweight.shape[1])
    groups = torch.arange(q.shape[0]) // group_size
    w = (q.float() - 8) * scales.float()[groups]
    return w.half().float() if round_half else w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--bf16', required=True)
    p.add_argument('--raw', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    torch.manual_seed(42)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.raw)
    payload = json.loads((raw/'corpus-concurrent-v1/thinking0-repeat0-session0.json').read_text())
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    bf = AutoTokenizer.from_pretrained(a.bf16, local_files_only=True)
    rendered = tok.apply_chat_template(payload['messages'], tokenize=False,
        add_generation_prompt=True, **payload['chat_template_kwargs'])
    prompt = tok.encode(rendered, add_special_tokens=False)
    assert len(prompt) == 4045, len(prompt)
    cases = []
    for arm in ['tp1-card1-mtp0-serial-session0-ctx100k', 'tp1-card1-mtp3-conv-adapter1-ctx100k']:
        rows = json.loads((raw/arm/'serial-session0-control/results.json').read_text())
        for row in rows:
            content = row['response']['choices'][0]['message']['content']
            output_ids = tok.encode(content, add_special_tokens=False)
            combined = tok.encode(rendered+content, add_special_tokens=False)
            assert tok.decode(output_ids, skip_special_tokens=False) == content
            cases.append({'arm':arm,'id':row['id'],'prompt_ids':prompt,
                'output_ids':output_ids,'combined_ids':combined,
                'concat_ids_equal':combined == prompt+output_ids,
                'bf16_same_combined_ids':bf.encode(rendered+content, add_special_tokens=False)==combined,
                'usage':row['response']['usage'],'suffix':content[-40:],
                'ambiguity':'Canonical re-encoding, not captured sampled token IDs; usage may include stop token.'})
    (out/'teacher-forced-ids.json').write_text(json.dumps(cases, indent=2, ensure_ascii=True)+'\n')
    # Independent packing construction covers negative signed int32 and all nibbles.
    values = torch.arange(256*16).reshape(256,16) % 16
    values = (values + torch.arange(256)[:,None]) % 16
    packed = torch.zeros(32,16,dtype=torch.int64)
    for n in range(8): packed |= values[n::8] << (4*n)
    packed = packed.to(torch.int32)
    scales = torch.tensor([[.0013]*16,[.1234]*16],dtype=torch.float16)
    expected = torch.empty(256,16)
    for k in range(256):
        for n in range(16): expected[k,n] = torch.tensor((int(values[k,n])-8)*float(scales[k//128,n]),dtype=torch.float16).float()
    actual = unpack_symmetric(packed,scales,torch)
    assert torch.equal(actual,expected)
    # Remove decorators from exact installed source, so neither hub nor FLA can dispatch.
    source = Path(a.source).read_text()
    names = {'l2norm','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'}
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    for n in nodes: n.decorator_list=[]
    ns={'torch':torch,'F':F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(a.source),'exec'),ns)
    q,k,v=[torch.randn(1,65,2,8)*.2 for _ in range(3)]
    g=-torch.rand(1,65,2)*.1; beta=torch.rand(1,65,2)
    common=dict(initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
    c,cs=ns['torch_chunk_gated_delta_rule'](q,k,v,g,beta,**common)
    r,rs=ns['torch_recurrent_gated_delta_rule'](q,k,v,g,beta,**common)
    torch.testing.assert_close(c,r,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(cs,rs,atol=2e-6,rtol=2e-5)
    # One representative FP32 projection; no real model weights are read.
    x=torch.randn(256,5120); w=torch.randn(17408,5120)
    times=[]
    with torch.inference_mode():
        for _ in range(3):
            start=time.perf_counter(); y=F.linear(x,w); times.append(time.perf_counter()-start)
    flops=2*256*5120*17408
    result={'passed':True,'torch':torch.__version__,'threads':torch.get_num_threads(),
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'fallback_dispatch':'AST-extracted exact function bodies with decorators removed; CPU tensors only',
        'gdn_chunk_recurrent_max_abs':float((c-r).abs().max()),
        'projection_shape':[256,5120,17408],'projection_seconds':times,
        'projection_gflops':[flops/t/1e9 for t in times],
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'token_cases':[{'arm':z['arm'],'id':z['id'],'prompt':len(z['prompt_ids']),
          'output':len(z['output_ids']),'usage_completion':z['usage']['completion_tokens'],
          'concat_equal':z['concat_ids_equal'],'bf16_ids_equal':z['bf16_same_combined_ids'],'suffix':z['suffix']} for z in cases],
        'packing_fixture':'all nibbles, signed packed int32, two groups; scalar FP16-rounding expected exact',
        'limitation':'Synthetic fixture proves implementation of declared math, not native packing/rounding identity.'}
    (out/'probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__': main()
