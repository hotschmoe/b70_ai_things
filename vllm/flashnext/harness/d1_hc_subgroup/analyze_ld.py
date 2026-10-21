"""Offline saved-Ld census only. Input pickles are trusted campaign artifacts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import pickle
import statistics
S=Path(__file__).resolve().parent
P=Path('/mnt/vm_8tb/b70/build/vllm-flashnext/o4/run-Ld')
x=pickle.loads((P/'steps.pkl').read_bytes())
steps=[s for s in x['0:b:0:0 #0']if len(s)==1938]
patterns={'hc_up':'w8_short_k<8>','hc_down':'w8_gemv_k<8, 0>',
          'hc_act':'_hc_act','hc_collapse':'_hc_collapse','topk':'TopKGating',
          'dispatch':'ep_dispatch(','bump':'ep_bump(','wait':'ep_wait(',
          'ple_gate':'_ple_gate','ple_conv':'_ple_conv'}
rows={}
for label,match in patterns.items():
    values=[[(b-a)for a,b,n in s if match in n]for s in steps]
    total=[sum(v)for v in values]
    rows[label]={'calls_per_step':statistics.median(map(len,values)),
                 'median_kernel_sum_us':statistics.median(total),
                 'early64_median_us':statistics.median(total[:64]),
                 'late64_median_us':statistics.median(total[-64:])}
# Source-mapped PLE stage is between first layer's combine and second
# layer's HC norm, exactly once per graph; includes token metadata/hash,
# gather/DQ and two GEMMs. First find known gate then preceding layer combine.
ple=[]
for s in steps:
    gate=next(i for i,(_,_,n)in enumerate(s)if n=='_ple_gate')
    first=max(i for i,(_,_,n)in enumerate(s[:gate])if n=='_hc_combine')+1
    last=next(i for i,(_,_,n)in enumerate(s[gate:],gate)if n=='_hc_norm')
    ple.append({'kernel_count':last-first,'kernel_sum_us':sum(b-a for a,b,n in s[first:last])})
result={'scope':'saved trace; instrumentation affects absolute times; no new measured gain',
        'input_sha256':hashlib.sha256((P/'steps.pkl').read_bytes()).hexdigest(),
        'full_trace_path':str(P/'trace/python.7.json'),'exact1938kernel_steps':len(steps),
        'config':'Ld original/unfused HC and nativeTopKGating; not Lc route+Q8F',
        'kernels':rows,'ple_stage_source_mapped':{
            'count':statistics.median(v['kernel_count']for v in ple),
            'median_kernel_sum_us':statistics.median(v['kernel_sum_us']for v in ple)},
        'Lc_existing_gain_excluded':{'La_reasoning_ms':16.25,'Lc_reasoning_ms':15.70,
            'combined_flags':['B70_EP_ROUTE','B70_DECODE_HC_Q8F'],
            'not_new_estimate':True}}
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
print(json.dumps(result,sort_keys=True))
