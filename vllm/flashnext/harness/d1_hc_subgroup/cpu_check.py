"""CPU-only ownership/order check; no Torch/SYCL/XPU imports or build."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess
S=Path(__file__).resolve().parent
H,K=2560,320
original={}
for group in range(H//2):
    for sg in range(4):
        for lane in range(16):
            lr=sg*2+lane//8
            hh=group*2+lr//4
            st=lr%4
            original[(st*H+hh,lane%8)]=tuple(range(lane%8,K//16,8))
candidate={}
for group in range(H//4):
    for sg in range(4):
        hh=group*4+sg
        for lane in range(16):
            for half in (0,1):
                st=lane//8+2*half
                key=(st*H+hh,lane%8)
                assert key not in candidate
                candidate[key]=tuple(range(lane%8,K//16,8))
assert original==candidate
assert len(candidate)==4*H*8
for row in (0,H-1,H,2*H-1,4*H-1):
    assert sorted(c for sub in range(8)for c in candidate[(row,sub)])==list(range(20))
# The XOR1/2/4 reductions never cross original8-lane partitions. Only
# original output lanes0/8 are selected; stream accumulation0,1,2,3 stays.
for lane in range(16):
    assert all((lane^offset)//8==lane//8 for offset in (1,2,4))
for f in S.iterdir():
    if f.suffix in ('.cpp','.py','.sh','.md'):
        f.read_bytes().decode('ascii')
        if f.suffix=='.py':ast.parse(f.read_text())
        if f.suffix=='.sh':subprocess.run(['bash','-n',str(f)],check=True)
result={'verdict':'CPU_STRUCTURE_PASS; no compile/numerical/GPU qualification',
        'row_lane_owners':len(candidate),'K16_chunks':20,'original_groups':H//2,
        'candidate_groups':H//4,'threads_per_group':64,'subgroup_size':16,
        'LPR':8,'same_row_chunk_order':True,'same_original_reduction_output_lanes':[0,8],
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in S.iterdir()if f.suffix in ('.cpp','.py','.sh')}}
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
print(json.dumps(result,sort_keys=True))
