"""Fused pack + upstream topk + exact serial threshold compaction."""
import os
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
LOADED=False

class PackedTopk:
    def __init__(self,scores_n,cap,device):
        global LOADED
        if not LOADED:
            torch.ops.load_library(os.environ.get('B70_Q1_FUSED_SO','/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r1/libq1_fused.so'))
            LOADED=True
        self.cap=cap
        self.scores_n=scores_n
        self.keys=None

    def __call__(self,scores,ids,step):
        assert scores.shape[1]==self.scores_n and ids.shape[1]==self.cap
        if self.keys is None:
            self.keys=torch.empty_like(scores,dtype=torch.int64)
        assert scores.shape==self.keys.shape
        torch.ops.b70_q1_packcompact_r1.pack(scores,self.keys,step)
        values,_=torch.topk(self.keys,self.cap,dim=-1,largest=True,sorted=True)
        return torch.ops.b70_q1_packcompact_r1.compact(scores,values,ids,step)
