"""Opt-in eager QSA prefill: avoid unused query and side-state work."""
import os
import torch


def rope_rows(runner,x,positions,token0,n):
    half=int(runner.batch.cos.shape[1])
    pair=torch.arange(half,device=x.device)
    p=positions[:,token0:token0+n].transpose(0,1).long()
    p=p[:,pair%3]
    c=runner.batch.cos[p,pair]
    s=runner.batch.sin[p,pair]
    x1=x[...,:half].clone()
    x2=x[...,half:2*half].clone()
    x[...,:half]=x1*c[:,None,:]-x2*s[:,None,:]
    x[...,half:2*half]=x2*c[:,None,:]+x1*s[:,None,:]


def append_request(runner,s,t0,t1,k,v,raw,positions,pos_cpu,nreq):
    if os.environ.get('B70_PREFILL_QSA_APPEND')=='batch':
        from .prefill_append import append_batch
        if append_batch(runner,s,t0,t1,k,v,raw,positions,pos_cpu,nreq):
            return
    if t1<=t0:
        return
    b=runner.batch
    g=b.geom
    b.pos.fill_(-1)
    b.pos[s]=int(pos_cpu[t0])
    runner._prepare_side(nreq)
    if positions.ndim==2:
        if b.mrope.numel()==0:
            b.mrope=torch.zeros(b.batch,b.max_cells,3,dtype=torch.int32,device=b.device)
        p=positions[0,t0:t1].long()
        b.mrope[s].index_copy_(0,p,positions[:,t0:t1].transpose(0,1).to(torch.int32))
    for t in range(t0,t1):
        b.pos[s]=int(pos_cpu[t])
        b.k[s].copy_(k[t].reshape(g.n_kv,g.head_dim).float())
        b.v[s].copy_(v[t].reshape(g.n_kv,g.head_dim).float())
        b.raw[s].copy_(raw[t].reshape(int(runner.indexer.index_kv_heads),g.idx_dim)[0].float())
        b.append_cells()
    runner._store_side(nreq)
