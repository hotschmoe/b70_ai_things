"""Text prefill append from existing native batch primitives.

Private pooled scratch pages prevent neighboring-block sentinel writes
from racing. Only completed pooled rows are copied back to live pages.
"""
import torch
from . import ops


def append_batch(runner,s,t0,t1,k,v,raw,positions,pos_cpu,nreq):
    b=runner.batch
    g=b.geom
    n=t1-t0
    if n<8:
        return False
    first=int(pos_cpu[t0])
    if pos_cpu[t0:t1]!=list(range(first,first+n)) or int(b.pos_base[s])!=0:
        return False
    if positions.ndim==2 and (
        not torch.equal(positions[0,t0:t1],positions[1,t0:t1]) or
        not torch.equal(positions[0,t0:t1],positions[2,t0:t1])):
        return False
    dev=b.device
    r=g.idx_block
    dim=g.idx_dim
    bs=b.block_size
    b.pos.fill_(-1)
    b.pos[s]=first
    runner._prepare_side(nreq)
    oldtail=b.tail[s].clone()
    if positions.ndim==2:
        if b.mrope.numel()==0:
            b.mrope=torch.zeros(b.batch,b.max_cells,3,dtype=torch.int32,device=dev)
        b.mrope[s].index_copy_(0,positions[0,t0:t1].long(),
            positions[:,t0:t1].transpose(0,1).to(torch.int32))
    keys=k[t0:t1].reshape(n,g.n_kv,g.head_dim).float().contiguous()
    values=v[t0:t1].reshape(n,g.n_kv,g.head_dim).float().contiguous()
    rows=raw[t0:t1].reshape(n,int(runner.indexer.index_kv_heads),dim)[:,0].float().contiguous()
    pos=torch.arange(first,first+n,device=dev,dtype=torch.int32)
    step=torch.empty(n,4,device=dev,dtype=torch.int32)
    ops.write_step_batch(pos,step,g.idx_top_k,r)
    tables=b.block_table[s:s+1].expand(n,-1).contiguous()
    ops.kv_append_batch(keys,values,b.pages,tables,step,bs,r,dim,1)
    if first==0:
        b.raw[s].copy_(rows[0])
        ops.write_step_batch(b.pos,b.step,g.idx_top_k,r)
        ops.indexer_append_batch(b.raw,b.k_norm,b.tail,b.dead,b.block_pos,
            b.pages,b.block_table,b.cos,b.sin,b.step,b.mrope,b.pos_base,g.eps,bs)
    ends_cpu=[p for p in range(first,first+n) if p%r==r-1]
    # Bound scratch even when hybrid-cache alignment chooses large pages.
    pool_rows=max(1,256*1024**2//int(b.pages.shape[1]))
    for pool_start in range(0,len(ends_cpu),pool_rows):
        chosen=ends_cpu[pool_start:pool_start+pool_rows]
        ends=torch.tensor(chosen,device=dev,dtype=torch.int32)
        nb=len(chosen)
        which=torch.arange(nb,device=dev)
        wanted=ends[:,None]-(r-1)+torch.arange(r-1,device=dev)[None,:]
        past=rows[(wanted-first).clamp(min=0).long()]
        tails=torch.where((wanted>=first)[:,:,None],past,oldtail[None]).contiguous()
        dead=b.dead[s:s+1].expand(nb,-1).contiguous()
        block_pos=torch.zeros(nb,device=dev,dtype=torch.int32)
        scratch=torch.empty(nb,b.pages.shape[1],device=dev,dtype=torch.uint8)
        table=torch.zeros(nb,b.max_blocks,device=dev,dtype=torch.int32)
        logical=(ends//bs).long()
        table[which,logical]=which.to(torch.int32)
        pstep=torch.empty(nb,4,device=dev,dtype=torch.int32)
        ops.write_step_batch(ends,pstep,g.idx_top_k,r)
        ops.indexer_append_batch(rows[(ends-first).long()].contiguous(),b.k_norm,
            tails,dead,block_pos,scratch,table,b.cos,b.sin,pstep,
            torch.empty(0,device=dev,dtype=torch.int32),
            torch.zeros(nb,device=dev,dtype=torch.int32),g.eps,bs)
        offset=(int(b.pages.shape[1])-bs//r*dim*4)//4
        cols=offset+((ends//r)%(bs//r))[:,None]*dim+torch.arange(dim,device=dev)[None,:]
        pooled=scratch.view(torch.float32)[which[:,None],cols]
        physical=b.block_table[s,logical].long()
        pages=b.pages.view(torch.float32)
        pages[physical[:,None],cols]=pooled
        last=chosen[-1]
        # The sequential op leaves one dead sentinel after the last pool,
        # except at a physical page boundary.
        if (last//r+1)*r//bs==last//bs:
            page=b.block_table[s,last//bs].long().reshape(1)
            col=offset+((last//r+1)%(bs//r))*dim+torch.arange(dim,device=dev)
            pages[page[:,None],col[None,:]]=b.dead[s:s+1]
        b.block_pos[s].copy_(block_pos[-1])
        del scratch
    last=first+n-1
    group=last//r*r
    final=[]
    for j in range(r-1):
        target=group+j if j<=last%r else group-r+j
        final.append(rows[target-first] if target>=first else oldtail[j])
    b.tail[s].copy_(torch.stack(final))
    b.k[s].copy_(keys[-1])
    b.v[s].copy_(values[-1])
    b.raw[s].copy_(rows[-1])
    b.pos[s]=last
    ops.write_step_batch(b.pos,b.step,g.idx_top_k,r)
    b._commit_length()
    runner._store_side(nreq)
    return True
