# Reused reviewed S43/v2 helpers; only Q6 shape parameter ni generalizes24/6144.
import hashlib
from pathlib import Path
import numpy as np
INF=np.float64(np.inf)
def need(ok,message):
    if not ok: raise RuntimeError(message)

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def gamma_up(n,u):
    # n*u and 1-n*u are exactly representable for the fixed counts used here.
    return np.nextafter(np.float64(n*u/(1-n*u)),INF)

def decode_q6(payload,nrows,ni):
    # Storage mapping from artifact/dequant.hpp, NOT either GPU lane/reduction tree.
    dt=np.dtype([('ql','u1',(128,)),('qh','u1',(64,)),('sc','i1',(16,)),('d','<f2')])
    need(dt.itemsize==210 and [dt.fields[n][1] for n in ('ql','qh','sc','d')]==[0,128,192,208],'Q6 layout')
    a=np.frombuffer(payload,dtype=dt).reshape(nrows,ni//256)
    d=a['d'].astype(np.float64)
    need(np.isfinite(d).all(),'nonfinite Q6 half scale')
    w=np.empty((nrows,ni//256,256),dtype=np.float64)
    si=np.arange(32)//16
    for half in range(2):
        lo0=a['ql'][:,:,half*64:half*64+32];lo1=a['ql'][:,:,half*64+32:half*64+64]
        hi=a['qh'][:,:,half*32:half*32+32]
        for part,lo in enumerate((lo0,lo1,lo0,lo1)):
            low=(lo&15) if part<2 else (lo>>4)
            q=((low|(((hi>>(2*part))&3)<<4)).astype(np.int16)-32)
            sc=a['sc'][:,:,half*8+si+part*2].astype(np.float64)
            w[:,:,half*128+part*32:half*128+(part+1)*32]=d[:,:,None]*sc*q
    need(np.isfinite(w).all(),'nonfinite decoded weight')
    nonzero=np.abs(d[d!=0])
    stat={'halves':int(d.size),'zero_scales':int((d==0).sum()),'min_nonzero_abs_half':float(nonzero.min()) if nonzero.size else None,'max_abs_half':float(np.abs(d).max())}
    return w.reshape(nrows,ni),stat
