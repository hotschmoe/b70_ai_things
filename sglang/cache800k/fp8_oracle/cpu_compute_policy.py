import sys,json,torch
# Run beside oracle.py using a CPU-enabled Torch interpreter.
from oracle import normalized_reference, reference

def decode(q,k,v,ks,vs,qtype,ptype):
 q=q.to(qtype).float()[0];k=k.float().repeat_interleave(6,1);v=v.float().repeat_interleave(6,1)
 vals=[];lses=[]
 for first,last in [(0,64),(64,67)]:
  m=torch.full((24,),float('-inf'));den=torch.zeros(24);acc=torch.zeros(24,256)
  for start in range(first,last,32):
   score=torch.einsum('hd,khd->hk',q,k[start:min(start+32,last)])*(ks/16)
   nm=torch.maximum(m,score.max(-1).values);alpha=(m-nm).exp();p=(score-nm[:,None]).exp()
   acc=acc*alpha[:,None]+torch.einsum('hk,khd->hd',p.to(ptype).float(),v[start:min(start+32,last)])
   den=den*alpha+p.sum(-1);m=nm
  vals.append(acc/den[:,None]);lses.append(m+den.log())
 weights=torch.stack(lses).softmax(0)
 return (sum(v*w[:,None] for v,w in zip(vals,weights))*vs).half().float()[None]

def extend(q,k,v,k0,v0,ks,vs,qtype,ptype):
 q=q.transpose(0,1).float();qlen=q.shape[1];prefix=67-qlen
 m=torch.full((24,qlen),float('-inf'));den=torch.zeros(24,qlen);acc=torch.zeros(24,qlen,256)
 for first,last,kk,vv,qt,pt,sc,vc in [(0,prefix,k,v,qtype,ptype,ks,vs),(prefix,67,k0,v0,torch.float16,torch.float16,1.,1.)]:
  kk=kk.float().repeat_interleave(6,1);vv=vv.float().repeat_interleave(6,1)
  for start in range(first,last,32):
   stop=min(start+32,last);scores=torch.einsum('hqd,khd->hqk',q.to(qt).float(),kk[start:stop])*(sc/16)
   allowed=torch.arange(start,stop)[None,:]<=prefix+torch.arange(qlen)[:,None]
   scores.masked_fill_(~allowed[None,:,:],float('-inf'))
   nm=torch.maximum(m,scores.max(-1).values);alpha=(m-nm).exp();prob=(scores-nm[:,:,None]).exp()
   acc=acc*alpha[:,:,None]+torch.einsum('hqk,khd->hqd',prob.to(pt).float(),vv[start:stop])*vc
   den=den*alpha+prob.sum(-1);m=nm
 return (acc/den[:,:,None]).transpose(0,1).half().float()

torch.set_num_threads(2);g=torch.Generator().manual_seed(8123);k0=torch.randn(67,4,256,generator=g).half();v0=torch.randn(67,4,256,generator=g).half();ks=float(torch.tensor(.029828752790178572));vs=float(torch.tensor(.02104317801339286));k=normalized_reference(k0,ks,'xpu').to(torch.float8_e4m3fn);v=normalized_reference(v0,vs,'xpu').to(torch.float8_e4m3fn);rows=[]
for qlen in (1,4):
 q=torch.randn(qlen,24,256,generator=g).half()
 for mode in (['decode','extend'] if qlen==1 else ['extend']):
  for label,km,vm in [('correct',1,1),('Kx2',2,1),('Vx2',1,2)]:
   rk,rv=k.float()*ks*km,v.float()*vs*vm
   if mode=='extend':rk=torch.cat([rk[:67-qlen],k0[67-qlen:].float()]);rv=torch.cat([rv[:67-qlen],v0[67-qlen:].float()])
   expected=reference(q,rk,rv,67-qlen)
   for policy,qt,pt in [('Q_FP8_P_FP8',torch.float8_e4m3fn,torch.float8_e4m3fn),('Q_FP16_P_FP16',torch.float16,torch.float16)]:
    actual=decode(q,k,v,ks*km,vs*vm,qt,pt) if mode=='decode' else extend(q,k,v,k0,v0,ks*km,vs*vm,qt,pt)
    rows.append({'mode':mode,'qlen':qlen,'case':label,'policy':policy,'max_abs':float((actual-expected).abs().max()),'rmse':float((actual-expected).square().mean().sqrt()),'mismatched':int((~torch.isclose(actual,expected,rtol=.03,atol=.015)).sum())})
assert len(rows)==18
assert all(row['mismatched']==0 for row in rows if row['policy']=='Q_FP16_P_FP16')
assert rows[0]['mismatched']==48 and abs(rows[0]['max_abs']-.03187394142150879)<1e-10
print(json.dumps({'scope':'CPU arithmetic model; grouped decode splits64+3, standardextend, tiles32; not actual Triton dot execution','rows':rows},indent=2))
