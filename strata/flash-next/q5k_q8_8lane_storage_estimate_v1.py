"""Source-indexed CPU NONCONTRACTED F32 grouped-row estimate for Q5_K/Q8_0.
Integer pieces/8lane schedule match source; native compiler FMA/rounding unproved.
No actual weights, activation observations or numerical PASS interface.
"""
import numpy as np
from independent_q8_1_activation_v1 import decode

def f(x):return np.float32(x)

def row_dot(raw,kind,packet,lanes=8):
 if kind not in ('Q5_K','Q8_0') or lanes!=8:raise ValueError('Only audited default SYCL8laneQ5K/Q8 role supported')
 width=176 if kind=='Q5_K' else 34;qk=256 if kind=='Q5_K' else 32;ipb=16 if kind=='Q5_K' else 4
 if not raw or len(raw)%width:raise ValueError('Weight row block extent differs')
 blocks=len(raw)//width;n=blocks*qk;image=decode(packet,1,n);codes=image['codes'].reshape(-1,32).astype(np.int64);dx=image['scale'].reshape(-1).astype(np.float32);parts=[];ideal_parts=[];coverage=np.zeros(n,dtype=int)
 for block in range(blocks):
  b=raw[block*width:(block+1)*width];d=f(np.frombuffer(b[:2],dtype='<f2')[0]);dm=f(np.frombuffer(b[2:4],dtype='<f2')[0]) if kind=='Q5_K' else None
  if not np.isfinite(d) or (dm is not None and not np.isfinite(dm)):raise ValueError('Nonfinite original blockscale')
  for part in range(ipb):
   if kind=='Q5_K':
    # iqs=2*part; bq8_offset=2*(part//4), q8 word index=part%4.
    start_group=2*(part//4);within=part%4;positions=list(range(4*within,4*within+4))+list(range(16+4*within,20+4*within));sd=f(0);sm=f(0);sd_ideal=0.;sm_ideal=0.
    for group in (start_group,start_group+1):
     scale=b[4+group]&63 if group<4 else (b[8+group]&15)|((b[group]>>6)<<4)
     minimum=b[8+group]&63 if group<4 else (b[8+group]>>4)|((b[4+group]>>6)<<4)
     w=np.asarray([((b[48+(group//2)*32+j]>>(4*(group%2)))&15)|(((b[16+j]>>group)&1)<<4) for j in positions]);x=codes[block*8+group,positions];dot=int(w@x);summation=int(x.sum());sd_ideal+=float(dx[block*8+group])*dot*scale;sm_ideal+=float(dx[block*8+group])*summation*minimum;sd=f(sd+f(dx[block*8+group]*f(dot*scale)));sm=f(sm+f(dx[block*8+group]*f(summation*minimum)));coverage[block*256+group*32+np.asarray(positions)]+=1
    parts.append(f(f(d*sd)-f(dm*sm)));ideal_parts.append(float(d)*sd_ideal-float(dm)*sm_ideal)
   else:
    positions=np.arange(part*8,part*8+8);w=np.frombuffer(b[2:],dtype=np.int8).astype(np.int64);dot=int(w[positions]@codes[block,positions]);parts.append(f(f(d*dx[block])*f(dot)));ideal_parts.append(float(d)*float(dx[block])*dot);coverage[block*32+positions]+=1
 if not np.all(coverage==1):raise AssertionError('Every original rowcolumn mustoccur exactlyonce')
 partial=np.zeros(8,dtype='<f4')
 for lane in range(8):
  for k in range(lane,len(parts),8):partial[lane]=f(partial[lane]+parts[k])
 for offset in (4,2,1):partial=f(partial+partial[np.arange(8)^offset])
 return {'unrounded_FP64_piece_algebra':sum(ideal_parts),'output_estimate':float(partial[0]),'all_columns_once':True,'row_columns':n,'piece_count':len(parts),'lanes':8,'stored_s_used':False,'source_schedule':'k=sub; k<nb*ipb; k+=8; XOR4/2/1','CPU_storage_arithmetic':'NONCONTRACTED F32 multiply/add/sub stage estimates','native_compiler_FMA_or_reductions_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None}
