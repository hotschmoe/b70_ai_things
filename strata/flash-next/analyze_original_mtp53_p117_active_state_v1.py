from pathlib import Path
import json,hashlib,struct,os,time
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--actor-root',type=Path,required=True)
parser.add_argument('--report-sha256',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert args.actor_root.is_absolute()and args.actor_root.resolve()==args.actor_root
assert args.output.is_absolute()and args.output.resolve()==args.output and not args.output.exists()and args.actor_root not in args.output.parents
actor_report=args.actor_root/'report.json'
assert hashlib.sha256(actor_report.read_bytes()).hexdigest()==args.report_sha256
parent=json.loads(actor_report.read_bytes());assert parent['passed']is True and parent['P117_observation_completed']is True
ROOT=args.actor_root
a=ROOT/'original-publication/p117/1-capture';b=ROOT/'original-publication/p117/2-capture'
def st(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def sha(v):return hashlib.sha256(v).hexdigest()
assert all(hashlib.sha256((p/'fields.json').read_bytes()).hexdigest()==parent['artifacts'][str((p/'fields.json').relative_to(ROOT))]for p in (a,b))
manifest=[json.loads((p/'fields.json').read_bytes())for p in (a,b)];rosters=[{v['name']:v for v in m['fields']}for m in manifest];reads=[]
def read(name,offset=0,count=None):
 result=[]
 for d,fields in zip((a,b),rosters):
  p=d/(name+'.bin');row=fields[name];before=st(p);assert p.resolve()==p and not p.is_symlink()and before[2]==row['bytes']and row['sha256']==parent['artifacts'][str(p.relative_to(ROOT))]
  hasher=hashlib.sha256()
  with p.open('rb')as f:
   for block in iter(lambda:f.read(8<<20),b''):hasher.update(block)
  assert hasher.hexdigest()==row['sha256']and st(p)==before
  n=row['bytes']-offset if count is None else count;assert 0<=offset<=row['bytes']and n<=row['bytes']-offset
  with p.open('rb')as f:f.seek(offset);raw=f.read(n)
  assert len(raw)==n and st(p)==before
  with p.open('rb')as f:f.seek(offset);echo=f.read(n)
  assert echo==raw and st(p)==before
  reads.append({'path':str(p),'offset':offset,'bytes':n,'current_range_sha256':sha(raw),'stat5':before,'producer_whole_field_sha256':row['sha256'],'whole_field_rehashed_by_this_audit':True})
  if offset==0 and n==row['bytes']:assert sha(raw)==row['sha256']
  result.append(raw)
 return result

def metrics(raw,kind,shape=None):
 x,y=[np.frombuffer(v,dtype=kind)for v in raw];assert x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all();bits_kind='<u4'if kind=='<f4'else'<u2';ux=x.view(bits_kind);uy=y.view(bits_kind);diff=ux!=uy;inds=np.flatnonzero(diff)
 xd=x.astype(np.float64);yd=y.astype(np.float64);delta=yd-xd;energy=float(np.dot(xd,xd));sign=0x80000000 if kind=='<f4'else 0x8000;mask=0xffffffff if kind=='<f4'else 0xffff
 ox=np.where((ux&sign)!=0,(~ux)&mask,ux|sign).astype(np.int64);oy=np.where((uy&sign)!=0,(~uy)&mask,uy|sign).astype(np.int64);ulp=np.abs(oy-ox)
 first=int(inds[0])if len(inds)else None
 return {'words':int(x.size),'different_bit_words':int(diff.sum()),'nmse_saved_denominator':float(np.dot(delta,delta)/energy)if energy else(0.0 if not delta.any()else None),'max_abs':float(np.max(np.abs(delta),initial=0)),'max_ulp':int(np.max(ulp,initial=0)),'first_different_word':first,'first_coordinate':list(map(int,np.unravel_index(first,shape)))if first is not None and shape else None,'first_saved':float(x[first])if first is not None else None,'first_fresh':float(y[first])if first is not None else None,'first_saved_bits':hex(int(ux[first]))if first is not None else None,'first_fresh_bits':hex(int(uy[first]))if first is not None else None,'first_ulp':int(ulp[first])if first is not None else None}
g=struct.unpack('<18q',read('stage0-geometry')[0]);assert g==(2560,48,4,128,16,48,4,10240,6144,24,2,256,4,128,4,320,512,640)
state_words=g[3]*g[5]*g[3];conv_words=g[7]*(g[6]-1)
results={'L0_recurrent_f32':metrics(read('stage0-gdn_conv',0,state_words*4),'<f4',(128,48,128)),'L0_conv_f32':metrics(read('stage0-gdn_conv',state_words*4,conv_words*4),'<f4',(10240,3)),'L1_PLE_history_f32':metrics(read('stage0-ple_history'),'<f4',(10240,9))}
for stage in range(2):
 for j in range(6):
  label='stage%d-qsa%d-'%(stage,j);layout=struct.unpack('<7q',read(label+'layout')[0]);assert layout==(0,120,2,256,4,30,128)
  layer=stage*24+3+4*j
  for kind in ('k','v'):
   raw=read(label+kind);arr=[np.frombuffer(v,dtype='<u2').reshape(30,2,4,256)for v in raw]
   active=[];padding=[]
   for x in arr:
    t=x.transpose(0,2,1,3).reshape(120,2,256);active.append(t[:117].copy().tobytes());padding.append(t[117:].copy().tobytes())
   results['L%d_%s_active117_f16'%(layer,kind)]=metrics(active,'<f2',(117,2,256));results['L%d_%s_padding117_120_raw'%(layer,kind)]={'bitwise_equal':padding[0]==padding[1],'bytes':len(padding[0]),'padding_numeric_interpretation':False}
  pooled=read(label+'pooled');results['L%d_indexer_completed29_f32'%layer]=metrics([v[:29*128*4]for v in pooled],'<f4',(29,128));results['L%d_indexer_moving_spare_f32'%layer]=metrics([v[29*128*4:]for v in pooled],'<f4',(1,128))
 # Tail contains3 slots per layer: only slot0 is current-block prefix at P117%4=1.
 tails=read('stage%d-indexer_tail'%stage);dead=read('stage%d-indexer_dead'%stage);block=read('stage%d-indexer_block_pos'%stage);assert block[0]==block[1]==struct.pack('<6i',*([112]*6))
 for j in range(6):
  layer=stage*24+3+4*j;off=j*3*128*4;results['L%d_tail_active1_f32'%layer]=metrics([v[off:off+128*4]for v in tails],'<f4',(1,128));results['L%d_tail_inactive_slots1_2_raw'%layer]={'bitwise_equal':tails[0][off+128*4:off+3*128*4]==tails[1][off+128*4:off+3*128*4],'future_overwritten_before_pooling':True,'numeric_interpretation':False};doff=j*128*4;results['L%d_indexer_dead_f32'%layer]=metrics([v[doff:doff+128*4]for v in dead],'<f4',(128,))
sdk=Path('/mnt/vm_8tb/b70/build/strata-native58-fresh52-v2-20261012T204001Z-id823bdv/source');sources=['src/core/conversation_state.cpp','sycl/src/core/conversation_snapshot.cpp','include/strata/core/layout.hpp','include/strata/core/session.hpp','include/strata/kernels/qsa.hpp','include/strata/kernels/ngram.hpp','sycl/src/kernels/cuda/verify_kernels.dp.cpp','sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp']
report={'schema':1,'scope':'Authenticated retired P117 saved-vs-fresh active-state diagnostic; current consumed field bytes rehashed; no model readmission/math/causality authority','actor_report_sha256':args.report_sha256,'source_script_sha256':sha(Path(__file__).read_bytes()),'source_sha256':{n:sha((sdk/n).read_bytes())for n in sources},'snapshot_metadata_sha256':{str(p/'fields.json'):sha((p/'fields.json').read_bytes())for p in (a,b)},'geometry':g,'P117':117,'KV_actual_format':'FP16 identity page-major [page][head][cell][dim]','KV_pagesize':4,'rounded_KV_cells':120,'indexer_block':4,'completed_pooled_rows':29,'moving_spare_row':29,'active_tail_slots':1,'stage1_ple_prev_active_operand':False,'results':results,'consumed_ranges':reads,'model_payload_read':False,'GPU_touched':False,'cause_qualified':False}
assert hashlib.sha256(actor_report.read_bytes()).hexdigest()==args.report_sha256
out=args.output;out.write_text(json.dumps(report,indent=2)+'\n',encoding='ascii');print(json.dumps({'report':str(out),'sha256':sha(out.read_bytes()),'first_meaningful_layer0':results['L0_recurrent_f32'],'L0_conv':results['L0_conv_f32'],'L1_PLE':results['L1_PLE_history_f32']},ensure_ascii=True))
