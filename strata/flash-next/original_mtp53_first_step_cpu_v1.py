"""Root-only CPU original34-weight reference; native captures are comparison ONLY."""
import os
for name in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,math,struct,time
from pathlib import Path
import numpy as np
import original_mtp_cpu_decode_v1 as decoder
from independent_q8_1_activation_v1 import encode,decode as packet_decode,NATIVE
HERE=Path(__file__).resolve().parent
INVENTORY_SHA='8267da37e921300a348813ecb985dc185cde87d1918fb69d65e6aa1013b3b9cb'
WEIGHT_SHA='8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for raw in iter(lambda:f.read(8<<20),b''):h.update(raw)
 return h.hexdigest()
def signature(p):
 s=Path(p).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def f32(x):return np.asarray(x,dtype=np.float32)
def rms(x,gamma):
 x=f32(x);mean=f32(np.mean(x.astype(np.float64)**2,axis=-1,keepdims=True));inverse=f32(1/np.sqrt(f32(mean+np.float32(1e-6))));return f32(f32(x*inverse)*gamma)
def sigmoid(x):return f32(1/(1+np.exp(-f32(x))))
def silu(x):return f32(f32(x)*sigmoid(x))
class OriginalSidecar:
 def __init__(self,inventory,path):
  require(sha(inventory)==INVENTORY_SHA,'Exact original header inventory required');inv=json.loads(Path(inventory).read_bytes());files=[f for f in inv['files']if '/MTP/'in f['path']];require(len(files)==1,'Original self-contained sidecar only');self.file=files[0];self.path=Path(path).resolve();self.stat=signature(self.path);require(str(self.path)==self.file['path']and self.stat[2]==2786204800 and sha(self.path)==WEIGHT_SHA and signature(self.path)==self.stat,'Fresh exact original publisher SHA/stat required')
  proposal=json.loads((HERE/'original-publisher-mtp53-primitive-proposal-v1.json').read_bytes());expected={t['name']:(t['type'],t['shape_ggml_order'],t['packed_bytes'])for t in proposal['original_tensor_inventory']};self.tensors={t['name']:t for t in self.file['tensors']};require(len(self.tensors)==34 and {n:(t['type'],t['shape_ggml_order'],t['packed_bytes'])for n,t in self.tensors.items()}==expected,'Exact original34 role/type/shape roster required')
  self.fd=os.open(self.path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);header=os.pread(self.fd,self.file['header_bytes_read'],0);require(hashlib.sha256(header).hexdigest()==self.file['header_sha256'],'Exact original header required');self.read_bytes=0
 def raw_rows(self,name,first,count):
  t=self.tensors[name];width=t['shape_ggml_order'][0];elements,block_bytes=decoder.GEOMETRY[t['type']];row_bytes=width//elements*block_bytes;total=math.prod(t['shape_ggml_order'][1:]) if len(t['shape_ggml_order'])>1 else 1
  require(width%elements==0 and 0<=first<total and 0<count<=total-first and count*width*8<=64<<20,'Bounded whole original rows required');require(signature(self.path)==self.stat,'Original source stat changed');raw=os.pread(self.fd,count*row_bytes,t['absolute_offset']+first*row_bytes);require(len(raw)==count*row_bytes and signature(self.path)==self.stat,'Original row read failed/changed');self.read_bytes+=len(raw);return raw,t
 def rows(self,name,first,count):
  raw,t=self.raw_rows(name,first,count);return decoder.decode(raw,t['type'],'ggml_f32').reshape(count,t['shape_ggml_order'][0])
 def vector(self,name):return f32(self.rows(name,0,1)[0])
 def linear(self,name,x,packet=False,expert=None):
  x=f32(x);shape=x.shape;xs=x.reshape(-1,shape[-1]);t=self.tensors[name];width=t['shape_ggml_order'][0];require(xs.shape[1]==width,'Original matrix input width differs');out_rows=t['shape_ggml_order'][1] if len(t['shape_ggml_order'])>1 else 1;base=0 if expert is None else expert*out_rows
  activation=xs.astype(np.float64);delta=None
  if packet and t['type']not in('F32','BF16'):
   packed=packet_decode(encode(xs.astype('<f4').tobytes(),len(xs),width,NATIVE),len(xs),width);activation=packed['reconstructed_activation']
   if t['type']=='Q5_0':delta=packed['stored_sum'].reshape(len(xs),-1)-activation.reshape(len(xs),-1,32).sum(axis=2)
  y=np.empty((len(xs),out_rows),dtype=np.float32)
  for start in range(0,out_rows,128):
   count=min(128,out_rows-start);raw,_=self.raw_rows(name,base+start,count);w=decoder.decode(raw,t['type'],'ggml_f32').reshape(count,width);block=w@activation.T
   if delta is not None:
    packets=np.frombuffer(raw,dtype=np.uint8).reshape(count,width//32,22);scale=packets[:,:,:2].copy().view('<f2').reshape(count,width//32).astype(np.float64);block+=(-16*scale)@delta.T
   y[:,start:start+count]=f32(block.T)
  return y.reshape(*shape[:-1],out_rows)
 def close(self):
  require(signature(self.path)==self.stat,'Original source changed after CPU math');os.close(self.fd)
def hc(reader,residual,prefix,packet,down='down',head=False):
 xn=rms(f32(residual).reshape(4,2560),reader.vector(prefix+'norm.weight').reshape(4,2560));low=silu(f32(reader.linear(prefix+down+'.weight',xn.reshape(-1),packet)*np.float32(.25)));gate=reader.linear(prefix+'up.weight',low,packet).reshape(4,2560);products=f32(xn*sigmoid(gate));mixed=products[0].copy()
 for i in(1,2,3):mixed=f32(mixed+products[i])
 mixed=f32(mixed*np.float32(.25));inject=None if head else reader.linear(prefix+'inject.weight',xn.reshape(-1),packet)
 return mixed,inject
def scatter(residual,block,inject):return f32(f32(residual).reshape(4,2560)+f32(f32(block)[None,:]*f32(2*sigmoid(f32(inject*.25)))[:,None]))
def first_step(reader,packet=False,full=False):
 # Independent fixture-owned input formula; no native hidden/state read.
 hidden=f32((np.arange(10240)*13%101-50)/64).reshape(4,2560);embedding=f32(reader.rows('token_embd.weight',248045,1)[0]);en=rms(embedding,reader.vector('blk.48.nextn.enorm.weight'));hn=rms(hidden,reader.vector('blk.48.nextn.hnorm.weight').reshape(4,2560));concat=np.concatenate((np.tile(en,(4,1)),hn),axis=1);eh=reader.linear('blk.48.nextn.eh_proj.weight',concat,packet);mixed,inject=hc(reader,eh,'blk.48.hc_attn_',packet)
 qfull=reader.linear('blk.48.attn_q.weight',mixed,packet).reshape(24,512);q=rms(qfull[:,:256],reader.vector('blk.48.attn_q_norm.weight'));k=rms(reader.linear('blk.48.attn_k.weight',mixed,packet).reshape(2,256),reader.vector('blk.48.attn_k_norm.weight'));v=reader.linear('blk.48.attn_v.weight',mixed,packet)
 # Position0 RoPE is identity. One causal cell => attention probability1.
 values=f32(v.astype(np.float16)).reshape(2,256);attn=np.repeat(values,12,axis=0);gated=f32(attn*sigmoid(qfull[:,256:]));attn_out=reader.linear('blk.48.attn_output.weight',gated.reshape(-1),packet)
 idx_raw=reader.linear('blk.48.indexer.k_proj.weight',mixed);idx_half=idx_raw.astype(np.float16);idx_query=rms(reader.linear('blk.48.indexer.q_proj.weight',mixed).reshape(4,128),reader.vector('blk.48.indexer.q_norm.weight'));after=scatter(eh,attn_out,inject);ffn_mixed,ffn_inject=hc(reader,after,'blk.48.hc_ffn_',packet);router=reader.linear('blk.48.ffn_gate_inp.weight',ffn_mixed);ids=np.lexsort((np.arange(512),-router))[:10].astype(np.int32);ex=f32(np.exp(f32(router-np.max(router))));prob=f32(ex/np.float32(np.sum(ex.astype(np.float64))));selected=prob[ids];weights=f32(selected/np.float32(max(float(np.sum(selected.astype(np.float64))),6.103515625e-5)))
 fields={'EH':eh.reshape(-1),'q':q.reshape(-1),'k':k.reshape(-1),'v':v,'idx_raw':idx_raw,'idx_half':idx_half,'idx_query':idx_query.reshape(-1),'attn_out':attn_out,'router':router,'expert_ids':ids,'expert_weights':weights}
 if full:
  parts=[]
  for expert in ids.tolist():
   gate=reader.linear('blk.48.ffn_gate_exps.weight',ffn_mixed,packet,expert);up=reader.linear('blk.48.ffn_up_exps.weight',ffn_mixed,packet,expert);parts.append(reader.linear('blk.48.ffn_down_exps.weight',f32(silu(gate)*up),packet,expert))
  sg=reader.linear('blk.48.ffn_gate_shexp.weight',ffn_mixed,packet);su=reader.linear('blk.48.ffn_up_shexp.weight',ffn_mixed,packet);shared=reader.linear('blk.48.ffn_down_shexp.weight',f32(silu(sg)*su),packet);shared=f32(shared*sigmoid(reader.linear('blk.48.ffn_gate_inp_shexp.weight',ffn_mixed))[0]);ffn_out=f32(weights[0]*parts[0])
  for weight,part in zip(weights[1:],parts[1:]):ffn_out=f32(ffn_out+f32(weight*part))
  ffn_out=f32(ffn_out+shared)
  next_hidden=scatter(after,ffn_out,ffn_inject);head,_=hc(reader,next_hidden,'blk.48.nextn.hc_head_',packet,head=True);logits=reader.linear('output.weight',head,packet);fields.update(ffn_out=ffn_out,next_hidden=next_hidden.reshape(-1),head_mixed=head,logits=logits)
 return fields
def compare(fields,capture,expected):
 result={}
 for name,own in fields.items():
  p=Path(capture)/('direct0-'+name+'.bin');dtype='<i4' if name=='expert_ids' else '<f2' if name=='idx_half' else '<f4';require(p.is_file()and not p.is_symlink(),'Original immutable target field required');before=signature(p);raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected['fields/'+p.name] and signature(p)==before,'Original native comparison SHA/stat differs');native=np.frombuffer(raw,dtype=dtype).reshape(own.shape);require(np.isfinite(own).all()and np.isfinite(native).all(),'Nonfinite comparison')
  x=own.astype(np.float64);d=x-native.astype(np.float64);metric={'shape':list(own.shape),'native_sha256':hashlib.sha256(raw).hexdigest(),'nmse':float(np.sum(d*d)/max(1e-30,np.sum(x*x))),'max_abs':float(np.max(np.abs(d))),'bitwise_equal':own.astype(dtype).tobytes()==raw}
  if name=='expert_ids':metric['exact_IDs_equal']=bool(np.array_equal(own,native))
  if name=='logits':
   best=int(np.argmax(own));native_best=int(np.argmax(native));metric.update(reference_top10=np.argsort(-own,kind='stable')[:10].tolist(),native_top10=np.argsort(-native,kind='stable')[:10].tolist(),reference_top1=best,native_top1=native_best,native_top1_rank_in_reference=int(np.sum(own>own[native_best]))+1,reference_top1_rank_in_native=int(np.sum(native>native[best]))+1)
  result[name]=metric
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--original-mtp',type=Path,required=True);p.add_argument('--capture-fields',type=Path);p.add_argument('--full-head',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Fresh CPU reference directory required');a.output.mkdir();started=time.monotonic();reader=OriginalSidecar(a.inventory,a.original_mtp);report={'scope':'position0/token248045 original34-owned-input reference, not main MTP verification','token':248045,'position':0,'input_hidden_provenance':'independent fixture formula (13*i mod101-50)/64; native capture not consumed','original_weight_sha256':WEIGHT_SHA,'lanes':{},'full_numerical_gate_passed':False,'GPU_execution_performed':False,'tolerance_fitted':False}
 targets=None
 if a.capture_fields:
  a.capture_fields=a.capture_fields.resolve();record=json.loads((a.capture_fields.parent/'report.json').read_bytes());require(record['loaded_component_passed']is True and record['passed']is True and record['owned_removed']is True and record['exit_code']==0 and record['post_original_sha256']==WEIGHT_SHA,'Original completed original-weight component required');targets=record['fields'];own=f32((np.arange(10240)*13%101-50)/64);require(hashlib.sha256(own.astype('<f4').tobytes()).hexdigest()==targets['fields/own-initial-hidden.f32'],'Independently generated owned input does not match fixture provenance')
 try:
  for label,packet in [('original_float_activation_f32_storage',False),('conditional_owned_native_Q8_1',True)]:
   fields=first_step(reader,packet,a.full_head);directory=a.output/label;directory.mkdir()
   for name,value in fields.items():value.astype('<i4' if name=='expert_ids' else '<f2' if name=='idx_half' else '<f4').tofile(directory/(name+'.bin'))
   report['lanes'][label]={'activation_quantization':NATIVE if packet else 'none','weight_decoder_lane':'ggml_f32','dot_accumulation':'independent NumPy float64 sum then F32 node store; not native/ggml reduction parity','Q5_0_raw_sum_affine_correction':packet,'native_reference_inputs_consumed':False,'routing_is_independently_computed':True,'fields':list(fields),'comparison':compare(fields,a.capture_fields,targets)if a.capture_fields else None}
 finally:reader.close()
 report.update(seconds=time.monotonic()-started,encoded_row_bytes_read=reader.read_bytes,full_first_head_computed=a.full_head,history_state_arrays_compared=False)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(a.output),'seconds':report['seconds'],'full_numerical_gate_passed':False}))
if __name__=='__main__':main()
