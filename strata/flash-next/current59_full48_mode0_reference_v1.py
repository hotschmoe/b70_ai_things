"""Independent original own-state prefix1 propagation; explicit current consumer lanes."""
import re
import numpy as np
from full48_owned_composition_storage_v1 import Full48OwnedComposition,OriginalStorageProjector,QsaGeometry,OwnedPle
from qsa_owned_state_storage_v1 import QsaOwnedState
from original_q80_q81_wide32_reference_v1 import original_rows,project_packets
from independent_q8_1_activation_v1 import encode
from native_storage_first_gdn_estimate_v1 import bf16_rne
from original_native_hc_bulk_f32_v1 import RowsDot32
import original_native_hc_f32_v1 as hc

def require(ok,msg):
 if not ok:raise ValueError(msg)
def xor32(values):
 v=np.asarray(values,dtype='<f4');positions=np.arange(32)
 for mask in(16,8,4,2,1):v=np.asarray(v+v[:,positions^mask],dtype='<f4')
 return v[:,0]

def bf16_source_rows(rows,x,backend,lane):
 """Bulk only transports exact per-lane FMA ordering; explicit native topology."""
 w=np.asarray(bf16_rne(rows),dtype='<f4');x=np.asarray(x,dtype='<f4');n=len(x)
 require(w.ndim==2 and w.shape[1]==n and n==2560 and np.isfinite(w).all()and np.isfinite(x).all(),'Current verifier BF16 consumers require own finiteN2560')
 if lane=='gdn_AB_group8':
  order=np.asarray([8*(i%32+32*(i//256))+((i//32)%8)for i in range(n)])
  return backend.rows(np.ascontiguousarray(w[:,order]),np.ascontiguousarray(x[order]))
 require(lane=='BF16_MMVF_pair2_block256','Unknown BF16 consumer topology')
 partials=[]
 # native_bf16.cpp n2560 chooses256 threads,5 pair iterations; no padding.
 for warp in range(8):
  order=np.asarray([2*(warp*32+i%32+256*(i//64))+((i//32)%2)for i in range(320)])
  partials.append(backend.rows(np.ascontiguousarray(w[:,order]),np.ascontiguousarray(x[order])))
 v=np.zeros((len(w),32),dtype='<f4');v[:,:8]=np.stack(partials,axis=1);return xor32(v)

class ConsumerProjector(OriginalStorageProjector):
 def __init__(self,p,backend,tile_bytes=64<<20):super().__init__(p,tile_bytes);self.backend=backend;self.contract_events=[]
 def record(self,name,lane,kind,activation,weight,unsupported=None):
  self.contract_events.append({'role':name,'original_kind':kind,'activation_storage':activation,'weight_storage':weight,'arithmetic_lane':lane,'unsupported':[]if unsupported is None else unsupported,'native_bitwise_qualified':False})
 def project(self,name,x,weight='original',activation='f32',expert=None):
  kind=self.kind(name);shape=self.p.shape(name);x=np.asarray(x,dtype='<f4');require(x.shape==(shape[0],)and np.isfinite(x).all(),'Independent own consumer operand differs')
  gdn=bool(re.fullmatch(r'blk\.\d+\.(attn_qkv|attn_gate|ssm_out)\.weight',name))
  qsa=bool(re.fullmatch(r'blk\.\d+\.attn_(q|k|v|output)\.weight',name))
  q80=kind=='Q8_0'and(gdn or qsa or name=='output.weight')
  if q80:
   require(weight=='original'and activation=='q8_1'and expert is None,'Only ordinary unpacked verifier Q80/Q81 supported')
   packet=encode(x.tobytes(),1,len(x));out=np.empty(shape[1],dtype='<f4');step=max(1,self.tile//(shape[0]//32*34+shape[0]*16))
   for at in range(0,shape[1],step):
    n=min(step,shape[1]-at);data=original_rows(self.p.reader,name,list(range(at,at+n)));out[at:at+n]=project_packets(data,packet,n,len(x))
   self.events.append({'role':name,'expert':None,'packet_sha256':__import__('hashlib').sha256(packet).hexdigest(),'stored_s_used':False})
   self.record(name,'Q80_Q81_half16_intdot_F32_product_add_XOR32_source_candidate',kind,activation,weight,['compiler_final_product_FMA_contraction_unqualified']);return out.astype(np.float64)
  ab=bool(re.fullmatch(r'blk\.\d+\.ssm_(alpha|beta)\.weight',name))
  mmvf=bool(re.fullmatch(r'blk\.\d+\.(indexer\.(q|k)_proj|ffn_gate_inp|ffn_gate_inp_shexp)\.weight',name))
  if ab or mmvf:
   require(kind in('F32','BF16')and activation=='f32'and expert is None,'BF16 consumer requires independent verifierF32 input')
   require(not ab or weight=='bf16','GDN AB caller must declare BF16 weights')
   out=np.empty(shape[1],dtype='<f4');step=max(1,self.tile//(shape[0]*8))
   for at in range(0,shape[1],step):
    n=min(step,shape[1]-at);rows=self.p.rows(name,range(at,at+n));out[at:at+n]=bf16_source_rows(rows,x,self.backend,'gdn_AB_group8'if ab else'BF16_MMVF_pair2_block256')
   self.record(name,'BF16_group8_FMA_XOR32'if ab else'BF16_MMVF_pair2_block256_FMA_subgroup_then_block_XOR32',kind,'f32','BF16_RNE',['device_FMA_gradual_semantics_unqualified']);return out.astype(np.float64)
  if name in('blk.1.ple_key.weight','blk.1.ple_value.weight'):
   require(kind=='Q8_0'and weight=='original'and activation=='f32','Source-exact PLE must not quantize activation')
   result=self.backend.project(self,name,x);self.record(name,'Original_Q80_decoded_F32_column_lane_FMA_XOR32',kind,'RAW_F32','Q80_original',['device_FMA_semantics_unqualified']);return result.astype(np.float64)
  # Explicit estimate contracts for grouped experts/shared products, not an
  # unlabelled generic BLAS fallback. Q51/K offsets use QUANTIZED code sums in
  # current iq_kernels.cpp; decoded original dot models that algebra, not their
  # integer lane/FMA/format scheduling. Stored rawQ81 sums are not substituted.
  allowed=bool(re.fullmatch(r'blk\.\d+\.ffn_(gate|up|down)_(exps|shexp)\.weight',name))
  require(allowed,'Unclassified native consumer refused '+name)
  self.record(name,'EXPLICIT_UNQUALIFIED_decoded_original_FP64_dot_F32_store',kind,activation,weight,['native_grouped_integer_dot_F32_lane_reduction_not_modeled','shared_consumer_grouping_and_format_schedule_unqualified'])
  return super().project(name,x,weight,activation,expert)

class InlineHalfMode0State(QsaOwnedState):
 def _pool(self,raw,pos):
  # Source mode0 inline convert<half,rte> then convert<float>, with no explicit
  # materialized input buffer. Existing own state stores only resulting values.
  # This models source-level RNE semantics, NOT actual compiled roundtrip proof.
  return super()._pool(raw,pos)

class Current59Full48OwnState(Full48OwnedComposition):
 def __init__(self,provider,identity,bulk_elf,bulk_sha,*,allow_unqualified_estimates):
  require(allow_unqualified_estimates is True,'Explicit estimate/nonlinear contracts must be acknowledged')
  super().__init__(provider,identity,runtime_options={'STRATA_SYCL_NATIVE_HC':'1'})
  self.rows_backend=RowsDot32(bulk_elf,bulk_sha);self.projector=ConsumerProjector(provider,self.rows_backend)
  for collection in(self.gdn,self.qsa,self.ffn):
   for model in collection.values():model.projector=self.projector
  self.ple=OwnedPle(provider,self.projector,identity)
 def hc_read(self,residual,stem,inject=True):
  self.projector.record(stem,'HC_RAW_F32_lane_FMA_XOR32_Q80weights_F32inject','Q80/F32','RAW_F32','original',['host_rsqrt_expf_vs_device_intrinsics_unqualified'])
  return self.rows_backend.hc_read_original(self.projector,residual,stem,self.g.eps,inject)
 def hc_write(self,residual,block,inject):return hc.hc_write_original(residual,block,inject)
 def tokens(self,ids):
  require(list(ids)==[248045],'Current initial recovery targets are literalprefix1 only; no long-prefill substitute')
  self.projector.events=[];self.ple.reset();states={l:m.initial_state()for l,m in self.gdn.items()};qstates={l:InlineHalfMode0State(QsaGeometry(),m.vector('indexer.k_norm.weight'),max_cells=4096)for l,m in self.qsa.items()}
  embedding=np.asarray(self.p.rows('token_embd.weight',[248045])[0],dtype='<f4');residual=np.broadcast_to(embedding,(4,2560)).copy();layers=[]
  for layer in range(48):
   before=residual.copy()
   if layer==1:residual=self.ple.advance(248045,residual)['postprojection']['result']
   attention=self.hc_read(residual,'blk.%d.hc_attn_'%layer)
   if layer in self.gdn:states[layer],detail=self.gdn[layer].mixer(attention['mixed'],states[layer],'verifier');block=detail['output']
   else:detail=self.qsa[layer].step(qstates[layer],248045,attention['mixed'],'verifier');block=detail['block_output']
   after=self.hc_write(residual,block,attention['inject']);ffn_read=self.hc_read(after,'blk.%d.hc_ffn_'%layer);ffn=self.ffn[layer].run(ffn_read['mixed'],'verifier');residual=self.hc_write(after,ffn['block_output'],ffn_read['inject'])
   layers.append({'layer':layer,'input':before,'attention':after,'ffn':residual.copy(),'router_ids_owned':ffn['router_ids_owned'],'QSA_selected_ids_owned':detail.get('selected_ids_owned')})
  final=self.hc_read(residual,'output_hc_',False);head=self.projector.project('output.weight',final['mixed'],activation='q8_1');self.rows_backend.recheck()
  return {'layers':layers,'head':head,'gdn_states_owned':states,'qsa_states_owned':qstates,'ple_history_owned':self.ple.history,'consumer_contracts':self.projector.contract_events,'packet_events':self.projector.events,'bulk_calls':self.rows_backend.calls,'captured_inputs_used':False,'mode0_inline_half_actual_compiler_qualified':False,'mode1_reference_relabelled':False,'unsupported':['host_vs_device_rsqrt_exp_log1p_rope_rounding','GDN_conv_norm_core_recurrence_F32_FMA_reduction_order','QSA_score_attention_FMA_softmax_reduction_and_ties','PLE_postprojection_norm_conv_reduction_and_nonlinearity','FFN_router_softmax_topk_ties_shared_sigmoid_and_rank_ordered_FMA_combine','grouped_Q4K_Q5K_Q51_Q80_integer_lane_schedule','mode0_inline_half_compiler_lowering_requires_native_confirmation','Q80_final_product_add_contraction_unqualified'],'numeric_pass_assigned':False,'full_model_math_qualified':False}
