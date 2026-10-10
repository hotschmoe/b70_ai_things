"""Independent original snapshots vs true native targets; no operand transfer."""
from pathlib import Path
import hashlib
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
import native_qsa40_reader_v3 as reader
from owned_layer3_qsa_control_v4 import producer_binding,arrays,candidate_arrays
from owned_layer3_qsa_helper_contract_v3 import recollect
require=reader.require

def metrics(left,right):
 require(left.shape==right.shape and np.isfinite(left).all()and np.isfinite(right).all(),'Complete finite matching target geometry required');x=left.astype(np.float64);y=right.astype(np.float64);den=float(np.square(y).sum());error=float(np.square(x-y).sum());maximum=float(np.abs(y).max());return {'shape':list(x.shape),'bitwise_equal':left.astype('<f4').tobytes()==right.astype('<f4').tobytes(),'NMSE':error/den if den else (0. if error==0 else None),'max_normalized_error':float(np.abs(x-y).max())/maximum if maximum else (0. if error==0 else None),'tolerance_used':False,'math_qualified':False}

def join(producer_root,targets):
 root=Path(producer_root).resolve();before=producer_binding(root);arrays(root);parent=read_unique(root/'report.json');work=read_unique(root/'reference-work/work-report.json');own=work['own_layer3_projection'];identity=hashlib.sha256(Path(parent['original_work_config']['model_identity']).read_bytes()).hexdigest();rows=recollect(own['records'],own['root'],identity,True);candidate=candidate_arrays(own,identity);results=[]
 projection={'hc_mixed':np.stack([x['own_HC_mixed']for x in rows]),'q_full_projected':np.stack([x['projections']['attn_q.weight'].reshape(24,2,256)for x in rows]),'k_projected':np.stack([x['projections']['attn_k.weight'].reshape(2,256)for x in rows]),'v_projected':np.stack([x['projections']['attn_v.weight'].reshape(2,256)for x in rows]),'indexer_key_projected':np.stack([x['projections']['indexer.k_proj.weight']for x in rows]),'indexer_query_projected':np.stack([x['projections']['indexer.q_proj.weight'].reshape(4,128)for x in rows]),'output_projected':np.stack([x['projections']['attn_output.weight']for x in rows]),'q_fused_RoPE':candidate['q_RoPE.f32'],'k_fused_RoPE':candidate['k_RoPE.f32'],'indexer_query_fused_RoPE':candidate['iq_RoPE.f32'],'gated':candidate['gated.f32']}
 for target in targets:
  frame=target['frame']
  if not frame['owner']:continue
  path=Path(target['metadata']);p=frame['first_position'];n=frame['rows'];items={item['name']:item for item in frame['fields']}
  for name,value in projection.items():
   expected=value[p:p+n].astype('<f4');actual=np.frombuffer(reader.consume(path.parent/items[name]['file']),dtype='<f4').reshape(expected.shape);results.append({'request':frame['request'],'first_position':p,'rows':n,'field':name,'own_input_origin':'independent_original_weight_projection_and_zero_history','native_target_is_math_input':False,**metrics(expected,actual)})
  if p==3:
   raw=reader.consume(path.parent/items['indexer_after']['file']);require(len(raw)==3076,'Exact indexer state bytes required')
   for name,offset,size,shape in [('tail',0,1536,(3,128)),('dead',1536,512,(128,)),('pooled',2048,1024,(2,128))]:results.append({'request':frame['request'],'field':'final_indexer_'+name,'native_target_is_math_input':False,**metrics(candidate['final_'+name],np.frombuffer(raw[offset:offset+size],dtype='<f4').reshape(shape))})
 require(len(results)==4*(3*11+3),'Complete allrequest/window own-localization roster required');require(canonical(producer_binding(root))==canonical(before),'Current own producer changed after comparison');return {'own_producer_binding':before,'comparisons':results,'native_captured_operands_used':False,'arithmetic_changed':False,'threshold_selected':None,'full_model_math_qualified':False}
