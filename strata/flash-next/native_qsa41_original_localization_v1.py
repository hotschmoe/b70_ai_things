"""Independent original snapshots vs true native targets; no operand transfer."""
from pathlib import Path
import hashlib
import numpy as np
from explore_full48_original_prefix1_v4 import compare_observation
from serial37_canonical_json_v3 import read_unique,canonical
import native_qsa41_targets_v1 as reader
from owned_layer3_qsa_control_v4 import arrays,candidate_arrays
from indexer_materialized41_masked_inputs_v2 import producer_binding as reference_binding,operation_witness
def producer_binding(root,model_association):
 proof=reference_binding(root);require(canonical(proof["current_postboot_reference"]["model_identity_association"])==canonical(model_association),"Exact admitted original model association required");return proof
from owned_layer3_qsa_helper_contract_v3 import recollect
require=reader.require

def metrics(left,right):
 require(left.shape==right.shape and np.isfinite(left).all()and np.isfinite(right).all(),'Complete finite matching target geometry required');x=left.astype(np.float64);y=right.astype(np.float64);den=float(np.square(y).sum());error=float(np.square(x-y).sum());maximum=float(np.abs(y).max());return {'shape':list(x.shape),'bitwise_equal':left.astype('<f4').tobytes()==right.astype('<f4').tobytes(),'NMSE':error/den if den else (0. if error==0 else None),'max_normalized_error':float(np.abs(x-y).max())/maximum if maximum else (0. if error==0 else None),'normalization_reference':'native_actual_as_in_frozen_localization_v9','tolerance_used':False,'math_qualified':False}

def layer3_join(producer_root,targets,model_association):
 root=Path(producer_root).resolve();before=producer_binding(root,model_association);arrays(root);parent=read_unique(root/'report.json');work=read_unique(root/'reference-work/work-report.json');own=work['own_layer3_projection'];identity=hashlib.sha256(Path(parent['original_work_config']['model_identity']).read_bytes()).hexdigest();rows=recollect(own['records'],own['root'],identity,True);candidate=candidate_arrays(own,identity);results=[]
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
 require(len(results)==4*(3*11+3),'Complete allrequest/window own-localization roster required');require(canonical(producer_binding(root,model_association))==canonical(before),'Current own producer changed after comparison');return {'own_producer_binding':before,'comparisons':results,'native_captured_operands_used':False,'arithmetic_changed':False,'threshold_selected':None,'full_model_math_qualified':False}


def load_owned(field,work_root):
 path=Path(field['path']);require(path.parent==work_root and path.resolve()==path and not path.is_symlink(),'Exact independent owned array path required');raw=reader.consume(path,1<<20);require(len(raw)==field['bytes']and hashlib.sha256(raw).hexdigest()==field['sha256'],'Independent owned consumed bytes/SHA differ');return np.frombuffer(raw,dtype='<f4').reshape(field['shape']).copy()

def full_prefix_join(producer_root,native):
 """All four repeated heads and4x48x3 phase rows against unchanged owned math."""
 root=Path(producer_root).resolve();before=reference_binding(root);report=read_unique(root/'reference-work/work-report.json');require(report['prefix4_attempted']is True and report['first_gate']['passed']is True,'Original exact first gate and completed independent prefix4 required');own=report['prefix4']['arrays'];work=root/'reference-work';results=[];seen=set()
 for frame in native['p30']['frames']:
  b=frame['binding'];p=b['first_position'];n=b['rows'];request=b['request']
  for field in frame['fields']:
   raw=reader.consume(field['path'],81920);require(len(raw)==field['bytes']and hashlib.sha256(raw).hexdigest()==field['sha256'],'Actual native P30 consumed bytes differ');target=np.frombuffer(raw,dtype='<f4').reshape(n,4,2560)
   for offset in range(n):
    key='p%d_l%d_%s'%(p+offset,field['layer'],field['phase']);coordinate=(request,key);require(coordinate not in seen,'Duplicate original/native phase row');seen.add(coordinate);expected=load_owned(own[key],work);results.append({'request':request,'key':key,'native_target_is_math_input':False,'normalization_reference':'native_actual_as_in_frozen_compare_observation','tolerance_used':False,**compare_observation(expected,target[offset])})
 roster={(request,'p%d_l%d_%s'%(p,layer,phase))for request in range(1,5)for p in range(4)for layer in range(48)for phase in('input','attention','ffn')};require(seen==roster,'Complete2304 original/native phase rows required')
 require([row['ordinal']for row in native['rows']]==[1,2,3,4],'Complete unique original/native head request roster required')
 head=load_owned(own['head'],work)
 for row in native['rows']:
  field=row['meta']['logits'][0];raw=reader.consume(field['path'],1<<20);require(len(raw)==int(field['bytes'])and hashlib.sha256(raw).hexdigest()==field['sha256'],'Actual complete native head bytes differ');value=np.frombuffer(raw,dtype='<f4').reshape(head.shape);results.append({'request':row['ordinal'],'key':'head','native_target_is_math_input':False,'normalization_reference':'native_actual_as_in_frozen_compare_observation','tolerance_used':False,**compare_observation(head,value)})
 require(len(results)==2308 and canonical(reference_binding(root))==canonical(before),'Complete4x577/current independent reference required')
 return {'comparisons':results,'original_reference_binding':before,'original_arithmetic_changed':False,'captured_operands_used':False,'tolerance_used':False,'full_model_math_qualified':False}

def join(producer_root,native):
 source=reader.source_binding();proof=reference_binding(producer_root);association=proof['current_postboot_reference']['model_identity_association'];source33=original_input33(native['source33'],association);layers=layer3_join(producer_root,native['targets'],association);full=full_prefix_join(producer_root,native)
 require(canonical(reference_binding(producer_root))==canonical(proof),'Independent original reference changed after all41 comparisons')
 require(reader.source_binding()==source and reader.actor_header(native['actor_binding'])==Path(native['actor_binding']['capture_root']),'Current actual41 source/actor binding changed after comparisons')
 return {'source_generation':41,'configured_materialized_mode':native['materialized_mode'],'observer_enabled':native['observer_enabled'],'layer3':layers,'full48_and_head':full,'original_input33':source33,'source40_alias_or_runtime_view_used':False,'reference_or_quantization_changed':False,'current_reference_operation_witness':operation_witness(),'full_model_math_qualified':False}


def original_input33(native,association):
 # Exact existing verification primitive, private explicit target reader namespace.
 import ast,copy
 import native_qsa40_input33_v9 as original
 from original_tensor_rows_postboot_v1 import BoundOriginalFull48RowsPostboot
 path=Path(original.__file__);node=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='verify');namespace={**vars(original),'reader':reader};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),str(path)+'[explicit41-target-provider]','exec'),namespace)
 provider=BoundOriginalFull48RowsPostboot(reader.qsa.HEADER.parent/'original-gguf-reference-foundation-plan-v1.json',association['historical_identity_path'],model_association=association)
 return namespace['verify'](native,association['historical_identity_path'],provider=provider)
