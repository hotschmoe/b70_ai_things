"""Preregistered QSA layer3 own-projection control; device ops still unobserved."""
import hashlib,json
from pathlib import Path
import numpy as np
from qsa_owned_state_storage_v1 import QsaGeometry
NORM_EXPRESSION='(sycl::rsqrt(mean_F32+epsilon_F32)*input_F32)*gamma_F32'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def f32(value):return np.asarray(value,dtype='<f4')
def xor32(values):
 a=f32(values);require(a.shape==(32,),'Exact32 lane tree required')
 for mask in (16,8,4,2,1):a=f32(a+a[np.arange(32)^mask])
 return float(a[0])
def norm256_argument(row,epsilon):
 """Actual <1024-column QSA two-stage256-thread reduction, not HC's tree."""
 x=f32(row);require(x.shape in ((128,),(256,)) and np.isfinite(x).all(),'Preregistered QSA128/256 columns only');eps=f32(epsilon);require(eps.shape==() and float(eps)>0 and np.isfinite(eps),'Positive F32 epsilon required');partial=np.zeros(256,dtype='<f4');partial[:len(x)]=f32(x*x);warp=np.asarray([xor32(partial[w*32:(w+1)*32]) for w in range(8)],dtype='<f4');final=np.zeros(32,dtype='<f4');final[:8]=warp;total=np.float32(xor32(final));mean=np.float32(total/np.float32(len(x)));arg=np.float32(mean+eps);return {'square_sum_F32':float(total),'mean_F32':float(mean),'argument_F32':float(arg),'threads':256,'subgroup':32,'stages':2,'device_reduction_observed':False}
def weighted_norm_from_own_rs(row,gamma,rs):
 x=f32(row);g=f32(gamma);require(x.shape==g.shape and x.shape in ((128,),(256,)) and np.isfinite(x).all() and np.isfinite(g).all() and type(rs) in (float,np.float32,np.float64) and np.isfinite(rs) and rs>0,'Own normalized argument/shape required');return f32(f32(np.float32(rs)*x)*g)
def fp16_pool_owned(keys,values):
 k=f32(keys);v=f32(values);require(k.shape==v.shape and k.ndim==3 and 1<=k.shape[0]<=4 and k.shape[1:]==(2,256) and np.isfinite(k).all() and np.isfinite(v).all(),'Bounded own1..4 FP16 KV geometry required');return k.astype('<f2').astype('<f4'),v.astype('<f2').astype('<f4')
def all_cell_ids(committed):
 require(type(committed) is int and 1<=committed<=4 and QsaGeometry().topk==2048,'Exact bounded original selection scope required');return list(range(committed))
def geometry_contract():return {'layer':3,'tokens':4,'query_heads':24,'KV_heads':2,'head_dim':256,'indexer_heads':4,'indexer_dim':128,'rotary_dims':64,'pool_block':4,'selection':'own identity IDs0..committed-1; all1..4 cells','score_chunksize':64,'query_heads_per_KV':12,'norm_threads':256,'norm_subgroup':32,'FP16_KV':True,'device_intrinsics_qualified':False,'actual_norm_RoPE_decode_intermediates_observed':False,'captured_operands_or_selected_IDs_used':False,'full_model_math_qualified':False}
def admission(manifest):
 require(manifest['geometry']==geometry_contract() and manifest['original_weights_and_embedding_only'] is True and manifest['own_zero_KV_history'] is True and manifest['captured_inputs_used'] is False and manifest['captured_states_used'] is False and manifest['captured_selected_IDs_used'] is False and manifest['ULP_fit_or_tolerance_used'] is False,'Own original QSA provenance/scope differs');roles=manifest['own_projection_roles'];require(roles==['attn_q.weight','attn_k.weight','attn_v.weight','indexer.k_proj.weight','indexer.q_proj.weight'],'Exact own original Q/K/V/indexer role roster required');require(manifest['primitive_labels']==['projected_own','norm_argument_own','normalized_own','RoPE_own','FP16_KV_own','indexer_own','scores_own','softmax_own','attention_FMA_own','native_gate_expression_own','Q81_packet_own'],'Exact preregistered seam order required');return {'source_contract_admitted':True,'actual_runtime_or_math_qualified':False}
