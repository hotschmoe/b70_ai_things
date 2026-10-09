"""Header inventory and bounded SYNTHETIC full48 state/schedule math fixture.
Not an actual original/native full-model reference; no model payload reader.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from original_gguf_vector_decoder_v2 import GEOMETRY
from original_math_scalar import gdn_step,ple_dilated_conv,sigmoid
from qsa_owned_state_storage_v1 import QsaOwnedState,QsaGeometry

INVENTORY=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/current-gguf-inventory.json')


def inventory_contract(path=INVENTORY):
 data=json.loads(Path(path).read_text());tensors=[tensor for file in data['files'] if '/UD-Q4_K_XL/' in file['path'] for tensor in file['tensors']];byname={tensor['name']:tensor for tensor in tensors}
 if len(tensors)!=1224 or len(byname)!=1224:raise ValueError('Complete selected original1224 source roster required')
 formats={tensor['type'] for tensor in tensors};missing=sorted(formats-set(GEOMETRY))
 gdn=[layer for layer in range(48) if 'blk.%d.attn_qkv.weight'%layer in byname];qsa=[layer for layer in range(48) if 'blk.%d.attn_q.weight'%layer in byname]
 if gdn!=[layer for layer in range(48) if layer%4!=3] or qsa!=list(range(3,48,4)):raise ValueError('Original36GDN/12QSA layer schedule differs')
 for layer in range(48):
  for half in ('attn','ffn'):
   for suffix in ('norm','down','up','inject'):
    if 'blk.%d.hc_%s_%s.weight'%(layer,half,suffix) not in byname:raise ValueError('Original HC half role missing')
  for suffix in ('gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp','gate_inp','gate_inp_shexp'):
   if 'blk.%d.ffn_%s.weight'%(layer,suffix) not in byname:raise ValueError('Original FFN role missing')
 exceptions={'Q5_K_gate_up_layers':sorted({int(name.split('.')[1]) for name,tensor in byname.items() if tensor['type']=='Q5_K'}),'Q8_0_expert_down_layers':sorted(int(name.split('.')[1]) for name,tensor in byname.items() if name.endswith('.ffn_down_exps.weight') and tensor['type']=='Q8_0')}
 if exceptions!={'Q5_K_gate_up_layers':[2],'Q8_0_expert_down_layers':[2,4,30,46,47]}:raise ValueError('Mixed original FFN exception map differs')
 return {'header_inventory_complete':True,'source_tensors':len(tensors),'decoder_formats':sorted(formats),'unsupported_decoder_formats':missing,'gdn_layers':gdn,'qsa_layers':qsa,'ple_layers':[1],'ffn_exception_map':exceptions,'inventory_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'native_full48_ready':False,'full_model_math_qualified':False,'model_payload_reads':0}


class SyntheticFull48Fixture:
 """Small real math primitives exercise ownership/order, not model numerics.
 Synthetic HC/FFN/embedding/head and geometry are deliberately distinct.
 """
 def run(self,ids):
  ids=list(ids)
  if len(ids) not in (1,2,4,8) or any(type(token) is not int for token in ids):raise ValueError('Bounded synthetic prefix1/2/4/8 required')
  gdn={layer:np.zeros((4,4,4)) for layer in range(48) if layer%4!=3};qsa={layer:QsaOwnedState(QsaGeometry(heads=4,kv_heads=2,dim=4,rot=4,idx_heads=4,idx_dim=4,topk=8),max_cells=16) for layer in range(3,48,4)};ple=np.zeros((9,16));trace=[];last2=[-1,-1]
  for pos,token in enumerate(ids):
   R=np.tile(np.sin(np.arange(4)+token*.05),(4,1));route='verifier' if pos==len(ids)-1 else 'prefill';layers=[]
   for layer in range(48):
    before=R.copy()
    if layer==1:
     ple,current=ple_dilated_conv(ple,R.reshape(-1),np.full((4,16),.01));R=R+.01*current.reshape(4,4)
    normalized=R/np.sqrt(np.mean(R*R,axis=1,keepdims=True)+1e-6);mixed=np.mean(normalized*.5,axis=0)
    if layer%4!=3:
     q=np.stack([mixed,np.roll(mixed,1)]);k=q*.1;v=np.tile(mixed,(4,1))*.2;gdn[layer],out=gdn_step(gdn[layer],q,k,v,np.full(4,-.1),np.full(4,.25));block=np.mean(out,axis=0)
    else:
     q=np.tile(mixed,(4,1));k=np.stack([mixed,mixed*.5]);v=k*.2;result=qsa[layer].advance(token,q,k,v,np.zeros((4,4)),mixed,np.tile(mixed,(4,1)));block=np.mean(result['attention_gated_F32'],axis=0)
    R=R+.01*block;post_attn=R.copy();ffn=mixed*sigmoid(mixed)*np.cos(layer+.1*np.arange(4));R=R+.01*ffn
    layers.append({'layer':layer,'family':'qsa' if layer%4==3 else 'gdn','route_label_only':route,'before':before,'after_attn':post_attn,'after_ffn':R.copy(),'state_owner':layer})
   last2=[last2[1],token];final=np.mean(R,axis=0);logits=np.cos(np.arange(16)[:,None]+np.arange(4)[None,:])@final;trace.append({'token':token,'position':pos,'layers':layers,'logits_toy16':logits})
  return {'scope':'SYNTHETIC dimensions4/heads4/vocab16 only; schedule/state math fixture, not actual model estimator','tokens':ids,'trace':trace,'gdn_states':gdn,'qsa_states':qsa,'ple_history':ple,'last_two':last2,'real_original_weights_used':False,'actual_native_storage_routes_executed':False,'full_model_math_qualified':False,'tolerance_gate':None}
