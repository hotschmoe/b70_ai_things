"""Original-owned layer0 attention/GDN replay from accepted IDs and zero state only.
No capture inputs/state/route/router values accepted. Native math unqualified.
"""
import hashlib,re
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import Geometry
from full48_owned_composition_storage_v1 import Full48OwnedComposition,OriginalStorageProjector,OwnedGdn,UNSUPPORTED
from full48_owned_composition_storage_v2 import derive_schedule,bind_dispatch_source,MATH_KEYS
from native_storage_first_gdn_estimate_v1 import f32
from independent_q8_1_activation_v1 import encode

F32_SHAPES={'residual_input':(4,2560),'attn_hc_normalized':(4,2560),'attn_hc_low':(320,),'attn_hc_gate':(4,2560),'attn_hc_inject':(4,),'attn_mixed':(2560,),'gdn_state_before':(128,48,128),'gdn_conv_before':(10240,3),'gdn_qkv':(10240,),'gdn_z':(6144,),'gdn_decay_beta':(2,48),'gdn_normalized_qkv':(10240,),'gdn_output_gated':(6144,),'gdn_block_output':(2560,),'gdn_state_after':(128,48,128),'gdn_conv_after':(10240,3),'residual_after_attn':(4,2560)}
PACKET_SHAPES={'attn_input_q81':2880,'gdn_output_q81':6912}

def require(ok,msg):
 if not ok:raise ValueError(msg)
def physical_recurrent(owned):
 value=f32(owned);require(value.shape==(48,128,128),'Own recurrent layout must be [head,i,j]');return value.transpose(1,0,2).copy()
def physical_conv(owned):
 value=f32(owned);require(value.shape==(3,10240),'Own history must be [history,channel]');return value.T.copy()

class OwnedLayer0GdnReplay:
 # Reuse frozen full48 stage equations without instantiating unrelated layers/MoE.
 sigmoid=staticmethod(Full48OwnedComposition.sigmoid)
 hc_read=Full48OwnedComposition.hc_read
 hc_write=Full48OwnedComposition.hc_write
 def __init__(self,provider,source_identity_sha256,args,env,source_root,tile_bytes=64<<20):
  require(re.fullmatch('[0-9a-f]{64}',source_identity_sha256) is not None,'Source identity SHA required')
  self.p=provider;self.identity=source_identity_sha256;self.g=Geometry();self.args=tuple(args);self.env=tuple(sorted(dict(env).items()));self.source_root=Path(source_root).resolve();self.binding=bind_dispatch_source(self.source_root)
  derive_schedule([19],self.args,dict(self.env))
  if provider.actual_source:require(getattr(provider,'source_identity_sha256',None)==self.identity and provider.source_contract['effective_expert_scale']==1.0 and provider.source_contract['original_rms_epsilon']==self.g.eps,'Actual original identity/geometry/source contract differs')
  self.projector=OriginalStorageProjector(provider,tile_bytes);self.projector.require('token_embd.weight','Q8_0',(2560,248320))
  math={key:dict(self.env)[key] for key in MATH_KEYS if key in dict(self.env)}
  self.gdn=OwnedGdn(provider,self.projector,0,self.g,math)
 def tokens(self,token_ids):
  ids=list(token_ids);schedule=derive_schedule(ids,self.args,dict(self.env));binding=bind_dispatch_source(self.source_root);require(binding==self.binding,'Source3 dispatch binding changed')
  by_position={p:w for w in schedule['windows'] for p in range(w['position'],w['position']+w['rows'])}
  self.projector.events=[];state=self.gdn.initial_state();rows=[]
  for position,token in enumerate(ids):
   window=by_position[position];route=window['math_route'];embedding=f32(self.p.rows('token_embd.weight',[token])[0]);require(embedding.shape==(2560,),'Original embedding shape differs');residual=np.broadcast_to(embedding,(4,2560)).copy();before={key:value.copy() for key,value in state.items()};start=len(self.projector.events)
   attention=self.hc_read(residual,'blk.0.hc_attn_');state,detail=self.gdn.mixer(attention['mixed'],state,route);after=self.hc_write(residual,detail['output'],attention['inject'])
   # HC primitive retains down; derive its own low stage without changing primitive.
   low=f32(attention['down']/4);low=f32(low*self.sigmoid(low))
   fields={'residual_input':residual,'attn_hc_normalized':attention['normalized'],'attn_hc_low':low,'attn_hc_gate':attention['gate'],'attn_hc_inject':attention['inject'],'attn_mixed':attention['mixed'],'gdn_state_before':physical_recurrent(before['recurrent']),'gdn_conv_before':physical_conv(before['conv']),'gdn_qkv':detail['qkv'],'gdn_z':detail['z'].reshape(-1),'gdn_decay_beta':np.stack([detail['log_decay'],detail['beta']]),'gdn_normalized_qkv':np.concatenate([detail['q'].reshape(-1),detail['k'].reshape(-1),detail['v'].reshape(-1)]),'gdn_output_gated':detail['normalized_gated_mathematical_F32'].reshape(-1),'gdn_block_output':detail['output'],'gdn_state_after':physical_recurrent(state['recurrent']),'gdn_conv_after':physical_conv(state['conv']),'residual_after_attn':after}
   for name,shape in F32_SHAPES.items():require(np.shape(fields[name])==shape and np.isfinite(fields[name]).all(),'Own field shape/nonfinite differs '+name)
   packets={name:encode(np.asarray(fields[source],dtype='<f4').tobytes(),1,cols) for name,source,cols in [('attn_input_q81','attn_mixed',2560),('gdn_output_q81','gdn_output_gated',6144)]}
   require(all(len(packets[name])==size for name,size in PACKET_SHAPES.items()),'Own packet byte extent differs')
   events=[dict(event,position=position,token=token) for event in self.projector.events[start:]]
   rows.append({'position':position,'token':token,'route':route,'normal_dispatch':window['normal_dispatch'],'declared_window_position':window['position'],'declared_window_rows':window['rows'],'declared_group_count':window['groups'],'embedding':embedding,'fields_owned':fields,'packets_owned':packets,'incoming_state_owned':before,'outgoing_state_owned':{key:value.copy() for key,value in state.items()},'mixer_details_owned':detail,'hc_details_owned':attention,'packet_events_owned':events})
  require(len(rows)==len(ids),'Complete own row trace required')
  return {'lane':'owned_original_layer0_GDN_source35_route_storage_unqualified_v1','ids':ids,'rows':rows,'declared_schedule':schedule,'source_dispatch_binding':binding,'source_identity_sha256':self.identity,'captured_inputs_used':False,'captured_states_or_routes_used':False,'state_initialized_from_zero':True,'actual_original_payload_used':self.p.actual_source,'nativebitwise_qualified':False,'full_model_math_qualified':False,'numerical_tolerance_assigned':False,'actual_T_group_rounding_qualified':False,'unsupported':list(UNSUPPORTED),'logical_recurrent_layout':'[head,i,j]','native_recurrent_layout':'[i,head,j]=own.transpose(1,0,2)','logical_history_layout':'[history,channel]','native_history_layout':'[channel,history]=own.transpose(1,0)','state_owned':{key:value.copy() for key,value in state.items()}}
