#!/usr/bin/env python3
"""CPU-only deterministic conditional-operator and source-binding controls."""
import copy,json,math,unittest,re,tempfile,hashlib
from independent_q8_1_activation_v1 import encode
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import verify_layer0_gdn_original_v1 as c

class ConditionalTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(378)
  self.v={name:rng.normal(0,.15,size=size) for name,size in c.F32.items()}
  self.gamma=rng.normal(1,.1,128);self.taps=rng.normal(0,.2,(10240,4))
  self.v['gdn_decay_beta']=np.concatenate([np.full(48,-.1),np.full(48,.3)])
  qkv=self.v['gdn_normalized_qkv'];state=self.v['gdn_state_before'].reshape(128,48,128).transpose(1,0,2)
  next_state,core=c.gdn_step(state,qkv[:2048].reshape(16,128),qkv[2048:4096].reshape(16,128),qkv[4096:].reshape(48,128),np.full(48,-.1),np.full(48,.3),lane='original_fp64')
  core/=math.sqrt(128)
  self.v['gdn_state_after']=next_state.transpose(1,0,2).reshape(-1)
  self.v['gdn_output_gated']=(core*self.gamma/np.sqrt(np.mean(core**2,axis=1,keepdims=True)+c.EPS)*c.sigmoid(self.v['gdn_z'].reshape(48,128))).reshape(-1)
 def test_gdn_positive(self):
  self.assertTrue(all(x['passed'] for x in c.conditional_gdn(self.v,self.gamma).values()))
 def test_incoming_state_mapping_negative(self):
  v=copy.deepcopy(self.v);v['gdn_state_before']=v['gdn_state_before'].reshape(128,48,128).transpose(1,0,2).reshape(-1)
  self.assertFalse(all(x['passed'] for x in c.conditional_gdn(v,self.gamma).values()))
 def test_normalized_input_negative(self):
  v=copy.deepcopy(self.v);v['gdn_normalized_qkv']*=1.2
  self.assertFalse(all(x['passed'] for x in c.conditional_gdn(v,self.gamma).values()))
 def test_gamma_weight_negative(self):
  self.assertFalse(c.conditional_gdn(self.v,self.gamma*1.1)['gdn_output_gated_supplied_state_qkv_beta_z_original_gamma']['passed'])
 def test_nonfinite_negative(self):
  gamma=self.gamma.copy();gamma[0]=np.nan
  with self.assertRaises(ValueError):c.conditional_gdn(self.v,gamma)
 def conv_expected(self):
  full=np.column_stack([self.v['gdn_conv_before'].reshape(10240,3),self.v['gdn_qkv']]);x=c.sigmoid(np.sum(full*self.taps,axis=1))*np.sum(full*self.taps,axis=1)
  for start in (0,2048):
   heads=x[start:start+2048].reshape(16,128);x[start:start+2048]=(heads/np.sqrt((heads**2).sum(axis=1,keepdims=True)+c.EPS)).reshape(-1)
  self.v['gdn_conv_after']=full[:,1:].reshape(-1);self.v['gdn_normalized_qkv']=x
 def test_conv_positive_and_tap_shift_negative(self):
  self.conv_expected();self.assertTrue(all(x['passed'] for x in c.conditional_convolution(self.v,self.taps).values()))
  self.assertFalse(all(x['passed'] for x in c.conditional_convolution(self.v,self.taps[:,::-1]).values()))
  self.v['gdn_conv_after']=np.roll(self.v['gdn_conv_after'],1)
  self.assertFalse(c.conditional_convolution(self.v,self.taps)['conv_history_shift_supplied_history_qkv']['passed'])
 def test_sum_vs_mean_epsilon_negative(self):
  self.conv_expected();self.v['gdn_normalized_qkv'][:4096]*=math.sqrt(128)
  self.assertFalse(c.conditional_convolution(self.v,self.taps)['conv_silu_qk_sum_l2_supplied_history_qkv_original_F32_taps']['passed'])
 def test_small_norm_epsilon_negative(self):
  self.v['gdn_qkv']*=1e-5;self.v['gdn_conv_before']*=1e-5;self.conv_expected()
  full=np.column_stack([self.v['gdn_conv_before'].reshape(10240,3),self.v['gdn_qkv']]);conv=np.sum(full*self.taps,axis=1);activated=conv*c.sigmoid(conv)
  for start in (0,2048):
   heads=activated[start:start+2048].reshape(16,128);self.v['gdn_normalized_qkv'][start:start+2048]=(heads/np.sqrt(np.sum(heads**2,axis=1,keepdims=True)+128*c.EPS)).reshape(-1)
  self.assertFalse(c.conditional_convolution(self.v,self.taps)['conv_silu_qk_sum_l2_supplied_history_qkv_original_F32_taps']['passed'])
 def test_nonfinite_supplied_state_negative(self):
  self.v['gdn_state_before'][0]=np.nan
  with self.assertRaises(ValueError):c.conditional_gdn(self.v,self.gamma)
 def test_projection_input_and_weight_negative(self):
  packets={'attn_input_q81':np.arange(2560,dtype=float),'gdn_output_q81':np.arange(6144,dtype=float)}
  outputs={'blk.0.attn_qkv.weight':'gdn_qkv','blk.0.attn_gate.weight':'gdn_z','blk.0.ssm_out.weight':'gdn_block_output'}
  def project(role,x):return np.full(c.F32[outputs[role]],x.sum())
  layer=SimpleNamespace(project=project)
  for role,key in outputs.items():self.v[key]=project(role,packets['gdn_output_q81' if 'ssm_out' in role else 'attn_input_q81'])
  self.assertTrue(all(x['passed'] for x in c.conditional_projections(layer,self.v,packets).values()))
  altered=SimpleNamespace(project=lambda role,x:project(role,x)*1.1)
  self.assertFalse(all(x['passed'] for x in c.conditional_projections(altered,self.v,packets).values()))
  packets['attn_input_q81']*=1.1
  self.assertFalse(c.conditional_projections(layer,self.v,packets)['attn_qkv_verified_packet_original_Q8_0']['passed'])
 def test_non_q8_weight_role_rejected(self):
  tensors={name:({'path':'fixture'},dict(type=kind,shape_ggml_order=list(shape),absolute_offset=0,packed_bytes=0),{}) for name,(kind,shape) in c.ROLES.items()}
  provider=SimpleNamespace(reader=SimpleNamespace(tensors=tensors));c.role_binding(provider)
  tensors['blk.0.attn_qkv.weight'][1]['type']='Q5_1'
  with self.assertRaises(ValueError):c.role_binding(provider)


class FrameBindingTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name)
  contract=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',Path(c.__file__).with_name('layer0_numerical_contract_v1.hpp').read_text())
  fields=[];observed=[];raws={}
  for index,(name,encoding,size) in enumerate(contract):
   seen=index<24 or index>=31;path=root/(name+'.bin');size=int(size)
   if seen:
    raw=bytes(size)
    if name in c.PACKETS:
     source,cols=c.PACKETS[name];raw=encode(raws[source],1,cols)
    path.write_bytes(raw);raws[name]=raw
    observed.append(dict(name=name,encoding=encoding,bytes=size,path=str(path),sha256=c.sha(path)))
   fields.append(dict(name=name,encoding=encoding,bytes=size,observed=seen,file=str(path) if seen else '',provenance='actual_buffer' if seen else 'UNOBSERVED_no_producer_hook'))
  self.frame=dict(schema=1,pid=11,request=2,stage=0,layer=0,rows=1,position=0,token=19,reused=0,gen_ids=[19],binding_sha256='a'*64,graph_key=2,completed_nonce=(11<<32)|2,request_replay_marker_verified=True,full_model_math_qualified=False,raw_fused_hidden_observed=False,complete_preregistered_layout=False,fields=fields)
  self.metadata=root/'frame.json'
  self.row=dict(prefix=1,raw=dict(ids=[19],fresh=1,stderr=['SFD request pid=11 request=2','L0Q8 frame pid=11 request=2 metadata='+str(self.metadata)]),layer0=dict(metadata=str(self.metadata),metadata_sha256='',frame=self.frame,observed=observed,full_model_math_qualified=False))
  self.publish()
 def publish(self):
  self.metadata.write_text(json.dumps(self.frame));self.row['layer0']['metadata_sha256']=c.sha(self.metadata)
 def test_bound_frame_positive(self):c.load_fields(self.row)
 def test_nonce_negative(self):
  self.frame['completed_nonce']+=1;self.publish()
  with self.assertRaises(ValueError):c.load_fields(self.row)
 def test_pid_negative(self):
  self.frame['pid']=12;self.publish()
  with self.assertRaises(ValueError):c.load_fields(self.row)
 def test_layout_provenance_negative(self):
  self.frame['fields'][0]['encoding']='LE_F32[10240]';self.publish()
  with self.assertRaises(ValueError):c.load_fields(self.row)
 def test_metadata_digest_negative(self):
  self.row['layer0']['metadata_sha256']='0'*64
  with self.assertRaises(ValueError):c.load_fields(self.row)
 def replace_field(self,name,raw):
  record=next(f for f in self.row['layer0']['observed'] if f['name']==name);Path(record['path']).write_bytes(raw);record['sha256']=c.sha(record['path'])
 def test_nonfinite_actual_field_negative(self):
  raw=np.zeros(2560,dtype='<f4');raw[0]=np.nan;self.replace_field('attn_mixed',raw.tobytes())
  with self.assertRaises(ValueError):c.load_fields(self.row)
 def test_packet_negative_even_updated_sha(self):
  record=next(f for f in self.row['layer0']['observed'] if f['name']=='attn_input_q81');raw=bytearray(Path(record['path']).read_bytes());raw[4]=1;self.replace_field('attn_input_q81',bytes(raw))
  with self.assertRaises(ValueError):c.load_fields(self.row)

if __name__=='__main__':unittest.main()
