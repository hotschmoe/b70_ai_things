#!/usr/bin/env python3
"""Synthetic conditional FFN/binding controls. Never read actual model weights."""
import copy,json,re,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import verify_layer0_ffn_original_v1 as c
from original_first_gdn_layer_v1 import FirstGdnLayer,Geometry
from independent_q8_1_activation_v1 import encode,decode
from original_gguf_vector_decoder_v2 import decode as decode_weight

class FakeProvider:
 def shape(self,name):return c.ROLES[name.removeprefix('blk.0.')][1]
 def rows(self,name,indices):
  slope=.100012 if 'shexp' in name else .000101
  return np.stack([np.full(2560,slope*(index+1)) for index in indices])

class FakeLayer:
 g=Geometry();sigmoid=staticmethod(FirstGdnLayer.sigmoid)
 def __init__(self,mixed):self.mixed=mixed
 def project(self,name,x,expert=None):
  cols=2560 if 'down' in name else 640;scale=.02 if expert is None else .002*(expert+1)
  return np.linspace(-.1,.1,cols)+np.asarray(x).sum()*scale
 def hc_read(self,r,half):return {'mixed':self.mixed,'inject':np.array([.1,.2,-.3,.4])}
 def hc_write(self,r,b,i):return FirstGdnLayer.hc_write(self,r,b,i)

class OperatorTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(117);mixed=rng.normal(0,.1,2560).astype('<f4').astype(np.float64);self.p=FakeProvider();self.l=FakeLayer(mixed)
  packet=encode(np.asarray(mixed,dtype='<f4').tobytes(),1,2560);x=decode(packet,1,2560)['reconstructed_activation']
  ids=np.arange(511,501,-1);hq=np.tile(np.linspace(-.15,.15,640),(10,1));sharedhq=np.linspace(-.1,.1,640).reshape(1,640)
  self.packets={'ffn_input_q81':x,'shared_hq81':sharedhq,'expert_hq81':hq}
  self.v={'ffn_mixed':mixed,'router_ids':ids,'shared_gate_up':np.stack([self.l.project('blk.0.ffn_gate_shexp.weight',x),self.l.project('blk.0.ffn_up_shexp.weight',x)]).reshape(-1),'expert_gate_up':np.stack([np.stack([self.l.project('blk.0.ffn_gate_exps.weight',x,expert=int(i)),self.l.project('blk.0.ffn_up_exps.weight',x,expert=int(i))]) for i in ids]).reshape(-1)}
  logits=c.bf16_projection(self.p,'ffn_gate_inp.weight',mixed);self.v['router_logits']=logits
  prob=np.exp(logits-logits.max());prob/=prob.sum();weights=prob[ids];weights/=weights.sum();self.v['router_weights']=weights
  down=np.stack([self.l.project('blk.0.ffn_down_exps.weight',hq[rank],expert=int(expert)) for rank,expert in enumerate(ids)]);shared=self.l.project('blk.0.ffn_down_shexp.weight',sharedhq.reshape(-1));gate=float(c.sigmoid(c.bf16_projection(self.p,'ffn_gate_inp_shexp.weight',mixed))[0]);self.v['ffn_block_output']=np.sum(down*weights[:,None],axis=0)+gate*shared
  residual=rng.normal(0,.2,(4,2560));self.v['residual_after_attn']=residual.reshape(-1);self.v['residual_after_ffn']=self.l.hc_write(residual,self.v['ffn_block_output'],self.l.hc_read(residual,'ffn')['inject']).reshape(-1)
 def result(self):return c.conditional_ffn(self.l,self.p,self.v,self.packets)
 def test_positive_conditional_scope(self):
  result=self.result();self.assertTrue(result['passed']);self.assertEqual(len(result['checks']),27);self.assertFalse(result['individual_down_outputs_observed']);self.assertFalse(result['native_exp_or_reduction_emulated'])
 def test_original_weight_or_input_perturb_rejected(self):
  old=self.l.project;self.l.project=lambda name,x,expert=None:old(name,x,expert)*1.1;self.assertFalse(self.result()['passed'])
 def test_gu_rank_perturb_rejected(self):
  self.v['expert_gate_up'][0]+=.1;self.assertFalse(self.result()['passed'])
 def test_hq_perturb_aggregate_rejected(self):
  self.packets['expert_hq81'][0]+=1.;self.assertFalse(self.result()['passed'])
 def test_router_bf16_seam_is_explicit(self):
  original=self.p.rows('blk.0.ffn_gate_inp.weight',range(512))@self.v['ffn_mixed'];self.assertFalse(c.check(original,self.v['router_logits'])['passed'])
 def test_router_weights_perturb_rejected(self):
  self.v['router_weights'][0]*=1.1;self.assertFalse(self.result()['passed'])
 def test_HC_write_perturb_rejected(self):
  self.v['residual_after_ffn']+=.1;self.assertFalse(self.result()['passed'])
 def test_nonfinite_rejected(self):
  self.v['ffn_block_output'][0]=np.nan
  with self.assertRaises(ValueError):self.result()
 def test_original_type_affine_generalization_rejected(self):
  tensors={'blk.0.'+name:({'path':'fixture'},dict(type=kind,shape_ggml_order=list(shape),absolute_offset=0,packed_bytes=0),[]) for name,(kind,shape) in c.ROLES.items()};provider=SimpleNamespace(reader=SimpleNamespace(tensors=tensors));c.original_roles(provider);tensors['blk.0.ffn_down_exps.weight'][1]['type']='Q5_0'
  with self.assertRaises(ValueError):c.original_roles(provider)
 def test_Q51_min_uses_code_sum_not_stored_sum(self):
  w=np.zeros(24,dtype=np.uint8);w[:4]=np.array([.25,.5],dtype='<f2').view(np.uint8);w[8:]=0x12;x=np.linspace(-1,2,32,dtype='<f4');packet=encode(x.tobytes(),1,32);decoded=decode(packet,1,32);codes=decoded['codes'].reshape(-1).astype(np.int64);q=np.r_[np.full(16,2),np.full(16,1)];scale=float(decoded['scale'][0]);want=scale*(.25*int(q@codes)+.5*int(codes.sum()))
  self.assertAlmostEqual(float(decode_weight(w.tobytes(),'Q5_1')@decoded['reconstructed_activation'].reshape(-1)),want,places=12);wrong=.25*scale*int(q@codes)+.5*float(decoded['stored_sum'][0]);self.assertGreater(abs(wrong-want),1e-5)

 def test_Q4K_min_uses_quantized_code_sum(self):
  w=np.zeros(144,dtype=np.uint8);w[:4]=np.array([.25,.125],dtype='<f2').view(np.uint8);w[4:8]=1;w[8:12]=2;w[12:16]=0x21;w[16:]=0x43
  x=np.linspace(-.9,1.7,256,dtype='<f4');decoded=decode(encode(x.tobytes(),1,256),1,256);codes=decoded['codes'].astype(np.int64);wcode=np.tile(np.r_[np.full(32,3),np.full(32,4)],4).reshape(8,32)
  expected=np.sum(decoded['scale']*(.25*np.sum(wcode*codes,axis=1)-.25*np.sum(codes,axis=1)))
  actual=decode_weight(w.tobytes(),'Q4_K')@decoded['reconstructed_activation'].reshape(-1);self.assertAlmostEqual(float(actual),float(expected),places=12)

class FrameTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name);self.directory=root/'case';layer0=self.directory/'layer0';layer0.mkdir(parents=True)
  contract=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',Path(c.__file__).with_name('layer0_numerical_contract_v1.hpp').read_text());fields=[];observed=[]
  for index,(name,encoding,size) in enumerate(contract):
   size=int(size);raw=bytes(size)
   if name=='router_ids':raw=np.arange(10,dtype='<i4').tobytes()
   if name=='expert_entry_map':raw=np.column_stack([np.arange(10),np.zeros(10),np.arange(10)]).astype('<i4').tobytes()
   path=layer0/(name+'.bin');path.write_bytes(raw);provenance='DERIVED_gpu_native_exp_from_actual_gate_up' if index in (25,29) else 'actual_buffer'
   fields.append(dict(name=name,encoding=encoding,bytes=size,observed=True,file='/results/layer0/'+path.name,provenance=provenance));observed.append(dict(name=name,encoding=encoding,bytes=size,path=str(path),sha256=c.sha(path),provenance=provenance))
  addresses=[0x2000000+i*0x1000000 for i in range(10)];tiers=[1,2]*5;self.meta=layer0/'frame.json'
  self.frame=dict(schema=1,pid=9,request=1,stage=0,layer=0,rows=1,gen_ids=[19],position=0,token=19,reused=0,request_replay_marker_verified=True,completed_nonce=(9<<32)|1,graph_key=2,binding_sha256='a'*64,producer_mapping_verified=True,full_model_math_qualified=False,raw_fused_hidden_observed=False,complete_preregistered_layout=True,fields=fields,expert_blob_addresses=addresses,expert_tiers=tiers)
  marker='L0Q8 frame pid=9 request=1 metadata=/results/layer0/frame.json'
  alloc=lambda kind,addr,size:'<--- %s(.hContext = 0x111, .size = %d, .ppMem = 0x1 (0x%x)) -> UR_RESULT_SUCCESS;\n'%(kind,size,addr)
  log=alloc('urUSMDeviceAlloc',0x1000,7023304)+'L0Q8 allocation stage=0 pointer=0x1000 bytes=7023304 owner_queue=verifier_cs\n'
  for address,tier in zip(addresses,tiers):log+=alloc('urUSMDeviceAlloc' if tier==1 else 'urUSMHostAlloc',address,3072000)
  log+=marker+'\n';self.log=self.directory/'engine.combined.log';self.log.write_text(log)
  live=c.live_blob_bindings(log,9,1,addresses,tiers)
  self.row=dict(prefix=1,raw=dict(ids=[19],fresh=1,stderr=['SFD request pid=9 request=1',marker]),layer0=dict(metadata=str(self.meta),metadata_sha256='',frame=self.frame,observed=observed,source_live_binding=live));self.publish()
 def publish(self):self.meta.write_text(json.dumps(self.frame));self.row['layer0']['metadata_sha256']=c.sha(self.meta)
 def test_complete33_bound_frame_positive(self):
  values,packets,binding=c.load_frame(self.row);self.assertEqual(len(binding['fields']),33);self.assertFalse(binding['raw_hidden_proven'])
 def test_nonce_rejected(self):
  self.frame['completed_nonce']+=1;self.publish()
  with self.assertRaises(ValueError):c.load_frame(self.row)
 def test_DERIVED_false_raw_claim_rejected(self):
  self.frame['fields'][25]['provenance']='actual_buffer';self.publish()
  with self.assertRaises(ValueError):c.load_frame(self.row)
 def test_mapping_rank_rejected(self):
  field=next(f for f in self.row['layer0']['observed'] if f['name']=='expert_entry_map');raw=np.frombuffer(Path(field['path']).read_bytes(),dtype='<i4').copy();raw[2]=1;Path(field['path']).write_bytes(raw.tobytes());field['sha256']=c.sha(field['path'])
  with self.assertRaises(ValueError):c.load_frame(self.row)
 def test_tier_or_live_range_rejected(self):
  self.frame['expert_tiers'][0]=2;self.publish()
  with self.assertRaises(ValueError):c.load_frame(self.row)
 def test_packet_sha_updated_corruption_rejected(self):
  field=next(f for f in self.row['layer0']['observed'] if f['name']=='ffn_input_q81');raw=bytearray(Path(field['path']).read_bytes());raw[4]=1;Path(field['path']).write_bytes(raw);field['sha256']=c.sha(field['path'])
  with self.assertRaises(ValueError):c.load_frame(self.row)
 def test_wrong_derived_equation_with_matching_packet_rejected(self):
  field=next(f for f in self.row['layer0']['observed'] if f['name']=='shared_hidden_DERIVED');values=np.ones(640,dtype='<f4');Path(field['path']).write_bytes(values.tobytes());field['sha256']=c.sha(field['path'])
  hq=next(f for f in self.row['layer0']['observed'] if f['name']=='shared_hq81');Path(hq['path']).write_bytes(encode(values.tobytes(),1,640));hq['sha256']=c.sha(hq['path'])
  with self.assertRaises(ValueError):c.load_frame(self.row)
 def test_no_raw_hidden_proof_from_DERIVED_HQ(self):
  _,_,binding=c.load_frame(self.row);self.assertTrue(binding['derived_HQ_packet_contracts']['shared_hq81']['packet_exact']);self.assertFalse(binding['raw_hidden_proven'])

if __name__=='__main__':unittest.main()
