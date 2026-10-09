#!/usr/bin/env python3
"""Tiny-file/synthetic exploration controls; no original model payload or GPU."""
import json,tempfile,time,unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import explore_full48_original_prefix1_v1 as explore
from run_source_upload_oracle_full_v2 import full_buffered_identity

class ExploreTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
 def write(self,p,value):explore.write(p,value)
 def tiny_identity(self):
  shards=[];files=[]
  for i in range(1,5):
   p=self.root/('Tiny-0000%d-of-00004.gguf'%i);p.write_bytes(bytes([i])*127);shards.append(p);files.append({'path':'UD-Q4_K_XL/'+p.name,'size':127,'sha256':explore.sha(p)})
  lockpath=self.root/'lock.json';lock={'revision':'CPU-synthetic','files':files};self.write(lockpath,lock);out=self.root/'identity.json';full_buffered_identity(lockpath,lock,shards,out,time.time()-1);return lockpath,shards,out
 def test_actual_producer_canonical5_tiny4_consumer(self):
  lock,shards,identity=self.tiny_identity();value=explore.identity_admission(identity,lock,shards,0);self.assertTrue(value['complete_four_publisher_hashes_verified']);self.assertTrue(value['current_stat_verified'])
 def test_stale_wrong_partial_and_legacy_identity_reject(self):
  lock,shards,identity=self.tiny_identity();original=explore.read(identity)
  for change in ('wronghash','missing','dictstat','ctime','chronology'):
   value=json.loads(json.dumps(original))
   if change=='wronghash':value['rows'][0]['sha256']='0'*64
   elif change=='missing':value['rows'].pop()
   elif change=='dictstat':value['rows'][0]['stat_before']={};value['rows'][0]['stat_after']={}
   elif change=='ctime':value['rows'][0]['stat_after'][-1]+=1
   else:value['finished']=value['started']-1
   self.write(identity,value)
   with self.assertRaises(ValueError):explore.identity_admission(identity,lock,shards,0)
  self.write(identity,original)
  with self.assertRaises(ValueError):explore.identity_admission(identity,lock,shards,time.time()+10)
 def test_current_file_change_rejects_stat_source(self):
  lock,shards,identity=self.tiny_identity();shards[0].write_bytes(b'X'*127)
  with self.assertRaises(ValueError):explore.identity_admission(identity,lock,shards,0)
 def test_metrics_observational_zero_and_first_difference(self):
  x=np.array([0.,.25,-.5],dtype='<f4');a=explore.compare_observation(x,x);self.assertTrue(a['bitwise_equal']);self.assertFalse(a['numeric_gate_assigned']);self.assertNotIn('passed',a)
  y=x.copy();y[1]+=.1;b=explore.compare_observation(x,y);self.assertEqual(b['first_bitwise_differing_index'],1);self.assertGreater(b['nmse'],0)
  with self.assertRaises(ValueError):explore.compare_observation([np.nan],[0])
 def test_signed_zero_is_bitwise_difference(self):
  value=explore.compare_observation([0.],[-0.]);self.assertFalse(value['bitwise_equal']);self.assertEqual(value['nmse'],0.)
 def tiny_owned(self):
  value=np.arange(8,dtype=float).reshape(4,2)/10;layers=[{'layer':i,'input':value,'attention':value+i/100,'ffn':value+i/50} for i in range(48)];head=np.arange(16)/16
  own={'ids':[19],'trace':[{'route':'verifier','layers':layers}],'first_generated_logits':head,'gdn_states_owned':{0:{'recurrent':np.zeros((2,2,2)),'conv':np.zeros((3,4))}},'qsa_states_owned':{},'ple_history_owned':np.zeros((9,8)),'last_two_owned':[-1,19],'captured_inputs_used':False,'captured_states_or_selected_ids_used':False,'full_model_math_qualified':False};native={'embedding':value.copy(),'post_attention_0':value.copy(),'head':head.copy(),**{'post_ffn_%d'%i:layers[i]['ffn'].copy() for i in range(48)}};return own,native
 def test_save_full48_synthetic_trace_no_numerical_pass(self):
  own,native=self.tiny_owned();result=explore.save_owned_and_compare(self.root,own,native);self.assertEqual(len(result['comparisons']),51);self.assertIsNone(result['first_bitwise_difference']);self.assertFalse(result['numeric_pass_claim']);self.assertFalse(result['full_model_math_qualified']);self.assertNotIn('passed',result)
  for row in result['arrays'].values():self.assertEqual(explore.sha(row['path']),row['sha256'])
  with self.assertRaises(ValueError):explore.save_owned_and_compare(self.root,own,native)
 def test_earliest_embedding_mismatch_before_layer_head(self):
  own,native=self.tiny_owned();native['embedding'][0,0]+=.01;native['head'][0]+=.2;result=explore.save_owned_and_compare(self.root,own,native);self.assertEqual(result['first_bitwise_difference']['scope'],'embedding')
 def test_admission_failure_never_reads_original_or_fullscan(self):
  out=self.root/'failed';args=['explore','--native-run-root',str(self.root),'--model-identity',str(self.root/'missing'),'--output',str(out)]
  with patch.object(sys,'argv',args),patch.object(explore,'dependency_binding',return_value={}),patch.object(explore,'finalized_binding',side_effect=ValueError('CPU synthetic missing finalized native parent')),patch.object(explore,'full_buffered_identity') as scan,patch.object(explore,'guard',return_value={'passed':True}) as guard,patch.object(explore,'preserve',return_value={'scope':'CPU synthetic both-page views'}),patch.object(explore,'BoundOriginalFull48Rows') as provider:
   self.assertEqual(explore.main(),1);scan.assert_not_called();self.assertEqual(guard.call_count,1);provider.assert_not_called()
  self.assertFalse(explore.read(out/'report.json')['numeric_pass_claim'])
 def native_fixture(self):
  directory=self.root/'native/child/candidatecombined_on';captures=directory/'captures';captures.mkdir(parents=True);l0=directory/'layer0';l0.mkdir();snapshot=directory.parent/'plan.snapshot.json';snapshot.write_text('{}');vectors=[];stderr=[]
  for layer in list(range(48))+[-1]:
   stage=0 if 0<=layer<32 else 1;bounds=(0,32) if stage==0 else (32,48);size=993280 if layer==-1 else 40960;phase='first_logits_before_sampler' if layer==-1 else 'first_window_residual';path=captures/('native-%d.f32'%layer);path.write_bytes(bytes(size));vector={k:str(v) for k,v in dict(pid=10,request=1,stage=stage,lb=bounds[0],le=bounds[1],phase=phase,layer=layer,pos=0,token=19,bytes=size,file='/results/captures/'+path.name,canonical='le_f32',reused=0,observed_only=1).items()};stderr.append('SFD vector '+' '.join(k+'='+v for k,v in vector.items()));vector.update(path=str(path),sha256=explore.sha(path));vectors.append(vector)
  fields=[]
  for name in ('residual_input','residual_after_attn'):
   path=l0/(name+'.f32');path.write_bytes(bytes(40960));fields.append(dict(name=name,path=str(path),sha256=explore.sha(path)))
  row={'prefix':1,'raw':{'ids':[19],'fresh':1,'stderr':stderr},'layer0':{'frame':{'pid':10,'request':1,'binding_sha256':explore.sha(snapshot)},'observed':fields},'meta':{'coverage':{'passed':True,'required_stage_ranges':{'0':[0,32],'1':[32,48]}},'ledger':{'cancelled':False,'actual_reused':0,'evaluated_prompt_rows':1},'residuals':vectors[:-1],'logits':vectors[-1:]},'external':{'passed':True}}
  requests=directory/'requests.json';self.write(requests,[row]+[{'prefix':n} for n in (2,4,8)]);plan={'prefixes':{'1':[19]},'args':['--layer-split','32','--split-device','1']};return self.root/'native',requests,row,plan
 def test_source_token_PID_pair_stage_native_capture_admission(self):
  root,requests,row,plan=self.native_fixture()
  with patch('verify_layer0_ffn_original_v1.load_frame',return_value=({}, {},{'scope':'CPU synthetic metadata producer'})):
   ids,values,binding=explore.native_prefix1(root,plan);self.assertEqual(ids,[19]);self.assertEqual(len(values),51);self.assertEqual(values['head'].size,248320)
   original=json.loads(json.dumps(row))
   for change in ('token','pid','missing','topology','SHA'):
    bad=json.loads(json.dumps(original))
    if change=='token':bad['raw']['ids']=[20]
    elif change=='pid':bad['meta']['residuals'][0]['pid']='11'
    elif change=='missing':bad['meta']['residuals'].pop()
    elif change=='topology':bad['meta']['coverage']['required_stage_ranges']={'0':[0,48]}
    else:bad['meta']['logits'][0]['sha256']='0'*64
    self.write(requests,[bad]+[{'prefix':n} for n in (2,4,8)])
    with self.assertRaises(ValueError):explore.native_prefix1(root,plan)
 def test_cli_prefix2_not_admitted_by_generation1(self):
  with patch.object(sys,'argv',['explore','--native-run-root',str(self.root),'--model-identity','missing','--prefix','2','--output',str(self.root/'x')]),self.assertRaises(SystemExit):explore.main()

if __name__=='__main__':unittest.main()
