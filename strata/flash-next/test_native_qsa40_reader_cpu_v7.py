"""Raw protocol and independent target comparisons, synthetic byte fixtures."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import native_qsa40_reader_v7 as q
import native_qsa40_input33_v7 as staged
import native_qsa40_localization_v7 as local
class Controls(unittest.TestCase):
 def raw(self):
  pid,request=123,1;digest=q.r.numeric.le32_digest(list(q.qsa.IDS));events=[{'event':'begin','pid':pid,'request':request,'tokens':4,'input_sha256_le32':digest},{'event':'stage_span','pid':pid,'request':request,'phase':'prefill','device':0,'lb':0,'le':48,'lo':0,'hi':4,'complete':False},{'event':'stage_span','pid':pid,'request':request,'phase':'prefill','device':0,'lb':0,'le':48,'lo':0,'hi':4,'complete':True},{'event':'committed_live','pid':pid,'request':request,'phase':'complete','finish':'length','ids_truncated':False,'tokens':4,'ids':list(q.qsa.IDS),'sha256_le32':digest,'published':False,'chain_updated':True,'live_reusable':False}]
  raw={'ids':list(q.qsa.IDS),'command':q.r.request_command(q.qsa.IDS,1,1,None),'fresh':1,'pin':None,'cancel_requested':None,'stop_sent':False,'output_ids':[7],'done':'DONE 1 4 x x length x x 0 x x x x x 4','stderr':['PCL '+json.dumps(e)for e in events]};meta={'ledger':{'actual_reused':0,'candidate_resume':0,'evaluated_prompt_rows':4,'evaluated_decode_rows':0,'generated':1,'cancelled':False,'finish':'length'},'selection':dict.fromkeys(('actual_reused','candidate_resume','parked_bytes','parked_entries','evictions','checkpoints'),0),'logits':[{'pid':'123','request':'1'}]};meta['selection']['reread']=False;return raw,meta
 def test_truthful_cacheOFF_tuple_and_complete_source_spans(self):
  raw,meta=self.raw()
  with patch.object(q.r.numeric,'extract_numeric',return_value=meta):
   q.numeric(raw,Path('/CPU_MOCK'),[(0,48)])
   for key,value in [('pin',0),('fresh',True),('done',raw['done'].replace('DONE 1 4','DONE 1 3'))]:
    changed=copy.deepcopy(raw);changed[key]=value;self.assertRaises(ValueError,q.numeric,changed,Path('/CPU_MOCK'),[(0,48)])
   for key,value in [('live_reusable',True),('chain_updated',False),('ids',[1,2,3,4]),('pid',124)]:
    changed=copy.deepcopy(raw);event=json.loads(changed['stderr'][-1][4:]);event[key]=value;changed['stderr'][-1]='PCL '+json.dumps(event);self.assertRaises(ValueError,q.numeric,changed,Path('/CPU_MOCK'),[(0,48)])
   changed=copy.deepcopy(raw);changed['stderr'].pop(1);self.assertRaises(ValueError,q.numeric,changed,Path('/CPU_MOCK'),[(0,48)])
 def test_packed_P30_windows_count_logical_rows_not_files(self):
  value={'frames':[{'binding':{'request':1,'first_position':0,'rows':2},'fields':[{'layer':3,'phase':'attention','path':'CPU'}]}]};keys=q.p30_vectors(value);self.assertEqual(list(keys),[(1,0,2,3,'attention')]);self.assertEqual(sum(k[2]for k in keys),2);value['frames'].append(value['frames'][0]);self.assertRaises(ValueError,q.p30_vectors,value)
 def test_independent_comparison_never_selects_tolerance(self):
  left=np.array([1,2],dtype='<f4');right=np.array([1,3],dtype='<f4');result=local.metrics(left,right);self.assertFalse(result['bitwise_equal']);self.assertFalse(result['tolerance_used']);self.assertFalse(result['math_qualified']);self.assertGreater(result['NMSE'],0);self.assertRaises(ValueError,local.metrics,left,np.array([float('nan'),0],dtype='<f4'))
 def test_repeated_source33_original_hash_and_embedding_control(self):
  from ple_owned_history_storage_v1 import ngram_rows,PleHashConsts
  ids=list(q.qsa.IDS);wanted=ngram_rows(ids[-1],ids[1:3],PleHashConsts())
  class Provider:
   def rows(self,role,indices):
    assert role=='per_layer_token_embd.weight'and indices==wanted
    return np.zeros((16,160),dtype='<f4')
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'ids').write_bytes(np.asarray(wanted,dtype='<u4').tobytes());(root/'embedding').write_bytes(bytes(10240));frame={'binding':{'request':1,'gen_ids':ids,'T':1,'pos':3,'prev':ids[1:3],'window_tokens':[ids[3]],'row_ids':wanted},'fields':{'row_ids':{'path':str(root/'ids')},'embedding':{'path':str(root/'embedding')}}};frames=[]
   for i in range(1,5):row=copy.deepcopy(frame);row['binding']['request']=i;frames.append(row)
   proof={'passed':True,'actual_matching_SFD_request_terminal_observed':True,'frames':frames};self.assertTrue(staged.verify(proof,root/'no_model',Provider())['passed']);(root/'embedding').write_bytes(np.ones(2560,dtype='<f4').tobytes());self.assertRaises(ValueError,staged.verify,proof,root/'no_model',Provider())
if __name__=='__main__':unittest.main()
