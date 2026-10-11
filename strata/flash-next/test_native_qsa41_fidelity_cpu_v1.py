import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import native_qsa41_targets_v1 as target
import native_qsa41_original_localization_v1 as own

class Controls(unittest.TestCase):
 def header(self,root):
  command=['docker','run','-e','STRATA_QSA_INDEXER_MATERIALIZED_HALF=1']
  values={'command':command,'result':{'passed':True,'error':None,'errors':[],'engine_rc':0,'return_code':0},'phase_receipt':{'passed':True,'error':None,'return_code':0}}
  from source41_sdk_binding_v1 import SDK
  proof={'engine_root':str(SDK),'actual_receipt_source_rows':67,'independently_required_source_rows':69,'old_source40_oracle_or_runtime_transferred':False}
  binding={'capture_root':str(root),'stages':[[0,0,48]],'materialized_mode':1,'observer_enabled':True,'binding_sha256':'fixture','engine_pid':123,'source41_sdk':proof}
  for role,value in values.items():
   path=root/(role+'.json');path.write_text(json.dumps(value));binding[role+'_path']=str(path);binding[role+'_sha256']=target.sha(path)
  return binding,proof
 def test_actual41_header_and_mode_type(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);binding,proof=self.header(root)
   with patch('source41_sdk_binding_v1.binding',return_value=proof):
    self.assertEqual(target.actor_header(binding),root)
    for key,value in [('materialized_mode',True),('observer_enabled',False),('stages',[[False,0,48]])]:
     bad=copy.deepcopy(binding);bad[key]=value
     with self.assertRaises(ValueError):target.actor_header(bad)
 def test_actual41_old_sdk_and_failed_result_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);binding,proof=self.header(root)
   with patch('source41_sdk_binding_v1.binding',return_value=proof):
    bad=copy.deepcopy(binding);bad['source41_sdk']['old_source40_oracle_or_runtime_transferred']=True
    with self.assertRaises(ValueError):target.actor_header(bad)
    path=Path(binding['result_path']);row=json.loads(path.read_text());row['return_code']=False;path.write_text(json.dumps(row));binding['result_sha256']=target.sha(path)
    with self.assertRaises(ValueError):target.actor_header(binding)
 def test_conflicting_mode_and_changed_receipt_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);binding,proof=self.header(root)
   with patch('source41_sdk_binding_v1.binding',return_value=proof):
    path=Path(binding['command_path']);row=json.loads(path.read_text());row+=['-e','STRATA_QSA_INDEXER_MATERIALIZED_HALF=0'];path.write_text(json.dumps(row));binding['command_sha256']=target.sha(path)
    with self.assertRaises(ValueError):target.actor_header(binding)
    Path(binding['phase_receipt_path']).write_text('{}')
    with self.assertRaises(ValueError):target.actor_header(binding)
 def test_array_consumed_bytes_sha_and_confined_path(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);path=root/'own.f32';raw=np.asarray([1,2],dtype='<f4').tobytes();path.write_bytes(raw);field={'path':str(path),'bytes':8,'shape':[2],'sha256':hashlib.sha256(raw).hexdigest()}
   self.assertEqual(own.load_owned(field,root).tolist(),[1,2]);path.write_bytes(bytes(8))
   with self.assertRaises(ValueError):own.load_owned(field,root)
 def fixture(self,root):
  own_fields={};frames=[];requests=[]
  for request in range(1,5):
   for position,rows in [(0,2),(2,1),(3,1)]:
    fields=[];raw=bytes(rows*40960)
    for layer in range(48):
     for phase in ('input','attention','ffn'):
      fields.append({'path':'/synthetic/n'+str(rows),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'layer':layer,'phase':phase})
      for offset in range(rows):own_fields['p%d_l%d_%s'%(position+offset,layer,phase)]={'synthetic':True}
    frames.append({'binding':{'request':request,'first_position':position,'rows':rows},'fields':fields})
   requests.append({'ordinal':request,'meta':{'logits':[{'path':'/synthetic/head','bytes':'4','sha256':hashlib.sha256(bytes(4)).hexdigest()}]}})
  own_fields['head']={'head':True};report={'prefix4_attempted':True,'first_gate':{'passed':True},'prefix4':{'arrays':own_fields}}
  native={'p30':{'frames':frames},'rows':requests};return report,native
 def test_full_actual_shaped_2308_roster_and_missing_phase(self):
  root=Path('/synthetic/own');report,native=self.fixture(root)
  def consume(path,*unused):return bytes(4)if str(path).endswith('head')else bytes(int(str(path)[-1])*40960)
  def load(field,*unused):return np.zeros(1,dtype='<f4')if field.get('head')else np.zeros((4,2560),dtype='<f4')
  with patch.object(own,'reference_binding',return_value={'fixture':'original'}),patch.object(own,'read_unique',return_value=report),patch.object(own.reader,'consume',side_effect=consume),patch.object(own,'load_owned',side_effect=load):
   self.assertEqual(len(own.full_prefix_join(root,native)['comparisons']),2308)
   native['p30']['frames'][0]['fields'].pop()
   with self.assertRaises(ValueError):own.full_prefix_join(root,native)
 def test_duplicate_head_roster_refused(self):
  root=Path('/synthetic/own');report,native=self.fixture(root);native['rows'][3]['ordinal']=1
  with patch.object(own,'reference_binding',return_value={'fixture':'original'}),patch.object(own,'read_unique',return_value=report),patch.object(own.reader,'consume',side_effect=lambda p,*a:bytes(int(str(p)[-1])*40960)),patch.object(own,'load_owned',return_value=np.zeros((4,2560),dtype='<f4')):
   with self.assertRaises(ValueError):own.full_prefix_join(root,native)
 def test_numeric_exact_pin_command_refused_before_payload(self):
  raw={'ids':list(target.qsa.IDS),'command':'GEN 1 fresh=1 pin=0','fresh':1,'pin':0,'cancel_requested':None,'stop_sent':False}
  with self.assertRaises(ValueError):target.numeric(raw,Path('/synthetic'),[(0,48)])
 def test_original_reference_has_no_threshold(self):
  value=own.metrics(np.asarray([1],dtype='<f4'),np.asarray([2],dtype='<f4'));self.assertFalse(value['math_qualified']);self.assertFalse(value['tolerance_used']);self.assertFalse(value['bitwise_equal']);self.assertEqual(value['NMSE'],.25);self.assertEqual(value['normalization_reference'],'native_actual_as_in_frozen_localization_v9')

if __name__=='__main__':unittest.main()
