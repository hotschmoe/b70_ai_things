"""Exact consumed bytes and producer-shaped own model/EOF/source chronology."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import own_hc_model_chronology_v2 as chronology
import owned_hc_device_rs_experiment_v2 as experiment
import qualify_owned_hc_device_rs_v2 as q
class Tests(unittest.TestCase):
 def test_exact_consumed_array_bound_and_stable(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'own.f32';raw=np.ones(4,dtype='<f4').tobytes();p.write_bytes(raw);binding={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':16};self.assertEqual(experiment.read_bound_array(binding,(4,),root).tobytes(),raw)
 def test_consumed_raw_cannot_borrow_later_path_hash(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'own.f32';raw=np.ones(4,dtype='<f4').tobytes();wrong=np.full(4,2.,dtype='<f4').tobytes();p.write_bytes(raw);binding={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':16}
   with patch.object(Path,'read_bytes',side_effect=[wrong,raw]):
    with self.assertRaises(ValueError):experiment.read_bound_array(binding,(4,),root)
 def test_postread_raw_change_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'own.f32';raw=np.ones(4,dtype='<f4').tobytes();p.write_bytes(raw);binding={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':16}
   with patch.object(Path,'read_bytes',side_effect=[raw,np.full(4,2.,dtype='<f4').tobytes()]):
    with self.assertRaises(ValueError):experiment.read_bound_array(binding,(4,),root)
 def report(self,prefix):
  work={'first_owned_terminal_epoch':3.,'computation_terminal_epoch':5.,'prefix4_attempted':prefix}
  if prefix:work['prefix4_computation_terminal_epoch']=4.
  return {'started_epoch':1.,'reference_phase':{'work':work,'model_computation_terminal_epoch':6.},'run_command':{'started_command_epoch':2.,'completed_epoch':7.,'finished_epoch':8.},'GPU_terminal_epoch':9.,'post_health':{'finished_epoch':10.},'post_journal':{'finished_epoch':11.},'post_full4':{'started':12.,'finished':13.},'finished_epoch':14.}
 def test_actual_first_and_prefix_model_before_phase_EOF_publisher(self):
  for prefix in (False,True):self.assertTrue(chronology.admit(self.report(prefix))['actual_original_model_phase_EOF_GPU_publisher_order'])
 def test_contradictory_model_phase_EOF_source_epochs_rejected(self):
  for kind in ('first','prefix','model','phase','EOF','GPU','full4','nonfinite','borrowedprefix'):
   r=self.report(True)
   if kind=='first':r['reference_phase']['work']['first_owned_terminal_epoch']=5.
   elif kind=='prefix':r['reference_phase']['work']['prefix4_computation_terminal_epoch']=6.
   elif kind=='model':r['reference_phase']['work']['computation_terminal_epoch']=7.
   elif kind=='phase':r['reference_phase']['model_computation_terminal_epoch']=8.
   elif kind=='EOF':r['run_command']['completed_epoch']=10.
   elif kind=='GPU':r['GPU_terminal_epoch']=7.
   elif kind=='full4':r['post_full4']['started']=8.
   elif kind=='nonfinite':r['reference_phase']['model_computation_terminal_epoch']=float('nan')
   else:r['reference_phase']['work']['prefix4_attempted']=False
   with self.assertRaises(ValueError):chronology.admit(r)
 def test_reader_and_producer_same_model_clock_gate(self):
  s=Path(q.__file__).read_text();self.assertIn('model_chronology(r)',s);self.assertIn('model_chronology(report)',s);self.assertIn('owned_hc_device_rs_experiment_v2',s)
 def test_original_frozen_prototype_preserved(self):self.assertEqual(q.sha(Path(q.__file__).with_name('owned-hc-device-rs-source-plan-v1.json')),'7cc5c66449f26bb1fd30cf091da31bfb33961e28092b8309a60062aacbca98c0')
if __name__=='__main__':unittest.main()
