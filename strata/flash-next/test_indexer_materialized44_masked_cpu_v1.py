"""Meaningful byte/owner/mask controls, synthetic only and no model/device work."""
import struct,tempfile,unittest
from pathlib import Path
import indexer_materialized44_masked_control_v1 as c
class Controls(unittest.TestCase):
 def fixture(self,root,mode):
  root.mkdir();raw=struct.pack('<512f',*[1.0001+i*.00003 for i in range(512)]);gamma=struct.pack('<128f',*([1.]*128));half,wide=c.ieee_rne(raw);feed=wide if mode else raw;common=struct.pack('<384f',*([.5]*384))+bytes(4)
  def state(rows):return b''.join(feed[row*512:(row+1)*512]for row in rows)+bytes(1536-len(rows)*512)+common
  states={'single4':state([0,1,2]),'steps4':state([0,1,2]),'batch4':state([0,1,2]),'accept0':bytes(3076),'accept1':state([0]),'accept2':state([0,1]),'grouped2':state([0,1]),'grouped_scalar2':state([0,1])}
  for name in c.SCENARIOS:
   for route in range(3):
    p=root/(name+'-route'+str(route));p.mkdir();h=bytearray(b'\x5a'*1024);w=bytearray(b'\x3d'*2048)
    if mode:
     for row in range(2 if name.startswith('accept')else 4):
      valid=c.positions(name)[row]>=0;h[row*256:(row+1)*256]=half[row*256:(row+1)*256]if valid else bytes(256);w[row*512:(row+1)*512]=wide[row*512:(row+1)*512]if valid else bytes(512)
    steps=list(struct.unpack('<16i',c.STEP_BYTES))
    for row,pos in enumerate(c.positions(name)):steps[row*4]=pos
    data={'raw.f32':raw,'gamma.f32':gamma,'positions.i32':struct.pack('<4i',*c.positions(name)),'steps.i32':struct.pack('<16i',*steps),'actual-half.f16':h,'actual-expanded.f32':w,'before.bin':bytes(6152),'after.bin':states[name]+(state([2,3])if name.startswith('grouped')else bytes(3076))}
    for key,value in data.items():(p/key).write_bytes(value)
  return raw,gamma
 def test_actual_shaped_all24_frames_and_separate_mode_sentinels(self):
  for mode in(0,1):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d)/'out';raw,gamma=self.fixture(root,mode);r=c.recollect(root,raw,gamma,mode);self.assertEqual(r['frames'],24);self.assertTrue(r['accepted0_1_2_masks_observed']);self.assertFalse(r['normal_model_graph_qualified'])
 def test_wrong_mask_wrongowner_noop_append_and_replay_negative(self):
  for kind in('mask','owner','noop','replay','alias'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d)/'out';raw,gamma=self.fixture(root,1)
    if kind=='mask':p=root/'accept0-route0/actual-expanded.f32';p.write_bytes(raw)
    elif kind=='owner':p=root/'accept2-route0/after.bin';data=p.read_bytes();p.write_bytes(data[:3076]+data[:3076])
    elif kind=='noop':
     for name in('accept1','accept2','grouped2','grouped_scalar2'):
      for route in range(3):(root/(name+'-route'+str(route))/'after.bin').write_bytes(bytes(6152))
    elif kind=='replay':p=root/'steps4-route2/after.bin';p.write_bytes(bytes(6152))
    else:p=root/'single4-route0/raw.f32';p.unlink();f=Path(d)/'else';f.write_bytes(raw);p.symlink_to(f)
    self.assertRaises(ValueError,c.recollect,root,raw,gamma,1)
 def test_CPP_links_actual_helper_and_masks_grouped_current_API(self):
  source=Path(__file__).with_name('indexer_materialized44_masked_gpu_v1.cpp').read_text();self.assertIn('native_qsa_indexer_materialize_input',source);self.assertIn('mode?half:nullptr,mode?wide:nullptr',source);self.assertIn('size_t(start)*128,512',source);self.assertIn('ps={{-1,-1,-1,-1}}',source);self.assertIn('ps={{0,1,0,1}}',source);self.assertIn('std::string(names[scenario])',source);self.assertIn('memory.release()',source);self.assertIn('for(int route=0;route<3',source);self.assertNotIn('Indexer40StoreHalf',source)
if __name__=='__main__':unittest.main()

class RoutingControls(unittest.TestCase):
 def test_actual_expected_compile_and_two_fresh_process_recipe(self):
  from unittest.mock import patch
  import qualify_indexer_materialized44_masked_v2 as q
  ns=q.derived();ns['sha']=lambda path:'CPU_SYNTHETIC_PLAN_SHA';argv,compile,runtime=ns['expected_recipes']('/CPU_SYNTHETIC/out','/CPU_SYNTHETIC/fixture',123)
  self.assertIn('/leaf/indexer_materialized44_masked_gpu_v1.cpp',argv);self.assertIn('20261011T165803Z-5xbo5_f0',str(compile));shell=runtime[-1];self.assertIn('STRATA_QSA_INDEXER_MATERIALIZED_HALF=0',shell);self.assertIn('STRATA_QSA_INDEXER_MATERIALIZED_HALF=1',shell);self.assertIn('--mode 0',shell);self.assertIn('--mode 1',shell);self.assertIn('cat /out/idx44mode0.after /out/idx44mode1.after',shell);self.assertNotIn('/out/half37',shell)
 def test_new_prerequisite_reader_shape_keeps_upload_only_authority(self):
  from unittest.mock import patch
  import qualify_indexer_materialized44_masked_v2 as q
  proof={'CPU_SYNTHETIC_FRESH_UPLOAD':True};record={'upload_receipt':'up','oracle_receipt':'or','pack_receipt':'pack','actual_fresh44_upload_binding':proof}
  with patch.object(q,'read',return_value=record),patch.object(q,'sha',return_value='CPU_SYNTHETIC_SHA'),patch('admit_source_upload44_v1.finalized_binding',return_value=proof),patch('source44_sdk_binding_v1.binding',side_effect=AssertionError('No actual SDK runtime bytes in tiny control')):
   prepared,binding=q.Baseline.finalized_binding('/CPU_SYNTHETIC/prerequisite');self.assertEqual(binding['baseline_root'],'/CPU_SYNTHETIC/prerequisite');self.assertFalse(prepared['current_model_baseline_qualified']);self.assertFalse(binding['model_baseline_or_constructor_proof_transferred']);self.assertEqual(binding['actual_fresh44_upload_binding'],proof)
 def test_complete_two_mode_marker_roster_no_truncation_or_duplicates(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'log';lines=[]
   for mode in(0,1):lines +=[f'INDEXER44_MASK_FRAME mode={mode} scenario={name} route={route} fields=8 restored=1\n'for name in c.SCENARIOS for route in range(3)]+[f'INDEXER44_MASK_RESULT mode={mode} frames=24 graph_retired=1 owned_freed=1 model_math=0\n']
   p.write_text(''.join(lines));self.assertEqual(c.native_trace(p)['actual_frames'],48)
   for bad in(''.join(lines).rstrip('\n'),''.join(lines)+lines[-1],''.join(lines).replace('mode=1','mode=0',1)):
    p.write_text(bad);self.assertRaises(ValueError,c.native_trace,p)
