import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import own_q81_quantizer43_control_v1 as c
import qualify_own_q81_quantizer43_v1 as q
from independent_q8_1_activation_v1 import encode,NATIVE,OFFICIAL
class Controls(unittest.TestCase):
 def fixture(self,root,kind='valid'):
  fixture=root/'fixture';fixture.mkdir();native=root/'build/native-output';native.mkdir(parents=True);raw=np.zeros(4*2560,dtype='<f4').tobytes();packet=encode(raw,4,2560);(fixture/'own-input.f32').write_bytes(raw);prior={'root':'synthetic'};f={'root':str(fixture),'record':{'rows':4,'cols':2560,'own_producer_binding':prior}}
  selected=[0]if kind=='mismatch'else[];actual=bytearray(packet)
  if selected:actual[4]=1
  (native/'consumed-raw.f32').write_bytes(raw);(native/'consumed-cpu.q81').write_bytes(packet)
  lines=['OWNQ81_DEVICE backend=level_zero affinity=0 fp16=1 vendor=synthetic driver=synthetic name=synthetic','OWNQ81_FP_CONFIG flags=1, observed_device_flags_only=1','OWNQ81_CONFIG rows=4 cols=2560 source_function_unchanged=1 shadow_not_internal=1']
  for r in range(3):
   d=native/('route'+str(r));d.mkdir();(d/'packet_before.q81').write_bytes(bytes(len(packet)));(d/'input.f32').write_bytes(raw);(d/'packet.q81').write_bytes(actual);(d/'mismatch_blocks.i32').write_bytes(np.asarray(selected,dtype='<i4').tobytes());(d/'shadow.f32').write_bytes(bytes(len(selected)*140));lines.append('OWNQ81_FRAME route=%d graph_replay=%d fields=5 mismatch_blocks=%d input_restored=1 packet_zero_restored=1'%(r,r,len(selected)))
  lines.append('OWNQ81_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 shadow_not_internal=1 full_model_math_qualified=0');log='\n'.join(lines)+'\n'
  if kind=='newline':log=log[:-1]
  if kind=='duplicate':log+=lines[-1]+'\n'
  (root/'runtime.log').write_text(log)
  if kind=='extra':(native/'route0/foreign.bin').write_bytes(b'foreign')
  if kind=='zero':(native/'route0/packet_before.q81').write_bytes(b'\x01'+bytes(len(packet)-1))
  if kind=='input':(native/'route1/input.f32').write_bytes(b'\x01'+raw[1:])
  if kind=='changed_packet':(native/'route2/packet.q81').write_bytes(b'\x01'+packet[1:])
  if kind=='alias':p=native/'route0/input.f32';p.unlink();p.symlink_to(fixture/'own-input.f32')
  return native,f,prior
 def run_case(self,kind):
  with tempfile.TemporaryDirectory()as t:
   native,f,prior=self.fixture(Path(t),kind)
   if kind in('valid','mismatch'):
    report=c.compare_routes(native,f,prior);self.assertTrue(report['direct_two_replays_bitwise']);self.assertEqual(report['cpu_native_packet_exact'],kind=='valid');self.assertFalse(report['full_model_math_qualified'])
   else:
    with self.assertRaises(ValueError):c.compare_routes(native,f,prior)
 def test_exact_packets_echoes_replays(self):self.run_case('valid')
 def test_mismatch_preserved_not_fake_packet_pass(self):self.run_case('mismatch')
 def test_truncated_marker(self):self.run_case('newline')
 def test_duplicate_marker(self):self.run_case('duplicate')
 def test_extra_file(self):self.run_case('extra')
 def test_nonzero_old_packet(self):self.run_case('zero')
 def test_changed_input(self):self.run_case('input')
 def test_replay_packet_mutation(self):self.run_case('changed_packet')
 def test_alias(self):self.run_case('alias')
 def test_declared_config_and_public_function_shape(self):
  text=Path(c.__file__).with_name('own_q81_quantizer43_gpu_v1.cpp').read_text();self.assertIn('native_quantize_q8_1(x,y,cols,4,&q)',text);self.assertIn('mismatches*32',text);self.assertIn('shadow[b*35+1]=d',text);self.assertEqual(len(c.config('output_gated')),64)
 def test_division_versus_reciprocal_and_half_divisor_negative(self):
  x=np.zeros(32,dtype='<f4');x[0]=31.6876964569;x[1]=3.8673961163;p=encode(x.tobytes(),1,32,NATIVE);self.assertNotEqual(p,encode(x.tobytes(),1,32,OFFICIAL));d=np.float32(np.max(abs(x))/127);half=np.float32(np.float16(d));self.assertNotEqual(d.tobytes(),half.tobytes())
 def test_half_away_not_even(self):
  x=np.zeros(32,dtype='<f4');x[0]=127;x[1]=-6.5;self.assertEqual(encode(x.tobytes(),1,32)[5],249);self.assertEqual(int(np.rint(x[1])),-6)
 def test_lifecycle_derivation_uses_new_single_recipe(self):
  ns=q.derived();code=__import__('inspect').getsource(q.derived);self.assertIn('/out/own-q81-quantizer43 --inputs /inputs',code);self.assertNotIn('indexer_materialized43_masked_control_v1',code);self.assertIn('finalized_binding',ns)
if __name__=='__main__':unittest.main()
