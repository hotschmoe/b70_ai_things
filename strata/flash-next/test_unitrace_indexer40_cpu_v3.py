import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import probe_unitrace_indexer40_v3 as p

class ClockCaptureTests(unittest.TestCase):
 def fixture(self,root):
  raw=root/'unitrace';raw.mkdir();pid=30;epoch=1791675511887210311
  common={'schema':1,'namespace':'b70-pti-clock-export-v1','source_base':'6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033','pid':pid,'writer_tid':pid}
  rows=[{**common,'sequence':1,'kind':'host_clock_origin','epoch_origin_ns':epoch,'monotonic_raw_minus_boot_i64_bits':1,'monotonic_raw_minus_tai_i64_bits':2},{**common,'sequence':2,'kind':'device_origin','device_handle':42,'descriptor_id':0,'host_ns':10000,'device_ticks':1000,**p.EXPECTED_DEVICE_CLOCK,'epoch_origin_ns':epoch}]
  gpu=[]
  for i in range(919):
   subset={'timer_mask':p.EXPECTED_DEVICE_CLOCK['timer_mask'],'timer_frequency_hz':19200000,'ns_per_cycle_ieee754_u64':p.EXPECTED_DEVICE_CLOCK['ns_per_cycle_ieee754_u64'],'raw_kernel_start_ticks':1002+i*20,'raw_kernel_end_ticks':1007+i*20,'submit_device_ticks':1000+i*20,'submit_host_ns':10000+i*1000}
   start,end=p.normalize_clock(subset)
   rows.append({**common,'sequence':i+3,'kind':'instance_normalization','instance_id':i+1,'command_id':i+1,'device_handle':42,'command_list_handle':43,'queue_handle':0,'engine_ordinal':0,'engine_index':0,'submitter_tid':pid,'append_host_ns':9999,**subset,'normalized_start_host_ns':start,'normalized_end_host_ns':end,'epoch_origin_ns':epoch,'host_time_origin_ns':10000,'graph_command':0})
   gpu.append({'ph':'X','cat':'gpu_op','args':{'id':str(i+1)}})
  (raw/'indexer-materia.30.json').write_text(json.dumps({'traceEvents':[{'ph':'M','name':'process_name','pid':pid,'args':{'name':'HOST<cpu_fixture>'}},*gpu]}))
  (root/'leaf.stderr').write_text('');self.save(root,rows);return rows
 def save(self,root,rows):
  (root/'unitrace/clock.30.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in rows))
 def test_actual_patched_source_admission(self):self.assertEqual(len(p.source_binding()),64)
 def test_feature_only_on_recipe(self):
  with patch.object(Path,'stat')as st:
   st.return_value.st_gid=991;off=p.recipe('off',Path('/tmp/off'),'a','a'*64);on=p.recipe('on',Path('/tmp/on'),'b','b'*64)
  self.assertFalse(any(x.startswith('UNITRACE_CLOCK_METADATA_PATH=')for x in off));self.assertIn('UNITRACE_CLOCK_METADATA_PATH='+p.CLOCK_PATH,on)
  self.assertIn(str(p.INSTALL)+':/tool:ro',on);self.assertEqual(p.tool_argv()[-len(p.helper_argv()):],p.helper_argv())
 def test_actual_shaped_full_clock_roster(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);self.fixture(root);r=p.clock_binding(root,'on');self.assertEqual(r['instance_records'],919);self.assertIs(r['clock_calibration_qualified'],False)
 def test_foreign_pid_sequence_and_missing_record(self):
  for kind in('pid','sequence','missing'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root)
    if kind=='missing':rows.pop()
    else:rows[-1][kind]=9999
    self.save(root,rows)
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_normalization_and_property_mutation(self):
  for kind in('normalization','frequency','UUID'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root)
    if kind=='normalization':rows[-1]['normalized_start_host_ns']+=1
    elif kind=='frequency':rows[1]['timer_frequency_hz']+=1
    else:rows[1]['uuid_word1_native_u64']=68
    self.save(root,rows)
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_legacy_boolean_rawerror_and_partial_line(self):
  for kind in('bool','error','partial'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root)
    if kind=='bool':rows[-1]['graph_command']=False;self.save(root,rows)
    elif kind=='error':(root/'leaf.stderr').write_text('[B70_CLOCK_EXPORT_ERROR] close failed\n')
    else:
     f=root/'unitrace/clock.30.jsonl';f.write_bytes(f.read_bytes()[:-1])
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_wrong_Chrome_owner_and_duplicate_ID(self):
  for kind in('owner','duplicate'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);self.fixture(root);f=root/'unitrace/indexer-materia.30.json';v=json.loads(f.read_bytes())
    if kind=='owner':v['traceEvents'][0]['pid']=31
    else:v['traceEvents'][-1]['args']['id']='1'
    f.write_text(json.dumps(v))
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_OFF_has_no_clock_metadata(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'unitrace').mkdir();self.assertEqual(p.clock_binding(root,'off'),{'clock_export_enabled':False});(root/'unitrace/clock.30.jsonl').write_text('{}\n')
   with self.assertRaises(ValueError):p.clock_binding(root,'off')
 def test_duplicate_JSON_clock_fields_rejected(self):
  with self.assertRaises(ValueError):p.read_json_line(b'{"pid":30,"pid":31}')
if __name__=='__main__':unittest.main()
