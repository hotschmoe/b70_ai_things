import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import probe_unitrace_indexer40_v5 as p
from test_unitrace_indexer40_cpu_v3 import ClockCaptureTests

class BridgeCaptureTests(unittest.TestCase):
 def fixture(self,root):
  old=ClockCaptureTests().fixture(root);common={k:old[0][k]for k in('schema','namespace','source_base','pid','writer_tid')};rows=[old[0]]
  def bridge(site,raw):return {**common,'kind':'host_clock_bridge','site':site,'raw_before_ns':raw,'monotonic_ns':raw+50,'raw_after_ns':raw+2}
  def query(site,device,host,ticks):
   return {**common,'kind':'global_timestamp_query_bracket','query_site':site,'device_handle':device,'return_code':0,'returned_pair_present':1,'host_ns':host,'device_ticks':ticks,'raw_before_ns':host-1,'raw_after_ns':host+1}
  rows.append(bridge(1,9990));rows.append(query(1,42,10000,1000));rows.append(bridge(101,10002));rows.append(old[1])
  for row in old[2:]:
   if row['instance_id']!=1:rows.extend([query(5,42,row['submit_host_ns'],row['submit_device_ticks']),bridge(105,row['submit_host_ns']+2)])
   rows.append(row)
  rows.append(bridge(2,1000000))
  self.save(root,rows);return rows
 def save(self,root,rows):
  for seq,row in enumerate(rows,1):row['sequence']=seq
  (root/'unitrace/clock.30.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in rows))
 def test_actual_new_source_and_build_admission(self):self.assertEqual(len(p.source_binding()),64)
 def test_both_flags_only_ON(self):
  with patch.object(Path,'stat')as st:
   st.return_value.st_gid=991;on=p.recipe('on',Path('/tmp/on'),'on','a'*64);off=p.recipe('off',Path('/tmp/off'),'off','a'*64)
  self.assertIn('UNITRACE_HOST_CLOCK_BRIDGE=1',on);self.assertIn('UNITRACE_CLOCK_METADATA_PATH='+p.CLOCK_PATH,on)
  self.assertFalse(any(x.startswith('UNITRACE_HOST_CLOCK_BRIDGE=')or x.startswith('UNITRACE_CLOCK_METADATA_PATH=')for x in off))
 def test_variable_sites_and_complete919_consumed(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);self.fixture(root);v=p.clock_binding(root,'on')['bridge_observations'];self.assertEqual(v['query_site_occurrences_observed'],{'1':1,'5':918});self.assertIs(v['all_six_static_sites_required'],False);self.assertIs(v['clock_alignment_qualified'],False)
 def test_missing_query_afterbridge_or_close_refused(self):
  for kind in ('query','after','close'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root)
    if kind=='query':rows.pop(2)
    elif kind=='after':rows.pop(3)
    else:rows.pop()
    self.save(root,rows)
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_CPU_bracket_reordered_and_bool_refused(self):
  for kind in ('order','bool','site'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root)
    if kind=='order':rows[2]['raw_after_ns']=1
    elif kind=='bool':rows[2]['return_code']=False
    else:rows[2]['query_site']=7
    self.save(root,rows)
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_foreign_thread_or_wrong_post_site_refused(self):
  for key,value in [('writer_tid',31),('site',106)]:
   with self.subTest(key=key),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root);rows[3][key]=value;self.save(root,rows)
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_returned_host_domain_outside_bracket_retained_not_calibrated(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);rows=self.fixture(root);rows[2]['raw_before_ns']=9990;rows[2]['raw_after_ns']=9991;self.save(root,rows);v=p.clock_binding(root,'on')['bridge_observations'];self.assertIs(v['returned_host_within_RAW_brackets_observed'],False);self.assertIs(v['physical_pair_uncertainty_qualified'],False)
 def test_failed_pair_semantics_and_no_false_success(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);rows=self.fixture(root);q={**rows[2],'query_site':6,'return_code':42,'returned_pair_present':0,'host_ns':0,'device_ticks':0};after={**rows[3],'site':106};rows[-1:-1]=[q,after];self.save(root,rows);v=p.bridge_grammar(rows);self.assertEqual(v['query_failures_retained'],1);self.assertIs(v['returned_host_within_RAW_brackets_observed'],False)
   with self.assertRaises(ValueError):p.clock_binding(root,'on')
   q['host_ns']=123;self.save(root,rows)
   with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_duplicate_schema_keys_and_normalization_still_refused(self):
  with self.assertRaises(ValueError):p.read_json_line(b'{"query_site":1,"query_site":5}')
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);rows=self.fixture(root);next(x for x in rows if x['kind']=='instance_normalization')['normalized_end_host_ns']+=1;self.save(root,rows)
   with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_moved_init_bridge_with_renumber_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);rows=self.fixture(root);initial=rows.pop(1);rows.insert(4,initial);self.save(root,rows)
   with self.assertRaises(ValueError):p.clock_binding(root,'on')
 def test_device_origin_query_ticks_handle_host_and_site_refused(self):
  for kind in ('ticks','handle','host','site'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root);origin=next(x for x in rows if x['kind']=='device_origin')
    if kind=='ticks':origin['device_ticks']+=1
    elif kind=='handle':origin['device_handle']+=1
    elif kind=='host':origin['host_ns']+=1
    else:rows[2]['query_site']=5;rows[3]['site']=105
    self.save(root,rows)
    with self.assertRaises(ValueError):p.clock_binding(root,'on')
if __name__=='__main__':unittest.main()
