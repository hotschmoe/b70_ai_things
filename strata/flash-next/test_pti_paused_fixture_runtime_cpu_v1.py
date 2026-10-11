import copy,struct,unittest
import pti_paused_fixture_runtime_v1 as r
class Tests(unittest.TestCase):
 def fixture(self):
  props=dict(r.clocks.EXPECTED_DEVICE_CLOCK);d={'kind':'device_origin','device_handle':7,**props};rows=[{'kind':'host_clock_origin'},d]
  events=[{'ph':'M','name':'process_name','pid':17,'args':{'name':'HOST<fixture>'}},{'ph':'X','cat':'cpu_op','name':'zeCommandListImmediateAppendCommandListsExp'}]
  for i,name in enumerate(('PausedRegularFixtureA','PausedRegularFixtureB'),1):
   row={'kind':'instance_normalization','instance_id':i,'device_handle':7,'timer_frequency_hz':props['timer_frequency_hz'],'timer_mask':props['timer_mask'],'ns_per_cycle_ieee754_u64':props['ns_per_cycle_ieee754_u64'],'raw_kernel_start_ticks':100+i*100,'raw_kernel_end_ticks':150+i*100,'submit_device_ticks':1,'submit_host_ns':1000}
   start,end=r.clocks.normalize_clock({k:row[k]for k in('timer_mask','timer_frequency_hz','ns_per_cycle_ieee754_u64','raw_kernel_start_ticks','raw_kernel_end_ticks','submit_device_ticks','submit_host_ns')});row.update(normalized_start_host_ns=start,normalized_end_host_ns=end);rows.append(row);events.append({'ph':'X','cat':'gpu_op','name':name,'args':{'id':str(i)}})
  for site in range(1,7):rows.append({'kind':'query_sampling_summary','query_site':site,'original_query_calls':0,'selected_brackets':0,'suppressed_brackets':0,'completed_brackets':0,'sample_every':1 if site<=2 else 512})
  rows.append({'kind':'host_clock_bridge','site':2})
  for i,row in enumerate(rows,1):row.update(schema=1,namespace='b70-pti-clock-export-v1',source_base='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033',pid=17,sequence=i)
  return rows,events,props
 def test_selected_children(self):
  rows,e,p=self.fixture();self.assertEqual(r.audit_rows(rows,e,True,2,p)['actual_child_counts'],{'PausedRegularFixtureA':1,'PausedRegularFixtureB':1})
 def test_warmup_forbidden(self):
  rows,e,p=self.fixture()
  with self.assertRaises(ValueError):r.audit_rows(rows,e,True,3,p)
 def test_changed_normalization(self):
  rows,e,p=self.fixture();rows[2]['normalized_end_host_ns']+=1
  with self.assertRaises(ValueError):r.audit_rows(rows,e,True,2,p)
 def test_wrong_device_or_child(self):
  rows,e,p=self.fixture();p['timer_frequency_hz']+=1
  with self.assertRaises(ValueError):r.audit_rows(rows,e,True,2,p)
  rows,e,p=self.fixture();e[-1]['name']='foreign'
  with self.assertRaises(ValueError):r.audit_rows(rows,e,True,2,p)
if __name__=='__main__':unittest.main()

class StrictClockTests(unittest.TestCase):
 def fixture(self,root,empty=False):
  import json
  from collections import Counter
  from test_unitrace_indexer40_cpu_v5 import BridgeCaptureTests
  rows=BridgeCaptureTests().fixture(root);keep=[]
  for row in rows:
   if row['kind']=='instance_normalization'and (empty or row['instance_id']>2):continue
   if row['kind']=='global_timestamp_query_bracket'and row['query_site']==5 and (empty or row['host_ns']!=11000):continue
   if row['kind']=='host_clock_bridge'and row['site']==105 and (empty or row['raw_before_ns']!=11002):continue
   keep.append(row)
  common={k:keep[0][k]for k in('schema','namespace','source_base','pid','writer_tid')};counts=Counter(x['query_site']for x in keep if x['kind']=='global_timestamp_query_bracket')
  summaries=[{**common,'kind':'query_sampling_summary','query_site':s,'sample_every':1 if s<=2 else 512,'original_query_calls':counts[s],'selected_brackets':counts[s],'completed_brackets':counts[s],'suppressed_brackets':0}for s in range(1,7)]
  keep[-1:-1]=summaries
  for i,row in enumerate(keep,1):row['sequence']=i
  BridgeCaptureTests().save(root,keep);chrome=next((root/'unitrace').glob('*.json'));v=json.loads(chrome.read_bytes());v['traceEvents']=[e for e in v['traceEvents']if e.get('ph')=='M'or(not empty and int(e['args']['id'])<=2)];chrome.write_text(json.dumps(v));return keep
 def test_complete_original_grammar_empty_and_two_children(self):
  import tempfile
  from pathlib import Path
  for empty in(False,True):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);self.fixture(root,empty);self.assertEqual(r.fixture_clock_port(r.clocks.EXPECTED_DEVICE_CLOCK)(root,'on')['instance_records'],0 if empty else 2)
 def test_origin_query_and_init_mutation_refused(self):
  import tempfile
  from pathlib import Path
  from test_unitrace_indexer40_cpu_v5 import BridgeCaptureTests
  for kind in('origin','init'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);rows=self.fixture(root)
    if kind=='origin':next(x for x in rows if x['kind']=='device_origin')['device_ticks']+=1
    else:rows[1]['writer_tid']+=1
    BridgeCaptureTests().save(root,rows)
    with self.assertRaises(ValueError):r.fixture_clock_port(r.clocks.EXPECTED_DEVICE_CLOCK)(root,'on')
