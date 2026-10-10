import importlib.util,json,re,tempfile,unittest,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
producer=module('h36_prepare',HERE/'prepare_host_critical_path36_v1.py')
collector=module('h36_collect',HERE/'collect_host_critical_path36_v1.py')
K=collector.KINDS

class HostTests(unittest.TestCase):
 def fixture(self,root):
  events=[];span=0;ctx={'dev':-1,'lb':-1,'le':-1,'T':0,'pos':-1,'queue':0,'graph':0}
  def record(kind,edge,s=0,value=0):events.append({'seq':len(events),'ns':100+len(events)*100,'span':s,'kind':K[kind],'edge':edge,'value':value,**ctx})
  def begin(kind):
   nonlocal span
   span+=1;record(kind,1,span);return span
  def end(kind,s):record(kind,2,s)
  request=begin('Request');reset=begin('Reset');end('Reset',reset);prompt=begin('PromptSpan');end('PromptSpan',prompt)
  for pos in range(3):
   ctx={'dev':0,'lb':0,'le':48,'T':1,'pos':pos,'queue':7,'graph':1}
   verify=begin('Verify');record('GraphCreate' if pos==0 else 'GraphHit',0)
   submit=begin('ReplaySubmit');end('ReplaySubmit',submit);wait=begin('ReplayWait');end('ReplayWait',wait);end('Verify',verify)
  ctx['pos']=3;record('TEmit',0,value=16)
  printf=begin('TPrintf');end('TPrintf',printf)
  readback=begin('LPReadback');end('LPReadback',readback);cpu=begin('LPCPU');end('LPCPU',cpu)
  flush=begin('StdoutFlush');end('StdoutFlush',flush)
  ctx={'dev':-1,'lb':-1,'le':-1,'T':0,'pos':-1,'queue':0,'graph':0}
  done=begin('DoneEmit');end('DoneEmit',done);end('Request',request)
  meta={'schema':36,'clock':'CLOCK_MONOTONIC','device_clock':'not_requested','binding':'a'*64,'pid':11,'request':1,'rid':0,'slotgen':0,'prompt':3,'fresh':1,'produced':1,'complete':True,'overflow':False,'thread_error':False,'clock_error':False,'events':len(events),'capacity':32768}
  footer={'footer':True,'records':len(events),'write_error':False}
  path=root/'trace.jsonl';path.write_text('\n'.join(json.dumps(x) for x in [meta,*events,footer])+'\n')
  log=root/'stderr.log';log.write_text('HOST_TRACE36_CLOSED request=1 success=1\nHOST_TRACE36_SESSION requests=1 thread_error=0 export_error=0 request_limit=0\n')
  raw={'ids':[1,2,3],'fresh':1,'output_ids':[16],'LP':['LP 0'],'done':'DONE 1 3 1 1 stop'}
  return path,log,raw
 def check(self,p,l,r):return collector.collect([p],l,'a'*64,[r],[[0,0,48]])
 def mutate(self,p,change):
  rows=[json.loads(x) for x in p.read_text().splitlines()];change(rows);p.write_text('\n'.join(map(json.dumps,rows))+'\n')
 def test_complete_host_only_fixture(self):
  with tempfile.TemporaryDirectory() as t:
   p,l,r=self.fixture(Path(t));report=self.check(p,l,r);self.assertTrue(report['passed']);self.assertFalse(report['actual_GPU_run_qualified']);self.assertFalse(report['speed_qualified'])
 def test_bad_metadata_flags(self):
  for key,val in [('overflow',True),('thread_error',True),('clock_error',True),('complete',False),('device_clock','fake_gpu_ns'),('binding','b'*64),('fresh',0)]:
   with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
    p,l,r=self.fixture(Path(t));self.mutate(p,lambda rows:rows[0].update({key:val}))
    with self.assertRaises(ValueError):self.check(p,l,r)
 def test_drop_duplicate_reverse_scope_or_clock(self):
  for mode in ['drop','duplicate','reverse','zero']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
    p,l,r=self.fixture(Path(t))
    def change(rows):
     if mode=='drop':rows.pop(5)
     if mode=='duplicate':rows.insert(5,rows[5].copy())
     if mode=='reverse':rows[4],rows[5]=rows[5],rows[4]
     if mode=='zero':rows[5]['ns']=0
    self.mutate(p,change)
    with self.assertRaises(ValueError):self.check(p,l,r)
 def test_unknown_graph_stage_token_input_association(self):
  for mode in ['graph','stage','position','token','printf','flush']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
    p,l,r=self.fixture(Path(t))
    def change(rows):
     e=next(e for e in rows[1:-1] if e['kind']==K['TEmit'])
     if mode=='graph':e['graph']=99
     if mode=='stage':e['dev']=1
     if mode=='position':e['pos']=4
     if mode=='token':e['value']=17
     if mode=='printf':next(e for e in rows[1:-1] if e['kind']==K['TPrintf'] and e['edge']==2)['span']=999
     if mode=='flush':next(e for e in rows[1:-1] if e['kind']==K['StdoutFlush'])['kind']=K['Sample']
    self.mutate(p,change)
    with self.assertRaises(ValueError):self.check(p,l,r)
 def test_missing_close_or_terminal_free_proof(self):
  for text in ['HOST_TRACE36_CLOSED request=1 success=0\n','HOST_TRACE36_CLOSED request=1 success=1\nHOST_TRACE36_SESSION requests=1 thread_error=1 export_error=0 request_limit=0\n']:
   with tempfile.TemporaryDirectory() as t:
    p,l,r=self.fixture(Path(t));l.write_text(text)
    with self.assertRaises(ValueError):self.check(p,l,r)
 def test_preexisting_graph_not_false_first_use(self):
  with tempfile.TemporaryDirectory() as t:
   p,l,r=self.fixture(Path(t));self.mutate(p,lambda rows:next(e for e in rows[1:-1] if e['kind']==K['GraphCreate']).update(kind=K['GraphPreexisting']))
   self.assertTrue(self.check(p,l,r)['passed'])
 def test_native_source_reconstruction_and_no_new_waits(self):
  old,new=producer.reconstruct()
  for path in [producer.G,producer.V]:
   waits=lambda s:re.findall(r'\b(?:wait|wait_and_throw|queues_wait_and_throw|ext_oneapi_graph)\s*\(',s)
   self.assertEqual(waits(old[path]),waits(new[path]),path)
  self.assertIn('o.layer_split.empty() && o.split_device.empty()',new[producer.G])
  self.assertIn('o.suffix_draft==0 && o.lookup_chain==0',new[producer.G])
  self.assertIn('::TPrintf)',new[producer.G]);self.assertIn('::StdoutFlush)',new[producer.G]);self.assertIn('::GraphFinalize)',new[producer.V])
  self.assertNotIn('::state().span',new[producer.G]);self.assertNotIn('++strata::program::host_trace36::state().span',new[producer.V])
 def test_complete_64_file_28_payload_source_closure(self):
  base=json.loads((HERE/'current-ple-prompt35-engine-build-plan-v1.json').read_text());built=json.loads((HERE/'host-critical-path36-engine-build-plan-v1.json').read_text());old,new=producer.reconstruct()
  self.assertEqual((len(built['expected_patched_source_sha256']),len(built['added_header_payloads']),len(built['patches']),len(built['build_targets']),len(built['runtime_python_sources'])),(64,28,36,8,6))
  for path,h in base['expected_patched_source_sha256'].items():self.assertEqual(producer.sha(producer.SDK/path),h,path)
  for path,h in built['expected_patched_source_sha256'].items():
   actual=__import__('hashlib').sha256(new[path].encode()).hexdigest() if path in new else producer.sha(producer.SDK/path)
   self.assertEqual(actual,h,path)
  self.assertEqual(producer.patch_bytes(),(HERE/'patches/0036-default-off-bounded-host-critical-path-trace.patch').read_bytes())
  self.assertEqual(built['runtime_python_sources'],base['runtime_python_sources'])

 def test_patch_applies_to_temp_only_and_payload_matches(self):
  old,new=producer.reconstruct()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for name,s in old.items():p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
   p=root/'change.patch';p.write_bytes(producer.patch_bytes())
   subprocess.run(['patch','--batch','-p1','-i',str(p)],cwd=root,check=True,capture_output=True)
   for name,s in new.items():self.assertEqual((root/name).read_text(),s)
 def test_actual_compiled_cpu_receipt_and_current_frozen_payload(self):
  root=Path('/mnt/vm_8tb/b70/build/hosttrace36-cpu-actual-v2-20261010');parent=json.loads((root/'parent-receipt.json').read_text());inside=json.loads((root/'inside-receipt.json').read_text())
  self.assertTrue(parent['passed']);self.assertTrue(parent['container_removed']);self.assertEqual(parent['container_terminal']['ExitCode'],0);self.assertFalse(parent['host_config']['Devices']);self.assertEqual(parent['host_config']['NetworkMode'],'none')
  self.assertTrue(inside['passed']);self.assertEqual(len(inside['cases']),9);self.assertEqual(producer.sha(root/'fixture'),inside['binary_sha256']);self.assertEqual(producer.sha(root/'inside-receipt.json'),parent['inside_receipt_sha256'])
  saved=json.loads((HERE/'host-critical-path36-cpu-receipt-v2.json').read_text());snapshot=json.loads(saved['source_snapshot_raw'])
  self.assertEqual(__import__('hashlib').sha256(saved['source_snapshot_raw'].encode()).hexdigest(),parent['source_snapshot_sha256'])
  self.assertEqual(producer.sha(root/'durable-copy-receipt.json'),saved['durable_copy_receipt_sha256'])
  for name,h in snapshot['source_files'].items():self.assertEqual(producer.sha(HERE/name),h)
  modes={x['mode']:x for x in inside['cases']};self.assertIn('success=0',modes['writefail']['stderr']);self.assertIn('thread_error=1',modes['foreign']['stderr']);self.assertIn('request_limit=1',modes['limit']['stderr'])

if __name__=='__main__':unittest.main()
