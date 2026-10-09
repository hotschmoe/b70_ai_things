"""Source35 reconstruction, row-layout and tiny strict collector controls only."""
import ast
import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import collect_prefix_residual30_v1 as oldcollector
import collect_prefix_residual30_v2 as collector
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((HERE/'current-ple-prompt35-engine-build-plan-v1.json').read_text());cls.old=json.loads((HERE/'current-ple-lastwindow34-engine-build-plan-v1.json').read_text());cls.tmp=tempfile.TemporaryDirectory(prefix='source35-cpu-');cls.source=Path(cls.tmp.name);needed=set(cls.plan['source_file_sha256'])
  for patch in cls.plan['patches']:
   text=(ROOT/patch['path']).read_text();needed.update(re.findall(r'^\+\+\+ b/(.*)$',text,re.M));needed.update(re.findall(r'^--- a/(.*)$',text,re.M))
  for name in needed:
   src=Path(cls.plan['source_root'])/name
   if src.is_file():dest=cls.source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  for name,digest in cls.plan['source_file_sha256'].items():assert sha(cls.source/name)==digest,name
  for index,patch in enumerate(cls.plan['patches']):
   path=ROOT/patch['path'];assert sha(path)==patch['sha256'];result=subprocess.run(['patch','--batch','--forward','-p1','-i',str(path)],cwd=cls.source,capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
   if index==33:cls.oldverify=(cls.source/'sycl/src/core/verify.cpp').read_text()
  cls.verify=(cls.source/'sycl/src/core/verify.cpp').read_text();cls.generate=(cls.source/'sycl/src/program/generate.cpp').read_text();cls.header=(cls.source/'sycl/include/strata/core/prefix_residual_capture.hpp').read_text();cls.vheader=(cls.source/'sycl/include/strata/core/verify.hpp').read_text()
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_pristine35_full63_header27_eightABI_sixPython(self):
  self.assertEqual((len(self.plan['patches']),len(self.plan['expected_patched_source_sha256']),len(self.plan['added_header_payloads']),len(self.plan['build_targets']),len(self.plan['runtime_python_sources'])),(35,63,27,8,6))
  for name,digest in self.plan['expected_patched_source_sha256'].items():self.assertEqual(sha(self.source/name),digest,name)
  for item in self.plan['added_header_payloads']:self.assertEqual(sha(self.source/item['path']),item['sha256'])
 def test_input34_and_USM_allocation_contract_unchanged(self):
  name='include/strata/core/ple_input_observer33.hpp';self.assertEqual(self.old['expected_patched_source_sha256'][name],self.plan['expected_patched_source_sha256'][name]);self.assertIn('r.active && pos0 + T == static_cast<int64_t>(r.ids.size())',self.verify)
  self.assertEqual(self.header.count('sycl::malloc_device('),1);self.assertIn('sycl::malloc_device(bytes_+16,q_)',self.header)
  contract='sycl/include/strata/core/prefix_residual_capture_contract.hpp';self.assertEqual(self.old['expected_patched_source_sha256'][contract],self.plan['expected_patched_source_sha256'][contract])
 def test_actual_prompt_call_select_run_dump_clear_order(self):
  start=self.generate.index('auto read_windows =');end=self.generate.index('// the batched path',start);body=self.generate[start:end]
  names=['ver.select_prefix30_prompt(T,q,win.data(),e)','ver.run(T, win.data(), q','ver.dump_prefix30_prompt(e)','ver.commit(T, e)']
  self.assertEqual(sorted(names,key=body.index),names);self.assertNotIn('select_fidelity_first',body)
  self.assertIn('const int T = (int) std::min<int64_t>(S, b - q)',body)
  self.assertIn('win ? read_windows(at, to, err) : read_part(at, to, err)',self.generate)
 def test_dedicated_T_AR_keys_capture_and_submit_agree(self):
  self.assertIn('prefix30_prompt_exec_[2][3]={}',self.vheader)
  self.assertEqual(self.verify.count('prefix30_prompt_ ? prefix30_prompt_exec_[ar_off_?1:0][T]'),2)
  capture=self.verify[self.verify.index('bool Verifier::capture('):self.verify.index('bool Verifier::capture_commit(')]
  self.assertLess(capture.index('if (exec_t != nullptr) return true'),capture.index('prefix30_snapshot_->begin_rows(T)'))
 def test_existing_fidelity_selector_dump_and_prefill_route_unchanged(self):
  start='void Verifier::select_fidelity_first(bool on)';end='bool Verifier::copy_logits('
  self.assertEqual(self.oldverify[self.oldverify.index(start):self.oldverify.index(end)],self.verify[self.verify.index(start):self.verify.index(end)])
  patch=(HERE/'patches/0035-default-off-current-prompt-verifier-prefix-residual-capture.patch').read_text()
  for forbidden in ('+    o.short_read', '+    o.spec', '+    o.prefill', '+    o.mtp'):self.assertNotIn(forbidden,patch)
 def test_actual_row_copy_locations_offsets_and_exact_coverage(self):
  self.assertIn('copy_rows("input",int(l),Rt(tb),n,tb,T,*cs)',self.verify)
  self.assertIn('copy_rows("attention",int(l),Rt(t),1,t,T,*cs)',self.verify)
  self.assertIn('copy_rows("ffn",int(l),Rt(tb),n,tb,T,*cs)',self.verify)
  self.assertIn('offset(layer,index)+std::size_t(first)*row_floats*4',self.header)
  self.assertIn('row_roster_[field*max_rows+row]',self.header)
  self.assertIn('P30 duplicate prompt verifier row slice',self.header)
  self.assertIn('sealed_rows_[rows]=true',self.header)
 def test_all_stage_select_dump_clear_and_run_identity_guards(self):
  for name in ('next_->select_prefix30_prompt(rows,pos,tokens,err)','next_->dump_prefix30_prompt(err)','next_->clear_prefix30_prompt()'):self.assertIn(name,self.verify)
  for name in ('r.ordinal!=prefix30_prompt_request_','T!=prefix30_prompt_rows_','pos0!=prefix30_prompt_pos_','tokens[row]!=r.ids[size_t(pos0+row)]','last_tokens_[row]!=r.ids[size_t(last_pos0_+row)]'):self.assertIn(name,self.verify)
 def test_new_graphs_retire_before_snapshot_in_all_owner_paths(self):
  for start,end in [('void Verifier::release_stage_mirror_graphs()', 'Verifier::~Verifier()'),('Verifier::~Verifier()', 'bool Verifier::init('),('void Verifier::release_fidelity_observer()', 'bool Verifier::select_prefix30_prompt(')]:
   body=self.verify[self.verify.index(start):self.verify.index(end,self.verify.index(start))]
   self.assertLess(body.index('for(auto& mode:prefix30_prompt_exec_)'),body.index('prefix30_snapshot_'))
 def test_OFF_selector_returns_before_any_new_GPU_operation(self):
  body=self.verify[self.verify.index('bool Verifier::select_prefix30_prompt('):self.verify.index('void Verifier::clear_prefix30_prompt()')]
  self.assertLess(body.index('if(!prefix_residual30::eligible())'),body.index('prefix30_snapshot_->begin('))
  clear=self.verify[self.verify.index('void Verifier::clear_prefix30_prompt()'):self.verify.index('bool Verifier::dump_prefix30_prompt(')]
  for forbidden in ('wait','memcpy','get_native','get_context','get_device','malloc'):self.assertNotIn(forbidden,clear)
 def test_normal_window_schedule_1_2_4_8_T1_T2_no_route_substitution(self):
  for n in (1,2,4,8):
   for S in (1,2):
    prompt=[];pos=0
    while pos<n-1:T=min(S,n-1-pos);prompt.append((pos,T));pos+=T
    rows=[p+r for p,T in prompt for r in range(T)]+[n-1]
    self.assertEqual(rows,list(range(n)));self.assertTrue(all(T in (1,2) and p+T<=n-1 for p,T in prompt))
 def test_T2_group_slices_cover_each_phase_layer_once(self):
  for split in (False,True):
   T=2;groups=[(0,T)] if not split else [(0,1),(1,2)];seen=set();buffer=np.full((3,48,8,4),-1,dtype=np.int32)
   def put(phase,layer,first,count):
    for row in range(first,first+count):
     key=(phase,layer,row)
     if key in seen:raise ValueError('duplicate actual row slice')
     seen.add(key);buffer[phase,layer,row]=phase*1000+layer*10+row
   for layer in range(48):
    for tb,te in groups:
     put(0,layer,tb,te-tb)
     for t in range(tb,te):put(1,layer,t,1)
     put(2,layer,tb,te-tb)
   self.assertEqual(len(seen),3*48*T)
   for phase in range(3):
    for layer in range(48):self.assertEqual(buffer[phase,layer,1,0],phase*1000+layer*10+1)
   self.assertTrue(np.all(buffer[:,:,T:]==-1))
   with self.assertRaises(ValueError):put(0,0,0,1)
 def test_partial_row_roster_and_wrong_shape_cannot_seal(self):
  want={(phase,layer,row) for phase in range(3) for layer in range(48) for row in range(2)}
  complete=set(want);complete.remove((1,17,1));self.assertNotEqual(complete,want)
  complete=set(want);complete.add((0,0,2));self.assertNotEqual(complete,want)
  self.assertIn('first+rows>total',self.header);self.assertIn('rows!=recording_rows_',self.header)
  self.assertIn('sealed_rows_[rows]=false',self.header)
 def test_selection_eligible_prefixes_and_request_bound(self):
  for part in ('r.ordinal>=1&&r.ordinal<=4','r.ids.size()==1||r.ids.size()==2||r.ids.size()==4||r.ids.size()==8','r.resume==0'):self.assertIn(part,self.header)


class CollectorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(prefix='prompt35-collector-');cls.root=Path(cls.tmp.name);cls.ids=[101,102,103,104];cls.requests={1:cls.ids};cls.binding='a'*64;cls.stages={0:(0,32),1:(32,48)};cls.metadata=[];lines=['SFD request pid=9 request=1 tokens=4 ids=101,102,103,104 shape=serial_text','SFD resume pid=9 request=1 reused=0 evaluated_prompt_pending=1']
  for pos,rows,route in ((0,2,'prompt_verifier'),(2,1,'prompt_verifier'),(3,1,'verifier')):
   for stage,(lo,hi) in cls.stages.items():
    stem='p30-9-r1-s%d-%s-p%d-n%d'%(stage,route,pos,rows);fields=[]
    for layer in range(lo,hi):
     for phase in collector.PHASES:
      path=cls.root/(stem+'-l%d-%s.f32'%(layer,phase));values=np.stack([np.full(10240,layer*10+pos+r,dtype='<f4') for r in range(rows)]);path.write_bytes(values.tobytes());fields.append({'layer':layer,'phase':phase,'bytes':rows*40960,'file':path.name,'encoding':'LE_F32[rows,4,2560]'})
    data={'schema':2,'pid':9,'request':1,'stage':stage,'lb':lo,'le':hi,'route':route,'native_hc_source':True,'first_position':pos,'rows':rows,'row_floats':10240,'source_extent_bytes':rows*40960,'nonce':collector.window_nonce(9,1,pos,rows,route),'binding_sha256':cls.binding,'gen_ids':cls.ids,'normal_dispatch':collector.ROUTES[route][1],'row_coverage_complete':True,'fields':fields,'full_model_math_qualified':False};path=cls.root/(stem+'.json');cls.metadata.append((path,data))
    if route=='prompt_verifier':lines.extend(['P30 prompt_select pid=9 request=1 stage=%d pos=%d T=%d ids=%s'%(stage,pos,rows,','.join(map(str,cls.ids[pos:pos+rows]))),'P30 prompt_returned pid=9 request=1 stage=%d pos=%d T=%d rc=0'%(stage,pos,rows)])
    lines.append('P30 frame pid=9 request=1 stage=%d route=%s rows=%d p0=%d metadata=%s'%(stage,route,rows,pos,path.name))
  cls.log=cls.root/'producer.log';cls.logtext='\n'.join(lines)+'\n'
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def setUp(self):
  self.rows=copy.deepcopy(self.metadata)
  for path,data in self.rows:path.write_text(json.dumps(data))
  self.log.write_text(self.logtext)
 def collect(self):return collector.collect(self.root,self.requests,self.stages,self.binding,self.log)
 def test_new_truthful_route_allrows_two_stages(self):
  proof=self.collect();self.assertTrue(proof['passed']);self.assertEqual(len(proof['frames']),6);self.assertTrue(proof['normal_dispatch_source_crossbinding_verified']);self.assertFalse(proof['GPU_graph_handle_source_row_association_observed'])
 def test_old_collector_rejects_schema2_truthful_prompt_verifier(self):
  with self.assertRaises(ValueError):oldcollector.collect(self.root,self.requests,self.stages,self.binding,self.log)
 def test_missing_earlier_window_rejected(self):
  path,data=self.rows[0];path.unlink()
  with self.assertRaises(ValueError):self.collect()
 def test_wrong_stale_window_nonce_rejected(self):
  path,data=self.rows[2];data['nonce']=collector.window_nonce(9,1,0,2,'prompt_verifier');path.write_text(json.dumps(data))
  with self.assertRaisesRegex(ValueError,'window nonce'):self.collect()
 def test_finalrow_mislabeled_prompt_route_rejected(self):
  path,data=self.rows[-1];data['route']='prompt_verifier';data['normal_dispatch']='prompt_verify';data['nonce']=collector.window_nonce(9,1,3,1,'prompt_verifier');path.write_text(json.dumps(data))
  with self.assertRaisesRegex(ValueError,'earlier actual'):self.collect()
 def test_wrong_dispatch_or_unsealed_rowcoverage_rejected(self):
  for key,value in [('normal_dispatch','prefill'),('row_coverage_complete',False)]:
   self.setUp();path,data=self.rows[0];data[key]=value;path.write_text(json.dumps(data))
   with self.assertRaises(ValueError):self.collect()
 def test_wrong_selected_actual_tokens_rejected(self):
  self.log.write_text(self.logtext.replace('ids=101,102\n','ids=101,103\n',1))
  with self.assertRaisesRegex(ValueError,'token association'):self.collect()
 def test_missing_or_failed_actual_return_rejected(self):
  self.log.write_text(self.logtext.replace('rc=0','rc=1',1))
  with self.assertRaisesRegex(ValueError,'failed actual'):self.collect()
 def test_return_before_select_rejected(self):
  lines=self.logtext.splitlines();lines[2],lines[3]=lines[3],lines[2];self.log.write_text('\n'.join(lines)+'\n')
  with self.assertRaisesRegex(ValueError,'association'):self.collect()
 def test_missing_producer_log_rejected(self):
  with self.assertRaises(ValueError):collector.collect(self.root,self.requests,self.stages,self.binding)
 def test_no_log_can_never_claim_normal_dispatch_proof(self):
  result=None
  try:result=collector.collect(self.root,self.requests,self.stages,self.binding,producer_log=None)
  except ValueError:pass
  self.assertTrue(result is None or result['normal_dispatch_source_crossbinding_verified'] is False)
  self.assertIn("'normal_dispatch_source_crossbinding_verified':producer_log is not None",Path(collector.__file__).read_text())
 def test_changed_finite_raw_bytes_change_recollection_proof(self):
  before=self.collect();path=self.root/self.rows[0][1]['fields'][0]['file'];raw=path.read_bytes();path.write_bytes(np.full(20480,999,dtype='<f4').tobytes());self.addCleanup(path.write_bytes,raw);after=self.collect();self.assertNotEqual(before,after)
 def test_missing_phase_or_wrong_field_extent_rejected(self):
  path,data=self.rows[0];data['fields'][0]['bytes']=40960;path.write_text(json.dumps(data))
  with self.assertRaises(ValueError):self.collect()
 def test_no_raw_snapshot_path_traversal(self):
  path,data=self.rows[0];data['fields'][0]['file']='../'+data['fields'][0]['file'];path.write_text(json.dumps(data))
  with self.assertRaises(ValueError):self.collect()
 def test_nonce_distinguishes_request_route_position_and_shape(self):
  values=[collector.window_nonce(9,request,pos,T,route) for request in (1,2,3,4) for pos in range(8) for T in (1,2) for route in collector.ROUTES];self.assertEqual(len(values),len(set(values)))
 def test_nonpilot_prefix_or_token_roster_rejected(self):
  for requests in ({1:[1,2,3]},{0:[1]},{1:[True]},{1:[248320]}):
   with self.assertRaises(ValueError):collector.collect(self.root,requests,self.stages,self.binding,self.log)


if __name__=='__main__':unittest.main()
