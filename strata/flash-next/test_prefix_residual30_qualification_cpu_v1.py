#!/usr/bin/env python3
"""Synthetic/tiny P30 runtime-gate controls; no Docker, weights, GPU or runtime writes."""
import copy,json,tempfile,unittest,ast
from pathlib import Path
from unittest.mock import patch
import numpy as np
import prefix_residual30_qualification_v1 as q
import qualify_prefix_residual30_v1 as parent
from audit_prefix30_lifecycle_v1 import audit_text

class MappingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.rows=[];self.frames=[]
  for request,n in enumerate(q.PREFIXES,1):
   ids=list(range(100,100+n));vectors=[];fields=[]
   for layer in range(48):
    raw=np.full(10240,np.float32(layer+n)).tobytes();a=self.root/f'p30-{n}-{layer}.f32';b=self.root/f'sfd-{n}-{layer}.f32';a.write_bytes(raw);b.write_bytes(raw)
    vectors.append({'pid':'9','request':str(request),'stage':'0','layer':str(layer),'pos':str(n-1),'token':str(ids[-1]),'bytes':'40960','path':str(b),'sha256':q.sha(b)})
    fields.append({'phase':'ffn','layer':layer,'bytes':40960,'path':str(a),'sha256':q.sha(a)})
   self.rows.append({'prefix':n,'meta':{'logits':[{'pid':'9','request':str(request)}],'residuals':vectors}});self.frames.append({'binding':{'route':'verifier','gen_ids':ids,'pid':9,'request':request,'stage':0,'first_position':n-1},'fields':fields})
  self.capture={'frames':self.frames}
 def test_all192_bitwise_actual_raw_pair_positive(self):self.assertEqual(len(q.cross_sfd(self.capture,self.rows)['pairs']),192)
 def test_samebinary49_fullvectors_control(self):
  head=self.root/'head.f32';head.write_bytes(np.zeros(248320,dtype='<f4').tobytes());vectors=self.rows[0]['meta']['residuals'];meta={'logits':[{'path':str(head),'sha256':q.sha(head)}],'residuals':vectors};raw={'ids':[100],'fresh':1,'output_ids':[101],'LP':['LP fixture'],'done':'DONE 0 0 0 0 length'}
  row={'raw':raw,'meta':meta};result=q.numeric.compare_numeric(row,copy.deepcopy(row));self.assertTrue(result['head']['bitwise_equal']);self.assertEqual(result['head']['floats'],248320);self.assertEqual(len(result['residuals']),48)
  changed=copy.deepcopy(row);changed['meta']['residuals'].pop()
  with self.assertRaises(ValueError):q.numeric.compare_numeric(row,changed)
 def test_dynamic_row_offset_even_valid_nonce_cannot_pass(self):
  field=self.frames[-1]['fields'][-1];p=Path(field['path']);raw=np.full(10240,np.float32(999)).tobytes();p.write_bytes(raw);field['sha256']=q.sha(p)
  with self.assertRaises(ValueError):q.cross_sfd(self.capture,self.rows)
 def test_missing_layer_rejected(self):
  self.frames[0]['fields'].pop()
  with self.assertRaises(ValueError):q.cross_sfd(self.capture,self.rows)
 def test_duplicate_layer_rejected(self):
  self.frames[0]['fields'].append(self.frames[0]['fields'][0])
  with self.assertRaises(ValueError):q.cross_sfd(self.capture,self.rows)
 def test_changed_file_with_stale_sha_rejected(self):
  Path(self.frames[0]['fields'][0]['path']).write_bytes(bytes(40960))
  with self.assertRaises(ValueError):q.cross_sfd(self.capture,self.rows)
 def test_wrong_stage_request_position_rejected(self):
  for key,value in [('stage',1),('request',99),('first_position',99)]:
   capture=copy.deepcopy(self.capture);capture['frames'][0]['binding'][key]=value
   with self.assertRaises(ValueError):q.cross_sfd(capture,self.rows)

class LifeTests(unittest.TestCase):
 def fixture(self,on=True):
  self.rows=[];ptr=0x1000;self.targets=[]
  for stage,(lb,le) in {0:(0,32),1:(32,48)}.items():
   ctx=hex(0x10+stage);device=hex(0x20+stage)
   for kind in (['SFD','SFD','P30','P30'] if on else ['SFD','SFD']):
    if kind=='SFD':head=(len([t for t in self.targets if t['stage']==stage and t['kind']=='SFD'])==0 and le==48);size=(le-lb)*40960+int(head)*248320*4
    else:size=(le-lb)*3*8*40960+16
    ptr+=0x100;pointer=hex(ptr);self.rows.append(f'<--- urUSMDeviceAlloc(.hContext = {ctx}, .hDevice = {device}, .size = {size}, .ppMem = 0xf ({pointer})) -> UR_RESULT_SUCCESS;')
    marker=f'{kind} allocation '+(f'pid=9 ' if kind=='P30' else '')+f'stage={stage} lb={lb} le={le} '+(f'pointer={pointer} ' if kind=='P30' else '')+f'bytes={size}'+(f' residuals1 logits{int(head)}' if kind=='SFD' else '')
    self.rows.append(marker);self.targets.append(dict(kind=kind,pointer=pointer,stage=stage,ctx=ctx,size=size))
  for t in self.targets:
   if t['kind']=='P30':self.rows.append(f'P30 release_begin pid=9 stage={t["stage"]} pointer={t["pointer"]} bytes={t["size"]}')
   self.rows.append(f'<--- urUSMFree(.hContext = {t["ctx"]}, .pMem = {t["pointer"]}) -> UR_RESULT_SUCCESS;')
   if t['kind']=='P30':self.rows.append(f'P30 release_returned pid=9 stage={t["stage"]}')
  return '\n'.join(self.rows)
 def audit(self,text,on=True):return audit_text(text,{0:(0,32),1:(32,48)},on)
 def test_both_sourceowner_sets_budget_contexts_frees_positive(self):
  result=self.audit(self.fixture());self.assertTrue(result['passed'],result['errors']);self.assertEqual(len(result['owners']),8);self.assertEqual(result['aggregate_peak_bytes'],99297344);self.assertFalse(result['graph_retirement_runtime_handle_association_observed'])
 def test_off_requires_noP30_withSFDowners_positive(self):self.assertTrue(self.audit(self.fixture(False),False)['passed'])
 def test_missing_free_and_double_free_rejected(self):
  self.fixture();i=next(i for i,x in enumerate(self.rows) if '<--- urUSMFree' in x)
  self.assertFalse(self.audit('\n'.join(self.rows[:i]+self.rows[i+1:]))['passed']);self.assertFalse(self.audit('\n'.join(self.rows[:i+1]+[self.rows[i]]+self.rows[i+1:]))['passed'])
 def test_failed_contextwrong_free_rejected(self):
  text=self.fixture();i=next(i for i,x in enumerate(self.rows) if '<--- urUSMFree' in x)
  for value in [self.rows[i].replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN'),self.rows[i].replace('0x10','0x99')]:
   rows=self.rows[:];rows[i]=value;self.assertFalse(self.audit('\n'.join(rows))['passed'])
 def test_missing_owner_stage_or_returned_rejected(self):
  text=self.fixture()
  for prefix in ['P30 allocation','P30 release_begin','P30 release_returned']:
   rows=self.rows[:];i=next(i for i,x in enumerate(rows) if x.startswith(prefix));rows.pop(i);self.assertFalse(self.audit('\n'.join(rows))['passed'])
 def test_unexpected_on_in_off_rejected(self):self.assertFalse(self.audit(self.fixture(),False)['passed'])
 def test_missing_device_roster_rejected(self):self.assertFalse(self.audit(self.fixture().replace('.hDevice = 0x21','.hDevice = 0x20'))['passed'])
 def test_aggregatebudget_source_extent_cannot_be_forged(self):
  text=self.fixture().replace('bytes=31457296','bytes=131457296')
  self.assertFalse(self.audit(text)['passed'])
 def test_head_owner_census_rejected(self):
  self.assertFalse(self.audit(self.fixture().replace('logits1','logits0'))['passed'])
 def test_empty_cannot_qualify(self):self.assertFalse(self.audit('')['passed'])

class AdmissionTests(unittest.TestCase):
 def test_missing_actualnewSDK_stops_before_model_read(self):
  with patch.object(q.c1,'combined_generation_gate',side_effect=ValueError('SDK absent')),patch.object(q.c1,'validate_prepared',side_effect=AssertionError('No model reads')):
   with self.assertRaises(ValueError):q.candidate_binding({'engine_root':'/tmp/CPU_missing','prepared':'/tmp/CPU_missing'})
 def test_manifest_driver_rejected_before_candidate(self):
  with patch.object(q,'candidate_binding',side_effect=AssertionError('No candidate access')):
   with self.assertRaises(ValueError):q.manifest_binding({'driver_sha256':'wrong','source_plan_sha256':q.PLAN_SOURCE_SHA})
 def test_parent_retains_proven_health_cleanup_hash_helpers(self):
  a=ast.parse(Path(parent.__file__).read_text());b=ast.parse(Path(parent.__file__).with_name('qualify_layer0_numerical_v9.py').read_text());functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
  for name in ('stat_signature','full_buffered_identity','finalizable'):self.assertEqual(functions(a)[name],functions(b)[name])
 def test_newproof_before_payload_validator_and_prepare(self):
  source=Path(q.__file__).read_text();candidate=source[source.index('def candidate_binding'):source.index('def manifest_binding')];self.assertLess(candidate.index('validate_final_source_proof'),candidate.index('c1.validate_prepared'))
  prepare=source[source.index('def prepare'):source.index('def cross_sfd')];self.assertLess(prepare.index('candidate_binding'),prepare.index('subprocess.check_output'));self.assertLess(prepare.index('candidate_binding'),prepare.index('verify_model_identity'))
 def test_parent_holdlease_passfds_two_page_posthash_and_cleanup_hooks(self):
  source=Path(parent.__file__).read_text()
  for text in ['pass_fds=(8,9)','ctrl.c1.leased([0,1])','preserve_source_pages','full_buffered_identity','post-health','owned_containers_terminal','prefix30-logical-lifecycle.json']:self.assertIn(text,source)
if __name__=='__main__':unittest.main()
