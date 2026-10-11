"""Meaningful tiny transport/storage/lifecycle controls; no model/device work."""
import json,struct,tempfile,unittest,subprocess,sys,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import indexer_materialized40_control_v2 as c
import qualify_indexer_materialized40_v2 as q
class Controls(unittest.TestCase):
 def test_integer_RNE_boundary_bits(self):
  values=[0.,-0.,1.,1+2**-11,1+3*2**-11,2**-24,2**-25,3*2**-25,2**-14,65504.,65520.]
  raw=struct.pack('<%df'%len(values),*values);bits,w=c.ieee_rne(raw)
  self.assertEqual(list(struct.unpack('<%dH'%len(values),bits)),[0,0x8000,0x3c00,0x3c00,0x3c02,1,0,2,0x400,0x7bff,0x7c00])
  with np.errstate(over='ignore'):self.assertEqual(w,np.asarray(values,dtype='<f4').astype('<f2').astype('<f4').tobytes())
  self.assertRaises(ValueError,c.ieee_rne,struct.pack('<f',float('nan')))
 def synthetic(self,base):
  fixture=base/'fixture';fixture.mkdir();raw=np.tile(np.asarray([1+2**-13,1+2**-11,-.12345,.23456],dtype='<f4'),128).tobytes();gamma=np.ones(128,dtype='<f4').tobytes();(fixture/'raw.f32').write_bytes(raw);(fixture/'gamma.f32').write_bytes(gamma);prior={'CPU_SYNTHETIC':True};record={'root':str(fixture),'record':{'own_producer_binding':prior}};build=base/'build';build.mkdir();out=build/'native-output';out.mkdir();echo=out/'input_echo';echo.mkdir()
  for n,b in [('config.txt',c.CONFIG),('raw.f32',raw),('gamma.f32',gamma)]:(echo/n).write_bytes(b)
  lines=['IDXMAT40_DEVICE backend=level_zero affinity=0 fp16=1 vendor=CPU driver=fixture name=synthetic\n','IDXMAT40_FP_CONFIG flags=1, observed_device_flags_only=1 compiler_lowering_unobserved=1\n','IDXMAT40_CONFIG actual_resolved_checked=1 rows=4 dim=128 windows=2,1,1 source_functions_unchanged=1\n'];half,widen=c.ieee_rne(raw)
  for mode in c.MODES:
   for route in range(3):
    p=out/(mode+'-route'+str(route));p.mkdir();tails=c.expected_tail(widen if'materialized'in mode else raw);states=[t+bytes(1540)for t in tails];data={'input.f32':raw,'gamma.f32':gamma,'materialized.f16':half,'expanded.f32':widen,'steps.i32':c.STEP_BYTES,'wrong_raw_store.f32':raw[:1536],'before_states.bin':bytes(3076)+states[0]+states[1],'after_states.bin':b''.join(states)}
    for n,b in data.items():(p/n).write_bytes(b)
    lines.append(f'IDXMAT40_FRAME mode={mode} route={route} graph_replay={route} fields=8 own_zero_restored=1\n')
  lines.append('IDXMAT40_RESULT frames=18 graph_retired=1 owned_allocations_freed=1 full_model_math_qualified=0 semantic_model_fix_qualified=0\n');(base/'runtime.log').write_text(''.join(lines));return out,record,prior
 def test_full_behavioral_source_tail_and_route_join(self):
  with tempfile.TemporaryDirectory()as d:
   out,f,p=self.synthetic(Path(d));r=c.compare_routes(out,f,p);self.assertTrue(r['materialized_tail_bitwise_contract']);self.assertTrue(r['complete_materialized_all3_API_state_bitwise']);self.assertTrue(r['wrong_raw_store_control_discriminated']);self.assertFalse(r['semantic_model_fix_qualified'])
 def test_rawstore_negative_fails_materialized_tail_gate(self):
  with tempfile.TemporaryDirectory()as d:
   out,f,p=self.synthetic(Path(d));dest=out/'steps_materialized-route0/after_states.bin';dest.write_bytes((out/'steps_raw-route0/after_states.bin').read_bytes());self.assertRaises(ValueError,c.compare_routes,out,f,p)
 def test_extra_truncated_alias_and_half_load_mutations(self):
  for kind in ('extra','newline','symlink','widen'):
   with tempfile.TemporaryDirectory()as d:
    b=Path(d);out,f,p=self.synthetic(b)
    if kind=='extra':(out/'foreign').write_bytes(b'bad')
    elif kind=='newline':log=b/'runtime.log';log.write_bytes(log.read_bytes()[:-1])
    elif kind=='symlink':file=out/'batch_raw-route0/input.f32';raw=file.read_bytes();file.unlink();foreign=b/'other';foreign.write_bytes(raw);file.symlink_to(foreign)
    else:file=out/'steps_raw-route0/expanded.f32';file.write_bytes(bytes(2048))
    self.assertRaises(ValueError,c.compare_routes,out,f,p)
 def test_constructor_original_source_and_current40_routing(self):
  ns=q.derived();self.assertEqual(ns['PLAN'],q.PLAN);self.assertEqual(ns['__file__'],q.__file__);self.assertEqual(ns['modules']()[0].SDK,q.SDK);self.assertRaises(ValueError,ns['modules']()[-1].finalized_binding,Path('/CPU_SYNTHETIC'),Path('/old-adjudication'))
  import qualify_owned_indexer_half_control_v2 as old
  self.assertNotEqual(old.PLAN,q.PLAN)
 def test_source_only_own_flags_actual40_and_argv(self):
  fields=q.builder();self.assertIn('-fp-model=precise',fields['FLAGS']);self.assertEqual(q.leaf_argv()[-1],'/out/indexer-materialized40');self.assertIn('/leaf/indexer_materialized40_gpu_v1.cpp',q.leaf_argv())
 def test_original_resolved_parameters_not_CONFIG_authority(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'reference-work').mkdir();metadata={'effective_epsilon_F32_hex':struct.pack('<f',1e-6).hex(),'effective_RopeScaling':{'type':'none','freq_base':1e7,'factor':1.,'freq_scale_in':0.,'orig_ctx':262144.,'ext_factor':0.,'attn_factor':1.,'beta_fast':32.,'beta_slow':1.}}
   path=root/'reference-work/work-report.json';path.write_text(json.dumps({'own_layer3_projection':{'metadata_binding':metadata}}));self.assertEqual(c.own_params(root)['effective_epsilon_F32_hex'],metadata['effective_epsilon_F32_hex'])
   metadata['effective_RopeScaling']['factor']=2.;path.write_text(json.dumps({'own_layer3_projection':{'metadata_binding':metadata}}));self.assertRaises(ValueError,c.own_params,root)
 def test_operation_owned_once_current_fullbytes_every_reuse(self):
  from types import SimpleNamespace
  class Epoch:
   def __init__(self,*a):self.calls=[]
   def ready(self):self.calls.append('ready')
   def fresh(self,label):self.calls.append(label);raise ValueError('changed complete bytes')
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);own=root/'own';own.mkdir();(own/'report.json').write_text('{}');prepared=root/'prepared';prepared.mkdir();(prepared/'prepared.json').write_text(json.dumps({'engine_receipt':str(root/'sdk/receipt.json'),'pack':str(root/'pack')}));admission=root/'admission.json';admission.write_text(json.dumps({'fresh_native_runtime_authorization':{'baseline_root':str(prepared)}}));plan=root/'plan.json';plan.write_text('{"files":{}}')
   with patch.object(c,'OWN',own),patch.object(c,'ADMISSION',admission),patch.object(c,'_operation',None),patch.object(q,'PLAN',plan),patch('native_qsa40_byte_epoch_v9.ByteEpoch',Epoch),patch.object(c,'_full_producer_binding',return_value={'CPU_SYNTHETIC':True})as full:
    self.assertEqual(c.producer_binding(own),{'CPU_SYNTHETIC':True});self.assertRaises(ValueError,c.producer_binding,own);full.assert_called_once_with(own)
 def test_stdlib_only_inside_runtime_copy_bootstrap(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);copy=root/'qualifier.py';copy.write_bytes(Path(q.__file__).read_bytes());helper=root/'helper';helper.write_bytes(b'ELF_CPU_SYNTHETIC');lib=root/'libfixture.so';lib.write_bytes(b'LIB_CPU_SYNTHETIC');bin=root/'bin';bin.mkdir();tool=bin/'ldd';tool.write_text('#!/bin/sh\necho "libfixture.so => '+str(lib)+' (0x1)"\n');tool.chmod(0o755)
   # The source bootstrap deliberately uses absolute ldd. Stub only that call
   # in a self-contained CPU wrapper, preserving import-isolation semantics.
   runner=root/'runner.py';runner.write_text('import runpy,sys,subprocess\nsubprocess.check_output=lambda *a,**k:'+repr('libfixture.so => '+str(lib)+' (0x1)\n')+'\nsys.argv='+repr([str(copy),'--inside-runtime',str(helper),'--binding-output',str(root/'binding.json')])+'\nrunpy.run_path('+repr(str(copy))+',run_name="__main__")\n')
   r=subprocess.run([sys.executable,'-I',str(runner)],capture_output=True,text=True,timeout=10);self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(json.loads((root/'binding.json').read_text())['actual_model_payload_read'])
if __name__=='__main__':unittest.main()

class WitnessControls(unittest.TestCase):
 def fixture(self,root):
  from types import SimpleNamespace
  view={'CPU_SYNTHETIC_BYTES':'sha'}
  labels=['entry','ready','current-use-2','current-use-3:predevice','current-use-4:postrun','current-use-5','current-use-6:postproof']
  times=[(1,2),(3,4),(5,6),(20,21),(40,41),(70,71),(80,81)]
  rows=[{'label':l,'owner_pid':123,'started_epoch':a,'finished_epoch':b,'view':view}for l,(a,b)in zip(labels,times)]
  witness={'schema':1,'owner_pid':123,'one_full_semantic_admission_per_owned_process':True,'whole_current_input_bytes_on_each_use':True,'generic_persistent_cached_PASS_used':False,'full_model_math_qualified':False,'calls':6,'roots':['/CPU_SYNTHETIC'],'files':['/CPU_SYNTHETIC/source'],'boundaries':rows}
  report={'producer_pid':123,'started_epoch':7,'finished_epoch':90,'compile_command':{'finished_epoch':19},'run_command':{'started_epoch':22,'finished_epoch':39},'GPU_terminal_epoch':42,'post_full4':{'finished':75},'post_pages':{'epoch':76},'artifact_sha256':{}}
  op={'owner_pid':os.getpid(),'epoch':SimpleNamespace(roots=[Path('/CPU_SYNTHETIC')],files=[Path('/CPU_SYNTHETIC/source')],initial=view)}
  return witness,report,op
 def admit(self,root,w,r,op):
  path=root/'producer-operation-witness.json';path.write_text(json.dumps(w,allow_nan=False)+'\n');digest=c.sha(path);r['producer_operation_witness_sha256']=digest;r['artifact_sha256']['producer-operation-witness.json']=digest
  with patch.object(c,'_operation',op):return c.admit_operation_witness(root,r)
 def test_complete_actual_receipt_role_join(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);w,r,op=self.fixture(root);self.assertTrue(self.admit(root,w,r,op)['original_predevice_postrun_postproof_recollected'])
 def test_entry_ready_only_and_omitted_boundary_refused(self):
  for mode in('entryready','omit'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);w,r,op=self.fixture(root)
    if mode=='entryready':w['boundaries']=w['boundaries'][:2];w['calls']=1
    else:w['boundaries'].pop(3);w['calls']-=1
    self.assertRaises(ValueError,self.admit,root,w,r,op)
 def test_typed_count_owner_order_and_duplicate_role_refused(self):
  for mode in('count','boolcount','boolowner','label','duplicate','foreignview','artifact'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);w,r,op=self.fixture(root)
    if mode=='count':w['calls']+=1
    elif mode=='boolcount':w['calls']=True
    elif mode=='boolowner':w['boundaries'][2]['owner_pid']=True;r['producer_pid']=1;w['owner_pid']=1
    elif mode=='label':w['boundaries'][2]['label']='current-use-100'
    elif mode=='duplicate':w['boundaries'][5]['label']='current-use-5:predevice'
    elif mode=='foreignview':w['boundaries'][3]['view']={'CPU_SYNTHETIC_BYTES':'changed'}
    else:r['producer_pid']=True
    self.assertRaises(ValueError,self.admit,root,w,r,op)
 def test_receipt_predevice_postrun_final_chronology_refused(self):
  for mode in('compile','runstart','runend','gputerminal','pages','reportfinish','ready'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);w,r,op=self.fixture(root)
    if mode=='compile':r['compile_command']['finished_epoch']=22
    elif mode=='runstart':r['run_command']['started_epoch']=20
    elif mode=='runend':r['run_command']['finished_epoch']=41
    elif mode=='gputerminal':r['GPU_terminal_epoch']=40
    elif mode=='pages':r['post_pages']['epoch']=82
    elif mode=='reportfinish':r['finished_epoch']=80
    else:r['started_epoch']=3
    self.assertRaises(ValueError,self.admit,root,w,r,op)
 def test_derived_producer_actual_role_calls_in_order(self):
  import dis
  main=q.derived()['main'];consts=main.__code__.co_consts
  for role in('predevice','postrun','postproof'):self.assertIn(role,consts)
  roles=[x.argval for x in dis.get_instructions(main)if x.opname=='LOAD_CONST'and x.argval in('predevice','postrun','postproof')];self.assertEqual(roles[:3],['predevice','postrun','postproof']);self.assertTrue(all(role=='postproof'for role in roles[3:]))
