"""Actual fresh43 dispatch/census/flags with only tiny source-level fixtures."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_source43_pti_latency_v1 as a
import source43_pti_runtime_v1 as tool
import validate_source43_pti_latency_v1 as r
from test_source43_normal_model_cpu_v1 import Controls as Shapes
class Controls(unittest.TestCase):
 def fixture(self):
  m,cfg,inputs=Shapes().fixture();m['runtime']['image']='CPU_SYNTHETIC_IMAGE';return m,cfg,inputs
 def test_only_tool_ON_and_both_normal_host_traces(self):
  m,cfg,inputs=self.fixture();runtime={'image':m['runtime']['image'],'install_root':'/CPU_SYNTHETIC/install'}
  with patch.object(a,'_RUNTIME',runtime),patch.object(tool,'inherited_env'):
   left,x=a.phase_recipe(a.PHASES[0],m,cfg,inputs,'a'*64);right,y=a.phase_recipe(a.PHASES[1],m,cfg,inputs,'a'*64)
  self.assertEqual(left,right);self.assertEqual(x['requests'],y['requests']);self.assertEqual(y['native_argv'][-len(x['native_argv']):],x['native_argv']);self.assertEqual(x['engine_generation'],43);self.assertEqual(x['native_env']['STRATA_CRITICAL_PATH_TRACE'],'1');self.assertEqual(x['native_env']['STRATA_DECODE_TIMING'],'1');self.assertNotIn('UNITRACE_MODEL_FINALIZE_V1',x['native_env']);self.assertEqual(y['native_env']['UNITRACE_MODEL_FINALIZE_V1'],'1');self.assertEqual(y['native_env']['STRATA_DIAGNOSTIC_PTI_FINALIZE'],'1');self.assertEqual(x['materialized_mode'],0);self.assertFalse(x['observer_enabled'])
 def test_omission_request_actually_rejects_LP(self):
  ns=a.omission.request_ports();self.assertNotIn('logprobs',ns['request_command']([7],64,1,None));self.assertTrue(issubclass(a.Protocol,a.protected.Protocol))
 def test_current_device_properties_strict_types_and_descriptor(self):
  p={'descriptor_id':0,'timer_frequency_hz':19200000,'kernel_valid_bits':64,'global_valid_bits':64,'timer_mask':(1<<64)-1,'ns_per_cycle_ieee754_u64':4632526894828006059,'uuid_word0_native_u64':3793977478,'uuid_word1_native_u64':11};self.assertEqual(tool.properties(p),p)
  for name in('descriptor_id','timer_frequency_hz','kernel_valid_bits'):
   q=dict(p);q[name]=False
   with self.assertRaises(ValueError):tool.properties(q)
 def test_pending_runtime_fails_before_any_source_ELF_read(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'intake.json';p.write_text('{"runtime_ready":false}')
   with patch.object(tool,'regular',side_effect=AssertionError('payload read')):
    with self.assertRaises(ValueError):tool.binding(p)
 def test_actual_census_dispatches_exact_both_live_boundaries(self):
  n=a.ports();code=n['native_phase'].__code__;self.assertIn('after_READY',code.co_consts);self.assertIn('after_last_DONE_before_QUIT',code.co_consts);self.assertIn('ready-observed.json',code.co_consts);self.assertIn('last-DONE-observed.json',code.co_consts);self.assertIs(n['Protocol'],a.Protocol);self.assertIs(n['intake'],a.intake);self.assertIs(n['life'],a.life);self.assertEqual(a.PHASES,('source43-pti-OFF','source43-pti-ON'));self.assertIs(r._PORTS['a'],a)
 def test_finalizer_marker_source_is_not_oldSESSION_waiver(self):
  source=Path(r.__file__).read_text();self.assertIn("markers==[('1','0')]",source);self.assertIn('pti.collect(directory,actor._RUNTIME)',source);self.assertIn('collector.collect(files',source)
if __name__=='__main__':unittest.main()

class RuntimeControls(unittest.TestCase):
 def test_producer_shaped_finalizer_exported_host_paths_and_actual_property_schema(self):
  import hashlib
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);build=root/'build';source=root/'source';prop=root/'properties';dep=root/'deps';prop.mkdir();dep.mkdir();(build/'install/bin').mkdir(parents=True);(build/'install/lib').mkdir();source.mkdir()
   (build/'install/bin/unitrace').write_bytes(b'\x7fELFtiny-tool');(build/'install/lib/libunitrace_tool.so').write_bytes(b'\x7fELFtiny-injection');ts=sha(build/'install/bin/unitrace');ls=sha(build/'install/lib/libunitrace_tool.so')
   (source/'clock.h').write_text('CPU_SYNTHETIC_HEADER');proposal=root/'proposal.json';proposal.write_text(json.dumps({'expected_patched_source_sha256':{'clock.h':sha(source/'clock.h')}}));ps=sha(proposal)
   br=build/'receipt.json';br.write_text(json.dumps({'passed':True,'build_rc':0,'error':None,'GPU_devices_exposed':False,'normal_owned_removed':True,'clock_source_root':str(source),'clock_source_plan_sha256':ps,'all_patched_source_bytes_checked_before_after':True,'image':'CPU_IMAGE'}));bs=sha(br)
   files=[]
   for i in range(9):
    p=dep/('dso'+str(i));p.write_bytes(b'\x7fELFCPUtiny');files.append({'original_container_path':'/container/lib'+str(i),'exported_host_path':str(p),'sha256':sha(p)})
   dr=dep/'receipt.json';dr.write_text(json.dumps({'passed':True,'return_code':0,'error':None,'GPU_devices_exposed':False,'normal_owned_removed':True,'image':'CPU_IMAGE','unitrace_sha256':ts,'injection_sha256':ls,'files':files}))
   clocks={'descriptor_id':0,'timer_frequency_hz':19200000,'kernel_valid_bits':64,'global_valid_bits':64,'timer_mask':(1<<64)-1,'ns_per_cycle_ieee754_u64':4632526894828006059,'uuid_word0_native_u64':3793977478,'uuid_word1_native_u64':11};(prop/'unitrace').mkdir();raw=prop/'unitrace/clock.1.jsonl';raw.write_text(json.dumps({'kind':'device_origin',**clocks})+'\n');script=prop/'b70_root_pti_finalizer_gpu0_clock_property_v1.py';script.write_text('CPU_SYNTHETIC_QUERY_SOURCE')
   qr=prop/'receipt.json';qr.write_text(json.dumps({'passed':True,'return_code':0,'clock_build_receipt_sha256':bs,'clock_source_plan_sha256':ps,'normal_owned_removed':True,'helper_or_model_executed':False,'ZE_AFFINITY_MASK':'0','pre_health_rc':0,'post_health_rc':0,'raw_files':{'clock.1.jsonl':{'sha256':sha(raw),'bytes':raw.stat().st_size}}}))
   cp=prop/'device-clock-properties.json';cp.write_text(json.dumps({'passed':True,'ZE_AFFINITY_MASK':'0','actual_API_query_only':True,'model_executed':False,'operation_counts_transferred':False,'clock_calibration_qualified':False,'receipt_sha256':sha(qr),'source_script_sha256':sha(script),'raw_clock_sha256':sha(raw),'clock_properties':clocks}))
   intake=root/'intake.json';intake.write_text(json.dumps({'schema':1,'runtime_ready':True,'source_root':str(source),'install_root':str(build/'install'),'metadata_cap_bytes':512<<20,'build_receipt_path':str(br),'build_receipt_sha256':bs,'dependency_receipt_path':str(dr),'dependency_receipt_sha256':sha(dr),'actual_device_clock_metadata_path':str(cp),'actual_device_clock_metadata_sha256':sha(cp),'actual_device_clock_properties':clocks}))
   values={'SOURCE':source,'BUILD':build,'BUILD_SHA':bs,'PROPOSAL':proposal,'PROPOSAL_SHA':ps,'TOOL_SHA':ts,'LIB_SHA':ls,'DEPS':dr,'PROPERTY':cp,'PROPERTY_SHA':sha(cp),'QUERY_SHA':sha(qr)}
   with __import__('contextlib').ExitStack()as stack:
    for key,value in values.items():stack.enter_context(patch.object(tool,key,value))
    result=tool.binding(intake);self.assertEqual(len(result['exported_dependencies']),9);self.assertEqual(result['clock_properties'],clocks)
    body=json.loads(dr.read_text());del body['files'][0]['exported_host_path'];dr.write_text(json.dumps(body));value=json.loads(intake.read_text());value['dependency_receipt_sha256']=sha(dr);intake.write_text(json.dumps(value))
    with self.assertRaises((ValueError,KeyError)):tool.binding(intake)

class CensusControls(unittest.TestCase):
 def fixture(self,root,change):
  import hashlib
  from pti_mapped_injection_census_v1 import select
  cmd=['CPU_SYNTHETIC_NATIVE'];(root/'command.json').write_text(json.dumps(cmd));image='CPU_IMAGE';binding='a'*64;terminal={'Id':'CPU_CONTAINER','Name':'/CPU_NATIVE'};maps='1000-2000 r-xp 00000000 00:01 7 /pti/lib/libunitrace_tool.so\n';payload=b'\x7fELFCPU_SYNTHETIC';digest=hashlib.sha256(payload).hexdigest()
  for i,phase in enumerate(('after_READY','after_last_DONE_before_QUIT')):
   d=root/phase;d.mkdir();(d/'mapped-injection.elf').write_bytes(payload);(d/'proc1.maps').write_text(maps);receipt={'passed':True,'phase':phase,'process':{'pid':1,'start_ticks':99},'original_container_path':'/pti/lib/libunitrace_tool.so','mapped_segments':select(maps),'backing_file_stat':[1,7,len(payload),2,3],'sha256':digest,'bytes':len(payload),'process_memory_bytes_read':False,'LevelZero_handle_proof':False,'versioned_symbol_origin_queried':False};(d/'receipt.json').write_text(json.dumps(receipt))
   row={'passed':True,'return_code':0,'container_id':terminal['Id'],'host_native_pid':77,'command':a.census.command(terminal['Id'],phase,digest),'metadata_stdout_bytes':0,'census':receipt,'exported_host_path':str(d/'mapped-injection.elf'),'exported_host_sha256':digest,'started_epoch':2.+i*3,'finished_epoch':2.5+i*3,'started_monotonic_ns':20+i*30,'finished_monotonic_ns':25+i*30}
   if change=='foreign_command'and i:row['command']=['FOREIGN']
   if change=='foreign_owner'and i:row['host_native_pid']=999
   (root/(phase+'-owned.json')).write_text(json.dumps(row));(root/(phase+'-exec.stdout')).write_bytes(b'')
   for boundary in('before','after'):(root/(phase+'-owner-'+boundary+'.json')).write_text(json.dumps({'Id':terminal['Id'],'State':{'Running':True,'Pid':77}}))
  (root/'ready-observed.json').write_text(json.dumps({'monotonic_ns':10,'epoch':1.5}));(root/'last-DONE-observed.json').write_text(json.dumps({'monotonic_ns':45,'epoch':4.5}));(root/'result.json').write_text(json.dumps({'stdout_capture':{'process_started_epoch':1.,'completed_epoch':7.}}));sent=[{'packet':'GEN64\n','monotonic_ns':30+i*3}for i in range(4)]+[{'packet':'QUIT\n','monotonic_ns':70}];(root/'native.stdin.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in sent));return digest,terminal,{'tool_runtime':{'image':image},'native_env':{'STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256':binding}}
 def run_fixture(self,change):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);digest,terminal,recipe=self.fixture(root,change)
   with patch.object(a.tool,'LIB_SHA',digest),patch.object(a.life.contract,'inspection'):
    if change:
     with self.assertRaises(ValueError):r.census_binding(root,recipe,terminal,[])
    else:self.assertTrue(r.census_binding(root,recipe,terminal,[])['actual_mapped_injection_before_after_joined'])
 def test_actual_saved_census_full_behavior(self):self.run_fixture(None)
 def test_foreign_saved_exec_command_rejected(self):self.run_fixture('foreign_command')
 def test_foreign_live_owner_rejected(self):self.run_fixture('foreign_owner')
