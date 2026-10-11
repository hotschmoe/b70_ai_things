"""Tiny source/recipe/retirement controls; no profiler/device execution."""
import copy
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import probe_vtune_indexer40_v4 as p

class Controls(unittest.TestCase):
 def command(self):
  with patch.object(p.Path,'stat',return_value=type('S',(),{'st_gid':991})()):return p.recipe('on',Path('/tmp/new-results'),'tiny-owner','abc')
 def inspection(self,c):
  return {'Id':'a'*64,'Name':'/tiny-owner','Config':{'Image':p.IMAGE,'Labels':{'b70.vtune.indexer40':'abc'},'Entrypoint':['/bin/bash'],'Cmd':c[-2:],'User':c[c.index('--user')+1],'WorkingDir':'','Env':[c[i+1]for i,v in enumerate(c)if v=='-e']},'HostConfig':{'Binds':[c[i+1]for i,v in enumerate(c)if v=='-v'],'NetworkMode':'none','Privileged':False,'DeviceRequests':[],'GroupAdd':['991'],'Memory':4<<30,'MemorySwap':4<<30,'NanoCpus':2_000_000_000,'PidsLimit':512,'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}]},'State':{'Running':False,'ExitCode':0,'OOMKilled':False}}
 def test_exact_help_supported_collector_and_leaf(self):
  argv=p.tool_argv();self.assertEqual(argv[argv.index('-collect')+1],'xpu-offload');self.assertIn('collect-programming-api=true',argv);self.assertIn('target-gpu=0:11:0.0',argv);self.assertEqual(argv[argv.index('--')+1:],p.helper_argv());self.assertNotIn('gpu-offload',argv)
 def test_only_small_mounts_no_model_build_or_sdk(self):
  c=self.command();mounts=[c[i+1]for i,v in enumerate(c)if v=='-v'];self.assertEqual(len(mounts),4);self.assertEqual(sum(x.endswith(':rw')for x in mounts),1);self.assertFalse(any('/model:'in x or '/pack:'in x or '/src:'in x or '/build:'in x for x in mounts));self.assertIn('SYCL_UR_TRACE=0',c)
  self.assertEqual(Path('/harness/strata/flash-next/probe.py').parents[2],Path('/harness'));self.assertIn('/harness/strata/flash-next/probe.py --inside on',c[-1])
 def test_actual_inspection_mount_permutation_and_mutations(self):
  c=self.command();obj=self.inspection(c);obj['HostConfig']['Binds'].reverse();p.owned(obj,c,obj['Id'])
  for kind in('duplicate','name','image','cmd','env','device','privileged','memory'):
   bad=copy.deepcopy(obj)
   if kind=='duplicate':bad['HostConfig']['Binds'][0]=bad['HostConfig']['Binds'][1]
   elif kind=='name':bad['Name']='/foreign'
   elif kind=='image':bad['Config']['Image']='foreign'
   elif kind=='cmd':bad['Config']['Cmd']=['-c','different']
   elif kind=='env':bad['Config']['Env']=['ZE_AFFINITY_MASK=1']
   elif kind=='device':bad['HostConfig']['Devices']=[]
   elif kind=='privileged':bad['HostConfig']['Privileged']=True
   else:bad['HostConfig']['Memory']=2<<30
   self.assertRaises(ValueError,p.owned,bad,c,obj['Id'])
 def test_duplicate_or_nonfinite_json_refused(self):
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'input.json'
   for raw in(' {"a":1,"a":2}','{"a":NaN}'):
    f.write_text(raw);self.assertRaises(ValueError,p.read,f)
 def test_raw_alias_and_bound_refused(self):
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'raw';f.write_bytes(b'abc');a=Path(d)/'alias';a.symlink_to(f);self.assertRaises(ValueError,p.file_binding,a);self.assertRaises(ValueError,p.file_binding,f,2)
 def test_ownread_atime_change_allowed_identity_changes_refused(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'raw';f.write_bytes(b'unchanged');real=f.stat();fields={k:getattr(real,k)for k in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_atime_ns')};before=SimpleNamespace(**fields);after=SimpleNamespace(**{**fields,'st_atime_ns':fields['st_atime_ns']+1000000})
   with patch.object(p.Path,'stat',side_effect=[before,after]):self.assertEqual(p.file_binding(f)['sha256'],p.hashlib.sha256(b'unchanged').hexdigest())
   for key in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'):
    changed=SimpleNamespace(**{**fields,key:fields[key]+1})
    with patch.object(p.Path,'stat',side_effect=[before,changed]):self.assertRaises(ValueError,p.file_binding,f)
 def test_receipt_body_and_actual_eof_bytes(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);command=['tiny','command'];p.write(root/'step.command.json',command)
   for stream in('stdout','stderr'):(root/('step.'+stream)).write_bytes(b'EOF\n')
   row={'command':command,'return_code':0,'started_epoch':1.,'finished_epoch':2.,**{s:p.file_binding(root/('step.'+s))for s in('stdout','stderr')}};p.write(root/'step.receipt.json',row);p.receipt_binding(root,'step',command,row)
   self.assertRaises(ValueError,p.receipt_binding,root,'step',['foreign'],row)
   row['return_code']=False;p.write(root/'step.receipt.json',row);self.assertRaises(ValueError,p.receipt_binding,root,'step',command)
 def test_complete18_frame_direct_and_replays(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);fixture=root/'fixture';fixture.mkdir();arm=root/'arm';arm.mkdir();native=arm/'build/native-output';native.mkdir(parents=True);echo=native/'input_echo';echo.mkdir()
   for name in('raw.f32','gamma.f32','config.txt'):(fixture/name).write_bytes(b'original');(echo/name).write_bytes(b'original')
   for m in p.MODES:
    for r in range(3):
     f=native/(m+'-route'+str(r));f.mkdir()
     for name,size in p.FIELDS.items():(f/name).write_bytes(bytes(size))
   markers=['IDXMAT40_DEVICE backend=level_zero affinity=0 fp16=1 vendor=Intel driver=sample name=sample\n','IDXMAT40_FP_CONFIG flags=1,2, observed_device_flags_only=1 compiler_lowering_unobserved=1\n','IDXMAT40_CONFIG actual_resolved_checked=1 rows=4 dim=128 windows=2,1,1 source_functions_unchanged=1\n']+[f'IDXMAT40_FRAME mode={m} route={r} graph_replay={r} fields=8 own_zero_restored=1\n'for m in p.MODES for r in range(3)]+['IDXMAT40_RESULT frames=18 graph_retired=1 owned_allocations_freed=1 full_model_math_qualified=0 semantic_model_fix_qualified=0\n']
   (arm/'leaf.stdout').write_text(''.join(markers));(arm/'leaf.stderr').write_text('')
   for label in('before','after'):p.write(arm/('runtime-'+label+'.json'),{'helper_libraries':{'files':[],'ldd_stdout':'same','ldd_stderr':''},'tool_libraries':{'files':[],'ldd_stdout':'same','ldd_stderr':''}})
   with patch.object(p,'FIXTURE',fixture):
    self.assertEqual(len(p.output_binding(arm)),18)
    (arm/'leaf.stdout').write_text(''.join(markers).rstrip('\n'));self.assertRaises(ValueError,p.output_binding,arm);(arm/'leaf.stdout').write_text(''.join(markers))
    f=native/'single_raw-route2/input.f32';f.write_bytes(b'x'+bytes(2047));self.assertRaises(ValueError,p.output_binding,arm)
 def test_health_requires_actual_pair_sentinels(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);a=root/'strict';b=root/'compiled';a.write_text('card 0: OK\ncard 1: OK\nxpu-health: HEALTHY (cards 0 1)\n');b.write_text('COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=0\n');rows=[{'stderr':{'path':str(a)}},{'stdout':{'path':str(b)}}];p.health_text(rows);b.write_text('COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=1\n');self.assertRaises(ValueError,p.health_text,rows)
 def test_removal_success_unknown_absence_retries_only_absence(self):
  c=self.command();obj=self.inspection(c);inspect_calls=[];runs=[]
  def inspect(identity):inspect_calls.append(identity);return obj
  def run(command,out,label,timeout,cleanup=False):
   runs.append(label)
   if label=='create':(out/'create.stdout').write_text('a'*64);return {'stdout':{'path':str(out/'create.stdout')}}
   return {'return_code':0}
  replies=[type('R',(),{'returncode':1,'stdout':'','stderr':'unknown error'})(),type('R',(),{'returncode':1,'stdout':'[]','stderr':'error: no such object: '+'a'*64})()]
  with tempfile.TemporaryDirectory()as d,patch.object(p,'recipe',return_value=c),patch.object(p,'inspected',side_effect=inspect),patch.object(p,'run_command',side_effect=run),patch.object(p.subprocess,'run',side_effect=replies),patch.object(p.time,'sleep'):
   # The temporary unknown census remains a normal-qualification failure,
   # despite eventual exact removal, and does not re-inspect removed owner.
   self.assertRaises(ValueError,p.arm_run,'on',Path(d)/'arm','abc')
  self.assertEqual(runs.count('remove'),1);self.assertEqual(runs.count('terminal-logs'),1);self.assertEqual(len(inspect_calls),2)
 def test_ASLR_only_rawldd_change_still_preserved_and_real_changes_refused(self):
  base={'helper_libraries':{'files':[{'path':'/lib/real.so','sha256':'same','bytes':32}],'ldd_stdout':'real.so => /lib/real.so (0x0123)\n','ldd_stderr':''},'tool_libraries':{'files':[],'ldd_stdout':'linux-vdso.so (0x0123)\n','ldd_stderr':''}}
  after=copy.deepcopy(base)
  for row in after.values():row['ldd_stdout']=row['ldd_stdout'].replace('0x0123','0x4567')
  p.runtime_equal(base,after);self.assertNotEqual(base,after)
  after['helper_libraries']['files'][0]['sha256']='foreign';self.assertRaises(ValueError,p.runtime_equal,base,after)
 def test_actual_two_ldd_calls_have_canonical_file_identity(self):
  before=p.libraries('/usr/bin/true');after=p.libraries('/usr/bin/true');p.runtime_equal({'helper_libraries':before,'tool_libraries':before},{'helper_libraries':after,'tool_libraries':after})
 def test_typed_complete_terminal_and_resources(self):
  obj={'RestartCount':0,'State':{'Running':False,'OOMKilled':False,'Dead':False,'Paused':False,'Restarting':False,'ExitCode':0,'Pid':0,'Status':'exited','Error':''}};p.normal_terminal(obj)
  for key,value in(('ExitCode',False),('Pid',False),('Error','failed'),('Dead',True),('Status','created')):
   bad=copy.deepcopy(obj);bad['State'][key]=value;self.assertRaises(ValueError,p.normal_terminal,bad)
  bad=copy.deepcopy(obj);bad['RestartCount']=1;self.assertRaises(ValueError,p.normal_terminal,bad)
  c=self.command();bad=self.inspection(c);bad['HostConfig']['Memory']=float(4<<30);self.assertRaises(ValueError,p.owned,bad,c,bad['Id'])
 def test_cleanup_runs_after_interrupt_without_cancelling_owned_retirement(self):
  with tempfile.TemporaryDirectory()as d,patch.object(p,'STOP',True):
   root=Path(d);row=p.run_command(['/usr/bin/true'],root,'cleanup',timeout=2,cleanup=True);self.assertEqual(row['return_code'],0)
   self.assertRaises(TimeoutError,p.run_command,['python3','-c','import time;time.sleep(1)'],root,'ordinary',2)
 def test_actual_metadata_rows_outputs_knobs_and_chronology(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d)
   for i,(label,argv)in enumerate((('version',[p.TOOL,'-version']),('help',[p.TOOL,'-help','collect','xpu-offload']))):
    p.write(root/('vtune-'+label+'.command.json'),argv);(root/('vtune-'+label+'.stdout')).write_text('2026.2'if label=='version'else 'collect-programming-api target-gpu');(root/('vtune-'+label+'.stderr')).write_text('')
    row={'command':argv,'return_code':0,'started_epoch':1.+i,'finished_epoch':1.5+i,**{s:{**p.file_binding(root/('vtune-'+label+'.'+s)),'path':'/results/vtune-'+label+'.'+s}for s in('stdout','stderr')}};p.write(root/('vtune-'+label+'.receipt.json'),row)
   p.vtune_metadata_binding(root,0.,4.)
   self.assertRaises(ValueError,p.vtune_metadata_binding,root,0.,2.)
   f=root/'vtune-help.receipt.json';saved=p.read(f);bad=copy.deepcopy(saved);bad['return_code']=False;p.write(f,bad);self.assertRaises(ValueError,p.vtune_metadata_binding,root,0.,4.);p.write(f,saved)
   (root/'vtune-help.stdout').write_text('unsupported');saved['stdout']={**p.file_binding(root/'vtune-help.stdout'),'path':'/results/vtune-help.stdout'};p.write(f,saved);self.assertRaises(ValueError,p.vtune_metadata_binding,root,0.,4.)
 def test_exact_observed_auxiliary_scope_and_required_negatives(self):
  row={'path':p.AUX_SHIM,'sha256':p.AUX_SHA,'needed':[],'filter':[],'auxiliary':p.AUX_NAMES};dynamic=[row,{'path':'/real.so','sha256':'bound','needed':['libc.so.6'],'filter':[],'auxiliary':[]}];files=[{'path':p.AUX_SHIM,'sha256':p.AUX_SHA}];result=p.auxiliary_admission(p.MISSING_AUX,dynamic,files);self.assertFalse(result['hard_required_dependency_gate_waived'])
  for mode in('name','hash','needed','filter','foreign_aux'):
   bad=copy.deepcopy(dynamic);missing=list(p.MISSING_AUX)
   if mode=='name':missing[-1]='libbroken.so'
   elif mode=='hash':bad[0]['sha256']='foreign'
   elif mode=='needed':bad[1]['needed']=[missing[0]]
   elif mode=='filter':bad[1]['filter']=[missing[0]]
   else:bad[1]['auxiliary']=[missing[0]]
   self.assertRaises(ValueError,p.auxiliary_admission,missing,bad,files)
 def test_actual_root_exact_environment_metadata_AUX_roster(self):
  record=p.read(p.F17/'vtune-exact-env-no-device-dependency-v1/stdout.json');rows=[{'path':r['path'],'sha256':r['sha256'],'needed':r['DT_NEEDED'],'auxiliary':r['DT_AUXILIARY'],'filter':r['DT_FILTER']}for r in record['resolved_dependencies']];missing=[re.match(r'(\S+)\s+=>',s)[1]for s in record['missing_names']];p.auxiliary_admission(missing,rows,rows);self.assertEqual(record['ordinary_version']['return_code'],0);self.assertEqual(record['explicit_loader_diagnostic']['return_code'],0)
 def test_public_consumer_reparses_original_dynamic_raw_receipts(self):
  actual=p.read(p.F17/'vtune-exact-env-no-device-dependency-v1/stdout.json');files=[];dynamic=[]
  for row in actual['resolved_dependencies']:
   if row['path']==p.TOOL:continue
   files.append({k:row[k]for k in('path','sha256','bytes')});d=row['ELF_dynamic'];dynamic.append({**{k:row[k]for k in('path','sha256')},**d,'needed':row['DT_NEEDED'],'auxiliary':row['DT_AUXILIARY'],'filter':row['DT_FILTER']})
  missing=[re.match(r'(\S+)\s+=>',s)[1]for s in actual['missing_names']];record={'ldd_command':['/usr/bin/ldd',p.TOOL],'ldd_return_code':0,'ldd_stdout':actual['ldd']['stdout'],'files':files,'actual_dynamic_metadata':dynamic,'explicit_auxiliary_scope':p.auxiliary_admission(missing,dynamic,files)};p.dependency_record_binding(p.TOOL,record)
  bad=copy.deepcopy(record);bad['actual_dynamic_metadata'][0]['needed']=['invented'];self.assertRaises(ValueError,p.dependency_record_binding,p.TOOL,bad)
  bad=copy.deepcopy(record);bad['ldd_return_code']=False;self.assertRaises(ValueError,p.dependency_record_binding,p.TOOL,bad)
if __name__=='__main__':unittest.main()
