"""Read-only fixture admission. No model, latency or alignment authority."""
import json
from pathlib import Path
import probe_pti_paused_regular_fixture_v1 as producer
import pti_paused_fixture_runtime_v1 as runtime
from serial37_canonical_json_v3 import canonical,read_unique
require=producer.require

def roster(root):
 result={};total=0
 for p in sorted(root.rglob('*')):
  require(not p.is_symlink(),'Artifact aliases refused')
  if p.is_file():
   raw,bound=runtime.bound(p,32<<20);total+=len(raw);require(total<=64<<20 and len(result)<1000,'Bounded fixture artifact roster');result[str(p.relative_to(root))]=bound
 return result

def finalized_binding(root,expected_report_sha256):
 root=Path(root);require(root.is_absolute()and root.resolve()==root,'Canonical closed fixture required');before=roster(root);source=producer.source_binding();require(producer.sha(root/'report.json')==expected_report_sha256,'Original fixture report changed');report=read_unique(root/'report.json');require(report['passed']is True and report['errors']==[]and report['source_plan_sha256']==source,'Actual success and current source required')
 require(producer.artifact_roster(root)==report['artifacts'],'Original producer-pinned full artifact roster changed')
 recipe=producer.recipe_for(Path(report['fixture_path']),report['fixture_sha256'],Path(report['properties_path']));u=producer.utility(recipe);property_record=read_unique(Path(report['properties_path']));properties=property_record['clock_properties'];require(report['properties_sha256']==producer.PROPERTY_SHA,'Exact original API-property bytes required');require(producer.sha(Path(report['properties_path']))==report['properties_sha256']and producer.sha(Path(report['fixture_path']))==report['fixture_sha256'],'Actual fixture/properties changed')
 for arm in('compile','off','on'):
  out=root/('build'if arm=='compile'else arm);command=read_unique(out/'create.command.json');active_recipe=producer.compile_recipe(Path(report['properties_path']))if arm=='compile'else recipe;active_u=producer.utility(active_recipe,compile_only=arm=='compile');expected=active_recipe(arm,out,command[command.index('--name')+1],source);require(canonical(command)==canonical(expected),'Original complete create recipe changed')
  created=read_unique(out/'created-inspection.json');terminal=read_unique(out/'terminal-inspection.json');active_u['owned'](created,command,created['Id']);active_u['owned'](terminal,command,created['Id']);producer.prior.normal_terminal(terminal);require(canonical(report['compile_terminal']if arm=='compile'else report['arms'][arm])==canonical(terminal['State']),'Original terminal/report mismatch')
  for label,argv in(('create',command),('start',['docker','start','-a',created['Id']]),('terminal-logs',['docker','logs',created['Id']]),('remove',['docker','rm',created['Id']])):producer.prior.receipt_binding(out,label,argv)
  absent=read_unique(out/'absence.json');require(type(absent['return_code'])is int and absent['return_code']==1 and ('no such object: '+created['Id'])in absent['stderr'].lower(),'Exact original ID absence required')
  if arm=='compile':
   require(report['compile_source_sha256']==producer.sha(producer.HERE/'pti_paused_regular_list_fixture_v1.cpp'),'Actual fixture source identity changed');continue
  inside=read_unique(out/'inside-receipt.json');require(inside['passed']is True and inside['return_code']==0 and inside['retention_enabled']is(arm=='on')and inside['graph_and_allocation_retired']is True,'Actual inside retirement missing')
  for bound in inside['raw_bindings']:
   declared=Path(bound['path']);require(declared.is_absolute()and declared.parts[:2]==('/', 'results')and '..'not in declared.parts,'Exact container raw prefix required');mapped=out.joinpath(*declared.parts[2:]);fresh=runtime.bound(mapped,32<<20)[1];fresh['path']=str(declared);require(fresh==bound,'Original consumed inside raw bytes changed')
  for name in('/pti/bin/unitrace','/pti/lib/libunitrace_tool.so','/leaf/fixture'):
   left=inside['libraries_before'][name];right=inside['libraries_after'][name];require(runtime.library_projection(left)==runtime.library_projection(right)and left['command']==['/usr/bin/ldd',name]and type(left['return_code'])is int and left['return_code']==0 and 'not found'not in left['stdout']+left['stderr']and bool(left['resolved_files']),'Actual complete dependency observations changed')
  child=inside['actual_child'];owner=inside['controller_owner'];require(type(child['pid'])is int and child['pid']>1 and type(child['start_ticks'])is int and child['start_ticks']>0 and child['parent_pid']==owner['pid']==1 and type(owner['start_ticks'])is int and owner['start_ticks']>0,'Original actual child/controller ownership required');require(inside['actual_child_exe']=='/leaf/fixture'and inside['actual_child_exe_sha256']==report['fixture_sha256'],'Actual child executable changed');require(inside['session']==runtime.control.session_name(owner['pid'],owner['start_ticks'],report['fixture_sha256']),'Actual named session source owner changed');runtime.control.check_transition(inside['before'],inside['after'],'resume');require(inside['before']['path']=='/dev/shm/uctrl'+inside['session']and inside['resume_argv']==['/pti/bin/unitrace','--resume',inside['session']]and type(inside['resume_return_code'])is int and inside['resume_return_code']==0,'Exact actual temporal control receipt changed')
  stdout,_=runtime.bound(out/'leaf.stdout',8<<20);require([line for line in stdout.splitlines()if line.startswith(b'PAUSED_LIST_')]==[b'PAUSED_LIST_WARM_DONE values=1,2',b'PAUSED_LIST_SELECTED_DONE values=2,4',b'PAUSED_LIST_RETIRED graph=1 allocation=1'],'Original graph/allocation markers changed')
  expected_inside=['/pti/bin/unitrace','--start-paused','--session',inside['session'],'--chrome-call-logging','--chrome-kernel-logging','--chrome-device-logging','-d','-s','-t','--pid','--tid','--verbose','--output-dir-path','/results/unitrace','/leaf/fixture'];require(inside['command']==expected_inside,'Actual original tool/leaf argv changed')
  clock=list((out/'unitrace').glob('clock.*.jsonl'));chrome=list((out/'unitrace').glob('*.json'));require(len(clock)==len(chrome)==1,'Exact clock/Chrome file roster required');require(clock[0].name=='clock.'+str(child['pid'])+'.jsonl','Actual writer child filename changed');raw,_=runtime.bound(clock[0],8<<20);data,_=runtime.bound(chrome[0],32<<20);rows=[runtime.clocks.read_json_line(x)for x in raw.splitlines()];require(all(r['pid']==child['pid']for r in rows),'Original clock child owner changed');strict=runtime.fixture_clock_port(properties)(out,'on');strict['raw']['path']='/results/'+str(Path(strict['raw']['path']).relative_to(out));require(canonical(strict)==canonical(inside['strict_clock']),'Original strict clock grammar changed');audit=runtime.audit_rows(rows,json.loads(data)['traceEvents'],arm=='on',inside['paused_last_sequence'],properties)
  require(all(canonical(inside[k])==canonical(v)for k,v in audit.items()),'Independently reconstructed coverage differs')
 for stage in('pre','post'):
  health=report[stage+'_health'];producer.health_text(health)
  producer.prior.receipt_binding(root,stage+'-strict',[str(producer.ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--card','0','--img',producer.prior.HEALTH],health[0])
  kernel=report[stage+'_kernel'];producer.prior.receipt_binding(root,stage+'-kernel',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'],kernel)
  import re
  require(not re.search(producer.prior.FAULT,Path(kernel['stdout']['path']).read_text(),re.I),'Raw kernel fault retained')
 require(roster(root)==before and producer.source_binding()==source,'Closing original artifacts/source changed');build=runtime.bound(producer.BUILD/'receipt.json',8<<20)[1];tool=runtime.bound(producer.INSTALL/'bin/unitrace',64<<20)[1];injection=runtime.bound(producer.INSTALL/'lib/libunitrace_tool.so',64<<20)[1]
 envelope={'intake_path':str(root/'report.json'),'intake_sha256':expected_report_sha256,'exact_intake_sha256':expected_report_sha256,'source_root':'/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-source-v1','source_plan_sha256':'42e4e3de36b3b376a254f46e0bc91648d9e4a64705421cd544e7f8ed68cd6c14','build':build,'tool':tool,'injection':injection,'clock_properties':properties,'actual_device_clock_properties':properties,'install_root':str(producer.INSTALL),'image':producer.IMAGE,'fresh512MiB_clock_budget':True,'explicit_model_finalizer_enabled':True,'query_sample_every':512,'paused_regular_list_retention_required':True,'fresh_fixture_runtime_dependency_observations':{arm:read_unique(root/arm/'inside-receipt.json')['libraries_after']for arm in('off','on')},'old_dependency_receipt_transferred':False,'fixture_artifacts':report['artifacts'],'fixture_source_plan_sha256':source,'full_model_coverage_qualified':False,'clock_alignment_qualified':False}
 return {'runtime_envelope':envelope,'fixture_success_revalidated':True,'report_sha256':expected_report_sha256,'actual_retention_OFF_ON_discriminated':True,'full_model_coverage_qualified':False,'clock_alignment_qualified':False,'physical_idle_qualified':False}

if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(a.output.is_absolute()and a.output.parent.resolve()==a.output.parent and not a.output.exists()and a.root.resolve()not in a.output.parents,'Fresh external canonical public output required');a.output.write_text(json.dumps(finalized_binding(a.root,a.expected_report_sha256),indent=2)+'\n')
