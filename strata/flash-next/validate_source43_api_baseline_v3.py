#!/usr/bin/env python3
"""Current genuine43 API provenance/lifecycle; no fullmath or shelf promotion."""
import argparse,ast,copy,json,tempfile
from pathlib import Path
import c1_source43_api_controller_v3 as c
import run_source43_api_baseline_v3 as a
import validate_decode40_host_profile_v3 as old
import validate_source41_model_cli_v3 as bytes_reader
from source43_api_evidence_v3 import bind,epoch
from serial37_canonical_json_v3 import canonical
require=c.require

def command(m,directory,name):
    node=next(n for n in ast.parse(Path(c.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='launch');parts=[copy.deepcopy(n)for n in node.body if isinstance(n,(ast.Assign,ast.AugAssign))and ((isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='command'for t in n.targets))or(isinstance(n,ast.AugAssign)and isinstance(n.target,ast.Name)and n.target.id=='command'))];require(len(parts)==2,'Exact original launch recipe statements');ns={**vars(c.original),'m':m,'directory':directory,'name':name};exec(compile(ast.unparse(ast.Module(parts,type_ignores=[])).replace('c1_api_trace.py','c1_api_trace43_concurrent_v1.py'),'[actual43-original-recipe]','exec'),ns);return ns['command']

def _body_finalized_binding(root):
    root=Path(root).resolve();source=c.source_binding();m=c.validate_prepared(root);report_sha=c.sha(root/'report.json');report=c.read(root/'report.json');require(report['passed']is True and report['errors']==[]and type(report['schema'])is int and report['schema']==1431,'Actual completed API parent required');require(report['source_plan_sha256']==c.sha(root/'source-plan.snapshot.json')==source and report['prepared_sha256']==c.sha(root/'prepared.json')and c.sha(root/'parent.snapshot.py')==c.sha(a.__file__),'Truthful actual parent/source/prepared snapshot required');require(not any(p.is_symlink()for p in root.rglob('*')),'Exact regular original evidence tree required');tree={str(p.relative_to(root)):c.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'};require(canonical(tree)==canonical(report['artifacts']),'Complete original artifact roster changed')
    old.health(root,'pre',report['pre_health']);old.health(root,'post',report['post_health'])
    from native_rms_kernel_journal_v5 import reject_faults
    for label in('pre','post'):
        old.command_row(root,label+'-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager']);reject_faults((root/(label+'-journal.log')).read_text())
    bytes_reader.full4(root,report,'pre');bytes_reader.full4(root,report,'post');bytes_reader.pages(report['post_before_pages'],Path(report['post_full4']['rows'][2]['path']));require(report['post_before_pages']['epoch']<=report['post_full4']['started'],'Actual post full4 page bracket');require(report['pre_after_pages']['epoch']<=report['pre_health']['started_epoch']<=report['pre_health']['finished_epoch']<=report['launch_started_epoch'],'Pre publisher/health before actual launch')
    native=bind(root);require(canonical(native)==canonical(report['native_API_evidence']),'Actual API/native evidence differs');launch=c.read(root/'launch.json');require(type(launch['owner_pid'])is int and launch['owner_pid']==report['supervisor_pid']and launch['prepared_sha256']==report['prepared_sha256']and launch['ready']is True and launch['pre_health_sha256']==c.sha(root/'pre-health.json'),'Actual supervisor/ready/prepared/prehealth owner differs');require(canonical(command(m,root,launch['container']))==canonical(c.read(root/'launch.command.json')),'Actual original launch recipe changed');saved_command=c.read(root/'launch.command.json');image=c.read(root/'runtime-image-inspection.json');stop=c.stop_receipt_checked(root);a.life.terminal(stop['terminal']);inspection=c.read(root/'terminal-inspection.json');require(terminal_owner_binding(inspection,native)and canonical(inspection['State'])==canonical(stop['terminal'])and inspection['Name'].lstrip('/')==launch['container']and type(inspection['RestartCount'])is int and inspection['RestartCount']==0 and inspection['Config']['Image']==m['runtime']['image']and inspection['Config']['Labels']['b70.c1.prepared']==report['prepared_sha256'],'Actual terminal owner/image/restart proof differs');expected_binds=[saved_command[i+1]for i,x in enumerate(saved_command)if x=='-v'];require(len(set(expected_binds))==len(expected_binds)and sorted(inspection['HostConfig']['Binds'])==sorted(expected_binds)and len(inspection['HostConfig']['Binds'])==len(expected_binds),'Exact mount bind bijection required');require(inspection['Config']['Cmd']==saved_command[saved_command.index(m['runtime']['image'])+1:]and inspection['Config']['Entrypoint']==image['Config']['Entrypoint']and inspection['Config']['WorkingDir']==image['Config']['WorkingDir']and inspection['Config']['User']=='1000:1000'and inspection['HostConfig']['NetworkMode']=='host'and type(inspection['HostConfig']['Memory'])is int and inspection['HostConfig']['Memory']==105*1024**3 and type(inspection['HostConfig']['MemorySwap'])is int and inspection['HostConfig']['MemorySwap']==105*1024**3 and inspection['HostConfig']['Privileged']is False and inspection['HostConfig']['ReadonlyRootfs']is False,'Actual native/API launch image/config/resource recipe differs');require(stop['clean_exit']is True and stop['engine_clean_exit_proven']is True and stop['engine_close_trace_error']is None,'Normal original API/native retirement required');supervisor=c.read(root/'launch-supervisor.json');exit=c.read(root/'supervisor-exit.json');require(supervisor['passed']is True and supervisor['container']==launch['container']and type(exit['return_code'])is int and exit['return_code']==report['supervisor_return_code']==0 and exit['pid']==launch['owner_pid'],'Actual original CLI exit required');require(epoch(native['native_closed_epoch'])<=epoch(stop['finished_epoch'])<=epoch(supervisor['finished_epoch'])<=epoch(report['actual_model_terminal_epoch'])<=epoch(report['post_health']['started_epoch'])<=epoch(report['finished_epoch']),'Actual model/CLI/health chronology differs');require(report['guardian']['passed']is True and report['guardian']['thread_retired']is True and report['guardian']['errors']==[]and report['guardian']['samples'],'Actual three-page guardian retirement required')
    for proof in report['guardian']['samples']:bytes_reader.pages(proof,Path(report['post_full4']['rows'][2]['path']))
    old.command_row(root,'api-client',[a.sys.executable,str(c.HERE/'source43_api_baseline_client_v1.py'),'--port',str(m['port']),'--output',str(root/'api-client')]);require(c.source_binding()==source and c.sha(root/'report.json')==report_sha,'Actual source/report changed during public admission');return {'passed':True,'report_sha256':c.sha(root/'report.json'),'source_plan_sha256':source,'source_generation':43,'materialized_mode':1,'actual_HTTP_overlap_passed':True,'native_API_evidence':native,'registry_association':c.registry_binding(),'old_prerequisite_current_registry_gate_passed':False,'full_model_math_qualified':False,'shelf_qualified':False,'native_parallelism_qualified':False}

def terminal_owner_binding(inspection,native):
    require(inspection['Id']==native['actual_native_host_owner']['container_id'],'Original live and terminal container incarnations differ');return True

def read_guard(root):
    require(root.is_dir()and not root.is_symlink()and not any(p.is_symlink()for p in root.rglob('*')),'Exact regular public read tree required');return {'source':c.source_binding(),'report':c.sha(root/'report.json'),'tree':{str(p.relative_to(root)):c.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'}}

def guarded_read(root,operation):
    before=read_guard(root);error=None;result=None;guard_error=None
    try:result=operation()
    except BaseException as exc:error=exc
    finally:
        try:require(canonical(read_guard(root))==canonical(before),'Original source/report/full raw tree changed during public read')
        except BaseException as exc:guard_error=exc
    if error is not None:
        if guard_error is not None:raise ValueError(str(error)+'; post-read guard: '+str(guard_error))from error
        raise error
    if guard_error is not None:raise guard_error
    return result

def _finalized_binding(root):return guarded_read(Path(root).resolve(),lambda:_body_finalized_binding(root))

def finalized_binding(root):
    root=Path(root).resolve();output=Path(tempfile.mkdtemp(prefix='source43-API-public-close-'));error=None;result=None;close_error=None
    try:result=_finalized_binding(root)
    except BaseException as exc:error=exc
    finally:
        try:
            lock=c.read(c.HERE/'model-lock.json');shards=[c.REPO/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];before=a.intake.guard(shards,output,'pre');value=a.life.q.upload_v2.full_buffered_identity(c.HERE/'model-lock.json',lock,shards,output/'full4.json',a.time.time());after=a.intake.guard(shards,output,'post');require(value['passed']is True,'Actual public failure-closing publisher bytes');c.source_binding()
        except BaseException as exc:close_error=exc
    if error is not None:
        if close_error is not None:raise ValueError(str(error)+'; public closing: '+str(close_error))from error
        raise error
    if close_error is not None:raise close_error
    result['public_fresh_closing']={'path':str(output/'full4.json'),'sha256':c.sha(output/'full4.json'),'pre_pages':before,'post_pages':after};return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);v=p.parse_args();c.write(v.output,finalized_binding(v.root))
