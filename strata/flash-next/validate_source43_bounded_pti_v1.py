#!/usr/bin/env python3
"""Actual43 source/lifecycle/clock/raw-model trace; no fixture count authority."""
import argparse,ast,copy,math,re
from pathlib import Path
import validate_source43_normal_model_cli_v1 as base
import run_source43_bounded_pti_v1 as actor
import collect_source43_sampled_model_pti_v1 as pti
import validate_source41_pti_latency_v3 as old
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require

def ports():
    namespace=base.ports();namespace['a']=actor
    # Deliberately modify only original reader phase/contract scope in its AST;
    # all current source/identity/health/EOF and CPU-closing predicates retained.
    node=copy.deepcopy(next(n for n in ast.parse(Path(base.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_finalized_binding'))
    branch=next(n for n in node.body if isinstance(n,ast.If)and ast.unparse(n.test)=='off');branch.orelse=[]
    source=ast.unparse(node).replace("phases = ('source40-control', 'source41-off') if report['phase'] == 'off-equivalence' else ('source41-original-fidelity',)","phases = a.PHASES").replace("'off-equivalence'","'host-latency'").replace("'OFF_comparison'","'H36_OFF_ON_comparison'").replace("'OFF_equivalence_passed'","'H36_OFF_ON_equivalence_passed'").replace("'source41_sdk': plan['source41']","'source43_sdk': plan['source43']")
    exec(compile(source,'[actual43-PTI-public-phase-contract]','exec'),namespace);return namespace
_PORTS=ports()

def census_binding(directory,recipe,terminal,requests):
    rows=[];root=Path(directory);command=c.read(root/'command.json');image=recipe['tool_runtime']['image'];binding=recipe['native_env']['STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256']
    for phase in('after_READY','after_last_DONE_before_QUIT'):
        saved=c.read(root/(phase+'-owned.json'));require(saved['passed']is True and type(saved['return_code'])is int and saved['return_code']==0 and saved['container_id']==terminal['Id']and type(saved['host_native_pid'])is int and saved['host_native_pid']>0,'Actual census command/owner terminal required')
        diagnostic=c.read(root/(phase+'-exec.receipt.json'));require(diagnostic['passed']is True and diagnostic['errors']==[]and diagnostic['actual_exec_started']is True and diagnostic['actual_exec_terminal']is True and type(diagnostic['return_code'])is int and diagnostic['return_code']==0 and canonical(diagnostic['command'])==canonical(saved['command'])and diagnostic['container_id']==terminal['Id'],'Actual successful census exec diagnostic required');require(c.sha(root/(phase+'-exec.stdout'))==diagnostic['metadata_stdout_sha256']and(root/(phase+'-exec.stdout')).stat().st_size==diagnostic['metadata_stdout_bytes'],'Original saved census stdout changed');expected=actor.census.command(terminal['Id'],phase,actor.tool.LIB_SHA);require(canonical(saved['command'])==canonical(expected),'Original exact census exec command changed')
        for boundary in('before','after'):
            owner=c.read(root/(phase+'-owner-'+boundary+'.json'));actor.life.contract.inspection(owner,terminal['Name'].lstrip('/'),command,image,binding,False);require(owner['Id']==terminal['Id']and owner['State']['Running']is True and type(owner['State']['Pid'])is int and owner['State']['Pid']==saved['host_native_pid'],'Original live native census ownership changed')
        require((root/(phase+'-exec.stdout')).stat().st_size==saved['metadata_stdout_bytes'],'Original census stdout changed')
        receipt=c.read(root/phase/'receipt.json');maps=(root/phase/'proc1.maps').read_text();require(canonical(receipt)==canonical(saved['census'])and receipt['passed']is True and receipt['phase']==phase and type(receipt['process']['pid'])is int and receipt['process']['pid']==1 and type(receipt['process']['start_ticks'])is int and receipt['process']['start_ticks']>0 and receipt['original_container_path']=='/pti/lib/libunitrace_tool.so'and receipt['process_memory_bytes_read']is False and receipt['LevelZero_handle_proof']is False and receipt['versioned_symbol_origin_queried']is False,'Actual mapped process/phase required')
        from pti_mapped_injection_census_v2 import select,covering_mount,associate
        require(type(receipt['schema'])is int and receipt['schema']==2,'Actual named Btrfs receipt schema required')
        mount_before=covering_mount((root/phase/'proc1.mountinfo').read_text(),receipt['original_container_path']);mount_after=covering_mount((root/phase/'proc1.mountinfo.after').read_text(),receipt['original_container_path'])
        require(canonical(mount_before)==canonical(receipt['covering_mount_before'])and canonical(mount_after)==canonical(receipt['covering_mount_after'])and receipt['expected_bind_root']==actor.census.EXPECTED_BIND_ROOT and receipt['expected_filesystem_source']==actor.census.EXPECTED_FILESYSTEM_SOURCE,'Original actual Btrfs mount/domain changed')
        fresh=associate(select(maps),receipt['backing_file_stat'],receipt['path_stat_after'],receipt['opened_fd_stat_before'],receipt['opened_fd_stat_after'],mount_before,mount_after,receipt['original_container_path'],actor.census.EXPECTED_BIND_ROOT,actor.census.EXPECTED_FILESYSTEM_SOURCE,receipt['sha256'],actor.tool.LIB_SHA,receipt['map_file_targets'])
        require(canonical(fresh)==canonical(receipt['btrfs_device_domain_association']),'Named mapped backing association differs')
        require(canonical(select(maps))==canonical(receipt['mapped_segments']),'Actual original maps/segments differ');path=root/phase/'mapped-injection.elf';require(path.is_file()and not path.is_symlink()and saved['exported_host_path']==str(path)and c.sha(path)==receipt['sha256']==saved['exported_host_sha256']==actor.tool.LIB_SHA and path.stat().st_size==receipt['bytes'],'Original mapped backing export SHA/path differs');rows.append(saved)
    require(rows[0]['host_native_pid']==rows[1]['host_native_pid']and canonical(rows[0]['census']['process'])==canonical(rows[1]['census']['process'])and canonical(rows[0]['census']['mapped_segments'])==canonical(rows[1]['census']['mapped_segments'])and rows[0]['census']['backing_file_stat']==rows[1]['census']['backing_file_stat'],'Native process/current mapping changed between boundaries')
    epoch=actor.contract.epoch
    for row in rows:require(epoch(row['started_epoch'])<=epoch(row['finished_epoch']),'Actual census chronology invalid')
    sent=[__import__('json').loads(line)for line in(root/'native.stdin.jsonl').read_text().splitlines()]
    ready=c.read(root/'ready-observed.json');done=c.read(root/'last-DONE-observed.json');capture=c.read(root/'result.json')['stdout_capture']
    values=[ready['monotonic_ns'],rows[0]['started_monotonic_ns'],rows[0]['finished_monotonic_ns'],sent[0]['monotonic_ns'],sent[-2]['monotonic_ns'],done['monotonic_ns'],rows[1]['started_monotonic_ns'],rows[1]['finished_monotonic_ns'],sent[-1]['monotonic_ns']]
    require(all(type(v)is int and v>0 for v in values)and values==sorted(values)and sent[-1]['packet']=='QUIT\n','Actual READY/census/GEN/DONE/census/QUIT monotonic ordering differs')
    require(epoch(capture['process_started_epoch'])<=epoch(ready['epoch'])<=rows[0]['started_epoch']and epoch(done['epoch'])<=rows[1]['started_epoch']<=rows[1]['finished_epoch']<=epoch(capture['completed_epoch']),'Actual mapped census source/EOF epochs differ')
    return {'actual_mapped_injection_before_after_joined':True,'native_process':rows[0]['census']['process'],'process_memory_bytes_read':False,'LevelZero_handle_proof':False}

def finalized_binding(root):
    root=Path(root).resolve();report=c.read(root/'report.json');require(report['phase']=='host-latency','Actual43 PTI scope required');saved=c.read(root/actor.PHASES[0]/'recipe.json')['tool_runtime'];actor.set_runtime(saved['intake_path'],saved['exact_intake_sha256']);require(canonical(actor._RUNTIME)==canonical(saved),'Actual finalizer runtime changed');require(canonical(actor.prerequisite(report['OFF_prerequisite_root']))==canonical(report['OFF_prerequisite']),'Actual fresh43 normal prerequisite changed')
    result=_PORTS['finalized_binding'](root);coverage={}
    for phase in actor.PHASES:
        directory=root/phase;recipe=c.read(directory/'recipe.json');terminal=c.read(directory/'result.json')['terminal'];pid=actor.native_pid(terminal,c.read(directory/'command.json'),recipe['PTI_enabled']);require(canonical(c.read(directory/'native-namespace.json'))==canonical({'native_pid':pid,'PTI_enabled':recipe['PTI_enabled'],'exec_branch_source_checked':True}),'Actual native namespace receipt changed');old.budget(directory,root)
        rows=c.read(directory/'requests.json');files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));value=actor.collector.collect(files,directory/'engine.combined.log',report['source_plan_sha256'],[r['raw']for r in rows],[[0,0,48]],pid);require(canonical(value)==canonical(c.read(directory/'host-coverage.json')),'Actual H36/LPdisabled coverage changed');coverage[phase]=value
        if not recipe['PTI_enabled']:require(not list((directory/'unitrace').iterdir()),'PTI OFF emitted profiler files')
        else:
            log=(directory/'engine.combined.log').read_text();markers=re.findall(r'^B70_PTI_MODEL_FINALIZE_V1 pid=(\d+) result=(\d+)$',log,re.M);require(markers==[('1','0')],'Exactly one actual owned successful explicit finalizer required');result['mapped_injection']=census_binding(directory,recipe,terminal,rows);result['PTI_observed_model_coverage']=pti.collect(directory,actor._RUNTIME)
    require(actor.closure()==report['source_plan_sha256']and _PORTS['tree'](root)==report['artifacts']and canonical(actor.tool.binding(saved['intake_path']))==canonical(saved),'Final source/runtime/raw artifact closure changed')
    result.update(cold_GEN4_scope=True,warm_process_decode_qualified=False,every_instance_submit_pair_bracketed=False,source_generation=43,materialized_mode=0,host_coverage_both_arms=coverage,actual_finalizer_closed=True,LPdisabled=True,periodic_budget_not_hard_quota=True,model_complete_device_coverage_qualified=False,H36_SYCL_to_LevelZero_mapping_qualified=False,physical_idle_qualified=False,speed_qualified=False,API_qualified=False,shelf_qualified=False,zero_added_waits=False);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
