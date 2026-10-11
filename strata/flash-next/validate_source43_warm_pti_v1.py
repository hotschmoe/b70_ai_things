#!/usr/bin/env python3
"""New warm IPC/graph evidence around unchanged strict actor/clock predicates."""
import ast,copy,json,hashlib,math
from pathlib import Path
import validate_source43_bounded_pti_v1 as old
import run_source43_warm_pti_v1 as actor
import pti_owned_temporal_control_v1 as helper
import source41_host_trace_contract_v2 as h36
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require

def control_binding(directory,recipe,terminal,report):
    directory=Path(directory);command=c.read(directory/'command.json');binding=report['source_plan_sha256'];session=helper.session_name(report['producer_pid'],report['producer_start_ticks'],binding);require(recipe['temporal_session']==session,'Original PID/start/source session differs')
    values=[]
    for label,action in actor.temporal_exec.PHASES.items():
        saved=c.read(directory/(label+'-owned.json'));raw=c.read(directory/(label+'.json'));diagnostic=c.read(directory/(label+'-exec.receipt.json'))
        require(saved['passed']is True and type(saved['return_code'])is int and saved['return_code']==0 and saved['container_id']==terminal['Id']and canonical(raw)==canonical(saved['control'])and c.sha(directory/(label+'.json'))==saved['control_sha256'],'Original successful temporal receipt/ID/bytes differ')
        require(canonical(saved['command'])==canonical(actor.temporal_exec.command(terminal['Id'],label,actor.tool.TOOL_SHA,session))and diagnostic['passed']is True and diagnostic['errors']==[]and diagnostic['actual_exec_started']is True and diagnostic['actual_exec_terminal']is True and canonical(diagnostic['command'])==canonical(saved['command'])and diagnostic['return_code']==0,'Original temporal exec command/EOF differs')
        require(c.sha(directory/(label+'-exec.stdout'))==diagnostic['metadata_stdout_sha256']and (directory/(label+'-exec.stdout')).stat().st_size==diagnostic['metadata_stdout_bytes'],'Original temporal stdout changed')
        for boundary in('before','after'):
            owner=c.read(directory/(label+'-owner-'+boundary+'.json'));actor.life.contract.inspection(owner,terminal['Name'].lstrip('/'),command,recipe['tool_runtime']['image'],binding,False)
            require(owner['Id']==terminal['Id']and owner['State']['Running']is True and owner['State']['Pid']==saved['host_native_pid']and owner['HostConfig']['IpcMode']=='private','Original live private-IPC owner differs')
        require(raw['session']==session and raw['action']==action and raw['passed']is True and raw['error']is None and raw['target']['pid']==1,'Actual source-owned temporal action differs');helper.check_transition(raw['before'],raw['after'],action)
        require(canonical(raw['command'])==canonical(['/pti/bin/unitrace','--session',session,'--resume']if action=='resume'else None),'Original inner temporal CLI differs')
        require(raw['GPU_queries_added']is False and raw['process_memory_read']is False,'Temporal control cannot add GPU query or process-memory access');values.append(saved)
    require(all(canonical(v['control']['target'])==canonical(values[0]['control']['target'])for v in values),'Native target identity changed across warm window')
    mapped=c.read(directory/'after_READY-owned.json')['census']['process'];require(values[0]['control']['target']['start_ticks']==mapped['start_ticks'],'Temporal target must join actual mapped injection target')
    sent=[json.loads(x)for x in(directory/'native.stdin.jsonl').read_text().splitlines()];require(len(sent)==3 and sent[-1]['packet']=='QUIT\n','Exactly two GEN plus normal QUIT required')
    warm=c.read(directory/'warmup-DONE-observed.json');selected=c.read(directory/'selected-DONE-observed.json');ready=c.read(directory/'after_READY-owned.json');last=c.read(directory/'after_last_DONE_before_QUIT-owned.json')
    order=[ready['finished_monotonic_ns'],values[0]['started_monotonic_ns'],values[0]['finished_monotonic_ns'],sent[0]['monotonic_ns'],warm['monotonic_ns'],values[1]['started_monotonic_ns'],values[1]['finished_monotonic_ns'],sent[1]['monotonic_ns'],selected['monotonic_ns'],values[2]['started_monotonic_ns'],values[2]['finished_monotonic_ns'],last['started_monotonic_ns'],last['finished_monotonic_ns'],sent[2]['monotonic_ns']]
    require(all(type(v)is int and v>0 for v in order)and order==sorted(order),'Actual paused warmup/resume/selected/QUIT order differs')
    return values

def clock_window(directory):
    directory=Path(directory);control=c.read(directory/'temporal-before-selected.json');endpoint=control['clock_before'];path=directory/'unitrace/clock.1.jsonl';before=path.stat();signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(not path.is_symlink()and before.st_size<=512<<20,'Bounded regular original clock required');count=0;seq=0
    with path.open('rb')as stream:
        for line in iter(lambda:stream.readline(4097),b''):
            require(len(line)<=4096 and line.endswith(b'\n'),'Complete source-bound clock row required');row=json.loads(line);seq+=1;require(type(row['sequence'])is int and row['sequence']==seq and row['pid']==1,'Original clock owner/order differs')
            if row['kind']=='instance_normalization':require(seq>endpoint['last_sequence'],'Paused warmup emitted an actual normalized operation');count+=1
        stream.seek(endpoint['tail_offset']);tail=stream.read(len(bytes.fromhex(endpoint['tail_hex'])))
    require(signature(path.stat())==signature(before)and tail.hex()==endpoint['tail_hex']and hashlib.sha256(tail).hexdigest()==endpoint['tail_sha256']and seq>endpoint['last_sequence']and count>0,'Actual resume clock boundary or selected operations absent')
    return dict(paused_warmup_normalizations_observed=0,observed_after_resume_normalizations=count,selected_request_semantic_operation_coverage_qualified=False,teardown_operations_may_be_included=True,clock_alignment_qualified=False)

def graph_binding(directory):
    paths=sorted((Path(directory)/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));require(len(paths)==2,'Exactly warmup/selected actual H36 files required')
    first,_=h36.typed_rows(paths[0],1);second,_=h36.typed_rows(paths[1],1);return actor.warm.graph_reuse(first,second,h36.K)

def finalized_binding(root):
    root=Path(root);report=c.read(root/'report.json');actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks'])
    namespace={**vars(old),'actor':actor};nodes=[copy.deepcopy(n)for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name in('ports','census_binding','finalized_binding')]
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[unchanged-current-model-runtime-and-clock-admission]','exec'),namespace);namespace['_PORTS']=namespace['ports']();result=namespace['finalized_binding'](root)
    graph={};window=None
    for phase in actor.PHASES:
        directory=root/phase;recipe=c.read(directory/'recipe.json');terminal=c.read(directory/'result.json')['terminal'];graph[phase]=graph_binding(directory);require(canonical(graph[phase])==canonical(c.read(directory/'warm-graph-proof.json')),'Original producer graph proof differs')
        if recipe['PTI_enabled']:control_binding(directory,recipe,terminal,report);window=clock_window(directory);require(canonical(window)==canonical(c.read(directory/'warm-clock-window.json')),'Original producer clock window differs')
        else:require(not list(directory.glob('temporal-*')),'OFF cannot execute tool temporal controls')
    require(actor.closure()==report['source_plan_sha256']and namespace['_PORTS']['tree'](root)==report['artifacts'],'Warm final source/artifact closure changed')
    result.update(cold_GEN4_scope=False,warm_same_process_graph_reuse=graph,observed_window=window,warm_process_decode_qualified=False,conversation_state_reuse_claimed=False);return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
