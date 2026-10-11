"""New source41 CLI admission; actual source40 control remains separately labeled."""
import json,re,tempfile
from pathlib import Path
import run_source41_model_cli_v4 as a
import validate_decode40_host_profile_v3 as existing
from serial37_canonical_json_v3 import canonical
c=a.c;require=a.require;epoch=a.contract.epoch

def tree(root):
    require(root.is_dir()and not root.is_symlink()and not any(p.is_symlink()for p in root.rglob('*')),'Exact regular evidence tree required')
    return {str(p.relative_to(root)):c.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'}

def pages(proof,shard):
    import source_page_watchdog_v3 as old
    require(proof['passed']is True and proof['path']==str(shard)and proof['stat_before']==proof['stat_after']==c.stat_signature(shard),'Current three-page path/stat differs');epoch(proof['epoch'])
    require(len(proof['rows'])==3,'All three original pages required')
    for row,(offset,sha)in zip(proof['rows'],(*old.KNOWN_PAGES,a.intake.EXTRA_PAGE)):
        path=Path(row['preserved_path']);require(path.is_file()and not path.is_symlink()and path.stat().st_size==4096 and c.sha(path)==sha==row['sha256']==row['expected_sha256']and row['offset']==offset and row['bytes']==4096 and row['passed']is True,'Original preserved page bytes/hash differ')

def full4(root,report,phase,prefix=''):
    lock=c.read(a.HERE/'model-lock.json');expected=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];value=c.read(root/(prefix+phase+'-full4.json'))
    require(canonical(value)==canonical(report[phase+'_full4'])and value['passed']is True and value['lock_sha256']==c.sha(a.HERE/'model-lock.json')and value['model_revision']==lock['revision']and len(value['rows'])==4,'Actual saved complete publisher identity differs')
    for row,want in zip(value['rows'],expected):
        path=a.ROOT/lock['destination']/want['path'];require(row['path']==str(path)and row['sha256']==row['expected_sha256']==want['sha256']and row['bytes']==want['size']and row['passed']is True and row['stat_before']==row['stat_after']==c.stat_signature(path),'Current exact publisher path/hash/stat differs')
    epoch(value['started']);epoch(value['finished']);require(value['started']<=value['finished'],'Full4 chronology invalid');pages(report[phase+'_pages'],Path(value['rows'][2]['path']))
    if phase=='pre':
        pages(report['pre_after_pages'],Path(value['rows'][2]['path']));require(report['pre_pages']['epoch']<=value['started']<=value['finished']<=report['pre_after_pages']['epoch'],'Actual pre page/full4 brackets differ')
    else:require(value['started']>=max(report['actual_model_terminal_epoch'],report['post_health']['finished_epoch'])and value['finished']<=report['post_pages']['epoch'],'Actual post terminal/health/full4/page bracket differs')
    return value

def command(root,phase,m,recipe,pid,binding):
    _,cmd=a.life.contract.native_command(root/phase,m,recipe,pid,binding);cmd[cmd.index('--name')+1]='b70-cli-'+phase+'-'+str(pid);cmd[-1]='exec 2>&1; '+cmd[-1];return cmd

def phase_binding(root,phase,m,cfg,inputs,report):
    directory=root/phase;binding=report['source_plan_sha256'];actual_m,recipe=a.phase_recipe(phase,m,cfg,inputs,binding)
    require(canonical(recipe)==canonical(c.read(directory/'recipe.json')),'Exact actual source40/41 recipe differs');cmd=command(root,phase,actual_m,recipe,report['producer_pid'],binding);require(canonical(cmd)==canonical(c.read(directory/'command.json')),'Actual native command differs')
    result=c.read(directory/'result.json');require(result['passed']is True and result['error']is None and result['errors']==[]and type(result['return_code'])is int and result['engine_rc']==result['return_code']==0 and result['forced_cleanup']is False,'Actual normal native terminal required')
    require(canonical(result)==canonical(report['results'][phase])and result['command_sha256']==c.sha(directory/'command.json')and type(result['engine_generation'])is int and result['engine_generation']==recipe['engine_generation']and type(result['materialized_mode'])is int and result['materialized_mode']==recipe['materialized_mode']and result['observer_enabled']is recipe['observer_enabled'],'Actual phase source/mode result differs')
    obj=result['terminal'];name=result['container'];a.life.contract.inspection(obj,name,cmd,actual_m['runtime']['image'],binding,False);a.life.terminal(obj['State']);require(result['removal']['removed']is True and result['removal']['recovery_used']is False and c.absent(name),'Actual original owned absence required')
    from native_qsa40_owned_launch_v9 import fence
    require(fence(directory),'Original attach/session EOF fence failed');capture=result['stdout_capture'];require(capture['passed']is True and capture['eof']is True and capture['reader_retired']is True and capture['error']is None and capture['path']==str((directory/'engine.combined.log').resolve())and capture['sha256']==c.sha(directory/'engine.combined.log')and capture['forced_cleanup']is False,'Actual original stream EOF/hash/error differs')
    limits=recipe['pipe_limits']
    for key in('recorded_bytes','recorded_lines','max_line_bytes','observed_bytes','observed_lines','pipe_byte_limit','pipe_line_limit','line_byte_limit'):require(type(capture[key])is int and capture[key]>=0,'Typed actual bounded pipe counter required')
    payload=(directory/'engine.combined.log').read_bytes();line_sizes=[len(line)for line in payload.splitlines(keepends=True)]
    require(capture['recorded_bytes']==capture['observed_bytes']==len(payload)<=capture['pipe_byte_limit']==limits['bytes']and capture['recorded_lines']==capture['observed_lines']==len(line_sizes)<=capture['pipe_line_limit']==limits['lines']and capture['max_line_bytes']==max(line_sizes,default=0)<=capture['line_byte_limit']==limits['line_bytes'],'Original actual pipe counts/caps/file bytes differ')
    hp=existing.health(root,phase+'-pre',report[phase+'-pre_health']);require(report['pre_full4']['finished']<=hp['started_epoch']<=hp['finished_epoch']<=result['created_epoch'] and 0<=result['created_epoch']-hp['finished_epoch']<=300 and result['finished_epoch']<=report['actual_model_terminal_epoch'],'Actual phase health/source/launch/terminal boundaries differ')
    rows=c.read(directory/'requests.json');views=a.contract.requests(rows,recipe['requests']);protocol=[]
    for row in rows:
        require(canonical(row['raw'])==canonical(c.read(directory/('request-'+str(row['ordinal'])+'.json'))),'Actual individual request differs');protocol.extend(row['raw']['stdout'])
    sent_raw=(directory/'native.stdin.jsonl').read_bytes();require(sent_raw.endswith(b'\n'),'Complete actual native input lines required');sent=[json.loads(line)for line in sent_raw.splitlines()];expected=[row['raw']['command']+'\n'for row in rows]+['QUIT\n'];require([r['packet']for r in sent]==expected and all(type(r['seq'])is int and r['seq']==i and type(r['monotonic_ns'])is int and r['monotonic_ns']>0 for i,r in enumerate(sent))and [r['monotonic_ns']for r in sent]==sorted(r['monotonic_ns']for r in sent),'Actual sent GEN/QUIT packet chronology differs')
    raw=(directory/'engine.protocol.log').read_bytes();combined=(directory/'engine.combined.log').read_bytes();require(raw.endswith(b'\n')and combined.endswith(b'\n'),'Original complete line termination required');lines=raw.decode('ascii').splitlines();require(lines and lines[0].startswith('READY')and lines[1:]==protocol,'Actual exact protocol/raw request partition differs')
    require([line for line in combined.decode('ascii').splitlines()if a.Protocol.is_protocol(line)]==lines,'Actual combined/protocol source crossjoin differs')
    guard=c.read(directory/'guardian-proof.json');require(guard['passed']is True and guard['errors']==[]and guard['thread_retired']is True and guard['samples'],'Actual active guardian retirement failed')
    for row in guard['samples']:pages(row,Path(report['post_full4']['rows'][2]['path']))
    startup=(directory/'startup/server-engine.log').read_bytes();require(combined.startswith(startup),'Actual startup prefix differs')
    startup_check=Path(tempfile.mkdtemp(prefix='source41-reader-startup-'));(startup_check/'server-engine.log').write_bytes(startup);fresh=a.life.prepared_ctrl.memory_admission(startup_check,m);saved=c.read(directory/'memory-admission.json');require(canonical({k:v for k,v in fresh.items()if k not in('epoch','log_path')})==canonical({k:v for k,v in saved.items()if k not in('epoch','log_path')}),'Actual startup memory gate changed')
    phase_receipt=c.read(directory/'phase_receipt.json');require(phase_receipt['passed']is True and phase_receipt['error']is None and type(phase_receipt['return_code'])is int and phase_receipt['return_code']==0 and phase_receipt['result_sha256']==c.sha(directory/'result.json')and phase_receipt['command_sha256']==c.sha(directory/'command.json')and phase_receipt['source_plan_sha256']==binding and phase_receipt['actual_phase']==phase,'Original phase receipt differs')
    from source41_model_cli_launch_binding_v1 import binding as launch_binding
    launch_binding(c.read(directory/'launch.json'),c.read(directory/'launch-retired.json'),capture,result,phase_receipt,{'producer_pid':report['producer_pid'],'producer_start_ticks':report['producer_start_ticks'],'actual_command':cmd,'actual_model_terminal_epoch':report['per_arm'][phase]['actual_model_terminal_epoch']})
    return views,result

def _finalized_binding(root):
    root=Path(root).resolve();report=c.read(root/'report.json');require(report['passed']is True and report['errors']==[]and type(report['producer_pid'])is int and report['producer_pid']>0,'Actual model CLI failed/untyped')
    source=a.closure();require(report['source_plan_sha256']==source==c.sha(root/'source-plan.snapshot.json')and c.sha(root/'producer.snapshot.py')==c.sha(a.__file__),'Actual source/producer snapshot differs');roster=tree(root);require(canonical(roster)==canonical(report['artifacts']),'Exact original evidence roster differs')
    plan=c.read(root/'plan.snapshot.json');require(c.sha(root/'plan.snapshot.json')==report['plan_sha256'],'Exact admitted plan snapshot SHA differs');m,cfg=a.admission(plan)
    require(canonical(m)==canonical(c.read(root/'prepared.snapshot.json'))and canonical(cfg)==canonical(c.read(root/'config.snapshot.json')),'Current original source40 software template differs');inputs=c.read(root/'inputs.json');require(canonical(inputs)==canonical(c.read(plan['inputs_path'])),'Current independently admitted token IDs differ')
    output=Path(tempfile.mkdtemp(prefix='source41-reader-tokenizer-'));current=dict(m);current['source']=str(a.intake.SDK/'source');fresh=a.life.tokenizer(output,current,source);require(canonical(fresh)==canonical(inputs)and canonical(c.read(output/'tokenizer-fixtures.json'))==canonical(c.read(root/'tokenizer-fixtures.json')),'Fresh original tokenizer/source output changed')
    tname,tcmd=a.life.contract.tokenizer_command(root,current,report['producer_pid'],source,a.life.__file__);require(canonical(tcmd)==canonical(c.read(root/'tokenizer.command.json')),'Actual tokenizer command changed');tr=c.read(root/'tokenizer.receipt.json');a.life.contract.inspection(tr['terminal'],tname,tcmd,current['runtime']['image'],source,True);a.life.terminal(tr['terminal']['State']);require(tr['error']is None and type(tr['return_code'])is int and tr['return_code']==0 and tr['removed']is True and tr['removal']['recovery_used']is False and c.absent(tname),'Original metadata retirement failed')
    for stream in('stdout','stderr'):existing.pipe(tr[stream],root/('tokenizer.'+stream+'.log'))
    speed=a.life.match_speed_inputs(report['speed_input_join']['path'],inputs,m);require(canonical(speed)==canonical(report['speed_input_join']),'Actual speed input origin differs')
    full4(root,report,'pre');full4(root,report,'post');phases=('source40-control','source41-off')if report['phase']=='off-equivalence'else('source41-original-fidelity',)
    arms=[]
    for phase in phases:
        identity=report['per_arm'][phase];full4(root,identity,'pre',phase+'-');full4(root,identity,'post',phase+'-');pre=report[phase+'-pre_health'];require(identity['pre_after_pages']['epoch']<=pre['started_epoch'],'Per-arm identity must precede fresh health');post=existing.health(root,phase+'-post',identity['post_health']);require(post['started_epoch']>=identity['actual_model_terminal_epoch'],'Per-arm post health before terminal');existing.command_row(root,phase+'-post-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'])
        arm,actual=phase_binding(root,phase,m,cfg,inputs,report);require(actual['finished_epoch']<=identity['actual_model_terminal_epoch'],'Actual phase terminal not inside per-arm closure');arms.append(arm)
        existing.command_row(root,phase+'-pre-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'])
    hp=existing.health(root,report['aggregate_post_health_label'],report['post_health']);require(hp['started_epoch']>=report['actual_model_terminal_epoch'],'Post health before original model terminal')
    require(report['aggregate_pre_from_actual_arm']==phases[0]and report['aggregate_post_from_actual_arm']==phases[-1]and report['aggregate_post_health_label']==phases[-1]+'-post','Exact first/last arm aggregate scope required')
    for phase,arm in [('pre',phases[0]),('post',phases[-1])]:require(canonical(report[phase+'_full4'])==canonical(report['per_arm'][arm][phase+'_full4'])and c.sha(root/(phase+'-full4.json'))==c.sha(root/(arm+'-'+phase+'-full4.json')),'Aggregate identity must be exact first/last real arm receipt')
    from native_rms_kernel_journal_v5 import reject_faults
    for phase in(*[p+'-pre'for p in phases],*[p+'-post'for p in phases]):reject_faults((root/(phase+'-journal.log')).read_text())
    off=report['phase']=='off-equivalence'
    if off:require(canonical(a.contract.equivalence(*arms))==canonical(report['OFF_comparison']),'Actual OFF comparison differs')
    else:require(canonical(__import__('validate_source41_model_cli_v3').finalized_binding(report['OFF_prerequisite_root']))==canonical(report['OFF_prerequisite'])and report['OFF_prerequisite']['OFF_equivalence_passed']is True,'Explicit independently closed original OFF admission absent')
    shard=Path(report['post_full4']['rows'][2]['path']);a.intake.guard([Path(r['path'])for r in report['post_full4']['rows']],Path(tempfile.mkdtemp(prefix='source41-reader-pages-')),'current')
    require(a.closure()==source and tree(root)==roster,'Final current source/original evidence changed')
    return {'passed':True,'root':str(root),'report_sha256':c.sha(root/'report.json'),'source_plan_sha256':source,'source41_sdk':plan['source41'],'OFF_equivalence_passed':off,'API_qualified':False,'full_model_math_qualified':False,'shelf_qualified':False}

def finalized_binding(root):
    """Fresh full4 closes every reference/source admission, including failures."""
    root=Path(root).resolve();output=Path(tempfile.mkdtemp(prefix='source41-public-close-'));result=None;error=None;closing=None;close_error=None
    try:result=_finalized_binding(root)
    except BaseException as exc:error=exc
    finally:
        try:
            lock=c.read(a.HERE/'model-lock.json');shards=[a.ROOT/lock['destination']/row['path']for row in lock['files']if row['path'].startswith('UD-Q4_K_XL/')]
            closing=a.life.q.upload_v2.full_buffered_identity(a.HERE/'model-lock.json',lock,shards,output/'full4.json',a.time.time());require(closing['passed'],'Actual CPU public closing publisher bytes failed');a.intake.guard(shards,output,'after')
        except BaseException as exc:close_error=exc
        c.write(output/'reader-close.json',{'original_error':None if error is None else type(error).__name__+': '+str(error),'closing_error':None if close_error is None else type(close_error).__name__+': '+str(close_error),'full4_path':str(output/'full4.json'),'full4_sha256':c.sha(output/'full4.json')if(output/'full4.json').is_file()else None})
    if error is not None:raise error
    if close_error is not None:raise close_error
    result['current_public_full4_rehashed']=True
    return result


def captured_binding(root):
    root=Path(root).resolve();admitted=finalized_binding(root);report=c.read(root/'report.json');require(report['phase']=='original-fidelity','Original-fidelity process required');directory=root/'source41-original-fidelity';recipe=c.read(directory/'recipe.json');lines=(directory/'engine.combined.log').read_text().splitlines();pids={int(m.group(1))for line in lines if (m:=re.search(r'^SFD request .*\bpid=(\d+)',line))}
    require(len(pids)==1,'Actual SFD engine PID must be unique');result={'capture_root':str(directory),'stages':[[0,0,48]],'materialized_mode':recipe['materialized_mode'],'observer_enabled':recipe['observer_enabled'],'binding_sha256':report['source_plan_sha256'],'engine_pid':pids.pop(),'source41_sdk':admitted['source41_sdk'],'public_actor_admission':admitted}
    for role in('command','result','phase_receipt'):
        path=directory/(role+'.json');result[role+'_path']=str(path);result[role+'_sha256']=c.sha(path)
    return result
