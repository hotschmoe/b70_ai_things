#!/usr/bin/env python3
"""Original producer/current metadata joins; CPU metadata verification outside tree."""
import hashlib,json,math,queue,time
from pathlib import Path
import run_decode40_host_profile_v2 as a
c=a.c;require=a.require


def same(x,y):return a.prepared_ctrl.base.same(x,y)


def artifact_tree(root):
    require(not any(p.is_symlink()for p in root.rglob('*')),'Regular nonsymlink original tree required')
    return {str(p.relative_to(root)):c.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'}


def pipe(proof,path):
    require(proof['passed']is True and proof['eof']is True and proof['retired']is True and proof['errors']==[]and proof['path']==str(path)and proof['sha256']==c.sha(path),'Original owned pipe EOF/hash/error differs')


def command_row(root,label,expected):
    row=c.read(root/(label+'.receipt.json'));require(same(row['command'],expected)and same(c.read(root/(label+'.command.json')),expected),'Actual original command differs')
    require(row['passed']is True and type(row['return_code'])is int and row['return_code']==0 and row['command_error']is None and row['phase_cleanup_error']is None and row['eof']is True and row['reader_retired']is True and row['error']is None,'Actual owned command terminal/EOF/error differs')
    require(row['path']==str(root/(label+'.log'))and row['sha256']==c.sha(root/(label+'.log'))and row['command_sha256']==c.sha(root/(label+'.command.json')),'Actual original command/log bytes differ')
    require(all(type(row[k])in(int,float)and math.isfinite(row[k]) for k in('started_command_epoch','started_epoch','completed_epoch','finished_epoch'))and row['started_command_epoch']<=row['started_epoch']<=row['completed_epoch']<=row['finished_epoch'],'Actual command chronology differs')
    return row


def health(root,label,report):
    value=c.read(root/(label+'-health.json'));require(same(value,report)and value['passed']is True and value['cards']==[0,1]and all(type(x)is int for x in value['cards']),'Exact actual health receipt/cards required')
    rows=[]
    for name,cmd in [('strict',[str(a.ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',a.q.HEALTH]),('compiled',[str(a.ROOT/'bin/xpu-collective-health'),'--img',a.q.HEALTH,'--p2p','0','--timeout','180'])]:rows.append(command_row(root,label+'-'+name,cmd))
    require(same(rows,value['files'])and value['started_epoch']<=min(r['started_command_epoch']for r in rows)and max(r['finished_epoch']for r in rows)<=value['finished_epoch'],'Original aggregate health boundaries differ')
    return value


def pages(value,path,stat):
    require(value['passed']is True and value['path']==str(path)and value['stat_before']==value['stat_after']==stat,'Known source-page path/stat differs')
    expected=list(a.q.watchdog.KNOWN_PAGES);require(len(value['rows'])==len(expected)==2,'Exactly both known pages required')
    for row,(offset,sha)in zip(value['rows'],expected):require(row['passed']is True and row['offset']==offset and row['bytes']==4096 and row['sha256']==row['expected_sha256']==sha,'Known source-page bytes/hash differ')


def source4(root,report):
    path=a.HERE/'model-lock.json';lock=c.read(path);expected=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')]
    for phase,before,after in [('pre','pre_pages','pre_after_pages'),('post','post_pages','post_after_pages')]:
        value=c.read(root/(phase+'-full4.json'));require(same(value,report[phase+'_full4'])and value['passed']is True and value['lock_sha256']==c.sha(path)and value['model_revision']==lock['revision']and len(value['rows'])==len(expected)==4,'Original complete publisher scan body/lock differs')
        for row,want in zip(value['rows'],expected):
            model=a.ROOT/lock['destination']/want['path'];require(row['path']==str(model)and row['passed']is True and row['sha256']==row['expected_sha256']==want['sha256']and row['bytes']==want['size']and row['stat_before']==row['stat_after']==c.stat_signature(model),'Original current publisher path/size/hash/stat differs')
        third=Path(value['rows'][2]['path']);pages(report[before],third,value['rows'][2]['stat_after']);pages(report[after],third,value['rows'][2]['stat_after'])
        require(report[before]['epoch']<=value['started']<=value['finished']<=report[after]['epoch'],'Actual full4/page brackets differ')
        if phase=='post':require(value['after_terminal_and_post_health_epoch']==max(report['owned_model_terminal_epoch'],report['post_health']['finished_epoch'])and value['started']>=value['after_terminal_and_post_health_epoch'],'Actual terminal/health/full4 boundary differs')
    # Current known pages remain byte-checked, never stat-only model authority.
    a.q.watchdog.guard(Path(report['post_full4']['rows'][2]['path']))


def replay(directory,saved):
    raw=(directory/'native.stdout.log').read_text().splitlines();times=[json.loads(x)for x in (directory/'native.stdout-times.jsonl').read_text().splitlines()]
    require(len(times)==len(raw)and all(r['seq']==i and type(r['seq'])is int and r['line']==line and type(r['monotonic_ns'])is int and r['monotonic_ns']>0 for i,(r,line)in enumerate(zip(times,raw))),'Original timestamp/line/sequence association differs')
    require(all(x['monotonic_ns']<=y['monotonic_ns']for x,y in zip(times,times[1:])),'Original pipe clocks reordered')
    lines=queue.Queue()
    for row in times:lines.put((row['line'],row['monotonic_ns']))
    class Stream:
        def __init__(self):self.packets=[]
        def write(self,packet):self.packets.append(packet.decode('ascii'))
        def flush(self):pass
        def close(self):pass
    stream=Stream();actual=a.protocol(stream,lines,saved,[False],lambda:(_ for _ in()).throw(ValueError('Original stdout ended before terminal')),time.monotonic()+2)
    inputs=[json.loads(x)for x in (directory/'native.stdin.jsonl').read_text().splitlines()]
    require(len(inputs)==len(stream.packets)==5 and [r['packet']for r in inputs]==stream.packets,'Original exact GEN/QUIT stdin packets differ')
    for old,new in zip(saved,actual):
        require(same({k:old[k]for k in('ids','fresh','max_new','output_ids','done','done_signature','native_pipe_token_ns')},{k:new[k]for k in('ids','fresh','max_new','output_ids','done','done_signature','native_pipe_token_ns')}),'Original protocol counts/budget/finish/timestamp projection differs')
    return [{k:r[k]for k in('ids','output_ids','done_signature')}for r in actual]


def finalized_binding(root,verification_output):
    root=Path(root).resolve();out=Path(verification_output).resolve();require(root not in out.parents and out!=root and not out.exists(),'New external CPU verification output required');out.mkdir(parents=True)
    report=c.read(root/'report.json');require(report['passed']is True and report['errors']==[]and type(report['producer_pid'])is int,'Original diagnostic failed or untyped producer')
    source=a.closure();require(report['source_plan_sha256']==source==c.sha(root/'source-plan.snapshot.json')and report['producer_sha256']==c.sha(a.__file__)==c.sha(root/'producer.snapshot.py'),'Exact current producer/source snapshot differs')
    roster=artifact_tree(root);require(roster==report['artifacts'],'Exact original artifact roster changed')
    directory=Path(report['prepared_directory']);m=a.prepared_ctrl.validate_prepared(directory);require(same(m,c.read(root/'prepared.snapshot.json')),'Current exact admitted preparation differs')
    a.prepared_ctrl._PORTS['snapshot_equal'](directory,c.read(root/'prepared-boundary.json'),m)
    cfg=c.read(directory/'server-config.json');require(same(cfg,c.read(root/'config.snapshot.json'))and c.sha(directory/'prepared.json')==report['prepared_sha256']and c.sha(directory/'server-config.json')==report['config_sha256'],'Complete config/prepared snapshot differs')
    original_tokenizer=c.read(root/'tokenizer-fixtures.json');inputs=c.read(root/'inputs.json');current_inputs=a.tokenizer(out,m,source);require(same(current_inputs,inputs),'Independent current tokenizer IDs differ')
    current_tokenizer=c.read(out/'tokenizer-fixtures.json');require(same(current_tokenizer,original_tokenizer),'Actual tokenizer template/files/runtime/fixtures differ')
    speed=a.match_speed_inputs(report['speed_input_join']['path'],inputs,m);require(same(speed,report['speed_input_join']),'Original speed input/source association differs')
    tname,tcmd=a.contract.tokenizer_command(root,m,report['producer_pid'],source,a.__file__);require(same(tcmd,c.read(root/'tokenizer.command.json')),'Original metadata command differs')
    tr=c.read(root/'tokenizer.receipt.json');a.contract.inspection(tr['terminal'],tname,tcmd,m['runtime']['image'],source,True);a.terminal(tr['terminal']['State']);require(tr['error']is None and tr['return_code']==0 and type(tr['return_code'])is int and tr['removed']is True and tr['removal']['recovery_used']is False and c.absent(tname),'Original CPU metadata retirement/absence differs')
    for stream in('stdout','stderr'):pipe(tr[stream],root/('tokenizer.'+stream+'.log'))
    model_terminal=[];arms={}
    for mode in('OFF','ON'):
        sub=root/mode.lower();hp=health(root,mode.lower()+'-pre',report[mode+'-pre_health'])
        plan=a.recipe.build(m,cfg,inputs,source,mode);require(same(plan,c.read(sub/'recipe.json')),'Original native recipe differs')
        name,cmd=a.contract.native_command(sub,m,plan,report['producer_pid'],source);require(same(cmd,c.read(sub/'command.json')),'Original complete native command differs')
        receipt=c.read(sub/'receipt.json');a.contract.inspection(receipt['terminal'],name,cmd,m['runtime']['image'],source,False);a.terminal(receipt['terminal']['State'])
        require(receipt['error']is None and receipt['return_code']==0 and type(receipt['return_code'])is int and receipt['removed']is True and receipt['forced_stop']is False and receipt['removal']['recovery_used']is False and c.absent(name),'Original native normal retirement/absence differs')
        require(0<=receipt['launch_epoch']-hp['finished_epoch']<=300 and receipt['finished_epoch']<=report['owned_model_terminal_epoch'],'Original real health freshness/terminal chronology differs');model_terminal.append(receipt['finished_epoch'])
        for stream in('stdout','stderr'):pipe(receipt[stream],sub/('native.'+stream+'.log'))
        startup=sub/'startup';raw=(startup/'server-engine.log').read_bytes();require((sub/'native.stderr.log').read_bytes().startswith(raw),'Original startup snapshot not actual stderr prefix')
        # Reexecute original unchanged admission outside immutable original tree.
        check=out/(mode.lower()+'-startup');check.mkdir();(check/'server-engine.log').write_bytes(raw);value=a.prepared_ctrl.memory_admission(check,m);saved=c.read(startup/'chunk-memory-admission.json')
        require(same({k:v for k,v in value.items()if k not in('epoch','log_path')},{k:v for k,v in saved.items()if k not in('epoch','log_path')}),'Actual startup memory capacity admission differs')
        requests=c.read(sub/'requests.json');require(len(requests)==4 and same([{k:r[k]for k in('fresh','ids','prompt_name','max_new')}for r in requests],plan['requests']),'Actual four bounded request corpus differs');arms[mode]=replay(sub,requests)
        if mode=='ON':
            traces=sorted((sub/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));actual=a.collector.collect(traces,sub/'native.stderr.log',source,requests,[[0,0,48]]);require(same(actual,c.read(sub/'host-coverage.json')),'Original H36 raw coverage recheck differs')
    hp=health(root,'post',report['post_health']);require(hp['started_epoch']>=max(model_terminal),'Original posthealth before actual native retirement')
    for label in('off-pre','on-pre','post'):
        row=command_row(root,label+'-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'])
        from native_rms_kernel_journal_v5 import reject_faults
        reject_faults((root/(label+'-journal.log')).read_text())
    source4(root,report);require(arms['OFF']==arms['ON']and arms['OFF'][0]==arms['OFF'][3]and arms['OFF'][1]==arms['OFF'][2],'Original OFF/ON/repeat outputs differ')
    require(a.closure()==source and artifact_tree(root)==roster,'Final current source/original tree changed')
    result={'passed':True,'scope':report['scope'],'report_sha256':c.sha(root/'report.json'),'source_plan_sha256':source,'original_tree_unchanged':True,'host_pipe_timestamp_and_H36_coverage_recollected':True,'current_full_model_rehashed_by_reader':False,'device_timestamps_observed':False,'physical_expert_traffic_observed':False,'API_qualification':False,'shelf_qualified':False}
    c.write(out/'binding.json',result);return result
