#!/usr/bin/env python3
"""Read-only producer-scope native diagnostic admission; no new model inference."""
import json,math
from pathlib import Path
import run_decode40_host_profile_v1 as actor
c=actor.c;require=actor.require


def finalized_binding(root):
    root=Path(root).resolve();report=c.read(root/'report.json');require(report['passed']is True and report['errors']==[],'Original diagnostic incomplete or failed')
    require(report['source_plan_sha256']==actor.closure()==c.sha(root/'source-plan.snapshot.json')and report['producer_sha256']==c.sha(actor.__file__)==c.sha(root/'producer.snapshot.py'),'Current exact producer/source snapshots differ')
    roster={str(p.relative_to(root)):c.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'}
    require(not any(p.is_symlink()for p in root.rglob('*'))and roster==report['artifacts'],'Exact regular original artifact tree differs')
    require(report['post_health']['passed']is True and report['post_health']['finished_epoch']<=report['post_full4']['started']and report['owned_model_terminal_epoch']<=report['post_full4']['started'],'Actual terminal/health/post-full4 chronology differs')
    lock=c.read(actor.HERE/'model-lock.json');want=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')]
    for key in ('pre_full4','post_full4'):
        value=report[key];require(value['passed']is True and len(value['rows'])==len(want)==4 and value['lock_sha256']==c.sha(actor.HERE/'model-lock.json'),'Publisher full4 scope differs')
        for row,expected in zip(value['rows'],want):require(row['passed']is True and row['sha256']==row['expected_sha256']==expected['sha256']and row['bytes']==expected['size']and row['stat_before']==row['stat_after'],'Actual publisher size/hash row differs')
    off=[];on=[]
    for mode,rows in (('off',off),('on',on)):
        directory=root/mode;receipt=c.read(directory/'receipt.json');require(receipt['return_code']==0 and type(receipt['return_code'])is int and receipt['error']is None and receipt['removed']is True and receipt['forced_stop']is False and receipt['removal']['recovery_used']is False,'Actual native normal retirement differs');actor.terminal(receipt['terminal']['State'])
        require(receipt['command_sha256']==c.sha(directory/'command.json'),'Original command changed')
        for stream in ('stdout','stderr'):
            proof=receipt[stream];require(proof['passed']is True and proof['errors']==[]and proof['eof']is True and proof['retired']is True and proof['sha256']==c.sha(directory/('native.'+stream+'.log')),'Original native EOF/raw log differs')
        saved=c.read(directory/'requests.json');require(len(saved)==4,'Exactly four native requests required')
        input_rows=[json.loads(line)for line in (directory/'native.stdin.jsonl').read_text().splitlines()];require(len(input_rows)==5 and input_rows[-1]['packet']=='QUIT\n','Actual original stdin sequence differs')
        emitted=[];current=[]
        for line in (directory/'native.stdout.log').read_text().splitlines():
            require(not line.startswith(('ERR','FATAL')),'Raw native error preserved')
            if line.startswith('T '):current.append(int(line.split()[1]))
            if line.startswith('DONE '):emitted.append((current,line));current=[]
        require(not current and len(emitted)==4,'Raw native terminal/token extent differs')
        for req,packet,(ids,done)in zip(saved,input_rows,emitted):
            require(packet['packet']=='GEN 64 seed=1 fresh=1 '+','.join(map(str,req['ids']))+'\n'and req['output_ids']==ids and req['done']==done and req['max_new']==64,'Original stdin/output/DONE association differs')
            require(req['done_signature']==done.split()[:3]+done.split()[5:],'Typed completion projection differs')
            rows.append({k:req[k]for k in('ids','output_ids','done_signature')})
        if mode=='on':
            paths=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]))
            actual=actor.collector.collect(paths,directory/'native.stderr.log',report['source_plan_sha256'],saved,[[0,0,48]])
            require(actual==c.read(directory/'host-coverage.json'),'Actual host intervals/raw coverage recollection differs')
    require(off==on and off[0]==off[3]and off[1]==off[2],'Original OFF/ON/within-arm outputs or counters differ')
    require(actor.closure()==report['source_plan_sha256']and roster=={str(p.relative_to(root)):c.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'},'Final source/artifact closure changed')
    return {'passed':True,'report_sha256':c.sha(root/'report.json'),'source_plan_sha256':report['source_plan_sha256'],'scope':report['scope'],'outputs_matched':True,'host_coverage_only':True,'current_full_model_rehashed_by_reader':False,'actual_device_timestamps_observed':False,'physical_expert_traffic_observed':False,'API_model_qualification':False,'shelf_qualified':False}
