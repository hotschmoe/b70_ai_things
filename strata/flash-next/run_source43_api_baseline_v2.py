#!/usr/bin/env python3
"""Small genuine43 API actor; eight calls, owned stop and fresh closing bytes."""
import argparse,os,signal,subprocess,sys,time
from pathlib import Path
import c1_source43_api_controller_v2 as c
import run_decode40_host_profile_v3 as life
import source43_normal_model_intake_v1 as intake
from source43_api_evidence_v2 import bind
import source43_api_live_owner_decode_v1 as joins
from source41_model_cli_contract_v1 import Guardian
from serial37_canonical_json_v3 import canonical
ROOT=c.REPO;HERE=c.HERE;require=c.require


def retire_supervisor(out,child,errors):
    # Never release the lease on a guessed daemon-creation cancellation.
    while True:
        try:
            if(out/'launch.json').exists():
                c.owned_stop(out);checked=c.stop_receipt_checked(out);require(checked['removed']is True,'Actual original owned removal required')
            elif child.poll()is None:time.sleep(1);continue
            child.wait();require(child.returncode==0,'Original owned launch supervisor failed');return
        except BaseException as e:
            message='retirement: '+type(e).__name__+': '+str(e)
            if message not in errors:errors.append(message)
            if child.poll()is not None and(out/'stop.json').exists():
                try:c.stop_receipt_checked(out);return
                except BaseException:pass
            c.write(out/'cleanup-pending.json',{'errors':errors,'epoch':time.time()});time.sleep(1)

def main():
    require(__debug__ and not os.environ.get('PYTHONOPTIMIZE'),'Optimized Python refused');p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--leased',action='store_true');a=p.parse_args();out=a.prepared.resolve();c.source_binding();c.registry_binding();cfg=c.read(out/'server-config.json');c.expected_config(cfg,cfg['port'])
    if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
    c.leased([0,1]);require(not(out/'report.json').exists(),'New API actor output required');m=c.validate_prepared(out);lock=c.read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];require([r['path']for r in m['model_shards']]==list(map(str,shards)),'Exact publisher shard roster');source=c.source_binding();report={'schema':1431,'producer_pid':os.getpid(),'passed':False,'errors':[],'started_epoch':time.time(),'source_plan_sha256':source,'prepared_sha256':c.sha(out/'prepared.json'),'hardware_model_math_qualified':False,'speed_qualified':False,'shelf_qualified':False};(out/'source-plan.snapshot.json').write_bytes(c.PLAN.read_bytes());(out/'parent.snapshot.py').write_bytes(Path(__file__).read_bytes());child=None;guardian=None;interrupted=[False]
    def stop_signal(*unused):interrupted[0]=True
    for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop_signal)
    def pages(label):return intake.guard(shards,out/'guardian',label+'-'+str(time.time_ns()))
    try:
        report['pre_pages']=pages('pre');report['pre_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/'pre-full4.json',time.time());require(report['pre_full4']['passed']is True,'Fresh complete pre publisher bytes');report['pre_after_pages']=pages('pre-after');report['pre_health']=life.health(out,'pre');life.journal(out,'pre',report['started_epoch'])
        cmd=[sys.executable,str(Path(c.__file__).resolve()),'launch','--prepared',str(out),'--pre-health',str(out/'pre-health.json'),'--ready-deadline','600','--max-runtime','7200'];c.write(out/'supervisor.command.json',cmd)
        with(out/'supervisor.log').open('w')as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(8,9))
        report['supervisor_pid']=child.pid;report['launch_started_epoch']=time.time();guardian=Guardian(lambda:pages('active'),lambda:stop_signal());guardian.start();deadline=time.monotonic()+620
        while not(out/'launch.json').exists()or not c.read(out/'launch.json').get('ready'):
            require(not interrupted[0]and child.poll()is None and time.monotonic()<deadline,'Owned readiness interrupted/terminal/deadline');pages('readiness');time.sleep(1)
        report['client_receipt']=life.command(out,[sys.executable,str(HERE/'source43_api_baseline_client_v1.py'),'--port',str(m['port']),'--output',str(out/'api-client')],'api-client',3000)
        report['native_owner']=joins.host_owner(out,m);report['independent_reply_decode']=joins.execute(out,m,source)
    except BaseException as e:report['errors'].append(type(e).__name__+': '+str(e))
    finally:
        interrupted[0]=False
        # Keep the pair lease until original exact owner stop/removal and CLI wait.
        if child is not None:
            retire_supervisor(out,child,report['errors'])
            report['supervisor_return_code']=child.returncode
        if guardian is not None:
            try:report['guardian']=guardian.close();require(report['guardian']['passed']is True,'Active source guardian failed')
            except BaseException as e:report['errors'].append('guardian: '+type(e).__name__+': '+str(e))
        report['actual_model_terminal_epoch']=time.time();c.write(out/'supervisor-exit.json',{'pid':None if child is None else child.pid,'return_code':None if child is None else child.returncode,'finished_epoch':report['actual_model_terminal_epoch']})
        try:report['native_API_evidence']=bind(out)
        except BaseException as e:report['errors'].append('API provenance: '+type(e).__name__+': '+str(e))
        try:report['post_health']=life.health(out,'post');life.journal(out,'post',report['started_epoch'])
        except BaseException as e:report['errors'].append('post-health: '+type(e).__name__+': '+str(e))
        try:
            report['post_before_pages']=pages('post-before');report['post_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/'post-full4.json',max(report['actual_model_terminal_epoch'],report.get('post_health',{}).get('finished_epoch',time.time())));report['post_pages']=pages('post');require(report['post_full4']['passed']is True,'Fresh failure-closing complete publisher bytes')
        except BaseException as e:report['errors'].append('closing-source: '+type(e).__name__+': '+str(e))
        try:require(c.source_binding()==source and c.sha(out/'source-plan.snapshot.json')==source and c.sha(out/'prepared.json')==report['prepared_sha256'],'Original source/prepared bytes changed');c.validate_prepared(out)
        except BaseException as e:report['errors'].append('closing-metadata: '+type(e).__name__+': '+str(e))
        report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(out)):c.sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='report.json'};c.write(out/'report.json',report)
    require(report['passed'],'Actual API actor failed; original evidence retained')
if __name__=='__main__':main()
