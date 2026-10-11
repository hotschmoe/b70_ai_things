#!/usr/bin/env python3
"""Small new source41 model CLI parent; existing protocol/lifecycle retained."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
import run_decode40_host_profile_v3 as life
import source41_model_cli_v2 as intake
import source41_model_cli_contract_v1 as contract
import native_qsa40_matched_experiment_v1 as observer
from native_source41_cli_protocol_v2 import Protocol
from serial37_canonical_json_v3 import canonical
ROOT=life.ROOT;HERE=life.HERE;PLAN=HERE/'source41-model-actor-source-plan-v2.json';c=life.c;require=intake.require


def closure():
    for path,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/path)==want,'Exact model CLI source changed '+path)
    # Cheap nested source/registry declaration before expensive SDK/reference checks.
    from admit_postboot_original_qsa_reference_v4 import source_binding
    source_binding();return c.sha(PLAN)


def admission(plan):
    original=c.read(plan['template_path']+'/prepared.json');metadata=life.prepared_ctrl.validate_prepared(Path(plan['template_path']))
    require(canonical(original)==canonical(metadata)and c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256'],'Actual source40 software template changed')
    sdk=intake.sdk_binding();require(canonical(sdk)==canonical(plan['source41']),'Current exact source41 SDK changed')
    require(plan['schema']==2 and plan['intake_producer_sha256']==c.sha(intake.__file__),'Exact explicit source41 intake required')
    from admit_source_upload41_v1 import finalized_binding
    proof=finalized_binding(plan['upload_receipt'],plan['oracle_receipt'],plan['pack_receipt'])
    require(canonical(proof)==canonical(plan['fresh41_upload']),'Current independent fresh source41 upload changed')
    from qualify_indexer_materialized41_masked_v2 import finalized_binding as masked
    require(canonical(masked(plan['masked_root']))==canonical(plan['fresh41_masked_component']),'Current independent masked component changed')
    require(c.sha(plan['actual_model_identity_path'])==plan['actual_model_identity_sha256'],'Complete identity receipt changed')
    require(c.sha(plan['inputs_path'])==plan['inputs_sha256'],'Independent authenticated input file changed')
    return metadata,c.read(Path(plan['template_path'])/'server-config.json')


def phase_recipe(phase,metadata,cfg,inputs,binding):
    if phase=='source40-control':
        m=dict(metadata);mode=0;observed=False
    else:
        # New minimal model descriptor, no source40 combined_generation/alias PASS view.
        m={k:metadata[k]for k in('cards','pack','runtime','profile')}
        m.update(source=str(intake.SDK/'source'),executable=str(intake.SDK/'build/strata'),alias='source41-native-CLI-diagnostic')
        mode=1 if phase=='source41-original-fidelity'else 0;observed=mode==1
    original=life.recipe.build(metadata,cfg,inputs,binding,'OFF');args=list(original['native_argv'][2:]);env=dict(original['native_env'])
    if phase!='source40-control':env['STRATA_QSA_INDEXER_MATERIALIZED_HALF']=str(mode)
    requests=[{'ordinal':i+1,'ids':inputs[name],'fresh':1,'max_new':64,'pin':None,'role':name}for i,name in enumerate(('short','prefill_about512','prefill_about512','short'))]
    if observed:
        requests=observer.request_roster();env=observer.arm_environment(env,True,binding,'/results/qsa')
    return m,{'mode':phase,'materialized_mode':mode,'observer_enabled':observed,'native_argv':['/build/strata','--serve',*args],'native_env':env,'requests':requests,'engine_generation':40 if phase=='source40-control'else 41}


def run_phase(out,phase,metadata,cfg,inputs,binding,shards,prehealth):
    directory=out/phase;directory.mkdir();m,plan=phase_recipe(phase,metadata,cfg,inputs,binding)
    for name in('captures','p30','ple-input','qsa'):(directory/name).mkdir()
    _,cmd=life.contract.native_command(directory,m,plan,os.getpid(),binding);name='b70-cli-'+phase+'-'+str(os.getpid());cmd[cmd.index('--name')+1]=name
    # Same original native protocol's merged FD, never a source40 command view.
    cmd[-1]='exec 2>&1; '+cmd[-1];c.write(directory/'command.json',cmd);c.write(directory/'recipe.json',plan)
    protocol=None;active=[];rows=[];error=None;forced=False;created=time.time();last_guard=[0]
    def guard():
        if time.monotonic()-last_guard[0]>=2:
            intake.guard(shards,out/'guardian',phase+'-'+str(time.time_ns()));last_guard[0]=time.monotonic()
    guardian=contract.Guardian(lambda:intake.guard(shards,out/'guardian',phase+'-active-'+str(time.time_ns())),lambda:os.kill(os.getpid(),signal.SIGUSR1))
    def cleanup():
        nonlocal forced
        forced=True
        if c.absent(name):return
        obj=life.inspect_owned(name,cmd,m['runtime']['image'],binding)
        if obj['State']['Running']:life.subprocess.run(['docker','stop','--time','30',obj['Id']],check=True,capture_output=True,timeout=45)
    try:
        require(0<=time.time()-prehealth['finished_epoch']<=300,'Actual fresh health before native launch required')
        guardian.start();protocol=Protocol(cmd,directory,cleanup,active=active)
        startup=directory/'startup';startup.mkdir();(startup/'server-engine.log').write_bytes((directory/'engine.combined.log').read_bytes());c.write(directory/'memory-admission.json',life.prepared_ctrl.memory_admission(startup,metadata))
        guard();(directory/'ARM').write_text('ARM new source41 model CLI original prefix schedule\n',encoding='ascii')
        for row in plan['requests']:
            guard();raw=protocol.request('request-'+str(row['ordinal']),row['ids'],fresh=1,max_new=row['max_new'],pin=None)
            c.write(directory/('request-'+str(row['ordinal'])+'.json'),raw);rows.append({'ordinal':row['ordinal'],'raw':raw});c.write(directory/'requests.json',rows)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        from native_source41_cli_protocol_v2 import deferred_source_signals
        with deferred_source_signals():
            try:
                guardian_proof=guardian.close();c.write(directory/'guardian-proof.json',guardian_proof)
                if guardian_proof['errors'] and error is None:error='Active guardian: '+str(guardian_proof['errors'])
            except BaseException as exc:
                error=(error+'; 'if error else'')+'Guardian close: '+type(exc).__name__+': '+str(exc)
                guardian.stop.set()
                if guardian.thread is not None:
                    while guardian.thread.is_alive():guardian.thread.join(.2)
            protocol=protocol or(active[0]if active else None)
            rc=None
            try:
                if protocol:rc=protocol.close(failed=error is not None)
            finally:
                if protocol:
                    obj=life.safe_retire(directory,name,cmd,m['runtime']['image'],binding,protocol.p,True if rc is None else error is not None);removal=life.remove_owned(directory,obj,name,cmd,m['runtime']['image'],binding)if obj['Id']is not None else {'removed':True,'recovery_used':True};c.write(directory/'terminal.json',obj)
            if protocol is None:obj=None;removal={'removed':False,'recovery_used':True}
    value={'schema':1,'passed':error is None and rc==0 and protocol is not None and protocol.capture['passed']and not forced and not removal['recovery_used'],'engine_generation':plan['engine_generation'],'materialized_mode':plan['materialized_mode'],'observer_enabled':plan['observer_enabled'],'error':error,'errors':[]if error is None else[error],'engine_rc':rc,'return_code':rc,'container':name,'command_sha256':c.sha(directory/'command.json'),'created_epoch':created,'finished_epoch':time.time(),'terminal':obj,'removal':removal,'stdout_capture':protocol.capture if protocol else None,'forced_cleanup':forced,'requests':len(rows)}
    c.write(directory/'result.json',value);c.write(directory/'phase_receipt.json',{'schema':1,'passed':value['passed'],'error':error,'return_code':rc,'actual_phase':phase,'engine_generation':plan['engine_generation'],'materialized_mode':plan['materialized_mode'],'observer_enabled':plan['observer_enabled'],'source_plan_sha256':binding,'command_sha256':c.sha(directory/'command.json'),'result_sha256':c.sha(directory/'result.json'),'started_epoch':created,'finished_epoch':value['finished_epoch']});require(value['passed'],'Actual native phase failed '+phase);life.terminal(obj['State']);guard();return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('off-equivalence','original-fidelity'),required=True);p.add_argument('--off-receipt',type=Path);p.add_argument('--leased',action='store_true');p.add_argument('--speed-trace',type=Path,required=True);a=p.parse_args()
    require(__debug__ and not sys.flags.optimize,'Ordinary Python required');source=closure();plan=c.read(a.plan)
    if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
    c.leased([0,1]);metadata,cfg=admission(plan);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'source-plan.snapshot.json').write_bytes(PLAN.read_bytes());(out/'producer.snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'plan.snapshot.json').write_bytes(a.plan.read_bytes())
    lock=c.read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];inputs=c.read(plan['inputs_path']);started=time.time();report={'schema':1,'producer_pid':os.getpid(),'source_plan_sha256':source,'plan_sha256':c.sha(a.plan),'phase':a.phase,'started_epoch':started,'passed':False,'errors':[],'results':{},'actual_model_graph_qualified':False,'API_qualified':False,'shelf_qualified':False}
    phases=('source40-control','source41-off')if a.phase=='off-equivalence'else('source41-original-fidelity',)
    if a.phase=='original-fidelity':
        require(a.off_receipt is not None,'Actual independently admitted OFF equivalence prerequisite required')
        from validate_source41_model_cli_v2 import finalized_binding
        proof=finalized_binding(a.off_receipt);require(proof['OFF_equivalence_passed']is True,'OFF prerequisite not closed');report['OFF_prerequisite']=proof;report['OFF_prerequisite_root']=str(a.off_receipt.resolve())
    for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled source41 parent interruption')))
    try:
        current=dict(metadata);current['source']=str(intake.SDK/'source')
        tokens=life.tokenizer(out,current,source);require(canonical(tokens)==canonical(inputs),'Fresh current tokenizer differs from admitted independent IDs');c.write(out/'inputs.json',tokens);report['speed_input_join']=life.match_speed_inputs(a.speed_trace,tokens,metadata)
        report['prepared_directory']=str(Path(plan['template_path']).resolve());c.write(out/'prepared.snapshot.json',metadata);c.write(out/'config.snapshot.json',cfg)
        report['per_arm']={}
        for phase in phases:
            arm={};report['per_arm'][phase]=arm
            arm['pre_pages']=intake.guard(shards,out/'guardian',phase+'-pre');arm['pre_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/(phase+'-pre-full4.json'),time.time());require(arm['pre_full4']['passed'],'Fresh complete per-arm publisher pre-scan failed');arm['pre_after_pages']=intake.guard(shards,out/'guardian',phase+'-pre-after')
            if phase==phases[0]:
                report['pre_pages']=arm['pre_pages'];report['pre_after_pages']=arm['pre_after_pages'];report['pre_full4']=arm['pre_full4'];(out/'pre-full4.json').write_bytes((out/(phase+'-pre-full4.json')).read_bytes());report['aggregate_pre_from_actual_arm']=phase
            h=life.health(out,phase+'-pre');life.journal(out,phase+'-pre',started);report[phase+'-pre_health']=h
            try:report['results'][phase]=run_phase(out,phase,metadata,cfg,inputs,source,shards,h)
            finally:
                arm['actual_model_terminal_epoch']=time.time();arm['post_health']=life.health(out,phase+'-post');life.journal(out,phase+'-post',started);after=max(arm['actual_model_terminal_epoch'],arm['post_health']['finished_epoch']);arm['post_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/(phase+'-post-full4.json'),after);arm['post_pages']=intake.guard(shards,out/'guardian',phase+'-post');require(arm['post_full4']['passed'],'Fresh complete per-arm publisher post-scan failed')
        report['actual_model_terminal_epoch']=report['per_arm'][phases[-1]]['actual_model_terminal_epoch']
        if a.phase=='off-equivalence':
            arms=[contract.requests(c.read(out/phase/'requests.json'),c.read(out/phase/'recipe.json')['requests'])for phase in phases];report['OFF_comparison']=contract.equivalence(*arms)
    except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        report.setdefault('actual_model_terminal_epoch',time.time())
        if not report['errors']:
            last=report['per_arm'][phases[-1]];report['post_health']=last['post_health'];report['post_full4']=last['post_full4'];report['post_pages']=last['post_pages'];report['aggregate_post_from_actual_arm']=phases[-1];report['aggregate_post_health_label']=phases[-1]+'-post';(out/'post-full4.json').write_bytes((out/(phases[-1]+'-post-full4.json')).read_bytes())
        else:
            # A fresh failure close is independent, including failed per-arm post checks.
            try:report['post_health']=life.health(out,'post');life.journal(out,'post',started);report['aggregate_post_health_label']='post'
            except BaseException as exc:report['errors'].append('post-health: '+str(exc))
            try:
                after=max(report['actual_model_terminal_epoch'],report.get('post_health',{}).get('finished_epoch',time.time()));report['post_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/'post-full4.json',after);report['post_pages']=intake.guard(shards,out/'guardian','post');require(report['post_full4']['passed'],'Fresh failure-closing complete model bytes failed')
            except BaseException as exc:report['errors'].append('post-source: '+str(exc))
        try:require(closure()==source,'Final current source changed')
        except BaseException as exc:report['errors'].append('final-source: '+str(exc))
        report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(x.relative_to(out)):c.sha(x)for x in sorted(out.rglob('*'))if x.is_file()and x.name!='report.json'};c.write(out/'report.json',report)
    require(report['passed'],'Actual constructor/model phase failed')

if __name__=='__main__':main()
