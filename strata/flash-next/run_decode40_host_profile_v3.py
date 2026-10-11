#!/usr/bin/env python3
"""ROOT-owned native four-request H36 diagnostic; never API qualification."""
import argparse,hashlib,json,math,os,queue,selectors,shlex,signal,subprocess,sys,threading,time
from pathlib import Path

# The CPU tokenizer entry imports only the original metadata worker.
if '--inside-tokenizer' in sys.argv:
    import qualify_api_positive_overlap_cpu_screen_v3 as tokenizer_worker
    tokenizer_worker.inside_tokenizer(Path(sys.argv[-2]),Path(sys.argv[-1]))
    raise SystemExit(0)

import c1_serve_controller_combined_v140_v3 as c
import qualify_c1_serving_combined_v140_v3 as q
import prefill_chunk64_512_control_v2 as prepared_ctrl
import prepare_decode40_host_profile_v1 as recipe
import collect_decode40_host_trace_v1 as collector
import decode40_host_contract_v3 as contract
from native_rms_phase_supervisor_v3 import supervise
ROOT=c.REPO;HERE=ROOT/'strata/flash-next';PLAN=HERE/'decode40-host-actor-source-plan-v3.json'
require=recipe.require
MAX_LINE=1<<20;MAX_PIPE=64<<20


def closure():
    value=c.read(PLAN)
    for path,digest in value['files'].items():require(c.sha(ROOT/path)==digest,'Consumed diagnostic source changed '+path)
    q.source_pins()
    return c.sha(PLAN)


class Pump:
    def __init__(self,stream,path,emit,timed=False):
        self.stream=stream;self.path=Path(path);self.emit=emit;self.timed=timed;self.errors=[];self.eof=False;self.bytes=0
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        try:
            with self.path.open('xb')as out:
                while True:
                    raw=self.stream.readline(MAX_LINE+1)
                    if not raw:self.eof=True;break
                    self.bytes+=len(raw)
                    if len(raw)>MAX_LINE or self.bytes>MAX_PIPE:
                        if not self.errors:self.errors.append('Native pipe line/total quota exceeded')
                        continue  # Continue draining until actual EOF; no blocked writer.
                    if not self.errors:
                        try:
                            ns=time.monotonic_ns();out.write(raw);out.flush();text=raw.decode('ascii').rstrip('\n')
                            if self.timed:self.emit(text,ns)
                            else:self.emit(text)
                        except BaseException as exc:self.errors.append(type(exc).__name__+': '+str(exc))
        except BaseException as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))
            try:
                while self.stream.read(65536):pass
                self.eof=True
            except BaseException as failure:self.errors.append(type(failure).__name__+': '+str(failure))
        finally:self.finished=time.time()
    def finish(self):
        self.thread.join(10)
        while self.thread.is_alive():
            print('H36 retains leases pending actual native pipe EOF',flush=True);self.thread.join(10)
        self.stream.close()
        row={'path':str(self.path),'sha256':c.sha(self.path)if self.path.is_file()else None,'bytes':self.bytes,'eof':self.eof,'retired':not self.thread.is_alive(),'errors':self.errors,'finished_epoch':self.finished}
        row['passed']=row['eof']and row['retired']and not row['errors']
        return row


def inspect_owned(name,cmd,image,label):
    return contract.inspection(c.inspected(name),name,cmd,image,label,name.startswith('b70-h36-tokenizer-'))


def terminal(state):
    require(type(state)is dict,'Actual container terminal was not observed')
    require(state['Running']is False and state['OOMKilled']is False and type(state['ExitCode'])is int and state['ExitCode']==0 and not state.get('Error')and state.get('Status')=='exited'and all(state.get(k)is False for k in('Paused','Restarting','Dead'))and type(state.get('Pid'))is int and state['Pid']==0,'Native normal terminal required')


def retire(name,cmd,image,label,proc,failed):
    # Never kill an outstanding Docker CLI and guess delayed daemon creation canceled.
    owned_id=None;last=None
    while proc.poll()is None:
        if not c.absent(name):
            obj=inspect_owned(name,cmd,image,label);owned_id=obj['Id']
            if failed and obj['State']['Running']:
                subprocess.run(['docker','stop','--time','15',name],check=True,capture_output=True,timeout=30)
        try:proc.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
    if not c.absent(name):
        obj=inspect_owned(name,cmd,image,label)
        if owned_id is not None:require(obj['Id']==owned_id,'Owned container ID changed')
        last=obj;require(not obj['State']['Running'],'Owned diagnostic remained live after CLI terminal')
    if last is None:
        require(c.absent(name),'Owned phase authoritative absence changed')
        return {'container_created_observed':False,'container_absent':True,'CLI_exit_code':proc.returncode,'State':None,'Id':None}
    return last


def safe_retire(out,name,cmd,image,label,proc,failed):
    while True:
        try:return retire(name,cmd,image,label,proc,failed)
        except BaseException as exc:
            c.write(Path(out)/'retirement-failure.json',{'error':type(exc).__name__+': '+str(exc),'container':name,'leases_retained':True,'epoch':time.time()})
            print('H36 retains leases: exact owned terminal inspection unresolved',flush=True);time.sleep(2)


def remove_owned(out,obj,name,cmd,image,label):
    errors=[];removed=False
    while True:
        if c.absent(name):
            return {'removed':True,'container_id':obj['Id'],'errors':errors,'recovery_used':bool(errors),'finished_epoch':time.time()}
        if removed:
            # Successful exact-ID rm followed by uncertain absence: retry only absence.
            time.sleep(2);continue
        try:
            current=inspect_owned(name,cmd,image,label);require(current['Id']==obj['Id']and not current['State']['Running'],'Removal refuses foreign/live container')
            subprocess.run(['docker','rm',obj['Id']],check=True,capture_output=True,timeout=30);removed=True
        except BaseException as exc:
            errors.append(type(exc).__name__+': '+str(exc));c.write(Path(out)/'removal-failure.json',{'errors':errors,'leases_retained':True});time.sleep(2)


def setup_failure_drain(out,name,cmd,image,label,proc,streams,error):
    """No new thread needed after partial Pump construction failure."""
    selector=selectors.DefaultSelector();handles=[]
    try:
        for role,stream in streams:
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ)
            path=Path(out)/(role+'-setup-recovery.log');handle=path.open('xb');handles.append((stream,handle));stream._h36_recovery=handle
        stopped=False
        while selector.get_map()or proc.poll()is None:
            if not stopped and not c.absent(name):
                obj=inspect_owned(name,cmd,image,label)
                if obj['State']['Running']:subprocess.run(['docker','stop','--time','15',name],check=True,capture_output=True,timeout=30)
                stopped=True
            for key,_ in selector.select(.1):
                try:raw=os.read(key.fileobj.fileno(),65536)
                except BlockingIOError:continue
                if not raw:selector.unregister(key.fileobj)
                elif key.fileobj._h36_recovery.tell()+len(raw)<=MAX_PIPE:key.fileobj._h36_recovery.write(raw)
        return {'setup_error':error,'original_CLI_return_code':proc.wait(),'actual_missing_streams_drained':True}
    finally:
        selector.close()
        for stream,handle in handles:handle.close();stream.close()


def command(out,argv,label,timeout):
    c.write(out/(label+'.command.json'),argv);row=supervise(argv,out/(label+'.log'),timeout)
    row['command_sha256']=c.sha(out/(label+'.command.json'));c.write(out/(label+'.receipt.json'),row)
    require(row['passed'],'Actual command failed '+label);return row


def health(out,label):
    begin=time.time();rows=[]
    for name,argv,timeout in [('strict',[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',q.HEALTH],210),('compiled',[str(ROOT/'bin/xpu-collective-health'),'--img',q.HEALTH,'--p2p','0','--timeout','180'],240)]:
        rows.append(command(out,argv,label+'-'+name,timeout))
    value={'passed':True,'started_epoch':begin,'finished_epoch':time.time(),'files':rows,'cards':[0,1]};c.write(out/(label+'-health.json'),value);return value


def journal(out,label,started):
    from native_rms_kernel_journal_v5 import reject_faults
    row=command(out,['journalctl','-k','--since','@'+str(int(started)),'--no-pager'],label+'-journal',30)
    reject_faults((out/(label+'-journal.log')).read_text());return row


def tokenizer(out,m,label):
    source=Path(m['source']);pack=Path(m['pack'])/'tokenizer'
    corpus=recipe.client.prompts();messages=[[{'role':'user','content':corpus[k]}]for k in ('short','prefill_about512')]
    c.write(out/'tokenizer-plan.json',{'corpus_messages':messages})
    files=['vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja']
    before={str(pack/f):c.sha(pack/f)for f in files}
    name,cmd=contract.tokenizer_command(out,m,os.getpid(),label,__file__);image=m['runtime']['image']
    c.write(out/'tokenizer.command.json',cmd)
    stdout=None;stderr=None;failed=True;error=None
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        stdout=Pump(proc.stdout,out/'tokenizer.stdout.log',lambda _:None);stderr=Pump(proc.stderr,out/'tokenizer.stderr.log',lambda _:None)
        require(proc.wait(timeout=120)==0,'Metadata tokenizer failed');failed=False
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        missing=[(role,stream)for role,pump,stream in [('stdout',stdout,proc.stdout),('stderr',stderr,proc.stderr)]if pump is None]
        if missing:setup_failure_drain(out,name,cmd,image,label,proc,missing,error)
        obj=safe_retire(out,name,cmd,image,label,proc,failed);a=stdout.finish()if stdout else {'passed':False,'setup_error':error};b=stderr.finish()if stderr else {'passed':False,'setup_error':error};c.write(out/'tokenizer.terminal.json',obj)
        removal=remove_owned(out,obj,name,cmd,image,label)if obj['Id']is not None else {'removed':True,'container_created_observed':False,'recovery_used':False}
        require(c.absent(name),'Metadata exact owned absence missing')
        c.write(out/'tokenizer.receipt.json',{'error':error,'return_code':proc.returncode,'terminal':obj,'stdout':a,'stderr':b,'removed':True,'removal':removal})
    terminal(obj['State']);require(error is None and a['passed']and b['passed']and not removal['recovery_used'],'Metadata tokenizer attempt failed '+str(error))
    require(not obj['HostConfig'].get('Devices')and not obj['HostConfig'].get('DeviceRequests'),'Metadata no-device recipe required')
    value=c.read(out/'tokenizer-fixtures.json');require(len(value['fixtures'])==2,'Exactly two actual tokenized prompts required')
    require(before=={str(pack/f):c.sha(pack/f)for f in files},'Tokenizer metadata bytes changed')
    require(all(value['tokenizer_file_sha256'][f]==before[str(pack/f)]for f in files),'Actual in-container tokenizer files differ')
    for expected,row in zip(messages,value['fixtures']):require(row['messages']==expected,'Actual template corpus changed')
    require(all(len(r['ids'])+64+8<=2048 for r in value['fixtures']),'Actual prompt/output/slack exceed unchanged context')
    return dict(zip(('short','prefill_about512'),[r['ids']for r in value['fixtures']]))


def match_speed_inputs(path,inputs,metadata):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=64<<20,'Bounded regular original speed trace');raw=path.read_bytes();require(len(raw)<=64<<20,'Bounded original speed metadata trace')
    parent=c.read(path.parent/'speed-parent.json');old=c.read(path.parent/'prepared.json')
    require(parent.get('passed')is True,'Original speed parent final admission required')
    require(old['engine_receipt_sha256']==metadata['engine_receipt_sha256']and old['executable_sha256']==metadata['executable_sha256'],'Actual speed source/ELF differs')
    rows=[json.loads(line)for line in raw.splitlines()];begins=[r for r in rows if r.get('kind')=='engine_begin']
    for ids in inputs.values():require(any(r['submitted_ids']==ids and r.get('rendered_matches_submitted')is True for r in begins),'Independent current tokenizer IDs differ from actual speed engine_begin')
    return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'speed_parent_sha256':c.sha(path.parent/'speed-parent.json'),'speed_prepared_sha256':c.sha(path.parent/'prepared.json'),'independent_tokenizer_matches_actual_speed_inputs':True}


def protocol(stream,lines,requests,interrupted,guard,deadline,on_send=lambda packet:None,on_ready=lambda:None):
    received=[]
    def next_line():
        while True:
            require(not interrupted[0]and time.monotonic()<deadline,'Native interrupt/deadline')
            try:
                row=lines.get(timeout=.2)
                if type(row)is tuple:
                    line,ns=row;require(type(ns)is int and ns>0,'Actual pipe timestamp required');next_line.last_ns=ns;return line
                next_line.last_ns=None;return row
            except queue.Empty:guard()
    while True:
        line=next_line()
        if line.split()and line.split()[0]=='READY':on_ready();break
        require(not line.startswith(('ERR','FATAL')),'Native startup error '+line)
    for req in requests:
        packet='GEN 64 seed=1 fresh=1 '+','.join(map(str,req['ids']))+'\n';on_send(packet);stream.write(packet.encode('ascii'));stream.flush()
        output=[];token_ns=[];done=None;start=time.time()
        while done is None:
            line=next_line()
            require(not line.startswith(('ERR','FATAL')),'Actual native request error '+line)
            if line.startswith('T '):
                fields=line.split();require(len(fields)==2 and fields[1].isdigit(),'Exact actual T format');output.append(int(fields[1]));token_ns.append(next_line.last_ns);require(len(output)<=64,'Native output cap exceeded')
            elif line.startswith('DONE '):
                f=line.split();require(len(f)>=6 and int(f[1])==len(output)and int(f[2])==len(req['ids'])and f[5]in('stop','length')and all(math.isfinite(float(x))and float(x)>=0 for x in f[3:5]),'Actual native DONE/count/finish differs')
                done=line
        received.append({**req,'output_ids':output,'native_pipe_token_ns':token_ns,'done':done,'done_signature':done.split()[:3]+done.split()[5:],'started_epoch':start,'finished_epoch':time.time()})
    on_send('QUIT\n');stream.write(b'QUIT\n');stream.flush();stream.close();return received


def native(out,m,mode,inputs,binding,interrupted,guard,health_finished):
    directory=out/mode.lower();directory.mkdir();(directory/'host-traces').mkdir();cfg=c.read(Path(m['_prepared'])/'server-config.json')
    plan=recipe.build(m,cfg,inputs,binding,mode);c.write(directory/'recipe.json',plan)
    image=m['runtime']['image'];label=binding;name,cmd=contract.native_command(directory,m,plan,os.getpid(),binding)
    c.write(directory/'command.json',cmd);lines=queue.Queue(maxsize=4096);error=None;received=[];failed=True
    require(0<=time.time()-health_finished<=300,'Fresh actual prehealth required before native launch')
    times=(directory/'native.stdout-times.jsonl').open('x');sent=(directory/'native.stdin.jsonl').open('x');seq=[0]
    proc=None;stdout=None;stderr=None;launch_epoch=None
    def timed_emit(line,ns):
        times.write(json.dumps({'seq':seq[0],'monotonic_ns':ns,'line':line})+'\n');times.flush();seq[0]+=1;lines.put_nowait((line,ns))
    try:
        launch_epoch=time.time();proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(8,9))
        stdout=Pump(proc.stdout,directory/'native.stdout.log',timed_emit,True);stderr=Pump(proc.stderr,directory/'native.stderr.log',lambda _:None)
        def checked_guard():
            require(not stdout.errors and not stderr.errors,'Native pump failed during live request');require(proc.poll()is None and not stdout.eof,'Actual native CLI/pipe terminated before READY or DONE');guard()
        if True:  # Already allocated before native Popen; preserve original protocol nesting.
            def retain(packet):sent.write(json.dumps({'epoch':time.time(),'packet':packet})+'\n');sent.flush()
            def startup_gate():
                deadline=time.monotonic()+10
                path=directory/'native.stderr.log'
                while b'MiB of VRAM free with everything loaded'not in path.read_bytes():
                    require(time.monotonic()<deadline and not stderr.errors,'Actual startup memory log unavailable');time.sleep(.05)
                require(path.stat().st_size<=16<<20,'Bounded original startup bytes required')
                startup=directory/'startup';startup.mkdir();(startup/'server-engine.log').write_bytes(path.read_bytes())
                prepared_ctrl.memory_admission(startup,m)
            received=protocol(proc.stdin,lines,plan['requests'],interrupted,checked_guard,time.monotonic()+1800,retain,startup_gate)
        require(proc.wait(timeout=120)==0,'Native CLI failed after QUIT');failed=False
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if proc is None:
            times.close();sent.close();raise RuntimeError('Actual native CLI failed to start; no Popen handle')
        missing=[(role,stream)for role,pump,stream in [('stdout',stdout,proc.stdout),('stderr',stderr,proc.stderr)]if pump is None]
        if missing:setup_failure_drain(directory,name,cmd,image,label,proc,missing,error)
        obj=safe_retire(directory,name,cmd,image,label,proc,failed)
        if not proc.stdin.closed:proc.stdin.close()
        a=stdout.finish()if stdout else {'passed':False,'setup_error':error};b=stderr.finish()if stderr else {'passed':False,'setup_error':error};times.close();sent.close();c.write(directory/'terminal.json',obj)
        removal=remove_owned(directory,obj,name,cmd,image,label)if obj['Id']is not None else {'removed':True,'container_created_observed':False,'recovery_used':False}
        require(c.absent(name),'Native exact owned absence missing')
        c.write(directory/'receipt.json',{'container':name,'launch_epoch':launch_epoch,'command_sha256':c.sha(directory/'command.json'),'error':error,'return_code':proc.returncode,'terminal':obj,'stdout':a,'stderr':b,'removed':True,'removal':removal,'forced_stop':failed,'finished_epoch':time.time()})
    terminal(obj['State']);require(error is None and a['passed']and b['passed']and not removal['recovery_used'],'Native diagnostic failed '+str(error));c.write(directory/'requests.json',received)
    require(len(received)==4 and received[0]['output_ids']==received[3]['output_ids']and received[1]['output_ids']==received[2]['output_ids']and received[0]['done_signature']==received[3]['done_signature']and received[1]['done_signature']==received[2]['done_signature'],'Within-arm repeated native outputs differ')
    if mode=='ON':
        paths=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]))
        value=collector.collect(paths,directory/'native.stderr.log',binding,received,[[0,0,48]]);c.write(directory/'host-coverage.json',value)
    return received


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--speed-trace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--leased',action='store_true');a=ap.parse_args()
    require(__debug__ and not sys.flags.optimize and not os.getenv('PYTHONOPTIMIZE'),'Ordinary Python required');recipe.client.observer_off(os.environ)
    cfg=c.read(a.prepared/'server-config.json');recipe.client.observer_off(cfg['env'])
    if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
    c.leased([0,1]) # Pair reservation permits mandatory compiled-pair health; model itself is pinned0.
    source=closure();m=prepared_ctrl.validate_prepared(a.prepared.resolve());boundary=prepared_ctrl._PORTS['metadata_snapshot'](a.prepared.resolve());prepared_ctrl._PORTS['snapshot_equal'](a.prepared.resolve(),boundary,m);cfg=c.read(a.prepared/'server-config.json');require(m['cards']==[0],'Onecard native host diagnostic only')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'source-plan.snapshot.json').write_bytes(PLAN.read_bytes());(out/'producer.snapshot.py').write_bytes(Path(__file__).read_bytes())
    c.write(out/'prepared-boundary.json',boundary);c.write(out/'prepared.snapshot.json',m);c.write(out/'config.snapshot.json',cfg);m['_prepared']=str(a.prepared.resolve());started=time.time();interrupted=[False]
    for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda *_:interrupted.__setitem__(0,True))
    lock_path=HERE/'model-lock.json';lock=c.read(lock_path);shards=[ROOT/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')]
    report={'schema':1,'scope':'native H36 four-request diagnostic; no APIserver/speed/fullmath qualification','source_plan_sha256':source,'producer_pid':os.getpid(),'producer_sha256':c.sha(__file__),'started_epoch':started,'passed':False,'errors':[],'workload_cards':[0],'reserved_cards':[0,1],'prepared_directory':str(a.prepared.resolve()),'prepared_sha256':c.sha(a.prepared/'prepared.json'),'config_sha256':c.sha(a.prepared/'server-config.json'),'API_model_identity_observed':False,'device_timestamps_observed':False,'physical_expert_traffic_observed':False,'old_C140_qualification_claimed':False}
    last_guard=[0]
    def guard():
        if time.monotonic()-last_guard[0]>2:
            q.upload_v2.guarded_pages(shards,out,'live-'+str(time.time_ns()));last_guard[0]=time.monotonic()
    try:
        report['pre_pages']=q.upload_v2.guarded_pages(shards,out,'pre');report['pre_full4']=q.upload_v2.full_buffered_identity(lock_path,lock,shards,out/'pre-full4.json',time.time());require(report['pre_full4']['passed'],'Pre full4 failed')
        report['pre_after_pages']=q.upload_v2.guarded_pages(shards,out,'pre-after')
        inputs=tokenizer(out,m,source);report['speed_input_join']=match_speed_inputs(a.speed_trace,inputs,m)
        c.write(out/'inputs.json',inputs)
        arms={}
        for mode in('OFF','ON'):
            prepared_ctrl._PORTS['snapshot_equal'](a.prepared.resolve(),boundary,c.read(a.prepared/'prepared.json'))
            report[mode+'-pre_health']=health(out,mode.lower()+'-pre');journal(out,mode.lower()+'-pre',started)
            arms[mode]=native(out,m,mode,inputs,source,interrupted,guard,report[mode+'-pre_health']['finished_epoch'])
        require([{k:r[k]for k in('ids','output_ids','done_signature')}for r in arms['OFF']]==[{k:r[k]for k in('ids','output_ids','done_signature')}for r in arms['ON']],'OFF/ON native IDs or DONE differ')
        report['outputs_matched']=True;report['owned_model_terminal_epoch']=time.time()
    except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        # Each phase retains its actual CLI/EOF/owned removal before returning here.
        report.setdefault('owned_model_terminal_epoch',time.time())
        try:report['post_health']=health(out,'post');journal(out,'post',started)
        except BaseException as exc:report['errors'].append('post-health: '+str(exc))
        try:
            after=max(report['owned_model_terminal_epoch'],report.get('post_health',{}).get('finished_epoch',time.time()))
            report['post_pages']=q.upload_v2.guarded_pages(shards,out,'post');report['post_full4']=q.upload_v2.full_buffered_identity(lock_path,lock,shards,out/'post-full4.json',after);report['post_after_pages']=q.upload_v2.guarded_pages(shards,out,'post-after');require(report['post_full4']['passed'],'Post full4 failed')
            require(closure()==source,'Final source closure changed');final_m=prepared_ctrl.validate_prepared(a.prepared.resolve());prepared_ctrl._PORTS['snapshot_equal'](a.prepared.resolve(),boundary,final_m);require(c.sha(a.prepared/'prepared.json')==report['prepared_sha256']and c.sha(a.prepared/'server-config.json')==report['config_sha256'],'Prepared/config metadata changed')
        except BaseException as exc:report['errors'].append('post-source: '+str(exc))
        report['artifacts']={str(p.relative_to(out)):c.sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='report.json'};report['finished_epoch']=time.time();report['passed']=not report['errors']and report.get('outputs_matched')is True;c.write(out/'report.json',report)
    require(report['passed'],'Native diagnostic failed; originals retained')


if __name__=='__main__':main()
