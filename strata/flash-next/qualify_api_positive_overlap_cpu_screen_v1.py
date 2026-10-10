#!/usr/bin/env python3
"""ROOT-executed, exclusive-RAM CPU diagnostic. Source preparation runs no inference."""
import argparse, ast, hashlib, importlib.util, json, os, re, signal, socket
import subprocess, sys, threading, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CONTAINER_RUNNER = '/harness/strata/flash-next/pilot.py'
PLAN = 'strata/flash-next/api-positive-overlap-cpu-screen-source-plan-v1.json'
TARGETS = {'llama-server','llama-completion','llama-tokenize','llama-debug','test-backend-ops','test-quantize-fns'}
FORBIDDEN = ('libsycl','libze_loader','libur_loader','libcuda','libhip','libvulkan','libopencl','libggml-')

def require(condition, message):
    if not condition: raise ValueError(message)
def read(path): return json.loads(Path(path).read_text())
def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n', encoding='ascii')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module

def dependency_binding(binary):
    result=subprocess.check_output(['/usr/bin/ldd',str(binary)],text=True)
    require('not found' not in result,'Intended-runtime dependency unresolved')
    paths=sorted({str(Path(x).resolve()) for line in result.splitlines() for x in line.split() if x.startswith('/')})
    require(paths and all(not any(b in p.lower() for b in FORBIDDEN) for p in paths),'Non-CPU dependency or empty dependency proof')
    return [{'path':p,'sha256':sha(p),'bytes':Path(p).stat().st_size} for p in paths]

def inside_server(config,execute=True):
    c=read(config); binary=Path(c['binary'])
    require(sha(binary)==c['ELF']['sha256'],'Server ELF changed inside intended runtime')
    actual=dependency_binding(binary)
    require(actual==sorted(c['ELF']['dependencies'],key=lambda x:x['path']),'Runtime libraries differ from all-six supplemental closure')
    require(not list(binary.parent.glob('libggml-*.so*')),'Executable-directory plugins present')
    require(not list(Path.cwd().iterdir()),'Runtime cwd must be empty')
    require(not any(k.startswith(('GGML_','LLAMA_ARG_','ONEAPI_','ZE_','SYCL_')) for k in os.environ),'Backend/argument overrides present')
    write('/results/runtime-binding.json' if execute else '/results/runtime-post-binding.json',{'binary_sha256':sha(binary),'dependencies':actual,'environment':dict(os.environ),'cwd_empty':True,'actual_GPU_touch':False})
    if execute:os.execv(str(binary),[str(binary),*c['argv']])

def original_build_gate(build, supplement, build_root):
    require(build.get('return_code')==0 and build.get('container_removed') is True and
            build.get('container_terminal',{}).get('Running') is False and
            build['container_terminal'].get('ExitCode')==0 and not build['container_terminal'].get('OOMKilled'),
            'Original compile must have successful terminal exit/removal')
    require(build.get('actual_GPU_touch') is False and build.get('actual_model_payload_read') is False,'Build scope differs')
    if build.get('passed') is not True:
        require(build.get('errors')==['AssertionError: '] and build.get('ELFs')=={},'Unknown failed-build case rejected')
        text=(Path(build_root)/'llama-completion.ldd.txt').read_text()
        missing=[x for x in text.splitlines() if 'not found' in x]
        require(len(missing)==1 and re.search(r'libgomp\.so\.1\s*=>\s*not found',missing[0]),'Only documented host libgomp loader failure accepted')
        require(supplement.get('original_build_receipt_passed') is False,'Supplement must explicitly retain failed original verdict')
        side=read(Path(build_root)/'host-loader-failure-binding-v1.json')
        require(side.get('passed') is True and side.get('original_build_receipt_passed') is False and side.get('original_reports_modified') is False and side.get('original_errors')==build['errors'],'Exact failure sidecar verdict differs')
        require(side['original_build_receipt_sha256']==sha(Path(build_root)/'receipt.json') and side['host_ldd_sha256']==sha(Path(build_root)/'llama-completion.ldd.txt') and side['original_bound_logs']==build['log_sha256'] and side['supplemental_all6_sha256']==sha(Path(build_root)/'elf-closure-v1.json'),'Host-loader failure sidecar bindings differ')
    require(supplement.get('passed') is True and set(supplement.get('ELFs',{}))==TARGETS,'Independent all-six intended-runtime closure missing')
    require(supplement.get('source_snapshot_rechecked') is True and supplement.get('actual_model_read') is False and
            supplement.get('actual_GPU_touch') is False and supplement.get('container_removed') is True and
            supplement.get('container_terminal',{}).get('ExitCode')==0 and supplement['container_terminal'].get('Running') is False,
            'Supplement terminal/source/scope proof incomplete')
    require(not supplement.get('container_devices') and supplement.get('container_image_id')==build['command'][build['command'].index('-w')+2], 'Supplement image/device binding differs')
    log=(Path(build_root)/'build.log').read_text()
    require(all(re.search(r'Built target '+re.escape(t)+r'\b',log) for t in TARGETS),'All-six build sentinels missing')
    for name,h in build.get('log_sha256',{}).items(): require(sha(Path(build_root)/name)==h,'Original build log changed')
    require(set(build.get('log_sha256',{}))=={'build.log','configure.log'},'Both original logs required')
    require('Manually-specified variables were not used' not in (Path(build_root)/'configure.log').read_text(),'Unused options rejected')
    for target,row in supplement['ELFs'].items():
        require(Path(row['path']).resolve()==(Path(build_root)/'build/bin'/target).resolve() and sha(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes'],'Supplement ELF changed')
        require(row['dependencies'] and all(not any(x in d['path'].lower() for x in FORBIDDEN) for d in row['dependencies']),'CPU dependencies required')
        require(any('libgomp.so' in d['path'] for d in row['dependencies']),'Intended OpenMP dependency proof required')
    smoke_root=Path(build_root)/'cpu-runtime-smoke-v1'; smoke=read(smoke_root/'receipt.json')
    require(smoke.get('passed') is True and smoke.get('return_code')==0 and smoke.get('removed') is True and smoke.get('terminal',{}).get('ExitCode')==0 and smoke['terminal'].get('Running') is False and not smoke.get('devices') and smoke.get('actual_GPU_touch') is False and smoke.get('actual_model_read') is False,'Actual CPU-only runtime smoke required')
    require(smoke['all6_closure_sha256']==sha(Path(build_root)/'elf-closure-v1.json') and smoke['runtime_log_sha256']==sha(smoke_root/'runtime.log'),'CPU smoke raw bindings differ')
    require(smoke['executed_commands']==[[str(Path(build_root)/'build/bin'/t),*args] for t,args in [('llama-server',['--version']),('llama-completion',['--version']),('test-quantize-fns',[])]],'Actual smoke commands differ')
    return True

def continuation_eligible(response):
    # Current actual CPU API: no stopped_eos/stopped_limit fields are invented.
    return (response['stop'] is True and response['stop_type']=='eos' and
            response['truncated'] is False and 8<=len(response['tokens'])<64 and
            bool(response['content'].strip()))

def selection_binding(cases,candidates):
    require(len(cases)==16,'All sixteen fresh-process observations required before selection')
    joined={}
    for row in cases:
        require(type(row['case'])is int and 0<=row['case']<8 and type(row['repeat'])is int and row['repeat']in (0,1),'Exact all-input/repeat roster required')
        key=(row['case'],row['repeat']);require(key not in joined,'Duplicate CPU screen case')
        require(row['passed']is True,'Every CPU screen lifecycle/measurement must qualify')
        response=row['response'];require(row['continuation_eligible']is continuation_eligible(response),'Saved continuation eligibility differs from actual response')
        joined[key]=row
    eligible={}
    for case in range(8):
        first,second=joined[(case,0)],joined[(case,1)]
        require(first['response']['tokens']==second['response']['tokens'] and first['response']['content']==second['response']['content'],'Fresh output repeat changed')
        eligible[case]=first['continuation_eligible']and second['continuation_eligible']
    selected=None
    if eligible[0]and eligible[1]:
        for candidate in candidates:
            if all(eligible[index+2]for index in candidate['target_indices']):
                selected=candidate['id'];break
    return {'selected_candidate':selected,'continuation_screen_passed':selected is not None,'complete_measurement_rows':16,'registered_quality_qualified':False}

def screen_recipe_binding(plan,authentic_fixture,authentic_binding):
    path=Path(plan['preregistered_screen_plan']['path'])
    require(sha(path)==plan['preregistered_screen_plan']['sha256'],'Preregistered all-input screen plan changed')
    screen=read(path);rows=authentic_fixture['fixtures']['warm']+authentic_fixture['fixtures']['target']
    require(screen['fixture_binding']==authentic_binding and len(screen['requests'])==8,'Actual screen/fixture association changed')
    require(screen['selected_candidate'] is None and screen['actual_CPU_screen_observed'] is False,'Preregistered plan must precede model observations')
    for index,(row,request) in enumerate(zip(rows,screen['requests'])):
        expected={'model':'hotschmoe-dd','prompt':row['ids'],'cache_prompt':False,'return_tokens':True,'stream':False,'temperature':0,'seed':1234,'n_predict':64,'repeat_penalty':1,'samplers':['temperature']}
        require(request['messages']==row['messages'] and request['rendered']==row['rendered'] and request['accepted_input_ids']==row['ids'] and request['request']==expected and request['fresh_process_repeats']==2,'Exact all-input request/seed/repeat changed')
    require(plan['corpus_messages']==[r['messages']for r in rows] and plan['candidate_order']==screen['candidate_order'],'Declared complete corpus order changed')
    return {'path':str(path),'sha256':sha(path),'requests':8,'fresh_process_repeats':2,'sampling_seed':1234,'old_stop_field_predicate_used':False}

def response_gate(response,ids,case,rendered):
    require(response.get('stop') is True and response.get('stop_type') in ('eos','limit') and response.get('truncated') is False,'Complete bounded CPU completion required')
    require(response.get('tokens_evaluated')==len(ids) and response.get('prompt')==rendered,'Actual prompt count/text differs')
    tokens=response.get('tokens');require(isinstance(tokens,list) and 0<len(tokens)<=64 and all(type(x)is int and 0<=x<248320 for x in tokens),'Complete actual bounded output IDs required')
    require(type(response.get('tokens_predicted'))is int and len(tokens)==response['tokens_predicted'],'Actual output count differs')
    require(type(response.get('content'))is str,'Actual response content required')
    settings=response.get('generation_settings',{})
    require(settings.get('temperature')==0 and settings.get('seed')==1234,'Preregistered greedy/seed1234 settings differ')
    require(settings.get('n_predict')==64 and settings.get('repeat_penalty')==1 and settings.get('samplers')==['temperature'] and settings.get('ignore_eos') is False,'Actual sampling/cap settings differ')
    return True

def model_identity(model_root,lock,output):
    rows=[]; result={'started_epoch':time.time(),'passed':False,'rows':rows,'read_mode':'full ordinary buffered original four-shard reads'}; write(output,result)
    for row in lock['files']:
        if not row['path'].startswith('UD-Q4_K_XL/'): continue
        p=Path(model_root)/row['path']; before=p.stat(); digest=sha(p); after=p.stat()
        signature=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
        rows.append({'path':str(p),'sha256':digest,'bytes':after.st_size,'stat_before':signature(before),'stat_after':signature(after),'passed':digest==row['sha256'] and after.st_size==row['size'] and signature(before)==signature(after)}); write(output,result)
    result['finished_epoch']=time.time(); result['passed']=len(rows)==4 and all(x['passed'] for x in rows); write(output,result)
    require(result['passed'],'Original full-four identity failure'); return result

def post_identity(page,guard_path,model_root,lock,output,after_epoch):
    before=page.preserve(guard_path,output,'post-terminal-pre-hash')
    try:result=model_identity(model_root,lock,output/'post-terminal-full-four.json')
    finally:after=page.preserve(guard_path,output,'post-terminal-post-hash')
    require(before['passed'] and after['passed'],'Post-terminal known-page guard failed')
    require(result['passed'] and result['started_epoch']>=after_epoch,'Fresh full-four scan/terminal ordering failed')
    return result

def known_page_target(shards):
    expected=['Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf'%i for i in range(1,5)]
    require(len(shards)==4 and [Path(p).name for p in shards]==expected,'Known-page guard requires exact original ordered four-shard roster')
    return shards[2]

def inside_tokenizer(plan_path,output,decode_input=None):
    plan=read(plan_path);sys.path[:0]=['/native-tools','/native-serve']
    from strata_tokenizer import Tokenizer
    from frontend import ChatTemplate
    import jinja2,regex
    tp=Path('/tokenizer');v=read(tp/'vocab.json');tokens=[None]*len(v)
    for token,i in v.items():tokens[i]=token
    cfg=read(tp/'tokenizer.json');tok=Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),read(tp/'token_type.json'),cfg['pre'],cfg['special_ids']);template=ChatTemplate(tp/'chat_template.jinja')
    fixtures=[]
    for messages in plan['corpus_messages']:
        rendered=template.render(messages,enable_thinking=False);ids=tok.encode(rendered,parse_special=True)
        fixtures.append({'messages':messages,'rendered':rendered,'ids':ids})
    decoded=[]
    if decode_input:
        for row in read(decode_input):
            require(all(type(i) is int and 0<=i<len(tokens) for i in row['tokens']),'Returned ID outside original vocabulary')
            text=tok.decode([i for i in row['tokens'] if tok.token_types[i] not in (3,4)],errors='strict')
            decoded.append({'case':row['case'],'repeat':row['repeat'],'text':text,'matches':text==row['content']})
    modules={m.__name__:{'path':m.__file__,'sha256':sha(m.__file__)} for m in [jinja2,regex]}
    write(output,{'fixtures':fixtures,'decoded_outputs':decoded,'tokenizer_file_sha256':{p.name:sha(p) for p in tp.iterdir() if p.is_file()},'python_runtime':{'executable':sys.executable,'version':sys.version,'modules':modules},'actual_GPU_touch':False,'actual_model_payload_read':False})

def metadata_tokenizer(plan,output,decode_input=None):
    name='b70-cpu-overlap-tokenizer-'+str(os.getpid())+'-'+str(time.time_ns())
    runner=Path(__file__).resolve();source=Path(plan['tokenizer_source']);destination=output/'tokenizer-fixtures.json' if decode_input is None else output/'decoded-output.json'
    cmd=['docker','run','--name',name,'--label','b70.api-overlap.cpu-screen='+plan['runner_sha256'],'--network','none','--read-only','--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','128','--user',str(os.getuid())+':'+str(os.getgid()),'--entrypoint','/usr/bin/env','-v',str(runner)+':'+CONTAINER_RUNNER+':ro','-v',str(output/'source-plan.snapshot.json')+':/pilot-plan.json:ro','-v',str(source/'tools/strata_tokenizer.py')+':/native-tools/strata_tokenizer.py:ro','-v',str(source/'serve/frontend.py')+':/native-serve/frontend.py:ro','-v',plan['tokenizer_pack']+':/tokenizer:ro','-v',str(output)+':/results:rw',plan['tokenizer_image'],'-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python',CONTAINER_RUNNER,'--inside-tokenizer','/pilot-plan.json','--token-output','/results/'+destination.name]
    if decode_input:cmd+=['--decode-input','/results/'+Path(decode_input).name]
    label='prepare' if decode_input is None else 'decode';write(output/('tokenizer-'+label+'-command.json'),cmd)
    try:result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    except BaseException:
        obj=inspect(name);require(owned(obj,name,plan['tokenizer_image'],plan['runner_sha256']),'Metadata cleanup refuses unowned container')
        if obj['State']['Running']:subprocess.run(['docker','stop','--time','10',name],check=True,capture_output=True,timeout=20)
        obj=inspect(name);write(output/('tokenizer-'+label+'-failure-terminal.json'),obj['State'])
        require(not obj['State']['Running'],'Metadata failure container still running')
        subprocess.run(['docker','rm',name],check=True,capture_output=True);raise
    obj=inspect(name)
    require(owned(obj,name,plan['tokenizer_image'],plan['runner_sha256']),'Metadata container ownership differs')
    require(not obj['State']['Running'],'Metadata container still running')
    subprocess.run(['docker','rm',name],check=True,capture_output=True)
    write(output/('tokenizer-'+label+'-receipt.json'),{'return_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'terminal':obj['State'],'container_removed':True,'devices':obj['HostConfig']['Devices'],'image':obj['Image']})
    require(result.returncode==0 and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled'] and not obj['HostConfig']['Devices'] and not obj['HostConfig'].get('DeviceRequests'),'Pinned metadata tokenizer failed')
    result=read(destination);require(result['tokenizer_file_sha256']=={Path(p).name:h for p,h in plan['source_bindings'].items() if Path(p).parent==Path(plan['tokenizer_pack'])},'Metadata export source identity differs')
    return result

def fixture_gate(plan,prepared):
    require(len(plan['corpus_messages'])==8 and len(prepared['fixtures'])==8,'All eight preregistered fixtures required')
    for messages,row in zip(plan['corpus_messages'],prepared['fixtures']):
        require(row['messages']==messages and isinstance(row['rendered'],str) and row['rendered'] and row['ids'] and all(type(i)is int and 0<=i<248320 for i in row['ids']) and len(row['ids'])+64<=2048,'Actual corpus template/IDs/shape differs')
    return True

def memory_snapshot(pid=None):
    host={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1)
        if k in ('MemAvailable','SwapFree','SwapTotal'): host[k]=int(v.split()[0])*1024
    r={'epoch':time.time(),'host':host,'pid':pid}
    if pid:
        status={}
        for line in Path('/proc',str(pid),'status').read_text().splitlines():
            if line.startswith(('VmRSS:','VmSwap:','VmPeak:','VmHWM:')): k,v=line.split(':',1);status[k]=int(v.split()[0])*1024
        r['process']=status
        cg=next(line.split('::',1)[1] for line in Path('/proc',str(pid),'cgroup').read_text().splitlines() if line.startswith('0::'))
        base=Path('/sys/fs/cgroup')/cg.lstrip('/')
        r['cgroup']={name:(base/name).read_text().strip() for name in ['memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.max','memory.events']}
    return r

def memory_gate(sample,baseline,plan):
    require(sample['host']['MemAvailable']>=plan['minimum_live_available_bytes'],'Host memory reserve exhausted')
    require(sample['host']['SwapFree']>=baseline['host']['SwapFree'],'Host swap use increased')
    if 'cgroup' in sample:
        require(int(sample['cgroup']['memory.max'])==plan['memory_cap_bytes'] and int(sample['cgroup']['memory.swap.max'])==0,'Actual cgroup bounds differ')
        require(int(sample['cgroup']['memory.swap.current'])==0 and sample.get('process',{}).get('VmSwap',0)==0,'Inference swap detected')
        events=dict(line.split() for line in sample['cgroup']['memory.events'].splitlines())
        require(all(int(events[k])==0 for k in ('max','oom','oom_kill')),'Cgroup memory pressure/OOM event')
    return True

def api(port,path,body=None,timeout=30):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)

def inspect(name): return json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=20))[0]

def owned(obj,name,image,label):
    return obj.get('Name')=='/'+name and obj.get('Image')==image and obj.get('Config',{}).get('Labels',{}).get('b70.api-overlap.cpu-screen')==label

class OwnedServerStop:
    """Serialize monitor/parent shutdown; never stop a foreign container."""
    def __init__(self,name,image,label):
        self.name,self.image,self.label=name,image,label
        self.lock=threading.Lock();self.terminal=None
    def stop(self):
        with self.lock:
            if self.terminal is not None:return self.terminal
            obj=inspect(self.name)
            require(owned(obj,self.name,self.image,self.label),'Controlled stop refuses unowned container')
            if obj['State']['Running']:
                subprocess.run(['docker','stop','--time','30',self.name],check=True,capture_output=True,timeout=45)
            obj=inspect(self.name)
            require(owned(obj,self.name,self.image,self.label) and not obj['State']['Running'],'Controlled stop ownership/terminal proof failed')
            self.terminal=obj
            return obj

def monitor_memory(pid,stopping,samples,errors,baseline,plan,owned_stop):
    while not stopping.is_set():
        try:
            sample=memory_snapshot(pid);samples.append(sample);memory_gate(sample,baseline,plan)
        except Exception as e:
            errors.append(str(e))
            # The HTTP caller may be blocked. Initiate shutdown in this thread,
            # closing the server connection instead of waiting for its deadline.
            try:owned_stop.stop()
            except Exception as stop_error:errors.append('memory controlled stop: '+str(stop_error))
            stopping.set()
            return
        stopping.wait(1)

def server_command(name,case_dir,plan,build_root,shards,runner,config):
    command=['docker','run','--name',name,'--label','b70.api-overlap.cpu-screen='+plan['runner_sha256'],'--network','host','--memory',str(plan['memory_cap_bytes']),'--memory-swap',str(plan['memory_cap_bytes']),'--cpus','8','--pids-limit','256','--user',str(os.getuid())+':'+str(os.getgid()),'--entrypoint','/usr/bin/env','-v',str(build_root)+':'+str(build_root)+':ro','-v',str(runner)+':'+CONTAINER_RUNNER+':ro','-v',str(case_dir)+':/results:rw','-w','/results/empty']
    for path in shards: command+=['-v',str(path)+':'+str(path)+':ro']
    return command+[plan['image'],'-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=8','/usr/bin/python3',CONTAINER_RUNNER,'--inside-server','/results/launch.json']

def runtime_recipe_gate(obj,command):
    config,host=obj['Config'],obj['HostConfig']
    image=obj['Image'];at=command.index(image)
    require(obj['Image']==image and config['Image']==image and config['Cmd']==command[at+1:] and config['Entrypoint']==['/usr/bin/env'],'Actual CPU image/argv/entrypoint changed')
    require(config['User']==command[command.index('--user')+1] and config['WorkingDir']==command[command.index('-w')+1],'Actual CPU user/cwd differs')
    require(host['NetworkMode']==command[command.index('--network')+1] and host['Memory']==host['MemorySwap']==int(command[command.index('--memory')+1]) and host['NanoCpus']==int(command[command.index('--cpus')+1])*10**9 and host['PidsLimit']==256,'Actual CPU resource recipe changed')
    require(not host.get('Devices') and not host.get('DeviceRequests') and not host.get('Privileged') and not host.get('GroupAdd'),'CPU-only device/privilege profile changed')
    mounts=[]
    for index,value in enumerate(command):
        if value=='-v':
            source,destination,mode=command[index+1].rsplit(':',2);mounts.append((source,destination,mode=='rw','bind'))
    require(sorted(mounts)==sorted((r['Source'],r['Destination'],r['RW'],r['Type'])for r in obj['Mounts']),'Exact CPU build/runner/result/model mounts changed')
    return True

def wrapper_version_preflight(plan,output,build_root,supplement):
    out=output/'server-wrapper-preflight';out.mkdir();(out/'empty').mkdir()
    name='b70-cpu-overlap-wrapper-'+str(os.getpid())+'-'+str(time.time_ns())
    row=supplement['ELFs']['llama-server'];write(out/'launch.json',{'binary':row['path'],'ELF':row,'argv':['--version']})
    recipe=dict(plan,memory_cap_bytes=2<<30)
    cmd=server_command(name,out,recipe,build_root,[],Path(__file__).resolve(),None)
    cmd[cmd.index('--network')+1]='none';cmd[cmd.index('--cpus')+1]='2';write(out/'command.json',cmd)
    try:result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    except BaseException:
        obj=inspect(name);require(owned(obj,name,plan['image'],plan['runner_sha256']),'Wrapper preflight cleanup refuses foreign container')
        if obj['State']['Running']:subprocess.run(['docker','stop','--time','10',name],check=True,capture_output=True,timeout=20)
        obj=inspect(name);write(out/'failure-terminal.json',obj['State']);require(not obj['State']['Running'],'Wrapper preflight still running')
        subprocess.run(['docker','rm',name],check=True,capture_output=True);raise
    obj=inspect(name);require(owned(obj,name,plan['image'],plan['runner_sha256']) and not obj['State']['Running'],'Wrapper preflight owned terminal proof differs')
    runtime_recipe_gate(obj,cmd)
    subprocess.run(['docker','rm',name],check=True,capture_output=True)
    write(out/'receipt.json',{'return_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'inspection':obj,'terminal':obj['State'],'container_removed':True,'devices':obj['HostConfig']['Devices'],'image':obj['Image'],'actual_model_read':False,'actual_inference':False})
    require(result.returncode==0 and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled'] and not obj['HostConfig']['Devices'] and not obj['HostConfig'].get('DeviceRequests') and obj['HostConfig']['NetworkMode']=='none' and obj['HostConfig']['Memory']==obj['HostConfig']['MemorySwap']==2<<30,'Actual CPU wrapper --version preflight failed')
    require(read(out/'runtime-binding.json')['binary_sha256']==row['sha256'],'Wrapper preflight actual ELF binding differs')
    return True

def port_preflight(port):
    # Linux httplib uses REUSEPORT. Check existing listeners explicitly before
    # probing both reuse policies, since REUSEPORT alone permits port sharing.
    for table in ('/proc/net/tcp','/proc/net/tcp6'):
        path=Path(table)
        if table.endswith('tcp6') and not path.exists():continue
        for line in path.read_text().splitlines()[1:]:
            fields=line.split()
            if fields[3]=='0A' and int(fields[1].split(':')[1],16)==port:
                raise OSError(98,'Actual TCP listener already owns requested pilot port')
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
        probe.bind(('127.0.0.1',port))
    # Probe closes before launch. This is advisory; actual owned startup/API
    # identity and normal teardown remain mandatory, not inferred from bind.

def execute_case(case,repeat,messages,native_rendered,ids,a,plan,supplement,shards,baseline):
    out=a.output/('case'+str(case)+'-repeat'+str(repeat));out.mkdir();(out/'empty').mkdir()
    name='b70-cpu-overlap-screen-'+str(os.getpid())+'-'+str(case)+'-'+str(repeat)
    row={'case':case,'repeat':repeat,'passed':False,'container':name,'errors':[],'started_epoch':time.time(),'memory_samples':[]}
    argv=['-m',str(shards[0]),*plan['server_argv'],'--port',str(a.port)]
    binary=Path(supplement['ELFs']['llama-server']['path'])
    write(out/'launch.json',{'binary':str(binary),'ELF':supplement['ELFs']['llama-server'],'argv':argv})
    command=server_command(name,out,plan,a.build_root,shards,Path(__file__).resolve(),None);write(out/'command.json',command)
    process=None; f=None; stop=threading.Event(); monitor=None; memory_errors=[]
    owned_stop=OwnedServerStop(name,plan['image'],plan['runner_sha256'])
    try:
        port_preflight(a.port)
        f=(out/'server.log').open('w');process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,pass_fds=(8,9))
        deadline=time.monotonic()+plan['case_deadline_seconds'];obj=None
        while time.monotonic()<deadline:
            if process.poll() is not None:raise ValueError('Server exited before readiness')
            try:
                obj=inspect(name)
                if obj['State']['Pid']>0:break
            except subprocess.CalledProcessError:pass
            time.sleep(.5)
        require(obj and owned(obj,name,plan['image'],plan['runner_sha256']),'Named container ownership absent')
        runtime_recipe_gate(obj,command);row['launch_inspection']=obj
        hc=obj['HostConfig'];require(not hc.get('Devices') and not hc.get('DeviceRequests') and not hc.get('Privileged') and not hc.get('GroupAdd'),'GPU/privileged grants forbidden')
        require(hc['Memory']==hc['MemorySwap']==plan['memory_cap_bytes'] and hc['PidsLimit']==256 and hc['NetworkMode']=='host','Container memory/network/ownership recipe differs')
        monitor=threading.Thread(target=monitor_memory,args=(obj['State']['Pid'],stop,row['memory_samples'],memory_errors,baseline,plan,owned_stop),daemon=True);monitor.start()
        while time.monotonic()<deadline:
            require(not memory_errors,'Memory gate failed: '+repr(memory_errors))
            if process.poll() is not None:raise ValueError('Server exited during initialization')
            try:
                health=api(a.port,'/health')
                if health.get('status')=='ok':break
            except Exception:pass
            time.sleep(1)
        else:raise ValueError('Server readiness deadline')
        models=api(a.port,'/v1/models');write(out/'models.json',models)
        require(len(models.get('data',[]))==1 and models['data'][0]['id']=='hotschmoe-dd' and set(models['data'][0]['aliases'])=={'hotschmoe-dd',plan['research_alias']},'Actual API primary/alias identity differs')
        props=api(a.port,'/props');write(out/'props.json',props)
        require(props['default_generation_settings']['n_ctx']==2048 and props['total_slots']==1 and props['model_alias']=='hotschmoe-dd' and props['model_path']==str(shards[0]),'Actual model/context/slot properties differ')
        require(hashlib.sha256(props['chat_template'].encode()).hexdigest()==plan['chat_template_sha256'],'Actual GGUF template differs from original export')
        body={'model':'hotschmoe-dd','messages':messages,'chat_template_kwargs':{'enable_thinking':False},'reasoning_format':'none'}
        write(out/'template-request.json',body); rendered=api(a.port,'/apply-template',body)['prompt'];write(out/'rendered.json',{'prompt':rendered})
        require(rendered==native_rendered,'Original exported template and llama.cpp rendered bytes differ')
        token_request={'content':rendered,'add_special':True,'parse_special':True};write(out/'tokenize-request.json',token_request)
        actual_ids=api(a.port,'/tokenize',token_request)['tokens'];write(out/'accepted-input-ids.json',actual_ids)
        require(actual_ids==ids and len(ids)+64<=2048,'Original exported tokenizer and llama.cpp accepted IDs differ')
        request={'model':'hotschmoe-dd','prompt':ids,'cache_prompt':False,'return_tokens':True,'stream':False,'temperature':0,'seed':1234,'n_predict':64,'repeat_penalty':1,'samplers':['temperature']}
        write(out/'completion-request.json',request)
        response=api(a.port,'/completion',request,timeout=max(1,int(deadline-time.monotonic())));write(out/'completion-response.json',response);row['response']=response
        require(not memory_errors,'Memory guard failed during request');response_gate(response,ids,case,rendered)
        subprocess.run(['docker','exec',name,'/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=8','/usr/bin/python3',CONTAINER_RUNNER,'--inside-check','/results/launch.json'],check=True,capture_output=True,timeout=30)
        require(read(out/'runtime-binding.json')['dependencies']==read(out/'runtime-post-binding.json')['dependencies'],'Actual runtime dependencies changed during inference')
        row['response']=response;row['continuation_eligible']=continuation_eligible(response);row['passed']=True
    except BaseException as e:row['errors'].append(type(e).__name__+': '+str(e))
    finally:
        stop.set()
        if monitor:monitor.join(timeout=5)
        if process is not None:
            try:
                obj=inspect(name);require(owned(obj,name,plan['image'],plan['runner_sha256']),'Cleanup refuses unowned container')
                owned_stop.stop()
                process.wait(timeout=30);obj=inspect(name)
                require(owned(obj,name,plan['image'],plan['runner_sha256']),'Terminal cleanup ownership differs');row['terminal']=obj['State'];row['client_return_code']=process.returncode
                runtime_recipe_gate(obj,command);row['terminal_inspection']=obj
                require(not obj['State']['Running'] and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled'] and process.returncode==0,'Server teardown/exit failed')
                subprocess.run(['docker','rm',name],check=True,capture_output=True);row['container_removed']=True
            except Exception as e:
                row['passed']=False;row['forced_cleanup']=True;row['errors'].append('cleanup: '+str(e))
                try:
                    obj=inspect(name)
                    if owned(obj,name,plan['image'],plan['runner_sha256']):
                        subprocess.run(['docker','kill',name],capture_output=True,timeout=30);obj=inspect(name);row['forced_terminal']=obj['State']
                        if not obj['State']['Running']:subprocess.run(['docker','rm',name],capture_output=True,timeout=30)
                except Exception as error:row['errors'].append('preserved cleanup failure: '+str(error))
        if f:f.close()
        row['container_terminal_observed']=process is None or row.get('terminal',{}).get('Running') is False or row.get('forced_terminal',{}).get('Running') is False
        row['finished_epoch']=time.time();row['memory_errors']=memory_errors
        row['passed']=row['passed'] and not memory_errors and row.get('container_removed') is True and bool(row['memory_samples'])
        row['file_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write(out/'case-receipt.json',row)
    return row

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inside-server',type=Path);parser.add_argument('--inside-check',type=Path);parser.add_argument('--inside-tokenizer',type=Path);parser.add_argument('--token-output',type=Path);parser.add_argument('--decode-input',type=Path)
    parser.add_argument('--fixture',type=Path);parser.add_argument('--build-root',type=Path);parser.add_argument('--source-receipt',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--port',type=int,default=28671);parser.add_argument('--leased',action='store_true');parser.add_argument('--wrapper-preflight',action='store_true');a=parser.parse_args()
    if a.inside_server:inside_server(a.inside_server);return
    if a.inside_check:inside_server(a.inside_check,execute=False);return
    if a.inside_tokenizer:inside_tokenizer(a.inside_tokenizer,a.token_output,a.decode_input);return
    signal.signal(signal.SIGTERM,lambda signum,frame: (_ for _ in ()).throw(InterruptedError('Parent terminated')))
    require(a.build_root and a.source_receipt and a.output and a.fixture,'Explicit build/source/authentic fixture/new output required')
    fixture_producer=load('cpu_overlap_fixture',ROOT/'strata/flash-next/produce_api_positive_overlap_corpus_v1.py')
    authentic_fixture,authentic_binding=fixture_producer.finalized_binding(a.fixture)
    if not a.wrapper_preflight and not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
    for card in (() if a.wrapper_preflight else (0,1)):require(os.path.samefile('/proc/self/fd/'+str(8+card),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Exclusive pair lease missing')
    a.build_root=a.build_root.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    plan_bytes=(ROOT/PLAN).read_bytes();plan=json.loads(plan_bytes);plan_sha256=hashlib.sha256(plan_bytes).hexdigest()
    (a.output/'source-plan.snapshot.json').write_bytes(plan_bytes)
    report={'schema':'api-positive-overlap-CPU-screen-v1','producer_pid':os.getpid(),'producer_argv':[str(Path(sys.executable).resolve()),*sys.argv],'producer_interpreter':str(Path(sys.executable).resolve()),'producer_interpreter_sha256':sha(Path(sys.executable).resolve()),'passed':False,'started_epoch':time.time(),'cases':[],'errors':[],'native_bitwise_authority':False,'registered_quality_qualified':False,'full48_math_qualified':False,'speed_qualified':False,'concurrency_qualified':False,'actual_GPU_touch':False,'source_plan_sha256':plan_sha256,'authentic_fixture_binding':authentic_binding,'sampling_seed':1234,'selected_candidate':None,'continuation_screen_passed':False,'old_response_proof_transferred':False}
    page=None;shards=[]
    try:
        require(sha(__file__)==plan['runner_sha256'],'Pilot source changed')
        require(a.build_root==Path(plan['build_root'])and a.source_receipt.resolve()==Path(plan['source_receipt'])and a.fixture.resolve()==Path(plan['fixture_root']),'Explicit pinned CPU build/source/authentic fixture required')
        report['screen_recipe_binding']=screen_recipe_binding(plan,authentic_fixture,authentic_binding)
        for path,h in plan['source_bindings'].items():require(sha(path)==h,'Frozen prerequisite changed: '+path)
        build=read(a.build_root/'receipt.json');supplement=read(a.build_root/'elf-closure-v1.json')
        require(build['plan_sha256']==supplement['plan_sha256']==plan['build_plan_sha256'],'V3 build plan differs')
        require(supplement['build_receipt_sha256']==sha(a.build_root/'receipt.json'),'Original build receipt differs')
        require(supplement['source_receipt_sha256']==build['source_receipt_sha256']==sha(a.source_receipt),'Actual source snapshot receipt differs')
        require(supplement['configure_cache_sha256']==sha(a.build_root/'build/CMakeCache.txt'),'Build cache differs')
        original_build_gate(build,supplement,a.build_root)
        source=read(a.source_receipt);producer=load('cpu_source_prepare',ROOT/'llamacpp/flash-next/prepare_cpu_reference_source_v2.py')
        producer.check_snapshot(source['snapshot'],read(ROOT/'llamacpp/flash-next/cpu-reference-source-build-plan-v3.json'))
        report['build_binding']={'original_build_receipt_passed':build['passed'],'receipt_sha256':sha(a.build_root/'receipt.json'),'supplement_sha256':sha(a.build_root/'elf-closure-v1.json'),'source_receipt_sha256':sha(a.source_receipt),'supplemental_all_six_passed':True,'host_loader_failure_sidecar_sha256':sha(a.build_root/'host-loader-failure-binding-v1.json'),'runtime_smoke_receipt_sha256':sha(a.build_root/'cpu-runtime-smoke-v1/receipt.json')}
        prepared_tokens=metadata_tokenizer(plan,a.output);fixture_gate(plan,prepared_tokens)
        authentic_rows=authentic_fixture['fixtures']['warm']+authentic_fixture['fixtures']['target']
        require(plan['corpus_messages']==[row['messages'] for row in authentic_rows],'New plan and authentic declared corpus differ')
        require(prepared_tokens['fixtures']==[{k:row[k] for k in ('messages','rendered','ids')} for row in authentic_rows],'Fresh metadata and authentic fixture differ')
        wrapper_version_preflight(plan,a.output,a.build_root,supplement)
        report['wrapper_preflight_passed']=True
        if not a.wrapper_preflight:
            baseline=memory_snapshot();require(baseline['host']['MemAvailable']>=plan['minimum_start_available_bytes'],'Exclusive-RAM start reserve inadequate');report['memory_before']=baseline
            page=load('cpu_source_page_guards',ROOT/'strata/flash-next/source_page_watchdog_v3.py');lock=read(ROOT/'strata/flash-next/model-lock.json')
            shards=[Path(plan['model_root'])/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];require(len(shards)==4,'Original four shards required');guard_path=known_page_target(shards)
            report['model_payload_read_attempted']=True
            require(page.preserve(guard_path,a.output,'pre-hash')['passed'],'Known pages failed before full identity')
            model_identity(plan['model_root'],lock,a.output/'pre-full-four.json');require(page.preserve(guard_path,a.output,'pre-inference')['passed'],'Known pages failed before inference')
            fixtures=[(x['messages'],x['rendered'],x['ids']) for x in prepared_tokens['fixtures']]
            for case,(messages,rendered,ids) in enumerate(fixtures):
                for repeat in (0,1):
                    require(page.preserve(guard_path,a.output,'case'+str(case)+'repeat'+str(repeat)+'-pre')['passed'],'Pre-case source page changed')
                    row=execute_case(case,repeat,messages,rendered,ids,a,plan,supplement,shards,baseline)
                    write(a.output/('case'+str(case)+'-repeat'+str(repeat))/'case-receipt.json',row)
                    report['cases'].append(row);write(a.output/'report.json',report)
                    require(page.preserve(guard_path,a.output,'case'+str(case)+'repeat'+str(repeat)+'-post')['passed'],'Post-case source page changed')
                    require(row['passed'],'Functional case/lifecycle/memory failure')
                first,second=report['cases'][-2:]
                require(first['response']['tokens']==second['response']['tokens'] and first['response']['content']==second['response']['content'],'Fresh-process deterministic repeat differs')
            decode_input=a.output/'output-id-requests.json';write(decode_input,[{'case':x['case'],'repeat':x['repeat'],'tokens':x['response']['tokens'],'content':x['response']['content']} for x in report['cases']])
            decoded=metadata_tokenizer(plan,a.output,decode_input)
            require(len(decoded['decoded_outputs'])==16 and all(x['matches'] for x in decoded['decoded_outputs']),'Original tokenizer output IDs/text differ')
            require(decoded['fixtures']==prepared_tokens['fixtures'],'Metadata tokenizer/template changed during pilot')
            report['independent_output_token_decode_passed']=True
            report['all_inputs_repeat_and_lifecycle_passed']=True
    except BaseException as e:report['errors'].append(type(e).__name__+': '+str(e))
    finally:
        report['last_case_terminal_epoch']=max((x.get('finished_epoch',0) for x in report['cases']),default=0)
        if page and shards:
            try:
                require(all(x.get('container_terminal_observed') is True for x in report['cases']),'Unknown/live owned container: defer full scan, preserve failure')
                post_identity(page,guard_path,plan['model_root'],read(ROOT/'strata/flash-next/model-lock.json'),a.output,report['last_case_terminal_epoch'])
                report['post_terminal_full_four_passed']=True
            except BaseException as e:report['errors'].append('post-source: '+str(e))
        try:
            require(sha(__file__)==plan['runner_sha256'],'Pilot runner changed during execution')
            require(sha(report['producer_interpreter'])==report['producer_interpreter_sha256'],'Producer interpreter changed during screen')
            require(screen_recipe_binding(plan,authentic_fixture,authentic_binding)==report['screen_recipe_binding'],'Screen recipe changed during execution')
            require(sha(ROOT/PLAN)==sha(a.output/'source-plan.snapshot.json')==plan_sha256,'Pilot source plan changed during execution')
            report['memory_after']=memory_snapshot();memory_gate(report['memory_after'],report.get('memory_before',report['memory_after']),plan)
            original_build_gate(read(a.build_root/'receipt.json'),read(a.build_root/'elf-closure-v1.json'),a.build_root)
            require(sha(a.build_root/'receipt.json')==report['build_binding']['receipt_sha256'] and sha(a.build_root/'elf-closure-v1.json')==report['build_binding']['supplement_sha256'],'Build receipts changed during pilot')
            require(sha(a.build_root/'host-loader-failure-binding-v1.json')==report['build_binding']['host_loader_failure_sidecar_sha256'] and sha(a.build_root/'cpu-runtime-smoke-v1/receipt.json')==report['build_binding']['runtime_smoke_receipt_sha256'],'Supplemental failure/smoke receipts changed during pilot')
            for path,h in plan['source_bindings'].items():require(sha(path)==h,'Source changed during pilot')
            source=read(a.source_receipt);producer.check_snapshot(source['snapshot'],read(ROOT/'llamacpp/flash-next/cpu-reference-source-build-plan-v3.json'))
        except BaseException as e:report['errors'].append('post-binding: '+str(e))
        try:
            _,current_binding=fixture_producer.finalized_binding(a.fixture)
            require(current_binding==authentic_binding,'Authentic fixture changed during screen')
        except BaseException as e:report['errors'].append('post-fixture: '+str(e))
        report['actual_CPU_inference_observed']=any(x.get('response') for x in report['cases'])
        report['finished_epoch']=time.time();report['passed']=(report.get('wrapper_preflight_passed') is True and not report['errors']) if a.wrapper_preflight else (report.get('all_inputs_repeat_and_lifecycle_passed') is True and report.get('post_terminal_full_four_passed') is True and len(report['cases'])==16 and not report['errors'])
        if report['passed'] and not a.wrapper_preflight:
            try:report.update(selection_binding(report['cases'],plan['candidate_order']))
            except BaseException as e:report['errors'].append('selection: '+str(e));report['passed']=False
        report['artifact_sha256']={str(p.relative_to(a.output)):sha(p)for p in a.output.rglob('*')if p.is_file()and p!=a.output/'report.json'}
        write(a.output/'report.json',report);print(json.dumps({'measurements_passed':report['passed'],'continuation_screen_passed':report['continuation_screen_passed'],'selected_candidate':report['selected_candidate'],'output':str(a.output),'registered_quality_qualified':False}))
    raise SystemExit(0 if report['passed'] else 1)

def artifact_binding(root,report):
    root=Path(root).resolve();files={}
    for path in root.rglob('*'):
        if path.is_file()and path!=root/'report.json':
            require(not path.is_symlink()and path.resolve().is_relative_to(root),'Evidence must remain confined regular files')
            files[str(path.relative_to(root))]=sha(path)
    require(files==report['artifact_sha256'],'Complete CPU evidence tree changed')
    return files

def finalized_binding(root):
    """Root-only read-only recollection, including current known source pages."""
    root=Path(root).resolve();report=read(root/'report.json');plan=read(root/'source-plan.snapshot.json')
    require(report['schema']=='api-positive-overlap-CPU-screen-v1' and report['passed']is True and report['errors']==[] and len(report['cases'])==16,'Complete fresh all16 CPU screen required')
    require(sha(ROOT/PLAN)==sha(root/'source-plan.snapshot.json')==report['source_plan_sha256']and sha(__file__)==plan['runner_sha256'],'Actual current screen source/plan changed')
    for path,h in plan['source_bindings'].items():require(sha(path)==h,'Current CPU prerequisite changed '+path)
    artifact_binding(root,report)
    producer=load('cpu_overlap_readonly_fixture',ROOT/'strata/flash-next/produce_api_positive_overlap_corpus_v1.py')
    fixture,binding=producer.finalized_binding(report['authentic_fixture_binding']['root'])
    require(binding==report['authentic_fixture_binding']and screen_recipe_binding(plan,fixture,binding)==report['screen_recipe_binding'],'Current authentic/preregistered fixture join changed')
    rows=fixture['fixtures']['warm']+fixture['fixtures']['target'];fixture_gate(plan,read(root/'tokenizer-fixtures.json'))
    require(read(root/'tokenizer-fixtures.json')['fixtures']==[{k:r[k]for k in ('messages','rendered','ids')}for r in rows],'Actual fresh metadata differs from authentic fixture')
    lock=read(ROOT/'strata/flash-next/model-lock.json');shards=[Path(plan['model_root'])/row['path']for row in lock['files']if row['path'].startswith('UD-Q4_K_XL/')]
    target=known_page_target(shards);page=load('cpu_overlap_readonly_pages',ROOT/'strata/flash-next/source_page_watchdog_v3.py')
    for phase in ('pre-full-four.json','post-terminal-full-four.json'):
        identity=read(root/phase);require(identity['passed']is True and len(identity['rows'])==4,'Complete original4 source proof missing')
        for actual,expected,path in zip(identity['rows'],[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')],shards):
            require(actual['path']==str(path)and actual['sha256']==expected['sha256']and actual['bytes']==expected['size']and actual['passed']is True and actual['stat_before']==actual['stat_after']==page.signature(path),'Current original role/shard stat identity changed')
    post=read(root/'post-terminal-full-four.json');require(post['started_epoch']>=report['last_case_terminal_epoch'],'PostCPU whole4 chronology changed')
    baseline=report['memory_before'];memory_gate(report['memory_after'],baseline,plan)
    for row in report['cases']:
        case,repeat=row['case'],row['repeat'];directory=root/('case'+str(case)+'-repeat'+str(repeat));raw=rows[case]
        require(read(directory/'case-receipt.json')==row,'Actual percase receipt changed')
        require(read(directory/'accepted-input-ids.json')==raw['ids'],'Actual accepted IDs differ from authentic corpus')
        response=read(directory/'completion-response.json');require(response==row['response'],'Raw response changed');response_gate(response,raw['ids'],case,raw['rendered'])
        name='b70-cpu-overlap-screen-'+str(report['producer_pid'])+'-'+str(case)+'-'+str(repeat)
        require(read(directory/'command.json')==server_command(name,directory,plan,Path(plan['build_root']),shards,Path(__file__).resolve(),None),'Actual full CPU command differs')
        require(row['container']==name and row['container_removed']is True and row['container_terminal_observed']is True and row['client_return_code']==0 and not row['errors']and not row['memory_errors'],'Original owned CPU case lifecycle failed')
        for inspection in ('launch_inspection','terminal_inspection'):
            obj=row[inspection];require(owned(obj,name,plan['image'],plan['runner_sha256']),'Actual owned CPU inspection changed');runtime_recipe_gate(obj,read(directory/'command.json'))
        require(row['terminal']==row['terminal_inspection']['State']and row['terminal']['Running']is False and row['terminal']['ExitCode']==0 and not row['terminal']['OOMKilled'],'Normal owned CPU terminal required')
        require(row['memory_samples'],'Actual memory samples missing')
        for sample in row['memory_samples']:memory_gate(sample,baseline,plan)
        require(read(directory/'runtime-binding.json')['dependencies']==read(directory/'runtime-post-binding.json')['dependencies'],'Runtime library recheck changed')
        pre=read(root/('case'+str(case)+'repeat'+str(repeat)+'-pre-source-pages.json'));after=read(root/('case'+str(case)+'repeat'+str(repeat)+'-post-source-pages.json'))
        require(pre['epoch']<=row['started_epoch']<=row['finished_epoch']<=after['epoch'],'Original percase page/terminal chronology changed')
    for path in root.glob('*-source-pages.json'):
        saved=read(path);require(saved['passed']is True and saved['path']==str(target)and saved['stat_before']==saved['stat_after']==page.signature(target),'Actual preserved pages/currentstat changed')
        require([(r['offset'],r['expected_sha256'])for r in saved['rows']]==list(page.KNOWN_PAGES),'Exact known-page roster changed')
        for row in saved['rows']:
            raw=Path(row['preserved_path']);require(raw.parent.resolve()==root and raw.stat().st_size==4096 and sha(raw)==row['sha256']==row['expected_sha256']and row['passed']is True,'Preserved raw known page changed')
    require(page.guard(target)['passed'],'Current known pages changed')
    selection=selection_binding(report['cases'],plan['candidate_order'])
    for key,value in selection.items():require(report[key]==value,'Final all-input selection changed')
    original_build_gate(read(plan['build_root']+'/receipt.json'),read(plan['build_root']+'/elf-closure-v1.json'),plan['build_root'])
    artifact_binding(root,report)
    return {'root':str(root),'report_sha256':sha(root/'report.json'),'source_plan_sha256':report['source_plan_sha256'],'fixture_binding':binding,**selection,'actual_GPU_positive_overlap_observed':False,'actual_full_cache_qualified':False}

if __name__=='__main__':main()
