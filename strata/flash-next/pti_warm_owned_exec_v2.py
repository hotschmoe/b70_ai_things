"""Private reuse of frozen exec-retirement supervisor, new IPC-only predicate."""
import ast,copy,json,hashlib
from pathlib import Path
import pti_census_owned_exec_v4 as old
import pti_sampled_census_exec_v1 as sampled
import pti_owned_temporal_control_v2 as helper
from serial37_canonical_json_v3 import canonical
HERE=Path(__file__).resolve().parent;require=helper.require
PHASES={'temporal-before-warmup':'observe-paused','temporal-before-selected':'resume','temporal-after-selected':'observe-resumed'}

def command(identifier,phase,sha,session):
    require(phase in PHASES,'Exact temporal phase required')
    return ['docker','exec',identifier,'/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python','/pti-temporal.py','--session',session,'--action',PHASES[phase],'--expected-tool-sha256',sha,'--output','/results/'+phase+'.json']

def run(directory,name,native_command,image,binding,phase,sha,session):
    node=copy.deepcopy(next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='run'))
    changed=0
    for n in ast.walk(node):
        if isinstance(n,ast.Constant)and n.value=='pti_mapped_injection_census_v2.py':n.value='pti_owned_temporal_control_v2.py';changed+=1
        if isinstance(n,ast.Constant)and n.value==':/pti-census.py:ro':n.value=':/pti-temporal.py:ro';changed+=1
    require(changed==2,'Exact protected exec readonly helper-mount port required')
    predicate=next(n for n in node.body if isinstance(n,ast.Try)and any(isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='value'for t in x.targets)for x in n.body))
    predicate.body=ast.parse("""
require(proc is not None and type(proc.returncode)is int and proc.returncode==0,'Owned temporal metadata exec failed')
receipt_path=directory/(phase+'.json');receipt=json.loads(receipt_path.read_bytes())
require(type(receipt['schema'])is int and receipt['schema']==1 and receipt['passed']is True and receipt['error']is None and receipt['session']==SESSION and receipt['action']==ACTION and receipt['target']['pid']==1 and type(receipt['target']['start_ticks'])is int and receipt['target']['start_ticks']>0 and receipt['GPU_queries_added']is False and receipt['process_memory_read']is False,'Actual original temporal control receipt required')
helper.check_transition(receipt['before'],receipt['after'],ACTION)
value={'passed':not errors,'command':cmd,'container_id':obj['Id'],'host_native_pid':obj['State']['Pid'],'started_epoch':started,'finished_epoch':time.time(),'started_monotonic_ns':started_mono,'finished_monotonic_ns':time.monotonic_ns(),'return_code':proc.returncode,'control':receipt,'control_path':str(receipt_path),'control_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'metadata_stdout_bytes':len(output),'process_memory_bytes_read':False}
""").body
    namespace={**vars(old),'helper':helper,'EXPECTED_HOST_ROOT':sampled.EXPECTED_HOST_ROOT,'SESSION':session,'ACTION':PHASES[phase],'command':lambda identifier,p,s:command(identifier,p,s,session)}
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[original-protected-exec-IPC-only]','exec'),namespace)
    return namespace['run'](directory,name,native_command,image,binding,phase,sha)
