"""Private reuse of frozen exec-retirement supervisor, new IPC-only predicate."""
import ast,copy,json,hashlib
from pathlib import Path
import pti_census_owned_exec_v4 as old
import pti_paused_census_exec_v1 as sampled
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

def mount_predicate(obj):
    """Execute the exact original run assertion with this current tool root."""
    node=next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='run')
    calls=[n for n in ast.walk(node)if isinstance(n,ast.Call)and getattr(n.func,'id',None)=='require'and len(n.args)==2 and isinstance(n.args[1],ast.Constant)and n.args[1].value=='Exact readonly helper and actual finalizer host bind required']
    require(len(calls)==1,'Exact original helper/install predicate required');expression=copy.deepcopy(calls[0])
    for n in ast.walk(expression):
        if isinstance(n,ast.Constant)and n.value=='pti_mapped_injection_census_v2.py':n.value='pti_owned_temporal_control_v2.py'
        if isinstance(n,ast.Constant)and n.value==':/pti-census.py:ro':n.value=':/pti-temporal.py:ro'
    eval(compile(ast.fix_missing_locations(ast.Expression(expression)),__file__+'[same-invoked-temporal-mount-predicate]','eval'),{**vars(old),'obj':obj,'HERE':HERE,'EXPECTED_HOST_ROOT':sampled.EXPECTED_HOST_ROOT})

def preflight_saved(directory):
    """Necessary metadata-only refusal; no runtime/model admission authority."""
    directory=Path(directory);require(directory.is_absolute()and directory.resolve()==directory,'Canonical original failed ON directory required')
    paths=[directory/'command.json',directory/'recipe.json',directory/'temporal-before-warmup-owner-before.json',directory.parent/'source-plan.snapshot.json',directory.parent/'report.json',HERE/'pti_owned_temporal_control_v2.py']
    rows={};bounds={}
    for path in paths:
        require(path.is_file()and not path.is_symlink()and path.stat().st_size<=4<<20,'Small original metadata/helper required');before=path.stat();raw=path.read_bytes();after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)and path.read_bytes()==raw,'Consumed metadata changed');rows[path]=raw;bounds[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    from serial37_canonical_json_v3 import unique_object,finite_constant
    def read(path):return json.loads(rows[path],object_pairs_hook=unique_object,parse_constant=finite_constant)
    command_row=read(paths[0]);recipe=read(paths[1]);obj=read(paths[2]);source=read(paths[3]);parent=read(paths[4]);require(parent['passed']is False,'Original failed parent remains failed')
    old.lifecycle.contract.inspection(obj,obj['Name'].lstrip('/'),command_row,recipe['tool_runtime']['image'],parent['source_plan_sha256'],False)
    require(obj['State']['Running']is True and type(obj['State']['Pid'])is int and obj['State']['Pid']>0,'Original saved live owner required');mount_predicate(obj)
    expected=source['files'][str((HERE/'pti_owned_temporal_control_v2.py').relative_to(HERE.parents[1]))];require(bounds[str(paths[5])]['sha256']==expected,'Exact original readonly helper bytes changed')
    require(recipe['tool_runtime']['install_root']==sampled.EXPECTED_HOST_ROOT,'Saved actual runtime and selected helper root differ')
    for path,raw in rows.items():require(path.read_bytes()==raw,'Closing original metadata changed')
    return {'schema':1,'metadata_preflight_only':True,'same_invoked_temporal_mount_predicate_satisfied':True,'actual_install_root':sampled.EXPECTED_HOST_ROOT,'original_parent_passed':False,'runtime_qualified':False,'model_qualified':False,'new_model_payload_read':False,'original_metadata_bindings':bounds}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--saved-on-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(a.output.is_absolute()and a.output.parent.resolve()==a.output.parent and not a.output.exists()and a.saved_on_directory.resolve()not in a.output.parents,'New external metadata-only result required');a.output.write_text(json.dumps(preflight_saved(a.saved_on_directory),indent=2)+'\n')
