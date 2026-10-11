"""Original numerical reader gates with explicit new producer/report identity."""
import ast
import hashlib
import time
from pathlib import Path
import native_qsa40_runtime_v9 as d
import validate_native_qsa40_v9 as previous
import owned_layer3_qsa_evidence_v4 as artifact_source
import native_qsa40_split_contract_v1 as split


def route(node,report_name):
    import copy
    node=copy.deepcopy(node);counts=[0,0,0]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='qualify_native_qsa40_v9.py':n.value='qualify_native_qsa40_split_v1.py';counts[0]+=1
        if isinstance(n,ast.Constant) and n.value=='report.json':n.value=report_name;counts[1]+=1
        if isinstance(n,ast.Compare) and ast.unparse(n.left)=="report['schema']" and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value==1:n.comparators[0].value=2;counts[2]+=1
    d.require(counts==[1,3,1], 'Exactly new parent filename/three report paths/one schema guard required')
    return node


def legacy_binding(root,report_name):
    split.no_leases();split.closure()
    d.require(report_name in ('cpu-candidate-report.json','report.json'), 'Exact private candidate/public report route required')
    tree=ast.parse(Path(previous.__file__).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finalized_binding')
    node=route(node,report_name)
    at=ast.parse(Path(artifact_source.__file__).read_text());art=next(n for n in at.body if isinstance(n,ast.FunctionDef)and n.name=='artifacts')
    count=0
    for n in ast.walk(art):
        if isinstance(n,ast.Constant) and n.value=='report.json':n.value=report_name;count+=1
    d.require(count==1,'Exactly one artifact report-file exclusion route required')
    ans=dict(vars(artifact_source));exec(compile(ast.fix_missing_locations(ast.Module(body=[art],type_ignores=[])),__file__,'exec'),ans)
    namespace=dict(vars(previous));namespace.update(__file__=__file__,__name__=__name__,artifacts=ans['artifacts'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    return namespace['finalized_binding'](root)


def closing_hash(root,boundary,output):
    """Actual fresh full4 after ALL CPU/reference payload reads; never stat reuse."""
    split.no_leases();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from source_page_watchdog_v3 import preserve
    from run_source_upload_oracle_full_v2 import full_buffered_identity
    lock_path=d.HERE/'model-lock.json';lock=d.read(lock_path)
    shards=[d.ROOT/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')]
    before=preserve(shards[2],output,'before');proof=full_buffered_identity(lock_path,lock,shards,output/'full4.json',boundary);after=preserve(shards[2],output,'after')
    d.require(proof['passed']is True and before['passed']is True and after['passed']is True and boundary<=proof['started']<=proof['finished'], 'CPU closing source4/pages failed')
    return {'boundary_epoch':boundary,'output':str(output.resolve()),'full4_sha256':d.sha(output/'full4.json'),
            'full4':proof,'before_pages':before,'after_pages':after,'GPU_touched':False,'source_stat_only_reused':False}


def finalized_binding(root,closing_output):
    split.no_leases();root=Path(root).resolve();report=d.read(root/'report.json')
    d.require(not Path(closing_output).resolve().is_relative_to(root), 'Independent CPU closing evidence must not mutate original result tree')
    d.require(report['parent_generation']=='qsa-split-v1'and report['numerical_admission_pending']is False and
              report['hardware_report_sha256']==d.sha(root/'hardware-report.json'), 'Public split qualification requires closed hardware and actual completed CPU phase')
    output=Path(closing_output);output.mkdir(parents=True,exist_ok=False)
    result=None;closing=None;error=None;closing_error=None
    try:
        hardware,_=split.hardware_binding(root)
        split.cpu_closing_binding(root,report,hardware)
        result=legacy_binding(root,'report.json')
        split.cpu_closing_binding(root,report,hardware)
        d.require(d.read(root/'report.json')==report,'Qualified parent changed during independent CPU admission')
    except BaseException as exc:error=exc
    finally:
        # Even failed original/reference reads get an actual fresh closing4.
        try:closing=closing_hash(root,time.time(),output/'source-close')
        except BaseException as exc:closing_error=exc
        if error is not None or closing_error is not None:
            d.write(output/'admission-failure.json',{'original_error':type(error).__name__+': '+str(error)if error else None,
                'closing_error':type(closing_error).__name__+': '+str(closing_error)if closing_error else None,
                'original_result_modified':False,'GPU_touched':False})
    if error is not None:raise error from closing_error
    if closing_error is not None:raise closing_error
    return {'original_binding':result,'fresh_CPU_closing_source':closing,'hardware_health_transferred_as_current':False,
            'GPU_lease_used_by_reader':False,'full_model_math_qualified':False}
