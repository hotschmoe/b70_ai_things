"""Public exact43 NUM10 diagnostic; fresh full4 closes all CPU capture reads."""
import ast,copy,re
from pathlib import Path
import validate_source41_model_cli_v5 as original
import run_source43_num10_zero_v3 as a
from serial37_canonical_json_v3 import canonical
c=a.c;require=a.require;epoch=original.epoch

def derived():
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('tree','pages','full4','command','phase_binding','_finalized_binding','finalized_binding')]
    final=next(n for n in nodes if n.name=='_finalized_binding')
    branch=next(n for n in final.body if isinstance(n,ast.If)and ast.unparse(n.test)=='off')
    require(branch.orelse,'Exact old cross-engine OFF prerequisite branch required');branch.orelse=[]
    text=ast.unparse(ast.Module(nodes,type_ignores=[]))
    text=text.replace("phases = ('source40-control', 'source41-off') if report['phase'] == 'off-equivalence' else ('source41-original-fidelity',)","phases = ('source43-num10-OFF', 'source43-num10-ON')")
    text=text.replace("report['phase'] == 'off-equivalence'","report['phase'] == 'num10-equivalence'").replace("'OFF_comparison'","'NUM10_observer_comparison'").replace("'source41_sdk': plan['source41']","'source43_sdk': plan['source43']").replace("'OFF_equivalence_passed': off","'NUM10_observer_equivalence_passed': off")
    namespace={**vars(original),**globals()};exec(compile(text,'[source43-NUM10-exact-reader-port]','exec'),namespace)
    original_final=namespace['_finalized_binding']
    def with_actual_raw(root):
        root=Path(root);source=a.closure();report_sha=c.sha(root/'report.json');roster=namespace['tree'](root);error=None;result=None
        try:return collect_actual(root)
        except BaseException as exc:error=exc;raise
        finally:
            try:
                require(c.sha(root/'report.json')==report_sha and namespace['tree'](root)==roster and a.closure()==source,'Closed source/report/full artifact roster changed during actual raw collection')
            except BaseException:
                if error is None:raise
    def collect_actual(root):
        result=original_final(root);root=Path(root)
        from source43_num10_numeric_v1 import numeric
        report=c.read(root/'report.json');pairs=[];observed=[]
        for phase in('source43-num10-OFF','source43-num10-ON'):
            directory=root/phase;rows=c.read(directory/'requests.json');numeric_rows=[]
            for row in rows:
                meta=numeric(row['raw'],directory,[(0,48)]);numeric_rows.append({'raw':row['raw'],'meta':meta})
                if phase.endswith('-ON'):
                    layer=a.case.collect(row['raw'],directory,report['source_plan_sha256']);observed.append({'ordinal':row['ordinal'],'raw':row['raw'],'layer0':layer})
            pairs.append(numeric_rows)
        from layer0_numerical_qualification_v10 import compare_numeric
        result['actual_full49_observer_pairs']=[compare_numeric(left,right)for left,right in zip(*pairs)]
        require(len(observed)==4,'All four actual zero-state layer0 frames required');result['captured_requests']=observed
        return result
    namespace['_finalized_binding']=with_actual_raw
    return namespace

def finalized_binding(root):return derived()['finalized_binding'](root)
def captured_binding(root):
    root=Path(root).resolve();admitted=finalized_binding(root);report=c.read(root/'report.json');directory=root/'source43-num10-ON';recipe=c.read(directory/'recipe.json');rows=admitted.pop('captured_requests')
    pids={row['layer0']['frame']['pid']for row in rows};require(len(pids)==1 and all(type(p)is int and p>0 for p in pids),'Unique actual frame owner PID required')
    result={'source_generation':43,'capture_root':str(directory),'plan_sha256':report['plan_sha256'],'materialized_mode':recipe['materialized_mode'],'NUM10_enabled':recipe['NUM10_enabled'],'QSA3_enabled':False,'observer_enabled':recipe['observer_enabled'],'binding_sha256':report['source_plan_sha256'],'engine_pid':pids.pop(),'source43_sdk':admitted['source43_sdk'],'requests':rows,'public_actor_admission':admitted,'full_model_math_qualified':False,'API_qualified':False}
    for role in('command','result','phase_receipt'):
        path=directory/(role+'.json');result[role+'_path']=str(path);result[role+'_sha256']=c.sha(path)
    return result
