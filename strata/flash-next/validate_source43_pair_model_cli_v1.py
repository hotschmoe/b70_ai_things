"""Exact actual43 one/pair native stage/full49 admission; full4 always closes."""
import ast,copy
from pathlib import Path
import validate_source43_normal_model_cli_v1 as base
import run_source43_pair_model_cli_v1 as a
from serial37_canonical_json_v3 import canonical
c=a.c;require=a.require;epoch=base.epoch

def ports():
    namespace=base.ports();namespace['a']=a;namespace['memory_admission']=a.memory_admission
    node=copy.deepcopy(next(n for n in ast.parse(Path(base.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='phase_binding'));count=0
    for call in ast.walk(node):
        if isinstance(call,ast.Call)and ast.unparse(call.func)=='a.life.prepared_ctrl.memory_admission':call.func=ast.Name('memory_admission',ast.Load());call.args=[ast.Name('startup_check',ast.Load()),ast.Name('actual_m',ast.Load()),ast.Name('recipe',ast.Load())];count+=1
    require(count==1,'One exact original readonly startup reconstruction call required');exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[paired43-readonly-startup-port]','exec'),namespace)
    original=namespace['_finalized_binding']
    def with_vectors(root):
        root=Path(root);source=a.closure();report_sha=c.sha(root/'report.json');tree=namespace['tree'](root);error=None
        try:
            result=original(root);pairs=[]
            for phase in a.PHASES:
                directory=root/phase;recipe=c.read(directory/'recipe.json');rows=c.read(directory/'requests.json');values=[]
                for row in rows:values.append({'raw':row['raw'],'meta':a.topology.numeric(row['raw'],directory,recipe['stages'])})
                pairs.append(values)
            from layer0_numerical_qualification_v10 import compare_numeric
            result['full49_topology_pairs']=[compare_numeric(left,right)for left,right in zip(*pairs)];require(len(result['full49_topology_pairs'])==4,'Exactly four full49 topology pairs required');return result
        except BaseException as exc:error=exc;raise
        finally:
            try:require(a.closure()==source and c.sha(root/'report.json')==report_sha and namespace['tree'](root)==tree,'Topology source/report/raw artifacts changed during tensor reads')
            except BaseException:
                if error is None:raise
    namespace['_finalized_binding']=with_vectors;return namespace

def finalized_binding(root):
    result=ports()['finalized_binding'](root);result.update(source_generation=43,materialized_mode=0,workload_cards_by_phase={'source43-one':[0],'source43-pair':[0,1]},matched_ctx_prefill_PLE=[2048,64,65536],actual_two_GPU_stage_work_and_full49_required=True,full_model_math_qualified=False,physical_buffer_release_qualified=False,API_qualified=False,shelf_qualified=False,speed_qualified=False);return result
