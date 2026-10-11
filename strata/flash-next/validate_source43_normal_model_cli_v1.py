"""Current fresh43 mode0 parity; original reader gates and fresh CPU byte close."""
import ast,copy
from pathlib import Path
import validate_source41_model_cli_v5 as original
import run_source43_normal_model_cli_v1 as a
from serial37_canonical_json_v3 import canonical
c=a.c;require=a.require;epoch=original.epoch

def ports():
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('tree','pages','full4','command','phase_binding','_finalized_binding','finalized_binding')]
    final=next(n for n in nodes if n.name=='_finalized_binding');final.body.insert(2,ast.parse("require(report['phase']=='off-equivalence','Normal43 control only; no source41 fidelity view')").body[0])
    branch=next(n for n in final.body if isinstance(n,ast.If)and ast.unparse(n.test)=='off');require(branch.orelse,'Exact old source41 fidelity prerequisite branch required');branch.orelse=[]
    source=ast.unparse(ast.Module(nodes,type_ignores=[])).replace("phases = ('source40-control', 'source41-off') if report['phase'] == 'off-equivalence' else ('source41-original-fidelity',)","phases = a.PHASES")
    source=source.replace("'source41_sdk': plan['source41']","'source43_sdk': plan['source43']")
    result={**vars(original),**globals()};exec(compile(source,'[actual43-normal-public-reader]','exec'),result);return result

def finalized_binding(root):
    result=ports()['finalized_binding'](root);result.update(source_generation=43,materialized_mode=0,source40_runtime_proof_transferred=False,normal43_fresh_constructor_and_bounded_outputs_observed=True,full_model_math_qualified=False,API_qualified=False,shelf_qualified=False);return result
