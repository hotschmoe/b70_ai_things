"""Explicit sampled-bracket schema; original operation ticks/Chrome joins retained."""
import ast,copy
from pathlib import Path
import collect_source41_model_pti_v1 as inherited
import pti_bounded_gen4_contract_v1 as scope
from serial37_canonical_json_v3 import canonical
require=inherited.require

def sampled_bridge(rows):
    result=scope.audit_sampled_rows(iter(rows))
    old=inherited.original
    node=copy.deepcopy(next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='bridge_grammar'))
    changed=0
    for n in ast.walk(node):
        if isinstance(n,ast.If)and ast.unparse(n.test)=="row['kind'] == 'instance_normalization'":
            require(len(n.body)==1 and 'Original instance submit pair lacks prior actual query'in ast.unparse(n.body[0]),'Exact per-instance bracket predicate required')
            n.body=[ast.Pass()];changed+=1
    require(changed==1,'Exactly one declared sampled per-instance bracket scope change required')
    namespace=dict(vars(old));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[sampled-bracket-origin-and-thread-joins]','exec'),namespace)
    joined=namespace['bridge_grammar'](rows);joined.update(result);joined['original_instance_submit_pairs_retained']=True
    return joined

def extend_clock(node):
    changed=0
    for n in ast.walk(node):
        if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='keys':
            require(isinstance(n.value,ast.Dict),'Exact original key dictionary required')
            n.value.keys.append(ast.Constant('query_sampling_summary'));n.value.values.append(ast.Set(elts=[ast.Constant(x)for x in('query_site','sample_every','original_query_calls','selected_brackets','completed_brackets','suppressed_brackets')]));changed+=1
    require(changed==1,'Exact summary-only key extension required');return node

def clock_port(properties):
    node=copy.deepcopy(next(n for n in ast.parse(Path(inherited.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='clock_port'))
    statement=next(n for n in node.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and getattr(n.value.func,'id',None)=='exec');at=node.body.index(statement)
    node.body[at:at]=ast.parse("node=extend_clock(node)\nnamespace['bridge_grammar']=sampled_bridge").body
    namespace={**vars(inherited),'extend_clock':extend_clock,'sampled_bridge':sampled_bridge}
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[unchanged-observed-model-count-port]','exec'),namespace)
    return namespace['clock_port'](properties)

node=copy.deepcopy(next(n for n in ast.parse(Path(inherited.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='collect'))
_NAMESPACE={**vars(inherited),'clock_port':clock_port}
exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[unchanged-model-instance-flow-coverage]','exec'),_NAMESPACE)
def collect(directory,runtime):
    result=_NAMESPACE['collect'](directory,runtime)
    result.update(query_diagnostic_brackets_sampled=True,every_instance_submit_pair_bracketed=False,clock_alignment_qualified=False,warm_process_decode_qualified=False)
    return result
