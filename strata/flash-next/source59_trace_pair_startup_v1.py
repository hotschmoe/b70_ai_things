"""Trace-only startup byte cap port; all original24/24 admission predicates unchanged."""
import ast,copy
from pathlib import Path
import run_source43_pair24_latency_v2 as original

def port():
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='memory_admission'))
 # The existing pair24 software first copies the actual pair-topology function
 # into `node`, then changes only its32/16 topology literals. Add one cap delta
 # to that actual copied AST before its existing transformation/compilation.
 insertion=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='text'for t in n.targets))
 node.body[insertion:insertion]=ast.parse("count=0\nfor item in ast.walk(node):\n if isinstance(item,ast.BinOp)and isinstance(item.op,ast.LShift)and isinstance(item.left,ast.Constant)and type(item.left.value)is int and item.left.value==16 and isinstance(item.right,ast.Constant)and item.right.value==20:\n  item.left.value=128;count+=1\nif count!=1:raise ValueError('Exact one original16MiB startup cap required')").body
 namespace=dict(vars(original));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[UR2-startup-cap-only]','exec'),namespace);return namespace['memory_admission']

def memory_admission(directory,metadata):return port()(Path(directory),metadata)
