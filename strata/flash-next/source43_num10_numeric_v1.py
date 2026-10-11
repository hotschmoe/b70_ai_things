"""Literal-prefix1 adaptation of unchanged raw full49 software predicates."""
import ast,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
import native_qsa41_targets_v1 as original
from source43_num10_zero_case_v1 import IDS
def numeric(raw,directory,stages):
    original.require(stages==[(0,48)],'Exactly source43 one-card48 layers required')
    node=next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='numeric')
    # Four is the previous prompt length, exclusively. No dtype, stage or token
    # math changes; literals in other functions are never transformed.
    count=0
    class PromptLength(ast.NodeTransformer):
        def visit_Constant(self,n):
            nonlocal count
            if type(n.value)is int and n.value==4:count+=1;return ast.copy_location(ast.Constant(1),n)
            return n
    node=PromptLength().visit(copy.deepcopy(node));original.require(count==8,'Exact eight old prefix-length literals required')
    ns={**vars(original),'qsa':SimpleNamespace(IDS=tuple(IDS))}
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[source43-prefix1-raw-numeric]','exec'),ns)
    return ns['numeric'](raw,Path(directory),stages)
