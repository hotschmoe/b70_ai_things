"""Fresh43 normal CLI intake; original software recipe is not ABI/math authority."""
import ast,copy
from pathlib import Path
import source41_model_cli_v2 as original
from source43_sdk_binding_v1 import SDK,binding as sdk_binding,require,sha
from admit_source_upload43_v1 import finalized_binding as upload_binding
EXTRA_PAGE=original.EXTRA_PAGE;guard=original.guard;HERE=original.HERE;ROOT=original.ROOT

def namespace():
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('recipes','prepare')]
    source=ast.unparse(ast.Module(nodes,type_ignores=[]))
    source=source.replace('source41','source43').replace('fresh41','fresh43').replace('materialized41','materialized43')
    source=source.replace('from run_source43_model_cli_v1 import closure','from run_source43_normal_model_cli_v1 import closure')
    result={**vars(original),**globals()};exec(compile(source,'[source43-normal-model-intake]','exec'),result);return result

def recipes(*args,**kwargs):return namespace()['recipes'](*args,**kwargs)
def prepare(*args,**kwargs):return namespace()['prepare'](*args,**kwargs)
