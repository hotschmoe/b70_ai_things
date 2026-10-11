"""Fresh43 model identity/intake; old41 software shape is not runtime proof."""
import ast,copy
from pathlib import Path
import source41_model_cli_v2 as software
from source43_sdk_binding_v1 import SDK,binding as sdk_binding,require,sha
from source43_num10_zero_case_v1 import recipe
from admit_source_upload43_v1 import finalized_binding as upload_binding
EXTRA_PAGE=software.EXTRA_PAGE;guard=software.guard;HERE=software.HERE;ROOT=software.ROOT

def recipes(metadata,config,inputs,sdk):
    require(inputs['short'][0]==inputs['prefill_about512'][0]==248045,'Fresh independently tokenized BOS/literal prefix differs')
    return {str(int(enabled)):recipe(metadata,config,'0'*64,enabled)for enabled in(False,True)}

def prepare(template,inputs_path,upload,oracle,pack,masked,current_identity,out):
    node=next(n for n in ast.parse(Path(software.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
    source=ast.unparse(node).replace('source41','source43').replace('fresh41','fresh43').replace('materialized41','materialized43')
    source=source.replace('from run_source43_model_cli_v1 import closure','from run_source43_num10_zero_v3 import closure')
    namespace={**vars(software),**globals()};exec(compile(source,'[source43-model-intake]','exec'),namespace)
    return namespace['prepare'](template,inputs_path,upload,oracle,pack,masked,current_identity,out)
