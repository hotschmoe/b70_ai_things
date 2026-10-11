"""Fresh44 model identity/intake; old41 software shape is not runtime proof."""
import ast,copy
from pathlib import Path
import source41_model_cli_v2 as software
from source44_sdk_binding_v1 import SDK,binding as sdk_binding,require,sha
from source44_num38_zero_case_v2 import recipe
from admit_source_upload44_v1 import finalized_binding as upload_binding
EXTRA_PAGE=software.EXTRA_PAGE;guard=software.guard;HERE=software.HERE;ROOT=software.ROOT

def recipes(metadata,config,inputs,sdk):
    require(inputs['short'][0]==inputs['prefill_about512'][0]==248045,'Fresh independently tokenized BOS/literal prefix differs')
    return {str(int(enabled)):recipe(metadata,config,'0'*64,enabled)for enabled in(False,True)}

def prepare(template,inputs_path,upload,oracle,pack,masked,current_identity,out):
    node=next(n for n in ast.parse(Path(software.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
    source=ast.unparse(node).replace('source41','source44').replace('fresh41','fresh44').replace('materialized41','materialized44')
    source=source.replace('from run_source44_model_cli_v1 import closure','from run_source44_num38_zero_v2 import closure')
    require(source.count('OFF actual source40 comparison; ON original fidelity reader')==1,'Exact inherited followup scope required')
    source=source.replace('OFF actual source40 comparison; ON original fidelity reader','both actual44 NUM38 OFF/ON; separate original numerical diagnostic')
    namespace={**vars(software),**globals()};exec(compile(source,'[source44-model-intake]','exec'),namespace)
    return namespace['prepare'](template,inputs_path,upload,oracle,pack,masked,current_identity,out)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in('template','inputs','upload','oracle','pack','masked','current-identity','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.template,a.inputs,a.upload,a.oracle,a.pack,a.masked,a.current_identity,a.output)
