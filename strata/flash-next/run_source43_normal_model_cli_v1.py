#!/usr/bin/env python3
"""Fresh source40/source43 mode0 four GEN64 parity; not full model qualification."""
import ast,copy
from pathlib import Path
import run_source41_model_cli_v5 as original
import source43_normal_model_intake_v1 as intake
from serial37_canonical_json_v3 import canonical
ROOT=original.ROOT;HERE=original.HERE;PLAN=HERE/'source43-normal-model-source-plan-v1.json';c=original.c;require=original.require;time=original.time
life=original.life;contract=original.contract;Protocol=original.Protocol
PHASES=('source40-control','source43-off')

def closure():
    for path,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/path)==want,'Fresh43 normal model source changed '+path)
    from admit_postboot_original_qsa_reference_v4 import source_binding
    source_binding();return c.sha(PLAN)

def ports():
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('admission','phase_recipe','run_phase','main')]
    main=next(n for n in nodes if n.name=='main');removed=[n for n in main.body if isinstance(n,ast.If)and ast.unparse(n.test)=="a.phase == 'original-fidelity'"]
    require(len(removed)==1,'Exact obsolete source41 fidelity branch required');main.body.remove(removed[0])
    recipe=next(n for n in nodes if n.name=='phase_recipe');recipe.body.insert(0,ast.parse("require(phase in PHASES,'Only actual40 control or actual43 mode0 allowed')").body[0])
    source=ast.unparse(ast.Module(nodes,type_ignores=[]))
    source=source.replace('admit_source_upload41_v1','admit_source_upload43_v1').replace('qualify_indexer_materialized41_masked_v2','qualify_indexer_materialized43_masked_v2')
    source=source.replace("['source41']","['source43']").replace("['fresh41_upload']","['fresh43_upload']").replace("['fresh41_masked_component']","['fresh43_masked_component']")
    source=source.replace('source41-native-CLI-diagnostic','source43-native-CLI-diagnostic').replace("'source41-off'","'source43-off'")
    source=source.replace("choices=('off-equivalence', 'original-fidelity')","choices=('off-equivalence',)")
    source=source.replace("phases = ('source40-control', 'source43-off') if a.phase == 'off-equivalence' else ('source41-original-fidelity',)","phases = PHASES")
    source=source.replace("'engine_generation': 40 if phase == 'source40-control' else 41","'engine_generation': 40 if phase == 'source40-control' else 43")
    result={**vars(original),**globals()};exec(compile(source,'[source43-normal-CLI-source40-control-port]','exec'),result);return result

def admission(*args,**kwargs):return ports()['admission'](*args,**kwargs)
def phase_recipe(*args,**kwargs):return ports()['phase_recipe'](*args,**kwargs)
def run_phase(*args,**kwargs):return ports()['run_phase'](*args,**kwargs)
def main():return ports()['main']()
if __name__=='__main__':main()
