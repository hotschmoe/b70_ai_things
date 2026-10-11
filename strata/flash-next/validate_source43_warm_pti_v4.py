#!/usr/bin/env python3
"""Original ON runtime gates plus explicitly preserved OFF-component binding."""
import ast,copy
from pathlib import Path
import validate_source43_warm_pti_v3 as original
import run_source43_warm_pti_v4 as actor
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require
NS={**vars(original),'actor':actor,'__file__':__file__}
nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[original-protected-ON-public]','exec'),NS)
control_binding=NS['control_binding'];clock_window=NS['clock_window'];graph_binding=NS['graph_binding']

def finalized_binding(root):
 report=c.read(Path(root)/'report.json');saved=report['OFF_component_prerequisite'];actor.set_baseline(saved['root'],saved['report_sha256']);require(canonical(saved)==canonical(actor.NS['_BASELINE']),'Original completed OFF component changed');result=NS['finalized_binding'](root);result.update(OFF_component_prerequisite=saved,baseline_whole_parent_passed=False,baseline_parent_errors_preserved=saved['original_parent_errors'],OFF_native_rerun=False);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
