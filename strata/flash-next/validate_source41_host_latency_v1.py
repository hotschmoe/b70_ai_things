#!/usr/bin/env python3
"""Original V3 current source/health/bytes/EOF gates plus exact H36 recollection."""
import argparse,ast,copy
from pathlib import Path
import validate_source41_model_cli_v3 as original
import run_source41_host_latency_v1 as actor
from serial37_canonical_json_v3 import canonical
require=actor.require;c=actor.c

def ports():
    namespace={**vars(original),'__file__':__file__,'__name__':__name__,'a':actor};nodes=[];counts={'phases':0,'off_scope':0,'comparison':0,'scope_flag':0}
    for node in ast.parse(Path(original.__file__).read_bytes()).body:
        if not isinstance(node,ast.FunctionDef)or node.name=='captured_binding':continue
        node=copy.deepcopy(node)
        if node.name=='_finalized_binding':
            phase_assign=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='phases'for t in n.targets)]
            require(len(phase_assign)==1 and isinstance(phase_assign[0].value,ast.IfExp),'Exact original reader phase roster required');phase_assign[0].value=ast.Tuple([ast.Constant(p)for p in actor.PHASES],ast.Load());counts['phases']+=1
            for n in ast.walk(node):
                if isinstance(n,ast.Constant)and n.value=='off-equivalence':n.value='host-latency';counts['off_scope']+=1
                if isinstance(n,ast.Constant)and n.value=='OFF_comparison':n.value='H36_OFF_ON_comparison';counts['comparison']+=1
                if isinstance(n,ast.Constant)and n.value=='OFF_equivalence_passed':n.value='H36_OFF_ON_equivalence_passed';counts['scope_flag']+=1
        nodes.append(node)
    require(counts=={'phases':1,'off_scope':1,'comparison':1,'scope_flag':2},'Exactly five named original reader scope/roster edits permitted '+str(counts))
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[explicit-V3-host-reader]','exec'),namespace);namespace['derivation_counts']=counts;return namespace
_PORTS=ports()

def finalized_binding(root):
    root=Path(root).resolve();report=c.read(root/'report.json');require(report['phase']=='host-latency','Only explicit H36 host-latency producer accepted')
    # Prior actual source40/source41 model equivalence is a distinct V3 admission.
    prior=original.finalized_binding(report['OFF_prerequisite_root']);require(prior['OFF_equivalence_passed']is True and canonical(prior)==canonical(report['OFF_prerequisite']),'Closed actual source41 OFF model prerequisite changed')
    binding=_PORTS['finalized_binding'](root);require(binding['H36_OFF_ON_equivalence_passed']is True,'Host control outputs/finish differ')
    for phase in actor.PHASES:
        directory=root/phase;files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]))
        if phase==actor.PHASES[0]:require(not list((directory/'host-traces').iterdir()),'Actual H36 OFF exported traces')
        else:
            rows=[r['raw']for r in c.read(directory/'requests.json')];actual=actor.collector.collect(files,directory/'engine.combined.log',report['source_plan_sha256'],rows,[[0,0,48]]);require(canonical(actual)==canonical(c.read(directory/'host-coverage.json')),'Original bounded H36 request/token/span/ownership coverage changed');binding['host_coverage']=actual
    require(actor.closure()==report['source_plan_sha256']and _PORTS['tree'](root)==report['artifacts'],'Final host source/original raw artifacts changed')
    binding.update(observer_export_cost_outside_H36_request_span=True,export_cost_measured=False,logprobs_policy='explicit20; includes LPReadback/LPCPU',graph_queue_identity_scope='SYCL identity; not Level Zero handles',device_clock_observed=False,physical_idle_qualified=False,physical_expert_bytes_observed=False,API_qualified=False,shelf_qualified=False,speed_qualified=False)
    return binding

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external host admission output required');c.write(a.output,finalized_binding(a.root))
if __name__=='__main__':main()
