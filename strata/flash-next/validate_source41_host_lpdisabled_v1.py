#!/usr/bin/env python3
"""Original V3 current source/health/bytes/EOF gates plus exact H36 recollection."""
import argparse,ast,copy
from pathlib import Path
import validate_source41_model_cli_v3 as original
import run_source41_host_lpdisabled_v1 as actor
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

def finalized_binding(root,lp20_root=None):
    root=Path(root).resolve();report=c.read(root/'report.json');require(report['phase']=='host-latency','Only explicit H36 host-latency producer accepted')
    # Prior actual source40/source41 model equivalence is a distinct V3 admission.
    prior=original.finalized_binding(report['OFF_prerequisite_root']);require(prior['OFF_equivalence_passed']is True and canonical(prior)==canonical(report['OFF_prerequisite']),'Closed actual source41 OFF model prerequisite changed')
    binding=_PORTS['finalized_binding'](root);require(binding['H36_OFF_ON_equivalence_passed']is True,'Host control outputs/finish differ')
    for phase in actor.PHASES:
        directory=root/phase;files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]))
        command=c.read(directory/'command.json');native_pid=actor.collector.native_namespace(c.read(directory/'result.json')['terminal'],command);require(canonical(c.read(directory/'native-namespace.json'))==canonical({'native_pid':native_pid,'scope':'direct exec bash replacement/no-init/private PID namespace','terminal_command_source_recollected':True}),'Original native namespace proof changed')
        if phase==actor.PHASES[0]:require(not list((directory/'host-traces').iterdir()),'Actual H36 OFF exported traces')
        else:
            rows=[r['raw']for r in c.read(directory/'requests.json')];actual=actor.collector.collect(files,directory/'engine.combined.log',report['source_plan_sha256'],rows,[[0,0,48]],native_pid);require(canonical(actual)==canonical(c.read(directory/'host-coverage.json')),'Original bounded H36 request/token/span/ownership coverage changed');binding['host_coverage']=actual
    require(actor.closure()==report['source_plan_sha256']and _PORTS['tree'](root)==report['artifacts'],'Final host source/original raw artifacts changed')
    if lp20_root is not None:
        import validate_source41_host_latency_v2 as lp20
        prior=lp20.finalized_binding(lp20_root);require(canonical(c.read(Path(lp20_root)/'plan.snapshot.json'))==canonical(c.read(root/'plan.snapshot.json')),'LP20/LPdisabled actual SDK/model/intake differs');comparisons=[]
        for phase in actor.PHASES:
            old_recipe=c.read(Path(lp20_root)/phase/'recipe.json');new_recipe=c.read(root/phase/'recipe.json');require(canonical(old_recipe['native_argv'])==canonical(new_recipe['native_argv']),'LP20/LPdisabled actual native arguments differ');old_env=dict(old_recipe['native_env']);new_env=dict(new_recipe['native_env']);old_env.pop('STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256',None);new_env.pop('STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256',None);require(canonical(old_env)==canonical(new_env),'LP20/LPdisabled actual native environment differs except declared trace binding')
            old=c.read(Path(lp20_root)/phase/'requests.json');new=c.read(root/phase/'requests.json')
            def view(rows):return [{'ids':r['raw']['ids'],'output_ids':r['raw']['output_ids'],'done_signature':r['raw']['done'].split()[:3]+r['raw']['done'].split()[5:]}for r in rows]
            require(canonical(view(old))==canonical(view(new)),'Closed LP20/LPdisabled useful IDs/finish differ');comparisons.append({'phase':phase,'matched_IDs_and_finish':True})
        binding['closed_LP20_comparison']={'binding':prior,'comparisons':comparisons,'LP_disabled_does_not_reproduce_LP20_rows':True}
    binding.update(observer_export_cost_outside_H36_request_span=True,export_cost_measured=False,logprobs_policy='omitted; native default -1, LPdisabled',graph_queue_identity_scope='SYCL identity; not Level Zero handles',device_clock_observed=False,physical_idle_qualified=False,physical_expert_bytes_observed=False,API_qualified=False,shelf_qualified=False,speed_qualified=False)
    return binding

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lp20-root',type=Path);a=p.parse_args();require(not a.output.exists(),'New external host admission output required');c.write(a.output,finalized_binding(a.root,a.lp20_root))
if __name__=='__main__':main()
