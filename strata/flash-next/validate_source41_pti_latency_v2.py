#!/usr/bin/env python3
"""Fresh-runtime/model lifecycle admission plus model-observed PTI raw coverage."""
import argparse,ast,copy,math
from pathlib import Path
import validate_source41_host_lpdisabled_v1 as base
import run_source41_pti_latency_v2 as actor
import collect_source41_model_pti_v1 as pti
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require

def ports():
    # Reuse the exact LPdisabled private reader derivation with this actor identity.
    node=copy.deepcopy(next(n for n in ast.parse(Path(base.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='ports'));namespace={**vars(base),'__file__':__file__,'__name__':__name__,'actor':actor};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[PTI-reader-namespace]','exec'),namespace);return namespace['ports']()
_PORTS=ports()

def budget(directory,root):
    value=c.read(directory/'raw-budget.json');require(value['passed']is True and value['errors']==[]and value['thread_retired']is True and type(value['cap_bytes'])is int and value['cap_bytes']==1<<30 and type(value['interval_seconds'])is int and value['interval_seconds']==1,'Original preregistered raw watcher terminal/budget required');require(value['samples'],'Actual raw samples required');peak=0;last=0
    for sample in value['samples']:
        require(type(sample['bytes'])is int and sample['bytes']>=0 and type(sample['monotonic_ns'])is int and sample['monotonic_ns']>=last and type(sample['epoch'])in(int,float)and math.isfinite(sample['epoch']),'Typed original raw sample required');last=sample['monotonic_ns'];peak=max(peak,sample['bytes']);require(sample['bytes']==sum(row['bytes']for row in sample['files'])and all(type(row['bytes'])is int and row['bytes']>=0 and not Path(row['path']).is_absolute()and '..'not in Path(row['path']).parts for row in sample['files']),'Original bounded raw sample file roster differs')
    require(value['peak_observed_bytes']==peak<=1<<30 and type(value['observed_overshoot_bytes'])is int and value['observed_overshoot_bytes']==0,'Actual observed raw overshoot invalid');require(actor.Watcher(root,lambda:None).close()['passed'],'Current raw root exceeds original budget');require(type(value['actual_postwrite_total_bytes'])is int and value['actual_postwrite_total_bytes']<=1<<30 and value['observed_postwrite_overshoot_bytes']==0,'Original postwrite raw close failed');return value

def finalized_binding(root):
    root=Path(root).resolve();report=c.read(root/'report.json');require(type(report['actual_postwrite_total_bytes'])is int and report['actual_postwrite_total_bytes']==__import__('source41_pti_raw_budget_v2').measure(root)['bytes']<=1<<30 and report['observed_postwrite_overshoot_bytes']==0,'Original final all-write raw disk close differs');require(report['phase']=='host-latency','Exact explicit PTI actor scope required');saved=c.read(root/actor.PHASES[0]/'recipe.json')['tool_runtime'];actor.set_runtime(saved['intake_path']);require(canonical(actor._RUNTIME)==canonical(saved),'Fresh profiler runtime differs from actual arm')
    prior=base.original.finalized_binding(report['OFF_prerequisite_root']);require(prior['OFF_equivalence_passed']is True and canonical(prior)==canonical(report['OFF_prerequisite']),'Original actual41 model prerequisite changed');result=_PORTS['finalized_binding'](root);coverages={}
    for phase in actor.PHASES:
        directory=root/phase;recipe=c.read(directory/'recipe.json');terminal=c.read(directory/'result.json')['terminal'];pid=actor.native_pid(terminal,c.read(directory/'command.json'),recipe['PTI_enabled']);require(canonical(c.read(directory/'native-namespace.json'))==canonical({'native_pid':pid,'PTI_enabled':recipe['PTI_enabled'],'exec_branch_source_checked':True}),'Actual profiler/native namespace proof differs');budget(directory,root)
        rows=[r['raw']for r in c.read(directory/'requests.json')];files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));coverage=actor.collector.collect(files,directory/'engine.combined.log',report['source_plan_sha256'],rows,[[0,0,48]],pid);require(canonical(coverage)==canonical(c.read(directory/'host-coverage.json')),'Actual H36 in both PTI arms changed');coverages[phase]=coverage
        if phase==actor.PHASES[0]:require(not list((directory/'unitrace').iterdir()),'Profiler OFF emitted raw tool files')
        else:
            # The primitive reads leaf.stderr only for fatal clock-export text.
            # A declared regular copy of the actual merged pipe is produced by root actor.
            result['PTI_observed_model_coverage']=pti.collect(directory,actor._RUNTIME)
    require(actor.closure()==report['source_plan_sha256']and _PORTS['tree'](root)==report['artifacts']and canonical(actor.tool.binding(saved['intake_path']))==canonical(saved),'Final source/runtime/original raw artifacts changed')
    result.update(host_coverage_both_arms=coverages,fresh_tool_runtime=saved,clock_budget_bytes=512<<20,total_raw_budget_bytes=1<<30,periodic_budget_not_hard_quota=True,physical_idle_qualified=False,model_complete_device_coverage_qualified=False,speed_qualified=False,API_qualified=False,shelf_qualified=False,zero_added_waits=False)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents,'New external profiling admission output required');c.write(a.output,finalized_binding(a.root))
if __name__=='__main__':main()
