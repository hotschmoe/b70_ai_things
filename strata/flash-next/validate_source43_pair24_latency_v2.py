#!/usr/bin/env python3
"""Original current reader plus descriptive small-path comparison."""
import ast,copy,math
from pathlib import Path
import validate_source43_warm_pti_v3 as original
import run_source43_pair24_latency_v2 as actor
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require
NS={**vars(original),'actor':actor,'__file__':__file__}
nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
class StageDelta(ast.NodeTransformer):
 def visit_List(self,n):
  if ast.unparse(n)=='[[0, 0, 48]]':return ast.parse('actor.STAGES',mode='eval').body
  return self.generic_visit(n)

exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[unchanged-protected-single-arm-reader]','exec'),NS)
graph_binding=NS['graph_binding'];control_binding=NS['control_binding'];clock_window=NS['clock_window']

def descriptive(root,baseline_root,report_sha,public_path,public_sha):
 root=Path(root);baseline_root=Path(baseline_root);public_path=Path(public_path)
 require(c.sha(baseline_root/'report.json')==report_sha and c.sha(public_path)==public_sha,'Exact completed baseline public/report bytes required')
 report=c.read(baseline_root/'report.json');public=c.read(public_path)
 import validate_source43_warm_off_component_v1 as component
 require(report['passed']is True and report['errors']==[]and public['passed']is True and public['report_sha256']==report_sha and public['root']==str(baseline_root),'Exact closed shortread2 single-card baseline required');actor.closure()
 phase=actor.PHASES[0];left=baseline_root/phase;right=root/phase
 for path in(left/'requests.json',left/'recipe.json',baseline_root/'config.snapshot.json',*sorted((left/'host-traces').glob('host-trace36-*.jsonl'))):
  require(not path.is_symlink()and c.sha(path)==report['artifacts'][str(path.relative_to(baseline_root))],'Original consumed baseline artifact changed')
 require(canonical(c.read(baseline_root/'config.snapshot.json'))==canonical(c.read(root/'config.snapshot.json')),'Same actual model config required')
 before=c.read(left/'recipe.json');after=c.read(right/'recipe.json')
 left_env=dict(before['native_env']);right_env=dict(after['native_env']);key='STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256';require(left_env.pop(key)==report['source_plan_sha256']and right_env.pop(key)==c.read(root/'report.json')['source_plan_sha256'],'Actual observer provenance source identities required')
 require(before['native_argv']+['--layer-split','24','--split-device','1','--trim-stage-weights']==after['native_argv']and canonical(before['requests'])==canonical(after['requests']),'Only balanced24/24 native topology flags allowed');require(left_env.pop('ZE_AFFINITY_MASK')=='0'and right_env.pop('ZE_AFFINITY_MASK')=='0,1'and canonical(left_env)==canonical(right_env),'Only paired device mask env delta allowed')
 import source41_host_trace_contract_v2 as h36
 paths=sorted((right/'host-traces').glob('host-trace36-*.jsonl'));require(len(paths)==2,'Warmup/selected host files required');rows,_=h36.typed_rows(paths[-1],1);events=rows[1:-1]
 prompt=[r for r in events if r['kind']==h36.K['PromptSpan']];require(len(prompt)==2,'Single prompt span required');begin,end=prompt[0]['ns'],prompt[1]['ns']
 require(not any(r['kind']==h36.K['Verify']and begin<=r['ns']<=end for r in events),'Prompt still ran verify windows')
 requests=c.read(right/'requests.json');baseline=c.read(left/'requests.json')
 # Use source-bound request views rather than assuming the raw DONE object schema.
 av=actor.contract.requests(requests,after['requests']);bv=actor.contract.requests(baseline,before['requests'])
 pp=[line for r in requests for line in r['raw']['stdout']if line.startswith('PP ')]
 require(pp and all(line.split()[1:3]==['53','54']for line in pp),'Actual prompt progress must cover53 of54 rows')
 from analyze_source41_h36_latency_v2 import metrics
 baseline_trace=sorted((left/'host-traces').glob('host-trace36-*.jsonl'))[-1];baseline_rows,_=h36.typed_rows(baseline_trace,1)
 observed={'baseline':metrics(baseline_rows),'pair24':metrics(rows)}
 return {'observed_host_metrics':observed,'nested_host_intervals_are_not_additive':True,'baseline_report_sha256':report_sha,'baseline_public_sha256':public_sha,'actual_prompt_verify_windows':0,'source_selected_prefill_rows':53,'source_selected_prefill_chunks':1,'source_shape_is_not_device_kernel_shape':True,'original_PP_lines':pp,'baseline_request_views':bv,'control_request_views':av,'observed_IDS_finish_equal_to_baseline':canonical(av)==canonical(bv),'numeric_head_residual_observed':False,'numeric_head_residual_differences':None,'tolerance_fitting_performed':False,'speed_promotion_qualified':False}

def finalized_binding(root,baseline_root,report_sha,public_path,public_sha):
 root=Path(root);report=c.read(root/'report.json');actor.set_api(report['API_prepared_root'],report['API_prepared_sha256']);actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks']);recipe=c.read(root/actor.PHASES[0]/'recipe.json');actor.set_runtime(recipe['tool_runtime']['intake_path'],recipe['tool_runtime']['exact_intake_sha256']);require(recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['host_trace_enabled']is False,'Actual latency-only traceOFF required')
 import validate_source43_bounded_pti_v1 as source_reader
 ns={**vars(source_reader),'actor':actor};node=copy.deepcopy(next(n for n in ast.parse(Path(source_reader.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='ports'));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[unchanged-native-constructor-public]','exec'),ns);helpers=ns['ports']();result=helpers['finalized_binding'](root)
 phase=actor.PHASES[0];directory=root/phase;require(not list((directory/'host-traces').glob('*')),'Disabled H36 cannot emit host traces');rows=c.read(directory/'requests.json');views=actor.contract.requests(rows,recipe['requests']);require(len(views)==2 and canonical(views[0])==canonical(views[1]),'Actual two fresh output repeats must agree')
 native=[]
 for row in rows:
  raw=row['raw'];require(raw['stdout'][-1]==raw['done']and raw['done'].startswith('DONE ')and len(raw['stdout_seconds'])==len(raw['stdout']),'Original native DONE/root wall roster required');native.append({'ordinal':row['ordinal'],'native_DONE':raw['done'],'stdout_seconds':raw['stdout_seconds'],'output_ids':raw['output_ids'],'scope':'native DONE and root-observed relative stdout stamps; not H36 or device timing'})
 require(c.sha(Path(baseline_root)/'report.json')==report_sha and c.sha(public_path)==public_sha,'Exact closed single baseline bytes required');before=c.read(Path(baseline_root)/'report.json');public=c.read(public_path);require(before['passed']is True and before['errors']==[]and public['passed']is True,'Closed default8 single baseline required');left=c.read(Path(baseline_root)/phase/'recipe.json');require(left['native_argv']+['--layer-split','24','--split-device','1','--trim-stage-weights']==recipe['native_argv']and canonical(left['requests'])==canonical(recipe['requests']),'Only native balanced topology flag delta allowed')
 left_env=dict(left['native_env']);right_env=dict(recipe['native_env']);key='STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256';require(left_env.pop(key)==before['source_plan_sha256']and right_env.pop(key)==report['source_plan_sha256'],'Actual source nonces differ');require(left_env.pop('ZE_AFFINITY_MASK')=='0'and right_env.pop('ZE_AFFINITY_MASK')=='0,1'and left_env.pop('STRATA_CRITICAL_PATH_TRACE')=='1'and right_env.pop('STRATA_CRITICAL_PATH_TRACE')=='0'and canonical(left_env)==canonical(right_env),'Only paired mask and explicit trace disable env deltas')
 result.update(native_observations=native,baseline_report_sha256=report_sha,baseline_public_sha256=public_sha,H36_coverage_qualified=False,graph_coverage_qualified=False,host_critical_path_coverage_qualified=False,device_coverage_qualified=False,full49_topology_qualified=False,physical_idle_qualified=False,speed_promotion_qualified=False)
 require(actor.closure()==report['source_plan_sha256']and {str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'}==report['artifacts'],'Final current source/raw tree changed');return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--baseline-report-sha256',required=True);p.add_argument('--baseline-public',type=Path,required=True);p.add_argument('--baseline-public-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external output required');c.write(a.output,finalized_binding(a.root,a.baseline_root,a.baseline_report_sha256,a.baseline_public,a.baseline_public_sha256))
