#!/usr/bin/env python3
"""Original current reader plus descriptive small-path comparison."""
import ast,copy,math
from pathlib import Path
import validate_source43_warm_pti_v3 as original
import run_source43_pair24_latency_v1 as actor
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
 report=c.read(Path(root)/'report.json');actor.set_api(report['API_prepared_root'],report['API_prepared_sha256']);node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='finalized_binding'));statement=next(n for n in node.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='nodes'for t in n.targets));at=node.body.index(statement);node.body[at+1:at+1]=ast.parse("nodes=[StageDelta().visit(n)for n in nodes]").body;scope={**NS,'StageDelta':StageDelta};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[protected24-24-final]','exec'),scope);result=scope['finalized_binding'](root);result.pop('H36_OFF_ON_equivalence_passed',None)
 result['pair24_latency_control']=descriptive(root,baseline_root,report_sha,public_path,public_sha)
 report=c.read(Path(root)/'report.json');require(actor.closure()==report['source_plan_sha256']and {str(p.relative_to(root)):c.sha(p)for p in sorted(Path(root).rglob('*'))if p.is_file()and p.name!='report.json'}==report['artifacts'],'Final control source/artifact tree changed');return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--baseline-report-sha256',required=True);p.add_argument('--baseline-public',type=Path,required=True);p.add_argument('--baseline-public-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external output required');c.write(a.output,finalized_binding(a.root,a.baseline_root,a.baseline_report_sha256,a.baseline_public,a.baseline_public_sha256))
