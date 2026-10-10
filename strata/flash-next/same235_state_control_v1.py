"""NEW exact same235 serial state control callable, root-owned API lifecycle only."""
import json
from pathlib import Path
import batch_api_cache_positive_buffered_v6 as source
import batch_api_client_same235_state_control_v1 as client
import api_shortwarm_positive_contract_v6 as contract
from generate_api_shortwarm_positive_case_v1 import case_from_fixture
HERE=Path(__file__).parent
PLAN=HERE/'same235-state-control-source-plan-v1.json'
def require(ok,message):
 if not ok:raise ValueError(message)
def source_binding():
 row=source.read(PLAN)
 for n,w in row['files'].items():require(source.sha(source.ROOT/n)==w,'Same235 control source changed '+n)
 return row['files']
def jobs():
 case=case_from_fixture(contract.FIXTURE_ROOT,18339);fixture,_=contract.fixture_binding();rows=[]
 for fresh in (False,True):
  for index in (0,1):rows.append({'control':'fresh'+str(int(fresh))+'-target'+str(index),'target_index':index,'messages':case['messages']['target'][index],'ids':case['api_token_ids']['target'][index],'request_policy':{'strata_fresh':fresh},'max_new':64,'warm_messages':case['messages']['warm'],'warm_max_new':32,'warm_request_policy':{'strata_fresh':True},'pin_present':False,'forced_continuation':False})
 require(all(len(row['ids'])==235 for row in rows),'Exact original235 targetIDs required');return rows

def collect_controls(url,model,positive_source_plan,out,trace_path=None):
 """Root callable inside NEW owned lifecycle, not a standalone qualified parent.
 Both scalar fresh0/fresh1 controls reset with same genuine warm47 cohort before
 each target. Current private API source configuration is not altered.
 """
 from same235_state_control_protocol_v1 import await_warm
 binding=source_binding();source.manifest_binding(positive_source_plan);source.c1.leased([0,1]);out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[]
 require(model=='hotschmoe-dd','Actual permanent primary name required')
 for job in jobs():
  directory=out/job['control'];directory.mkdir();warm=client.cohort(url,model,job['warm_messages'],directory/'warm',max_new=32,request_policy=job['warm_request_policy']);require(warm['client_transport_completed'],'Actual resetwarm transport failed')
  predecessor=await_warm(Path(trace_path) if trace_path is not None else out.parent/'api-native-trace.jsonl',warm,job);target=client.cohort(url,model,[job['messages']],directory/'target',max_new=64,request_policy=job['request_policy']);require(target['client_transport_completed'] and not target['real_client_cancel_requested'],'Actual natural serialtarget transport failed');rows.append({'declared_job':job,'actual_warm_client':warm,'actual_warm_native_predecessor':predecessor,'actual_target_client':target})
 require(source_binding()==binding,'Control source changed during actual API requests');report={'schema':1,'source_binding':binding,'positive_configuration_origin_plan':positive_source_plan,'rows':rows,'serial_state_control_collection_only':True,'native_consumed235_reset_counters_proof_still_required':True,'matched_API_concurrent_fresh_control_still_required':True,'underlying_EOS_cause_established':False,'actual_cached_state_handoff_qualified':False,'full_model_math_qualified':False,'latency_qualified':False,'scope':'same235 serial fresh0/fresh1 aftersamewarm47; NEW owned parent/terminal/source/journal/health/full4 joins required'};source.write(out/'client-controls.json',report);return report
