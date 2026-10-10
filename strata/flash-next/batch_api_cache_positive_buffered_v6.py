"""NEW shortwarm source consumer; frozen positive V2 non-input gates retained."""
import argparse,copy
from pathlib import Path
import batch_api_cache_positive_v2 as original
import api_shortwarm_positive_contract_v6 as contract
import generate_api_shortwarm_positive_case_v1 as generator
import run_batch_api_cache_positive_buffered_v6 as api
ROOT=original.ROOT;HERE=original.HERE;c1=original.c1;sha=original.sha;read=original.read;write=original.write;require=original.require
lane_contract=original.lane_contract;expected_stage_ranges=original.expected_stage_ranges;verify_model_identity=original.verify_model_identity
SOURCE_PLAN=contract.SOURCE_PLAN

def source_gate():
 for n,w in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/n)==w,'Shortwarm immutable source changed '+n)

def case_admission(case):
 expected=generator.case_from_fixture(contract.FIXTURE_ROOT,case['port']);require(case==expected,'Exact NEW authentic shortwarm case/metadata required');require(case['genuine_runtime_preparable'] is False and case['new_shortwarm_runtime_successor_required'] is True and case['matched_buffer_only_A_B_input_equivalent'] is False,'Shortwarm source case scope differs')
 return expected

def namespace():
 ns=dict(vars(original));ns.update(__file__=__file__,contract=contract,api=api,SOURCE_PLAN=SOURCE_PLAN)
 # Exact original source body retains all recipe/identity/current source gates;
 # only its declared authentic-case provider changes in this NEW generation.
 source=Path(original.__file__).read_text();start=source.index('def manifest_binding(plan):');end=source.index('\ndef prepare(',start);body=source[start:end]
 old="from generate_api_cache_positive_case_v2 import case_from_fixture\n require(plan['authentic_positive_case']==case_from_fixture(plan['port'])"
 new="from generate_api_shortwarm_positive_case_v1 import case_from_fixture\n require(plan['authentic_positive_case']==case_from_fixture(plan['shortwarm_fixture_root'],plan['port'])"
 require(body.count(old)==1,'Exact authentic-case successor integration failed');exec(compile(body.replace(old,new),str(original.__file__)+'[NEW-shortwarm-V6]','exec'),ns)
 return ns

def manifest_binding(plan):
 require(plan.get('buffered_trace_generation')==3 and plan.get('terminal_association_generation')==2 and plan.get('buffered_wrapper_generation')==6 and plan.get('shortwarm_case_generation')==1 and plan.get('shortwarm_fixture_root')==str(contract.FIXTURE_ROOT),'Explicit NEW buffered shortwarm producer required')
 require(plan['buffered_source_plan_sha256']==sha(SOURCE_PLAN) and plan['buffered_runner_sha256']==sha(Path(api.__file__)) and plan['buffered_tracer_sha256']==sha(HERE/'batch_api_trace_v3.py') and plan['matched_buffer_IO_savings_qualified'] is False and plan['matched_buffer_only_A_B_input_equivalent'] is False,'Shortwarm buffered source/scope differs')
 case_admission(plan['authentic_positive_case']);return namespace()['manifest_binding'](plan)

def prepare(a):
 source_gate();case=case_admission(read(a.spec));require(a.kind=='api' and a.lane=='source35','Exact shortwarm source35 API required')
 ns=namespace();source=Path(original.__file__).read_text();start=source.index('def prepare(a):');end=source.index('\ndef run(',start)
 def admission(plan):
  plan.update(buffered_trace_generation=3,terminal_association_generation=2,buffered_wrapper_generation=6,shortwarm_case_generation=1,shortwarm_fixture_root=str(contract.FIXTURE_ROOT),api_warm_max_new_by_request=case['api_warm_max_new_by_request'],api_max_new_by_request=case['api_max_new_by_request'],buffered_source_plan_sha256=sha(SOURCE_PLAN),buffered_runner_sha256=sha(Path(api.__file__)),buffered_tracer_sha256=sha(HERE/'batch_api_trace_v3.py'),matched_buffer_IO_savings_qualified=False,matched_buffer_only_A_B_input_equivalent=False)
  return manifest_binding(plan)
 # Original prepare computes profile before its admission hook; declare the NEW
 # case profile fields in its local candidate via one checked source insertion.
 old='candidate=dict(case,args=args,env=env);scope=contract.profile(candidate)'
 new="candidate=dict(case,args=args,env=env,shortwarm_fixture_root=str(contract.FIXTURE_ROOT));scope=contract.profile(candidate)"
 body=source[start:end];require(body.count(old)==1,'Exact shortwarm prepare scope integration failed');ns['manifest_binding']=admission;exec(compile(body.replace(old,new),str(original.__file__)+'[NEW-shortwarm-prepare-V6]','exec'),ns);return ns['prepare'](a)

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=api.run(plan,a.output,a.pre_health,bool(plan['diagnostic']));require(result['plan_sha256']==sha(a.output/'plan.snapshot.json'),'Actual shortwarm producer snapshotSHA differs');return 0 if result['collection_and_teardown_passed'] else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--lane',choices=['source35'],required=True);a.add_argument('--kind',choices=['api'],required=True);a.add_argument('--diagnostic',type=int,choices=[0,1],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--spec',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
