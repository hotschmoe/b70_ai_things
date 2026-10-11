"""ROOT ONLY source-derived shared actor with distinct live identity ingress."""
import ast,json,hashlib
from pathlib import Path
import run_full_cache_shared_runtime_v9 as original
import live_cache_identity_controller_v2 as ctrl

def owner_packet(plan,out):
 prepared=ctrl.read(Path(plan['prepared'])/'prepared.json');artifact=ctrl.read(Path(plan['prepared'])/'artifact-identity.json');manifest=ctrl.read(out/'artifact-identity.json');host=[]
 host.append((out/'artifact-identity.json','/results/artifact-identity.json'))
 source=Path(prepared['engine_receipt']).parent/'source'
 for relative in ('serve/server.py','serve/frontend.py','serve/artifact_identity.py','serve/batch_request_identity.py','tools/strata_tokenizer.py','tools/gguf_reader.py'):host.append((source/relative,'/src/'+relative))
 for name in artifact['tokenizer_files']:host.append((Path(plan['pack'])/'tokenizer'/name,'/pack/tokenizer/'+name))
 watched=[{'path':target,'bytes':p.stat().st_size,'sha256':ctrl.sha(p)}for p,target in host]
 value={'namespace':plan['live_identity_namespace'],'loaded_artifact_identity_sha256':ctrl.sha(out/'artifact-identity.json'),'watched_files':watched,'weight_file_bytes_monitored_per_prepare':False,'actual_load_binding':{'prepared_sha256':plan['prepared_sha256'],'baseline_binding':plan['baseline_binding'],'model_identity_receipt':plan['model_identity'],'native_pack_receipt_sha256':prepared['pack_receipt_sha256']},'new_guard_source':plan['live_identity_source'],'original_native_cache_key_qualified':False};ctrl.write(out/'live-identity-owner.json',value);return value

def adapted_source():
 source=Path(original.__file__).read_text();changes={
 'import full_cache_shared_runtime_v9 as ctrl':'import live_cache_identity_controller_v2 as ctrl',
 '/controller/full_cache_shared_api_trace_v9.py':'/controller/full_cache_live_identity_api_v2.py',
 " try:\n  from finite_tokenizer_owned_execution_v2 import execute as settle_owned_launch":" owner_packet(plan,out)\n try:\n  from finite_tokenizer_owned_execution_v2 import execute as settle_owned_launch",
 "   if phase['name']=='history_pinned':":"   if phase['name']=='prime1':\n    from run_live_cache_identity_probes_v2 import run as reject_foreign_namespaces,loaded_probes\n    result['live_namespace_refusals']=reject_foreign_namespaces(url,plan['live_identity_namespace'],phase['rows'][0]['messages'],plan['research_alias'],out/'api-native-trace.jsonl',out/'live-identity-guard.jsonl',out/'live-namespace-refusals')\n    result['live_loaded_owner_refusals']=loaded_probes(url,phase['rows'][0]['messages'],plan['research_alias'],out/'api-native-trace.jsonl',out/'live-identity-guard.jsonl',out/'live-identity-owner.json',out/'live-loaded-owner-refusals')\n   if phase['name']=='history_pinned':"}
 for before,after in changes.items():ctrl.require(source.count(before)==1,'Exact original actor adaptation point differs '+before);source=source.replace(before,after)
 ast.parse(source);return source
def run(*args):
 namespace=dict(vars(original));namespace.update(__file__=__file__,__name__='new_live_identity_actor_v1',owner_packet=owner_packet);exec(compile(adapted_source(),str(Path(original.__file__))+'[NEW live owner ingress]', 'exec'),namespace);return namespace['run'](*args)

# Pure original recipe/trace helpers used by the public admission; execution
# still uses exact adapted source above.
def __getattr__(name):return getattr(original,name)

def command_recipe(*args):
 values=dict(vars(original));values.update(__file__=__file__,__name__="live_identity_recipe_admission",owner_packet=owner_packet);exec(compile(adapted_source(),__file__,"exec"),values);return values["command_recipe"](*args)
