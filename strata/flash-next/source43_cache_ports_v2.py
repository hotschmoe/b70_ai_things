"""Explicit private source ports. No import/read/stat/hash globals are patched."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Every route is declared; historical numerical/parser primitives stay original.
MODULES={'full_cache_shared_runtime_v12':'full_cache43_runtime_v2','full_cache_shared_baseline_admission_v12':'full_cache43_baseline_v1','registry_c140_chunk173_association_v1':'registry_source43_cache175_association_v1','source_page_watchdog_v3':'source43_cache_watchdog_v1'}
for stem in ('run_full_cache_shared_runtime','qualify_full_cache_shared_runtime','validate_full_cache_shared_runtime','full_cache_shared_health_handoff','full_cache_shared_closed_source','full_cache_shared_history_decode','full_cache_shared_reply_decode','full_cache_shared_actor_retirement','full_cache_shared_memory_capture','full_cache_shared_metadata_retirement','full_cache_shared_source40_ownership','full_cache_shared_plan_snapshot','full_cache_shared_api_trace','full_cache_shared_fresh40','run_full_cache_shared_fresh40','qualify_full_cache_shared_fresh40','validate_full_cache_shared_fresh40','full_cache_shared_fresh_retirement','full_cache_shared_client_recollection','full_cache_shared_launch_binding'):
 MODULES[stem+'_v12']=stem.replace('full_cache_shared','full_cache43').replace('source40','source43').replace('fresh40','fresh43')+'_v2'

def source(old):
 path=HERE/(old+'.py');raw=path.read_bytes();plan=json.loads((HERE/'full-cache-shared-runtime-source-plan-v12.json').read_bytes());wanted=plan['files']['strata/flash-next/'+old+'.py'];
 if hashlib.sha256(raw).hexdigest()!=wanted:raise ValueError('Frozen original V12 source changed '+old)
 return raw

def transform(raw):
 tree=ast.parse(raw)
 class Route(ast.NodeTransformer):
  def visit_Import(self,n):
   for a in n.names:a.name=MODULES.get(a.name,a.name)
   return n
  def visit_ImportFrom(self,n):n.module=MODULES.get(n.module,n.module);return n
  def visit_Constant(self,n):
   if type(n.value)is str:
    value=n.value
    for old,new in MODULES.items():value=value.replace(old,new)
    value=value.replace('fullcache-shared40-v3','fullcache-shared43-v1').replace('fresh40','fresh43').replace('shared40','shared43')
    value=value.replace('source40','source43')
    # Owned metadata names/labels remain the original software protocol tags;
    # actual current SDK/source generation is separately mandatory in ctrl.
    n.value=value
   return n
 return ast.fix_missing_locations(Route().visit(tree))

def load(old,namespace):
 tree=transform(source(old));tree.body=[n for n in tree.body if not(isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and ast.unparse(n.test)=="__name__ == '__main__'")];exec(compile(tree,namespace['__file__']+'[explicit-original-V12-port]','exec'),namespace)
 if 'adapted_source'in namespace:
  generated=namespace['adapted_source']
  def routed_generated(*args,**kwargs):return ast.unparse(transform(generated(*args,**kwargs)))
  namespace['adapted_source']=routed_generated
