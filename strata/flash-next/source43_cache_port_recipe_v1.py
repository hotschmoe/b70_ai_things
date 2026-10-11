"""Source-only exact V12 recipe port; it grants no actor or API authority."""
import ast,copy,json
from pathlib import Path
import full_cache_shared_runtime_v12 as original
from source43_api_baseline_profile_v1 import require
from serial37_canonical_json_v3 import canonical

def recipe(config,capacity,selected_rids):
 require(type(capacity)is int and capacity in(2,4,6),'Strict native capacity must be exact2/4/6');require(type(selected_rids)is list and 0<len(selected_rids)<=6 and all(type(r)is int and r>0 for r in selected_rids)and len(set(selected_rids))==len(selected_rids),'Independent bounded capture-RID roster required');require(config['env'].get('STRATA_QSA_INDEXER_MATERIALIZED_HALF')=='1','Fresh43 mode1 configuration required')
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='recipe'));require(ast.unparse(node.body[0])=="cfg = read(Path(prepared) / 'server-config.json')",'Exact sole input read required');node.args.args[0].arg='cfg';node.body.pop(0);changes=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Tuple)and len(n.elts)==2 and isinstance(n.elts[0],ast.Constant)and n.elts[0].value=='--batch':require(isinstance(n.elts[1],ast.Constant)and n.elts[1].value==2,'Exact original batch2 literal');n.elts[1]=ast.Name(id='capacity',ctx=ast.Load());changes.append('--batch')
 require(changes==['--batch'],'Only declared batch-capacity recipe change');namespace={**vars(original),'capacity':capacity};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[explicit43-V12-cache-recipe]','exec'),namespace);args,env=namespace['recipe'](copy.deepcopy(config),selected_rids);require(env['STRATA_QSA_INDEXER_MATERIALIZED_HALF']=='1','Mode1 must remain exact');return args,env

def proposal(config,active_count,selected_rids):
 require(type(active_count)is int and active_count in(1,2,4,6),'Actual population progression1/2/4/6 required');capacity=2 if active_count==1 else active_count;args,env=recipe(config,capacity,selected_rids);cfg=copy.deepcopy(config);cfg.update(args=args,env=env,parallel=capacity);return {'schema':1,'status':'source-only recipe; not an executable runtime or signed manifest','source_generation':43,'materialized_mode':1,'target_actual_active_population':active_count,'native_batch_capacity':capacity,'one_active_is_subset_of_strict_batch2_not_batch1':active_count==1,'selected_capture_RIDs':selected_rids,'selected_RID_count_is_not_active_population':True,'configuration_proposal':cfg,'API_launch_allowed':False,'full_cache_runtime_qualified':False,'required_actual_population_proof':'same owner/window FC39 BSTEP_before rows plus matched BSTEP/BT/BDONE and full49; no RID-count substitution','remaining':['genuine completed API43 baseline/current registry association','new truthful cache configuration/manifest/controller and current43 source admission','new source43 whole9 plus persisted/fresh49 producer/public routes','actual4/6 corpus, cohort admission and active-window proof','exact startup capacity and fixed8GiB chain/512MiB transfer admission; no fallback']}
