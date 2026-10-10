"""CPU/source-only complete local import/delegation ledger for shared V3."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
RUNTIME_LEDGER=HERE/'full-cache-shared-runtime-source-plan-v3.json'
TESTS=sorted(p.stem for p in HERE.glob('test_full_cache_shared*cpu_v2.py'))+sorted(p.stem for p in HERE.glob('test_full_cache_shared*cpu_v3.py'))+['test_registry_c140_shared_association_cpu_v3']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def closure():
 seeds=list(HERE.glob('*full_cache_shared*v3.py'))+[HERE/'registry_c140_shared_association_v3.py',HERE/'test_registry_c140_shared_association_cpu_v3.py',Path(__file__),HERE/'full-cache-shared-runtime-v3-progress.md',HERE/'full-cache-shared-source40-registry-proposal-v3.yaml',HERE/'c1-combined-v140-parent-source-plan-v3.json',HERE/'full-cache-shared-source-plan-v2.json',HERE/'full-cache-shared-batch0-persisted-proposal-source-plan-v2.json']
 seeds += [HERE/(name+'.py') for name in TESTS]
 pending=list(seeds);seen=set()
 while pending:
  path=pending.pop().resolve()
  if path==RUNTIME_LEDGER:continue
  if path in seen:continue
  if not path.is_file() or not path.is_relative_to(ROOT):raise ValueError('Required repository source missing or foreign: '+str(path))
  raw=path.read_bytes();raw.decode('ascii');seen.add(path)
  if path.suffix=='.json':
   value=json.loads(raw)
   for key in ('files','dependency_sha256'):
    if type(value.get(key))is dict:
     for name in value[key]:
      target=ROOT/name if (ROOT/name).is_file() else HERE/name
      if target.is_file() and target.resolve().is_relative_to(ROOT):pending.append(target)
  if path.suffix!='.py':continue
  tree=ast.parse(raw,filename=str(path))
  for node in ast.walk(tree):
   names=[]
   if isinstance(node,ast.Import):names=[n.name for n in node.names]
   elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
   elif isinstance(node,ast.Constant) and type(node.value)is str:
    text=node.value
    if text.endswith(('.py','.json','.yaml','.md','.hpp','.patch')):
     target=ROOT/text if (ROOT/text).is_file() else HERE/text
     if target.is_file() and target.resolve().is_relative_to(ROOT):pending.append(target)
    if text.isidentifier():names=[text]
   for name in names:
    target=HERE/(name.replace('.','/')+'.py')
    if target.is_file():pending.append(target)
 return {str(p.relative_to(ROOT)):digest(p) for p in sorted(seen)}

def plan():
 files=closure()
 return {'schema':3,'source_scope':'New source40V2 full shared bounded actor/suite, CPU/source review only','engine_plan_sha256':'d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6','required_baseline_parent_generation':1403,'required_upload_plan':'native-source-upload-plan-full-source40-v3.json','files':files,'cpu_command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest '+' '.join(TESTS)+' -q','actual_model_runtime_observed':False,'full_cache_runtime_qualified':False,'full_model_math_qualified':False,'physical_residency_qualified':False,'latency_qualified':False,'required_separate_experiments':['actual changed-model LIVE-cache refusal, distinct from wrong disk fingerprint','actual physical host/device/per-expert backing and residency','whole actor physical/transient peak coverage'],'runtime_prerequisites':['actual current C1401403 one/pair serving baseline and source40 uploadV3','exact baseline165+C140four+purpose2 canonical171 association','all whole named actor/capture passes and every independently owned fresh49 group','all current source/identity/pages/health/journal/owned terminal gates']}
if __name__=='__main__':
 target=RUNTIME_LEDGER
 if target.exists():raise SystemExit('Refuse to overwrite an existing frozen runtime source ledger')
 target.write_text(json.dumps(plan(),indent=2,ensure_ascii=True)+'\n',encoding='ascii')
