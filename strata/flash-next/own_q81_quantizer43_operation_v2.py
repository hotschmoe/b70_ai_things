"""Private current43/new-Q81 operation witness plus exact consumed fixture."""
import ast,copy,os
from pathlib import Path
import indexer_materialized43_masked_inputs_v2 as original
from serial37_canonical_json_v3 import read_unique
require=original.require
_context=None
_fixture=None
FIXTURE_FILES=('config.txt','cpu-native.q81','input-binding.json','own-input.f32','source-plan.snapshot.json')
def register_fixture(root):
 global _fixture
 root=Path(root);require(root.is_absolute()and not any(p.is_symlink()for p in(root,*root.parents))and root.is_dir(),'Exact consumed fixture owner required');require({p.name for p in root.iterdir()}==set(FIXTURE_FILES)and all(p.is_file()and not p.is_symlink()for p in root.iterdir()),'Complete five-file consumed fixture required');require(sum(p.stat().st_size for p in root.iterdir())<=64<<20,'Bounded consumed fixture required before byte entry');row=(os.getpid(),root)
 require(_fixture is None or _fixture==row,'Fixture/owner cannot switch within original operation');_fixture=row
 if _context is not None:require(_context['_operation']is None or _context['_operation']['owner_pid']==os.getpid(),'Foreign existing operation refused')
 return root

def context():
 global _context
 require(_fixture is not None and _fixture[0]==os.getpid(),'Consumed fixture must be registered before original entry/ready')
 if _context is None:
  source=Path(original.__file__).read_text();old='from qualify_indexer_materialized43_masked_v2 import PLAN';require(source.count(old)==1,'Exact original43 witness PLAN import required');source=source.replace(old,'from qualify_own_q81_quantizer43_v2 import PLAN')
  needle="source=[ROOT/p if not Path(p).is_absolute()else Path(p)for p in read_unique(PLAN)['files']]+[PLAN]";require(source.count(needle)==1,'Exact original43 source roster required');source=source.replace(needle,needle+"+[Path(name)for name in read_unique(PLAN)['actual_consumed_source_bindings']]+[_fixture_root/name for name in _fixture_names]")
  names=('producer_binding','operation_witness','admit_operation_witness');nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names];require({n.name for n in nodes}==set(names),'Exact original43 witness function roster required')
  _context={**vars(original),'_operation':None,'_fixture_root':_fixture[1],'_fixture_names':FIXTURE_FILES,'_derived_AST':{n.name:ast.dump(n)for n in nodes}};exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),str(original.__file__)+'[own-Q81-current43-fixture-operation]','exec'),_context)
 return _context

def producer_binding(root,role=None):return context()['producer_binding'](root,role)
def operation_witness():return context()['operation_witness']()
def admit_operation_witness(root,report):return context()['admit_operation_witness'](root,report)
