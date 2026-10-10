"""AST Undo, complete call routing and tiny immutable upload-subset controls."""
import ast,copy,hashlib,importlib,inspect,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import prepare_upload_logical_gate_ports_v2 as producer
import c137_sdk37_logical_ports_v2 as port
import c137_sdk37_digest_ports_v1 as old
import logical_free_immutable_epoch_v2 as narrow
import logical_free_require_case_epoch_v2 as compact
import parse_usm_logical_free_trace as parser
from test_logical_free_snapshot_cpu_v1 import TRACE
HERE=Path(__file__).resolve().parent
class Undo(ast.NodeTransformer):
 def __init__(self):self.reverse={v:k for k,v in producer.MAP.items()}
 def visit_Import(self,n):
  for row in n.names:
   if row.name in self.reverse:
    row.name=self.reverse[row.name]
    if row.asname==row.name:row.asname=None
  return n
 def visit_ImportFrom(self,n):
  if n.module in self.reverse:n.module=self.reverse[n.module]
  return n
 def visit_FunctionDef(self,n):
  for key in reversed(producer.KEYWORDS):
   if n.args.kwonlyargs and n.args.kwonlyargs[-1].arg==key:n.args.kwonlyargs.pop();n.args.kw_defaults.pop()
  if n.name=='original_upload_gate':
   original=next(f for f in producer.functions('c137_sdk37_digest_ports_v1')if f.name==n.name);loop=next(f for f in n.body if isinstance(f,ast.For)and isinstance(f.target,ast.Name)and f.target.id=='row');original_loop=next(f for f in original.body if isinstance(f,ast.For)and isinstance(f.target,ast.Name)and f.target.id=='row');start=next(i for i,f in enumerate(original_loop.body)if isinstance(f,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='logical_path'for t in f.targets));loop.body=loop.body[:-1]+copy.deepcopy(original_loop.body[start:]);n.body=n.body[2:];index=next(i for i,f in enumerate(original.body)if isinstance(f,ast.ImportFrom)and f.module=='parse_usm_logical_free_trace');n.body.insert(index,copy.deepcopy(original.body[index]))
  return self.generic_visit(n)
 def visit_Call(self,n):
  self.generic_visit(n)
  if isinstance(n.func,ast.Name)and n.func.id=='original_producer_file':return ast.Name(id='__file__',ctx=ast.Load())
  if isinstance(n.func,ast.Name)and n.func.id=='original_module_file':return ast.Attribute(value=ast.Name(id=n.args[1].value,ctx=ast.Load()),attr='__file__',ctx=ast.Load())
  if isinstance(n.func,ast.Name)and n.func.id=='__import__'and n.args and isinstance(n.args[0],ast.Constant)and n.args[0].value in self.reverse:n.args[0].value=self.reverse[n.args[0].value]
  if isinstance(n.func,ast.Attribute)and n.func.attr=='import_module'and n.args and isinstance(n.args[0],ast.Constant)and n.args[0].value in self.reverse:n.args[0].value=self.reverse[n.args[0].value]
  n.keywords=[k for k in n.keywords if k.arg not in producer.KEYWORDS];return n
class Controls(unittest.TestCase):
 def test_static_producer_reconstructs_all16_exact_generated_sources(self):
  for target,row in producer.produce(write=False).items():self.assertEqual(hashlib.sha256((HERE/(target+'.py')).read_bytes()).hexdigest(),row['generated_sha256'])
 def test_full_AST_Undo_preserves_every_other_original_predicate_and_return(self):
  for original,target in producer.MAP.items():
   tree=ast.parse((HERE/(target+'.py')).read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom)and n.module in ('upload_logical_port_identity_v2','logical_free_require_case_epoch_v2','original_upload_logical_requirement_v2'))]
   if original=='batch_numerical_execution_v50':tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom)and n.module=='batch_numerical_proofs_logical_ports_v2')]
   undo=Undo().visit(tree);self.assertEqual(ast.dump(undo,include_attributes=False),ast.dump(producer.original_tree(original),include_attributes=False),target)
 def test_resolved_complete_callgraph_mandatory_keywords_and_no_foreign_kwargs(self):
  for target in producer.MAP.values():
   module=importlib.import_module(target);tree=ast.parse((HERE/(target+'.py')).read_text());aliases={}
   for n in ast.walk(tree):
    if isinstance(n,ast.Import):
     for a in n.names:aliases[a.asname or a.name]=a.name
    if isinstance(n,ast.ImportFrom):
     for a in n.names:
      if a.name!='*':aliases[a.asname or a.name]=(n.module,a.name)
   for n in ast.walk(tree):
    if not isinstance(n,ast.Call):continue
    obj=None;fn=n.func
    if isinstance(fn,ast.Name):
     obj=getattr(module,fn.id,None)
     if isinstance(aliases.get(fn.id),tuple):m,name=aliases[fn.id];obj=getattr(importlib.import_module(m),name)
    elif isinstance(fn,ast.Attribute)and isinstance(fn.value,ast.Name):
     if isinstance(aliases.get(fn.value.id),str):obj=getattr(importlib.import_module(aliases[fn.value.id]),fn.attr,None)
     elif hasattr(module,fn.value.id):obj=getattr(getattr(module,fn.value.id),fn.attr,None)
    elif isinstance(fn,ast.Attribute)and isinstance(fn.value,ast.Call)and isinstance(fn.value.func,ast.Name)and fn.value.func.id=='__import__':obj=getattr(importlib.import_module(fn.value.args[0].value),fn.attr)
    provided={k.arg for k in n.keywords if k.arg in producer.KEYWORDS}
    try:sig=inspect.signature(obj)
    except (TypeError,ValueError):self.assertFalse(provided,target+' unresolved '+ast.unparse(fn));continue
    wanted=set(producer.KEYWORDS)&set(sig.parameters);self.assertEqual(provided,wanted,target+' '+ast.unparse(fn))
    for key in wanted:self.assertIs(sig.parameters[key].default,inspect.Parameter.empty,'Mandatory epoch/roster may not silently default')
 def test_both_dynamic_and_static_historical_serial_routes_are_ported(self):
  text=(HERE/'serial_reader_logical_ports_v2.py').read_text();self.assertIn("__import__('serial_selection_logical_ports_v2')",text);self.assertIn('serial_vectors',producer.CALLS)
  for name in ('c1_serve_controller_combined_v137.py','batch_numerical_execution_v50.py','qualify_batch_numerical_v50.py'):self.assertNotIn('logical_ports', (HERE/name).read_text())
 def fixture(self,base):
  upload=base/'upload';upload.mkdir();oracle_root=base/'oracle';oracle_root.mkdir();engine=base/'engine.json';engine.write_text('{}');pack=base/'pack.json';pack.write_text('{}');source=oracle_root/'oracle.cpp';source.write_text('tiny synthetic source');binary=oracle_root/'source-upload-oracle';binary.write_bytes(b'\x7fELF tiny fake source-only control');roster=base/'roster';roster.write_text('synthetic390 metadata');rreceipt=base/'roster.json';rreceipt.write_text(json.dumps({'RESULT':{'rows':[{'name':'synthetic'+str(i)}for i in range(390)]}}));inventory=base/'inventory.json';inventory.write_text(json.dumps({'inventory_complete':True,'errors':[],'files':[]}));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();plan={'schema':2,'oracle_source_sha256':sha(source),'source_roster':{'path':str(roster),'sha256':sha(roster),'receipt':str(rreceipt),'receipt_sha256':sha(rreceipt)},'inventory':str(inventory),'inventory_sha256':sha(inventory)};planpath=oracle_root/'plan.snapshot.json';planpath.write_text(json.dumps(plan));oracle={'passed':True,'libraries_unchanged':True,'engine_receipt_sha256':sha(engine),'plan_sha256':sha(planpath),'oracle_source_sha256':sha(source),'binary_sha256':sha(binary)};oraclepath=oracle_root/'receipt.json';oraclepath.write_text(json.dumps(oracle));cases=[];inputs=[]
  names=['full390_card0','full390_card1','full390_same_device24_24','full390_two_device24_24','full390_actual_model_static_bounds']
  for name in names:
   log=upload/(name+'.log');log.write_text(TRACE);logical=upload/(name+'-logical-free.json');v=parser.parse_trace(TRACE);v['negative_controls']=parser.negative_controls(TRACE,True);logical.write_text(json.dumps(v));inputs.extend([log,logical]);cases.append({'case':name,'state':{'ExitCode':0,'OOMKilled':False,'Running':False},'report':{'stages':[{'unique_allocations':0}],'bounds':'static'}})
  receipt=upload/'receipt.json';report={'passed':True,'pre_health_passed':True,'post_health_passed':True,'image':port.BASE_IMAGE,'oracle_receipt_sha256':sha(oraclepath),'pack_receipt_sha256':sha(pack),'oracle_schema':2,'oracle_plan_sha256':sha(planpath),'source_roster_sha256':sha(roster),'cases':cases};receipt.write_text(json.dumps(report));inputs.extend([narrow.PARSER,narrow.WORKER,*(HERE/name for name in narrow.PROGRAM_SOURCE_NAMES),*(HERE/name for name in compact.SOURCES)]);expected=[(p,sha(p))for p in inputs];epoch=compact.RequireLogicalEpoch(expected,60);sdk=__import__('types').SimpleNamespace(digest_for=lambda role,path,digest:digest);return receipt,oraclepath,engine,pack,epoch,expected,sdk
 def call(self,parts):
  receipt,oracle,engine,pack,epoch,roster,sdk=parts
  with patch.object(port,'require_oracle'),patch.object(port,'full_source_case_gate')as shapes:return port.original_upload_gate(receipt,oracle,engine,pack,sdk_epoch=sdk,logical_epoch=epoch,logical_roster=roster),shapes.call_count
 def test_five_case_original_output_identical_and_outer_case_checks_repeat(self):
  with tempfile.TemporaryDirectory()as d:
   parts=self.fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts
   with patch.object(old,'require_oracle'),patch.object(old,'full_source_case_gate'):original=old.original_upload_gate(receipt,oracle,engine,pack,sdk_epoch=sdk)
   for i in range(4):value,checks=self.call(parts);self.assertEqual(value,original);self.assertEqual(checks,5)
   self.assertEqual(epoch.executions,5);self.assertEqual(epoch.calls,5);self.assertEqual(epoch.require_calls,20);self.assertEqual(epoch.first_result_isolation_copies,5)
 def test_changed_live_image_terminal_health_inventory_and_bounds_still_fail(self):
  for change in ('image','state','health','inventory','bounds'):
   with tempfile.TemporaryDirectory()as d:
    parts=self.fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;self.call(parts);report=json.loads(receipt.read_text())
    if change=='image':report['image']='foreign'
    elif change=='state':report['cases'][0]['state']['Running']=True
    elif change=='health':report['post_health_passed']=False
    elif change=='bounds':report['cases'][-1]['report']['bounds']='dynamic'
    else:
     p=Path(json.loads((oracle.parent/'plan.snapshot.json').read_text())['inventory']);p.write_text('{}')
    receipt.write_text(json.dumps(report));self.assertRaises(ValueError,self.call,parts)
 def test_epoch_and_roster_are_mandatory_not_saved_success_fallback(self):
  self.assertRaises(TypeError,port.original_upload_gate,Path('/synthetic'),Path('/synthetic'),Path('/synthetic'),Path('/synthetic'),sdk_epoch=None)
  self.assertRaises(ValueError,port.original_upload_gate,Path('/synthetic'),Path('/synthetic'),Path('/synthetic'),Path('/synthetic'),sdk_epoch=None,logical_epoch=None,logical_roster=[])
 def test_compact_token_is_immutable_and_full_results_keep_mutation_isolation(self):
  with tempfile.TemporaryDirectory()as d:
   parts=self.fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;self.call(parts);log=receipt.parent/'full390_card0.log';logical=receipt.parent/'full390_card0-logical-free.json';full=epoch.collect(log,logical,1,current_roster=roster);full['checked']['passed']=False;self.assertIs(epoch.require_case(log,logical,1,current_roster=roster),True);self.assertTrue(epoch.collect(log,logical,1,current_roster=roster)['checked']['passed'])
 def test_compact_changed_count_or_failed_epoch_cannot_reuse_token(self):
  with tempfile.TemporaryDirectory()as d:
   parts=self.fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;self.call(parts);log=receipt.parent/'full390_card0.log';logical=receipt.parent/'full390_card0-logical-free.json';self.assertRaises(ValueError,epoch.require_case,log,logical,2,current_roster=roster);self.assertRaises(ValueError,epoch.require_case,log,logical,1,current_roster=roster)
 def test_foreign_epoch_subclass_cannot_replace_requirement_with_callback(self):
  class Foreign(compact.RequireLogicalEpoch):
   def require_case(self,*args,**kwargs):return True
  fake=object.__new__(Foreign)
  self.assertRaises(ValueError,port.original_upload_gate,Path('/synthetic'),Path('/synthetic'),Path('/synthetic'),Path('/synthetic'),sdk_epoch=None,logical_epoch=fake,logical_roster=[])
 def test_every_transitive_gate_refuses_omitted_epoch_before_body_execution(self):
  for target in producer.MAP.values():
   module=importlib.import_module(target)
   for n in ast.parse((HERE/(target+'.py')).read_text()).body:
    if not isinstance(n,ast.FunctionDef)or n.name not in producer.CALLS:continue
    sig=inspect.signature(getattr(module,n.name));kwargs={name:None for name,p in sig.parameters.items()if p.kind in (p.POSITIONAL_OR_KEYWORD,p.KEYWORD_ONLY) and name not in producer.KEYWORDS}
    self.assertRaisesRegex(TypeError,'logical_epoch',sig.bind,**kwargs)
 def test_instance_method_shadows_are_rejected_and_epoch_stays_failed(self):
  for name in ('collect','require_case','owned','finalize'):
   with tempfile.TemporaryDirectory()as d:
    parts=self.fixture(Path(d));epoch=parts[4]
    self.assertRaises(ValueError,setattr,epoch,name,lambda *a,**k:{'original_upload_logical_subset_passed':True});self.assertTrue(epoch.failed)
    self.assertRaises(ValueError,compact.RequireLogicalEpoch.require_case,epoch,parts[0].parent/'full390_card0.log',parts[0].parent/'full390_card0-logical-free.json',2,current_roster=parts[5])
 def test_class_method_shadows_cannot_be_installed(self):
  for name in ('collect','require_case','owned','finalize'):self.assertRaises(ValueError,setattr,compact.RequireLogicalEpoch,name,lambda *a,**k:True)
 def test_known_other_function_global_substitution_is_not_accepted(self):
  import serial37_canonical_json_v3 as canonical_module
  with tempfile.TemporaryDirectory()as d:
   parts=self.fixture(Path(d))
   with patch.object(canonical_module,'canonical',canonical_module.finite_constant):self.assertRaises(ValueError,self.call,parts)
   self.assertTrue(parts[4].failed)
 def test_compact_require_global_and_invoke_and_worker_dispatch_shadows_fail(self):
  import logical_free_worker_owned_v2 as owned
  for module,name in ((compact,'require'),(narrow,'invoke'),(owned,'execute')):
   with tempfile.TemporaryDirectory()as d:
    parts=self.fixture(Path(d))
    with patch.object(module,name,lambda *a,**k:True):self.assertRaises(ValueError,self.call,parts)
    self.assertTrue(parts[4].failed);self.assertEqual(parts[4].executions,0)
 def test_genuine_packet_provenance_must_remain_after_cached_success(self):
  with tempfile.TemporaryDirectory()as d:
   parts=self.fixture(Path(d));self.call(parts);parts[4].commands.clear();self.assertRaises(ValueError,self.call,parts)
 def test_paired_guard_and_collect_substitution_is_refused_before_token(self):
  import logical_free_method_binding_v2 as guard
  with tempfile.TemporaryDirectory()as d:
   parts=self.fixture(Path(d));epoch=parts[4]
   with patch.object(guard,'binding',lambda *a,**k:True),patch.object(narrow.LogicalEpoch,'collect',lambda *a,**k:{'original_upload_logical_subset_passed':True}):self.assertRaises(ValueError,self.call,parts)
   self.assertTrue(epoch.failed);self.assertEqual(epoch.executions,0)
 def test_private_compiler_cache_injection_is_not_public_API(self):
  with tempfile.TemporaryDirectory()as d:
   epoch=self.fixture(Path(d))[4]
   for name in ('_known_method_cache','_guard_source_codes'):self.assertRaises(ValueError,setattr,epoch,name,{})
if __name__=='__main__':unittest.main()
