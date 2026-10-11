"""Explicit historical C113/C137 metadata predicates with postboot byte association.
No runtime/launch functions are exported and no original globals are patched.
"""
import ast,copy,hashlib,importlib,json
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
import postboot_original_model_association_v1 as model
HERE=Path(__file__).resolve().parent
CURRENT_FULL4_SHA='e9e43a28d660ac6c67dd3338667db03d8198ec1803d49b8a51f0bd914267ed21'
ORIGINAL_SOURCE_SHA={'c1_serve_controller_combined_v13': 'f2057108c8ed7992ddbb71facdc81f78302ccf142090f4d73b3a5063c6d8c149', 'c1_serve_controller_combined_v137': 'ad129cf6f9a0832615df88e985437168af0c40082049dc9c2e05dac27147ac53', 'qualify_c1_serving_combined_v13': '7400b7a73499f93645e17047ceb5b5da64017796e88ff9a7d05c4742a0ff3bb2', 'qualify_c1_serving_combined_v137_v3': '46e348d958242587dda704ff8b61c32a04aaf7269e77dc277910bebd48443778', 'qualify_c1_serving_combined_v137_v4': '703273a654783bbbe61902c3c0c0f2471bc783ea740a01754a86fab36fdc8fc2', 'layer0_numerical_qualification_v10': '5951972606ed0f05a5f3251d8e04bd8d8827c97473462a3f0baf25e248e5c74e', 'prefix_residual30_qualification_v4': '56e9c34dd2aa5d444487da7b0955f46925e8b20c270e65c72c6caa6fe0f442c2'}
CONTROLLERS={'c1_serve_controller_combined_v13','c1_serve_controller_combined_v137'}
PARENTS={'qualify_c1_serving_combined_v13','qualify_c1_serving_combined_v137_v3','qualify_c1_serving_combined_v137_v4'}
MANIFESTS={'layer0_numerical_qualification_v10','prefix_residual30_qualification_v4'}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def recheck(association):
 require(association['current_identity_sha256']==CURRENT_FULL4_SHA,'Exact actual postboot full4 producer receipt required')
 actual=model.association(association['historical_identity_path'],association['current_identity_path'],current_expected_sha256=CURRENT_FULL4_SHA)
 require(canonical(actual)==canonical(association),'Explicit current postboot association changed');return association

def associated(association,path,recorded):
 model.historical_stat(association,path,recorded);return True

def historical_sentinel(association,shards,recorded,known):
 require(type(recorded)is dict and set(recorded)=={'schema','path','pages','read_mode'} and recorded['schema']==3 and type(recorded['schema'])is int and recorded['read_mode']=='buffered read only; no invalidation/write/repair','Exact original sentinel metadata contract required')
 selected=[row for row in shards if '00003-of-00004' in row['path']];require(len(selected)==1 and recorded['path']==selected[0]['path'],'Original third-shard sentinel association differs');associated(association,selected[0]['path'],selected[0]['stat'])
 require(len(recorded['pages'])==2 and [(r['offset'],r['expected_sha256'])for r in recorded['pages']]==list(known) and all(r['bytes']==4096 and type(r['bytes'])is int and r['sha256']==r['expected_sha256']for r in recorded['pages']),'Original exact both-page sentinel hashes differ')
 # Whole current publisher bytes are identical; this is NOT a new page probe.
 return recorded

def before_view(path,view):
 original=read_unique(path)
 if view is None:return original
 require('started_epoch'not in original and set(view)==set(original)|{'started_epoch'} and all(canonical(view[k])==canonical(v)for k,v in original.items())and type(view['started_epoch'])in(int,float) and view['started_epoch']>0,'Only original named missing-started metadata repair permitted')
 return view

class Port(ast.NodeTransformer):
 def __init__(self,selected,parent=False):self.selected=set(selected);self.parent=parent;self.stat_edges=0
 def visit_Name(self,node):
  if node.id=='__file__':return ast.copy_location(ast.Name('_ORIGINAL_SOURCE',ast.Load()),node)
  return node
 def visit_Compare(self,node):
  node=self.generic_visit(node);parts=[node.left,*node.comparators]
  current=[i for i,x in enumerate(parts)if isinstance(x,ast.Call)and ((isinstance(x.func,ast.Name)and x.func.id=='stat_signature')or(isinstance(x.func,ast.Attribute)and x.func.attr=='stat_signature'))]
  if not current:return node
  require(len(current)==1 and all(isinstance(op,ast.Eq)for op in node.ops),'Only exact historical stat equality edges can be ported');i=current[0];require(len(parts)>=2,'Original historical stat counterpart missing');recorded=parts[i-1 if i else 1];call=parts[i];remaining=[x for j,x in enumerate(parts)if j!=i];self.stat_edges+=1
  gate=ast.Call(ast.Name('_associated',ast.Load()),[ast.Name('_model_association',ast.Load()),call.args[0],recorded],[])
  if len(remaining)==1:return ast.copy_location(gate,node)
  return ast.copy_location(ast.BoolOp(ast.And(),[ast.Compare(remaining[0],[ast.Eq()for _ in remaining[1:]],remaining[1:]),gate]),node)
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Name)and node.func.id=='original_page_sentinel':
   node.func=ast.Name('_historical_sentinel',ast.Load());node.args=[ast.Name('_model_association',ast.Load()),*node.args,ast.Subscript(ast.Name('m',ast.Load()),ast.Constant('source_page_sentinel'),ast.Load()),ast.Name('KNOWN_SOURCE_PAGES',ast.Load())];return node
  if isinstance(node.func,ast.Name)and node.func.id in self.selected:node.keywords.append(ast.keyword('_model_association',ast.Name('_model_association',ast.Load())))
  if isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='c1'and node.func.attr in ('validate_prepared','metadata_admission_gate'):
   node.func=ast.Name('_controller_validate'if node.func.attr=='validate_prepared'else '_controller_metadata',ast.Load());node.keywords.append(ast.keyword('model_association',ast.Name('_model_association',ast.Load())))
  if isinstance(node.func,ast.Name)and node.func.id=='validate_final_source_proof'and 'validate_final_source_proof'not in self.selected:node.func=ast.Name('_parent_validate',ast.Load());node.keywords.append(ast.keyword('model_association',ast.Name('_model_association',ast.Load())))
  return node

def derive(original,selected,association,extra=None):
 raw=Path(original.__file__).read_bytes();require(original.__name__ in ORIGINAL_SOURCE_SHA and hashlib.sha256(raw).hexdigest()==ORIGINAL_SOURCE_SHA[original.__name__],'Exact frozen original function source required');tree=ast.parse(raw);functions=[copy.deepcopy(n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in selected];require({n.name for n in functions}==set(selected),'Exact original readonly function roster required');port=Port(selected)
 for n in functions:n.args.kwonlyargs.append(ast.arg('_model_association'));n.args.kw_defaults.append(None);port.visit(n)
 namespace={**original.__dict__,'__file__':__file__,'_ORIGINAL_SOURCE':original.__file__,'_associated':associated,'_historical_sentinel':historical_sentinel};namespace.update(extra or {});exec(compile(ast.fix_missing_locations(ast.Module(functions,type_ignores=[])),original.__file__+'[explicit-postboot-historical]', 'exec'),namespace)
 return namespace,{'source':original.__file__,'source_sha256':hashlib.sha256(raw).hexdigest(),'selected':list(selected),'stat_edges':port.stat_edges}

def controller(original,association):
 require(original.__name__ in CONTROLLERS,'Only declared historical controllers supported');selected=('strict_v2_upload_provenance','upload_gate','metadata_admission_gate','validate_prepared');return derive(original,selected,association)
def metadata_admission_gate(prepared,model_association):
 recheck(model_association);generation=prepared['combined_generation']['controller_generation'];name='c1_serve_controller_combined_v13'if generation==13 else 'c1_serve_controller_combined_v137'if generation==137 else None;require(name is not None,'Only declared historical metadata controller generation supported');ns,_=controller(importlib.import_module(name),model_association);result=ns['metadata_admission_gate'](prepared,_model_association=model_association);recheck(model_association);return result

def validate_prepared(path,model_association):
 recheck(model_association);raw=read_unique(Path(path)/'prepared.json');generation=raw['combined_generation']['controller_generation'];name='c1_serve_controller_combined_v13'if generation==13 else 'c1_serve_controller_combined_v137'if generation==137 else None;require(name is not None,'Only exact historical C113/C137 prepared generations supported');original=importlib.import_module(name);ns,_=controller(original,model_association);result=ns['validate_prepared'](Path(path),_model_association=model_association);recheck(model_association);return result

def validate_final_source_proof(path,prepared,model_association,parent_module=None,candidate_final=None,extra=None):
 recheck(model_association);name=parent_module or ('qualify_c1_serving_combined_v13'if prepared['combined_generation']['controller_generation']==13 else 'qualify_c1_serving_combined_v137_v4');require(name in PARENTS,'Declared historical final-parent generation required');original=importlib.import_module(name);ns,_=derive(original,('validate_final_source_proof',),model_association,extra);result=ns['validate_final_source_proof'](path,prepared,candidate_final,_model_association=model_association);recheck(model_association);return result

def manifest_binding(original_ctrl,plan,model_association):
 recheck(model_association);require(original_ctrl.__name__ in MANIFESTS,'Only exact NUM10/P30 readonly manifest supported')
 extra={'_controller_validate':validate_prepared,'_controller_metadata':metadata_admission_gate,'_parent_validate':validate_final_source_proof};ns,_=derive(original_ctrl,('candidate_binding','manifest_binding'),model_association,extra)
 result=ns['manifest_binding'](plan,_model_association=model_association);recheck(model_association);return {'original_binding':result,**scope(model_association)}

def scope(association):return {'historical_evidence_only':True,'postboot_association':association,'old_current_stat_gate_passed':False,'new_known_page_probe_performed':False,'historical_GPU_health_transferred':False,'current_runtime_qualified':False,'original_reports_changed':False}
