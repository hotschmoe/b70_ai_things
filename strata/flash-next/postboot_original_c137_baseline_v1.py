"""Historical C137 predicates: explicit current-byte association, never live grant."""
import copy,json,hashlib,ast,types
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
import postboot_original_native_manifest_v1 as port
import c1_serve_controller_combined_v137 as c
import adjudicate_c137_missing_started_v3_v1 as legacy
import c137_journal_binding_v3 as journal3

def old_tree(root,saved,association):
 root=Path(root).resolve();paths={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()};port.require(paths==set(saved),'Exact original historical evidence tree changed')
 old_dev={x['historical_stat5'][0]for x in association['mapping']};new_dev={x['current_stat5'][0]for x in association['mapping']}
 for name,binding in saved.items():
  path=root/name;port.require(not path.is_symlink()and c.sha(path)==binding['sha256'],'Historical evidence bytes/path differ');now=c.stat_signature(path);old=binding['stat5'];port.require(all(type(v)is int for v in old+now)and old[1:]==now[1:]and old[0]in old_dev and now[0]in new_dev,'Evidence artifact may change only associated filesystem st_dev')

def parent_view_gate(root,proof,identity,parent,before):
 # The original metadata omission is repaired only via an explicit argument.
 source=Path(journal3.__file__).read_bytes();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='parent_binding');node=copy.deepcopy(node)
 class View(ast.NodeTransformer):
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='c'and n.func.attr=='read'and len(n.args)==1 and isinstance(n.args[0],ast.Name)and n.args[0].id=='path':return ast.Call(ast.Name('_before_view',ast.Load()),[n.args[0],ast.Name('_named_before',ast.Load())],[])
   return n
 node=View().visit(node);ns={**journal3.__dict__,'_before_view':port.before_view,'_named_before':before};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[historical1373-explicit-before-argument]','exec'),ns);return ns['parent_binding'](root,proof,identity,parent)

def admit_adjudication(receipt,association):
 port.recheck(association);receipt=Path(receipt).resolve();saved=read_unique(receipt);root=Path(saved['original_run_root']).resolve();port.require(saved['adjudication_kind']=='C137_actual90362_missing_started_v3_v1'and root==legacy.RUN and not receipt.is_relative_to(root),'Only original independently adjudicated90362 artifact admitted');legacy.source_binding()
 for name,want in legacy.PINS.items():port.require(c.sha(root/name)==want,'Pinned original failed artifact changed')
 old_tree(root,saved['original_tree_binding'],association);parent=c.read(root/'parent-qualification.json');before=c.read(root/'parent-before-proof.json');proof=c.read(root/'c1-source-identity-proof-v137-v3.json');raw=c.read(root/'qualification-controller-v137-v3.json');identity=c.read(root/'c1-post-model-identity-v137-v3.json');prepared=port.validate_prepared(root,association)
 parent_view,before_view=legacy.metadata_views(parent,before,proof);bounded=legacy.raw_controller_binding(root,raw,prepared)
 final=copy.deepcopy(raw);final.update(c1_parent_generation=1373,c1_parent_controller_sha256=legacy.OLD_SHA,post_full4_source_qualified=True,c1_source_identity_proof={'path':str(root/'c1-source-identity-proof-v137-v3.json'),'sha256':legacy.PINS['c1-source-identity-proof-v137-v3.json']},controller_qualification_sha256=legacy.PINS['qualification-controller-v137-v3.json'])
 expected=saved['derived_final_metadata'];port.require(canonical(final)==canonical(expected)and canonical(saved['source_proof'])==canonical(proof)and canonical(saved['bounded_actual_screen'])==canonical(bounded),'Original derived metadata/source/screen association differs')
 extra={'parent_binding':lambda directory,p,i,parent=None:parent_view_gate(directory,p,i,parent,before_view)}
 port.validate_final_source_proof(root,prepared,association,parent_module='qualify_c1_serving_combined_v137_v3',candidate_final=final,extra=extra);parent_view_gate(root,proof,identity,parent_view,before_view);journal3.journal_binding(root,proof);journal3.health_binding(root,proof,'pre');journal3.health_binding(root,proof,'post')
 port.require(saved['original_parent_passed']is False and saved['original_reports_modified']is False and saved['passed']is True and saved['source_binding']==legacy.source_binding()and saved['adjudicator_sha256']==c.sha(legacy.__file__),'Original named adjudication/source identity differs');old_tree(root,saved['original_tree_binding'],association);port.recheck(association)
 return prepared,saved

def finalized_binding(root,model_association,adjudication_receipt=None):
 root=Path(root).resolve();port.recheck(model_association)
 if adjudication_receipt is not None:
  prepared,proof=admit_adjudication(adjudication_receipt,model_association);port.require(Path(proof['original_run_root']).resolve()==root,'Foreign original baseline adjudication');binding={'baseline_kind':'explicit_readonly_adjudicated_actual90362','baseline_root':str(root),'adjudication_receipt':str(Path(adjudication_receipt).resolve()),'adjudication_receipt_sha256':c.sha(adjudication_receipt),'adjudication':proof,'original_parent_passed':False,'engine_receipt_sha256':prepared['engine_receipt_sha256']}
 else:
  port.require(c.read(root/'qualification.json').get('c1_parent_generation')==1374,'Exact original parent1374 required');prepared=port.validate_prepared(root,model_association);proof=port.validate_final_source_proof(root,prepared,model_association);binding={'baseline_kind':'actual_current_parent1374','baseline_root':str(root),'qualification_sha256':c.sha(root/'qualification.json'),'parent_sha256':c.sha(root/'parent-qualification.json'),'source_proof':proof,'engine_receipt_sha256':prepared['engine_receipt_sha256']}
 port.recheck(model_association);return prepared,{'original_binding':binding,**port.scope(model_association)}
