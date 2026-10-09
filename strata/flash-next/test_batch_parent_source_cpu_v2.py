#!/usr/bin/env python3
"""Pinned real parent functions on four tiny CPU files; no actual model/health."""
import ast,hashlib,json,tempfile,time
from pathlib import Path
import qualify_batch_numerical_v2 as p

def dump_function(path,name):
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);return ast.dump(node,include_attributes=False)

def main():
 here=Path(__file__).resolve().parent
 assert dump_function(here/'qualify_layer0_numerical_v9.py','full_buffered_identity')==dump_function(here/'qualify_batch_numerical_v2.py','full_buffered_identity')
 assert dump_function(here/'qualify_layer0_numerical_v9.py','stat_signature')==dump_function(here/'qualify_batch_numerical_v2.py','stat_signature')
 with tempfile.TemporaryDirectory() as name:
  root=Path(name);paths=[];files=[]
  for i in range(4):
   path=root/('shard'+str(i));raw=(b'original-source-'+bytes([i]))*31;path.write_bytes(raw);paths.append(path);files.append({'path':'UD-Q4_K_XL/'+path.name,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  lock={'revision':'tiny-independent-CPU-control','files':files};lock_path=root/'lock.json';lock_path.write_text(json.dumps(lock));after=time.time()-.01
  receipt=p.full_buffered_identity(lock_path,lock,paths,root/'good.json',after);assert receipt['passed'] and len(receipt['rows'])==4 and all(len(row['stat_before'])==5 for row in receipt['rows'])
  paths[2].write_bytes(paths[2].read_bytes().replace(b'original',b'changed!',1));bad=p.full_buffered_identity(lock_path,lock,paths,root/'bad.json',after);assert not bad['passed'] and not bad['rows'][2]['passed']
  try:p.full_buffered_identity(lock_path,lock,paths,root/'future.json',time.time()+10)
  except ValueError:pass
  else:raise AssertionError('Future terminal boundary accepted')
 parent={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[],'post_health_finished_epoch':20}
 child={'collection_and_teardown_passed':True,'finished_epoch':19};post={'passed':True,'started':21};assert p.finalizable(parent,child,post)
 controls=2
 for key,value in [('child_return_code',1),('interrupted',True),('owned_containers_terminal',False),('forced_cleanup',True),('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False),('errors',['actual failure'])]:
  bad=dict(parent);bad[key]=value;assert not p.finalizable(bad,child,post);controls+=1
 assert not p.finalizable(parent,child,{'passed':True,'started':19.5});controls+=1
 assert not p.finalizable(parent,{'collection_and_teardown_passed':False,'finished_epoch':19},post);controls+=1
 print(json.dumps({'passed':True,'negative_controls':controls,'actual_four_tiny_file_full_hash':True,'immutable_v9_hash_functions_AST_equal':True,'scope':'CPU functions only; not actual model identity/serving/GPU/health qualification'},ensure_ascii=True))
if __name__=='__main__':main()
