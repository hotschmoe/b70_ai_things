"""Narrow source/dependency scheduling controls; no archives/binaries/compiler/GPU."""
import hashlib,json,unittest
from pathlib import Path
import build_native62_qsa35_incremental_v1 as b
import source62_qsa35_sdk_binding_v1 as sdk
from native62_incremental_ninja_audit_v1 import audit
class Controls(unittest.TestCase):
 def test_actual_root_overlay_and_proposal(self):
  p=json.loads(b.CHILD_PLAN.read_bytes());a=json.loads(b.PARENT_PLAN.read_bytes());self.assertEqual(hashlib.sha256(b.CHILD_PLAN.read_bytes()).hexdigest(),b.CHILD_PLAN_SHA);self.assertEqual(sdk.PLAN_SHA,b.CHILD_PLAN_SHA);self.assertEqual(set(p['expected_patched_source_sha256']),set(a['expected_patched_source_sha256']));self.assertEqual(len(p['expected_patched_source_sha256']),76);self.assertEqual({n for n in a['expected_patched_source_sha256']if a['expected_patched_source_sha256'][n]!=p['expected_patched_source_sha256'][n]},b.CHANGED);r=Path(p['actual_overlay_root']);self.assertEqual(hashlib.sha256((r/'proposal.json').read_bytes()).hexdigest(),p['actual_overlay_recipe_sha256'])
  for n in b.CHANGED:self.assertEqual(hashlib.sha256((r/n).read_bytes()).hexdigest(),p['expected_patched_source_sha256'][n])
 def test_actual_saved_dependency_consumers(self):
  p=json.loads(b.CHILD_PLAN.read_bytes());root=Path(p['source62_parent_actual_sdk']);text=(root/'build/incremental-final-dependencies.txt').read_text();current=None;consumers=set()
  for line in text.splitlines():
   if ': #deps 'in line:current=line.split(': #deps ',1)[0]
   elif line.strip().startswith('/src/')and line.strip()[5:]in b.CHANGED:consumers.add(current)
  self.assertEqual(consumers,set(p['expected_changed_objects']));self.assertEqual(consumers,b.DIRECT_OBJECTS);self.assertEqual(len(consumers),2)
 def test_reused_exact_auditor_two_object_one_archive_schedule(self):
  objects={'verify.o':'v','generate.o':'g','reused.o':'r'};proof={'object_sha256':objects,'source_tree':{'observer.hpp':'old','identity.hpp':'old','keep.hpp':'unchanged'},'changed_files':['observer.hpp','identity.hpp'],'expected_child_sources':{'observer.hpp':'new','identity.hpp':'new'},'changed_objects':['verify.o','generate.o'],'scheduled_archives':['libstrata_engine.a'],'ggml_tree':{},'build_dependency_sha256':{}}
  deps='verify.o: #deps 1\n    /src/observer.hpp\n\ngenerate.o: #deps 1\n    /src/identity.hpp\n\nreused.o: #deps 1\n    /src/keep.hpp\n';rows=['icpx -o verify.o -c verify.cpp','icpx -o generate.o -c generate.cpp','cmake -E rm -f libstrata_engine.a && ar qc libstrata_engine.a verify.o']+['icpx -o target'+str(i)+' obj.o'for i in range(8)];dry='\n'.join('['+str(i+1)+'/11] '+x for i,x in enumerate(rows));r=audit(proof,deps,dry,lambda n:'new'if n.endswith(('observer.hpp','identity.hpp'))else'unchanged');self.assertTrue(r['passed']);self.assertEqual(len(r['scheduled_commands']),11)
  with self.assertRaises(ValueError):audit(proof,deps,dry+'\n[12/12] icpx -o reused.o -c foreign.cpp',lambda n:'new'if n.endswith(('observer.hpp','identity.hpp'))else'unchanged')
if __name__=='__main__':unittest.main()
