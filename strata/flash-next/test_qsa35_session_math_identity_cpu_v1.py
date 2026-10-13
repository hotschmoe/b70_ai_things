"""Bounded actual source apply/undo and legacy/new scope; no compilation/devices."""
import difflib,subprocess,tempfile,unittest
from pathlib import Path
import prepare_qsa35_session_math_identity_v1 as s
class Controls(unittest.TestCase):
 def test_actual_source_patch_apply_reverse(self):
  funcs={s.OBSERVER:s.observer,s.IDENTITY:s.identity,s.VERIFY:s.verifier};patch=[]
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);original={}
   for name,fn in funcs.items():
    b=(s.SDK/'source'/name).read_text();original[name]=b;p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(b);patch.extend(difflib.unified_diff(b.splitlines(True),fn(b).splitlines(True),fromfile='a/'+name,tofile='b/'+name))
   pp=root/'patch.diff';pp.write_text(''.join(patch));subprocess.run(['git','apply','--check',str(pp)],cwd=root,check=True,capture_output=True);subprocess.run(['git','apply',str(pp)],cwd=root,check=True,capture_output=True)
   for name,fn in funcs.items():self.assertEqual((root/name).read_text(),fn(original[name]))
   subprocess.run(['git','apply','--reverse',str(pp)],cwd=root,check=True,capture_output=True)
   for name in funcs:self.assertEqual((root/name).read_text(),original[name])
 def test_legacy_predicates_and_actual35_scope(self):
  original=(s.SDK/'source'/s.OBSERVER).read_text();x=s.observer(original);self.assertEqual(x[x.index('inline const std::array<field,29>&'):x.index('inline int index')],original[original.index('inline const std::array<field,29>&'):original.index('inline int index')]);legacy=original.split('void begin(')[1].split('route_=')[0];self.assertIn(legacy.split('if(!r.active')[1],x);self.assertIn('int layer=3;',x);self.assertIn('if(!enabled)return;',x);self.assertIn('r.ids.size()!=1||r.ids[0]!=248045',x);self.assertIn('stage==1&&lb==24&&le==48',x);self.assertIn('settings().layer==3?2048:4096',x);self.assertIn('(settings().layer-lb_)/4',x);self.assertEqual(35//4,8);self.assertEqual((35-24)//4,2)
 def test_identity_exact_two_new_contracts(self):
  b=(s.SDK/'source'/s.IDENTITY).read_text();x=s.identity(b);self.assertEqual(x.replace('sycl-state-contract-v8','sycl-state-contract-v7').replace(',\n        "STRATA_HC_EXPLICIT_SQRT_DIVIDE","STRATA_PF_GDN_DECODER_PROJ"',''),b);self.assertEqual(x.count('"STRATA_HC_EXPLICIT_SQRT_DIVIDE"'),1);self.assertEqual(x.count('"STRATA_PF_GDN_DECODER_PROJ"'),1);self.assertIn('value?value:"<unset>"',x)
if __name__=='__main__':unittest.main()
