import unittest,ast
from pathlib import Path
import run_original_mtp53_host_timing_v1 as actor
import build_original_mtp53_host_timing_v1 as builder
class Controls(unittest.TestCase):
 def test_actual_constructor_and_serialized_bounds(self):
  cfg=actor.read(actor.BASE);argv=actor.native_options(cfg,'on',actor.read(actor.FIXTURE)['cases'][0]['ids']);self.assertEqual(argv[argv.index('--max-new')+1],'512');self.assertEqual(argv[argv.index('--prefill')+1],'512')
  source=Path(actor.__file__).read_text();self.assertIn("'STRATA_ORIGINAL_MTP53_PHASE_TIMING':'1'",source)
  for flag in('STRATA_ORIGINAL_MTP53_TEACHER_HEAD_SKIP','STRATA_VERIFY_HC_T2_REUSE','STRATA_EXPERT_PAIR_REUSE'):self.assertIn("'"+flag+"':'0'",source)
  self.assertIn('size>8<<20',source);self.assertIn('arm(\'on\',0,0)',source)
  tree=ast.parse(source);self.assertTrue(any(isinstance(n,ast.FunctionDef)and n.name=='natural_rows'for n in tree.body))
  recipe={'archive_filter':{'remove_members':['verify.cpp.o']},'compile_commands':[{'name':'main.host-timing.o','argv':['icpx','-c','main']},{'name':'verify.host-timing.o','argv':['icpx','-c','verify']}],'link_argv':['icpx','main.o','verify.o','engine.filtered.a']}
  script=builder.shell(recipe);self.assertEqual(sum(line.startswith('icpx -c ')for line in script.splitlines()),2);self.assertIn('ar d /out/libstrata_engine.without_verify.a verify.cpp.o',script)
if __name__=='__main__':unittest.main()
