import unittest
import run_original_mtp53_teacher_head_v1 as r
class Counters(unittest.TestCase):
 def test_exact_counters(self):
  for mode,calls,skips,flag in [('off',2048,0,0),('on',1,2047,1)]:
   row=f'MTP53_TEACHER_HEAD_COUNTS mode={flag} teacher_rows=2047 calls={calls} skips={skips} committed_cells=2048 fresh_head=1'
   self.assertEqual(r.teacher_counts(row,mode)['calls'],calls)
   for corrupt in [row.replace('fresh_head=1','fresh_head=0'),row.replace('teacher_rows=2047','teacher_rows=2046'),row+' extra=0',row+'\n'+row]:
    with self.assertRaises(ValueError):r.teacher_counts(corrupt,mode)
 def test_actual_mtp_enabled_both_arms(self):
  from pathlib import Path
  source=Path(r.__file__).read_text()
  self.assertIn("'STRATA_ORIGINAL_MTP53_PAIR_MODE':'controls'if mode=='controls'else'on'",source)
  self.assertIn("natural_rows((d/'native.stdout').read_text(),'on',ids)",source)
  self.assertIn("'STRATA_EXPERT_PAIR_REUSE':'0'",source)
