import unittest,tempfile,subprocess,re
from pathlib import Path
import prepare_pti_paused_regular_lists_v1 as a
class Tests(unittest.TestCase):
    def test_full_apply_reverse(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);f=root/a.HEADER;f.parent.mkdir(parents=True);f.write_bytes(a.original());patch=root/'patch';patch.write_text(a.patch_text());subprocess.run(['git','apply',str(patch)],cwd=root,check=True,capture_output=True);self.assertEqual(f.read_bytes(),a.changed());subprocess.run(['git','apply','--reverse',str(patch)],cwd=root,check=True,capture_output=True);self.assertEqual(f.read_bytes(),a.original())
    def test_existing_query_and_normalization_code_unchanged(self):
        old=a.original().decode();new=a.changed().decode();pattern=r'ZE_FUNC\(zeDeviceGetGlobalTimestamps\)\([^;]+;';self.assertEqual(re.findall(pattern,old),re.findall(pattern,new));self.assertEqual(len(re.findall(pattern,old)),6)
        for begin,end in(('  inline void GetHostTime(', '  void PrintCommandCompleted('),('  void PrepareToExecuteCommandListsLocked(', '\n  ')):
            if begin=='  inline void GetHostTime(':
                left=old[old.index(begin):old.index(end)];right=new[new.index(begin):new.index(end)];self.assertEqual(left,right)
    def test_default_off_and_regular_only(self):
        text=a.changed().decode();self.assertIn('value==nullptr',text);self.assertIn('!found->second->immediate_',text);self.assertIn('options_.metric_query||options_.metric_stream',text)
    def test_both_fixture_replays_same_executable(self):
        text=Path(a.__file__).with_name('pti_paused_regular_list_fixture_v1.cpp').read_text();self.assertEqual(text.count('q.ext_oneapi_graph(executable).wait_and_throw();'),2);self.assertIn('PAUSED_LIST_WARM_DONE values=1,2',text);self.assertIn("std::getchar()!='R'",text);self.assertIn('PAUSED_LIST_RETIRED graph=1 allocation=1',text)
    def test_actual_child_roster_missing_duplicate_refused(self):
        import pti_paused_fixture_coverage_v1 as c
        rows=[dict(instance_id=i,raw_kernel_start_ticks=10*i,raw_kernel_end_ticks=10*i+2)for i in(1,2)];events=[dict(name=name,args={'id':str(i)})for i,name in enumerate(('PausedRegularFixtureA','PausedRegularFixtureB'),1)]
        self.assertEqual(c.child_coverage(rows,events,True)['selected_child_instance_ids'],[1,2])
        with self.assertRaises(ValueError):c.child_coverage(rows[:1],events[:1],True)
        self.assertFalse(c.child_coverage([],[],False)['full_model_coverage_qualified'])
    def test_wrong_actual_backend_route_refused(self):
        import pti_paused_fixture_coverage_v1 as c
        events=[dict(ph='X',cat='cpu_op',name='zeCommandListImmediateAppendCommandListsExp')];self.assertTrue(c.regular_replay_path(events)['actual_regular_list_submission_observed'])
        with self.assertRaises(ValueError):c.regular_replay_path(events+[dict(ph='X',cat='cpu_op',name='zeCommandListAppendGraphExt')])
if __name__=='__main__':unittest.main()
