import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source41_model_cli_v1 as s
import prefill_chunk64_512_control_v2 as template

class Controls(unittest.TestCase):
    def test_fresh41_record_no_old_runtime_or_alias(self):
        cfg=template.expected_config('chunk64',{'inherited_math_env':{}},18082);m={'cards':[0],'alias':cfg['aliases'][0]}
        modes=s.recipes(m,cfg,{'short':[1,2],'prefill_about512':[3]*524},{'engine_receipt_sha256':'fresh41'})
        self.assertEqual(modes['0']['argv_container'],modes['1']['argv_container']);self.assertTrue(modes['0']['argv_host'][0].startswith(str(s.SDK)))
        for mode in('0','1'):
            row=modes[mode];self.assertEqual(row['env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],mode);self.assertIn('source41',row['research_record_id']);self.assertFalse(row['eval_registry_alias_claimed']);self.assertFalse(row['source40_runtime_proof_transferred']);self.assertEqual(row['primary_future_API_alias'],'hotschmoe-dd')
            self.assertEqual(len(row['input_requests']),4);self.assertTrue(all(r['max_new']==64 for r in row['input_requests']))
    def test_bad_profile_math_or_trace_refused(self):
        cfg=template.expected_config('chunk64',{'inherited_math_env':{}},18082);m={'cards':[0],'alias':cfg['aliases'][0]}
        for kind in('card','math','trace'):
            a=copy.deepcopy(m);b=copy.deepcopy(cfg)
            if kind=='card':a['cards']=[False]
            elif kind=='math':b['env']['STRATA_HC_Q8']='1'
            else:b['env']['SYCL_UR_TRACE']='2'
            with self.assertRaises(ValueError):s.recipes(a,b,{'short':[1],'prefill_about512':[2]},{'engine_receipt_sha256':'fresh41'})
    def fixture(self,passed):
        import source_page_watchdog_v3 as old
        pages=(*old.KNOWN_PAGES,s.EXTRA_PAGE)
        return {'passed':passed,'path':'third','stat_before':[1,2,3,4,5],'stat_after':[1,2,3,4,5],'rows':[{'offset':offset,'bytes':4096,'sha256':sha,'expected_sha256':sha,'passed':passed,'data':bytes(4096)}for offset,sha in pages],'epoch':1}
    def test_guard_preserves_all_three_views_before_failure(self):
        import source_page_watchdog_v3 as old
        shards=[Path('a'),Path('b'),Path('Qwen-00003-of-00004.gguf'),Path('d')]
        with tempfile.TemporaryDirectory()as t,patch.object(old,'inspect_pages',return_value=self.fixture(False)):
            with self.assertRaises(ValueError):s.guard(shards,Path(t),'failure')
            self.assertEqual(len(list(Path(t).glob('*.raw'))),3);self.assertTrue((Path(t)/'failure-pages.json').is_file())
    def test_guard_scope_no_repair_no_fullidentity(self):
        import source_page_watchdog_v3 as old
        shards=[Path('a'),Path('b'),Path('Qwen-00003-of-00004.gguf'),Path('d')]
        with tempfile.TemporaryDirectory()as t,patch.object(old,'inspect_pages',return_value=self.fixture(True)):
            value=s.guard(shards,Path(t),'pass');self.assertFalse(value['full_model_identity_qualified']);self.assertTrue(value['no_DONTNEED_or_invalidation_or_source_write']);self.assertEqual(value['rows'][-1]['offset'],33530449920)

if __name__=='__main__':unittest.main()
