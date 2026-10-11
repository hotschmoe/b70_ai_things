import unittest,re,tempfile,subprocess
from pathlib import Path
import prepare_pti_clock_sampled_queries_v1 as p
import pti_bounded_gen4_contract_v1 as c
class Tests(unittest.TestCase):
    def test_query_calls_unchanged(self):
        old=p.original();new=p.changes();pattern=r'ZE_FUNC\(zeDeviceGetGlobalTimestamps\)\([^;]+;'
        self.assertEqual(re.findall(pattern,old[p.COLLECTOR].decode()),re.findall(pattern,new[p.COLLECTOR].decode()))
        self.assertEqual(len(re.findall(pattern,old[p.COLLECTOR].decode())),6)
    def test_normalization_unchanged(self):
        old=p.original()[p.COLLECTOR].decode();new=p.changes()[p.COLLECTOR].decode();start=old.index('  inline void GetHostTime(');end=old.index('  inline void LogCommandCompleted(',start)
        ns=new.index('  inline void GetHostTime(');ne=new.index('  inline void LogCommandCompleted(',ns);self.assertEqual(old[start:end],new[ns:ne])
    def test_full_patch_apply_undo(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            for name,raw in p.original().items():q=root/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
            patch=root/'patch';patch.write_text(p.patch_text());subprocess.run(['git','apply',str(patch)],cwd=root,check=True,capture_output=True)
            for name,raw in p.changes().items():self.assertEqual((root/name).read_bytes(),raw)
            subprocess.run(['git','apply','--reverse',str(patch)],cwd=root,check=True,capture_output=True)
            for name,raw in p.original().items():self.assertEqual((root/name).read_bytes(),raw)
    def test_sampling(self):
        self.assertTrue(all(p.selected(1,n)for n in range(1,1026)))
        self.assertEqual([n for n in range(1,1026)if p.selected(4,n)],[1,513,1025])
        with self.assertRaises(ValueError):p.selected(True,1)
    def test_counter_closure(self):
        rows=[dict(query_site=s,sample_every=1 if s<=2 else 512,original_query_calls=1025,selected_brackets=1025 if s<=2 else 3,completed_brackets=1025 if s<=2 else 3,suppressed_brackets=0 if s<=2 else 1022)for s in range(1,7)]
        self.assertFalse(c.sampling_summary(rows)['clock_alignment_qualified']);rows[3]['completed_brackets']=2
        with self.assertRaises(ValueError):c.sampling_summary(rows)
    def test_gen4_exact_ids_and_repeat(self):
        inputs={'short':list(range(54)),'prefill_about512':list(range(524))};recipe={'PTI_enabled':True,'native_env':{},'requests':[dict(ordinal=i+1,ids=inputs[k],fresh=1,pin=None,max_new=64,role=k)for i,k in enumerate(('short','prefill_about512','prefill_about512','short'))]}
        new=c.construct(recipe,inputs);self.assertEqual([r['max_new']for r in new['requests']],[4]*4);self.assertEqual(recipe['requests'][0]['max_new'],64);self.assertFalse(new['warm_process_decode_qualified'])
        recipe['requests'][0]['ids']=[1]*54
        with self.assertRaises(ValueError):c.construct(recipe,inputs)
    def test_default_off_source(self):
        h=p.changes()[p.HEADER].decode();self.assertIn('flag==nullptr',h);self.assertIn('completed!=selected',h);self.assertIn('(512u << 20)',h);self.assertIn('value.size() > 4096',h)
    def fixture_rows(self):
        data=[dict(kind='host_clock_origin'),dict(kind='device_origin'),dict(kind='global_timestamp_query_bracket',query_site=1),dict(kind='host_clock_bridge',site=101),dict(kind='instance_normalization',instance_id=7)]
        for site in range(1,7):
            seen=1 if site==1 else 0
            data.append(dict(kind='query_sampling_summary',query_site=site,sample_every=1 if site<=2 else 512,original_query_calls=seen,selected_brackets=seen,completed_brackets=seen,suppressed_brackets=0))
        data.append(dict(kind='host_clock_bridge',site=2))
        return [dict(r,pid=9,sequence=i+1)for i,r in enumerate(data)]
    def test_actual_row_counts(self):
        rows=self.fixture_rows();self.assertIn('Emit("host_clock_origin"', (p.BASE/'tools/unitrace/src/unitimer.h').read_text());self.assertEqual(c.audit_sampled_rows(iter(rows))['instance_normalizations'],1)
        rows[3]['site']=102
        with self.assertRaises(ValueError):c.audit_sampled_rows(iter(rows))
    def test_dropped_summary_and_owner(self):
        rows=self.fixture_rows();rows.pop(7)
        for i,r in enumerate(rows):r['sequence']=i+1
        with self.assertRaises(ValueError):c.audit_sampled_rows(iter(rows))
        rows=self.fixture_rows();rows[4]['pid']=10
        with self.assertRaises(ValueError):c.audit_sampled_rows(iter(rows))
if __name__=='__main__':unittest.main()
