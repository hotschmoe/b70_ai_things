import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import hashlib
import unittest
from unittest import mock
import native_qsa40_split_contract_v1 as s
import qualify_native_qsa40_split_v1 as q
import validate_native_qsa40_split_v1 as r


class Controls(unittest.TestCase):
    def test_duplicated_lease_fd_refused(self):
        lock=SimpleNamespace(st_dev=1,st_ino=2)
        with mock.patch.object(s.os,'stat',return_value=lock),mock.patch.object(s.os,'listdir',return_value=['57']),mock.patch.object(s.os,'fstat',return_value=lock):
            with self.assertRaises(ValueError):s.no_leases()

    def test_unrelated_fd_allowed(self):
        with mock.patch.object(s.os,'stat',return_value=SimpleNamespace(st_dev=1,st_ino=2)),mock.patch.object(s.os,'listdir',return_value=['57']),mock.patch.object(s.os,'fstat',return_value=SimpleNamespace(st_dev=1,st_ino=3)):
            s.no_leases()

    def test_cpu_refuses_before_original_reads_if_leased(self):
        with mock.patch.object(s,'no_leases',side_effect=ValueError('lease')),mock.patch.object(s,'hardware_binding')as binding:
            with self.assertRaises(ValueError):q.cpu_phase(Path('/missing'))
            binding.assert_not_called()

    def test_hardware_keeps_original_owner_health_full4(self):
        original=next(n for n in ast.parse(Path(q.old.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
        new=q.hardware_node();a=next(n for n in original.body if isinstance(n,ast.Try));b=next(n for n in new.body if isinstance(n,ast.Try))
        self.assertEqual(ast.dump(a.finalbody[0]),ast.dump(b.finalbody[0]))
        self.assertEqual(ast.dump(a.finalbody[1]),ast.dump(b.finalbody[1]))
        self.assertEqual([ast.dump(n)for n in a.body[:5]],[ast.dump(n)for n in b.body[:5]])
        self.assertEqual([ast.dump(n)for n in a.finalbody[2].body[:-1]],[ast.dump(n)for n in b.finalbody[2].body[:-1]])
        text=ast.unparse(b)
        self.assertNotIn('comparisons(',text);self.assertNotIn('verify(',text);self.assertNotIn('join(plan',text)
        self.assertNotIn('finalized_binding(out)',ast.unparse(new))

    def test_reader_undo_is_exact_original_gate_body(self):
        original=next(n for n in ast.parse(Path(r.previous.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='finalized_binding')
        new=r.route(original,'cpu-candidate-report.json')
        for n in ast.walk(new):
            if isinstance(n,ast.Constant)and n.value=='qualify_native_qsa40_split_v1.py':n.value='qualify_native_qsa40_v9.py'
            if isinstance(n,ast.Constant)and n.value=='cpu-candidate-report.json':n.value='report.json'
            if isinstance(n,ast.Compare)and ast.unparse(n.left)=="report['schema']"and isinstance(n.comparators[0],ast.Constant):n.comparators[0].value=1
        self.assertEqual(ast.dump(original),ast.dump(new))

    def test_extra_reader_path_replacement_rejected(self):
        node=next(n for n in ast.parse(Path(r.previous.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='finalized_binding')
        node.body.append(ast.Expr(value=ast.Constant(value='report.json')))
        with self.assertRaises(ValueError):r.route(node,'cpu-candidate-report.json')

    def test_pending_hardware_is_never_public_qualification(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'report.json').write_text(json.dumps({'parent_generation':'qsa-split-v1','numerical_admission_pending':True}))
            with mock.patch.object(s,'no_leases'),mock.patch.object(r,'legacy_binding')as original:
                with self.assertRaises(ValueError):r.finalized_binding(root,root.parent/'unused-close')
                original.assert_not_called()

    def test_cpu_failure_does_not_rewrite_hardware(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);hardware={'errors':[],'passed':False,'numerical_admission_pending':True};(root/'hardware-report.json').write_text(json.dumps(hardware));before=(root/'hardware-report.json').read_bytes()
            with mock.patch.object(s,'no_leases'),mock.patch.object(s,'hardware_binding',return_value=(hardware,{})),mock.patch.object(q.d,'manifest',side_effect=ValueError('original proof changed')),mock.patch.object(r,'closing_hash',return_value={'scope':'synthetic failure-closing hash'})as closing,mock.patch.object(s,'closure'):
                self.assertEqual(q.cpu_phase(root),1)
                closing.assert_called_once()
            final=json.loads((root/'report.json').read_text());self.assertFalse(final['passed']);self.assertTrue(final['numerical_admission_pending']);self.assertIn('original proof changed',final['errors'][0]);self.assertEqual((root/'hardware-report.json').read_bytes(),before)

    def test_public_failure_after_original_reads_still_closes(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t)/'original';root.mkdir();(root/'hardware-report.json').write_text('{}')
            report={'parent_generation':'qsa-split-v1','numerical_admission_pending':False,'hardware_report_sha256':r.d.sha(root/'hardware-report.json')};(root/'report.json').write_text(json.dumps(report))
            output=Path(t)/'closing'
            with mock.patch.object(s,'no_leases'),mock.patch.object(s,'hardware_binding',return_value=({},{})),mock.patch.object(s,'cpu_closing_binding'),mock.patch.object(r,'legacy_binding',side_effect=ValueError('after GGUF read')),mock.patch.object(r,'closing_hash',return_value={'fresh':'synthetic'})as closing:
                with self.assertRaisesRegex(ValueError,'after GGUF read'):r.finalized_binding(root,output)
                closing.assert_called_once()
            self.assertIn('after GGUF read',json.loads((output/'admission-failure.json').read_text())['original_error'])

    def test_original_and_closing_errors_both_preserved(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t)/'original';root.mkdir();(root/'hardware-report.json').write_text('{}');(root/'report.json').write_text(json.dumps({'parent_generation':'qsa-split-v1','numerical_admission_pending':False,'hardware_report_sha256':r.d.sha(root/'hardware-report.json')}))
            out=Path(t)/'closing'
            with mock.patch.object(s,'no_leases'),mock.patch.object(s,'hardware_binding',return_value=({},{})),mock.patch.object(s,'cpu_closing_binding'),mock.patch.object(r,'legacy_binding',side_effect=ValueError('original failed')),mock.patch.object(r,'closing_hash',side_effect=ValueError('close failed')):
                with self.assertRaisesRegex(ValueError,'original failed'):r.finalized_binding(root,out)
            proof=json.loads((out/'admission-failure.json').read_text());self.assertIn('original failed',proof['original_error']);self.assertIn('close failed',proof['closing_error'])

    def closing_fixture(self,base):
        root=base/'result';root.mkdir();directory=root/'cpu-closing-source';directory.mkdir();here=base/'source';here.mkdir();files=[];rows=[]
        for i in range(4):
            rel='UD-Q4_K_XL/shard'+str(i);path=base/'models'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bytes([i]));digest=r.d.sha(path)
            files.append({'path':rel,'sha256':digest,'size':1});rows.append({'path':str(path),'passed':True,'bytes':1,'sha256':digest,'expected_sha256':digest,'stat_before':s.d.c.stat_signature(path),'stat_after':s.d.c.stat_signature(path)})
        lock={'files':files,'destination':'models','revision':'synthetic'};(here/'model-lock.json').write_text(json.dumps(lock))
        proof={'schema':1,'passed':True,'lock_sha256':r.d.sha(here/'model-lock.json'),'model_revision':'synthetic','rows':rows,'after_terminal_and_post_health_epoch':4,'started':5,'finished':6}
        (directory/'full4.json').write_text(json.dumps(proof));closing={'output':str(directory),'boundary_epoch':4,'full4':proof,'full4_sha256':r.d.sha(directory/'full4.json'),'GPU_touched':False,'source_stat_only_reused':False};known=[]
        for offset in(0,4096):known.append((offset,hashlib.sha256(bytes(4096)).hexdigest()))
        for label,key,epoch in(('before','before_pages',4.5),('after','after_pages',6.5)):
            g={'passed':True,'epoch':epoch,'path':rows[2]['path'],'stat_before':rows[2]['stat_before'],'stat_after':rows[2]['stat_after'],'rows':[]}
            for offset,digest in known:
                p=directory/(label+'-buffered-page-'+str(offset)+'.raw');p.write_bytes(bytes(4096));g['rows'].append({'offset':offset,'expected_sha256':digest,'sha256':digest,'passed':True,'bytes':4096,'preserved_path':str(p)})
            closing[key]=g;(directory/(label+'-source-pages.json')).write_text(json.dumps(g))
        report={'CPU_closing_source':closing,'cpu_started_epoch':2,'numerical_admission_finished_epoch':3,'CPU_read_phase_finished_epoch':4,'cpu_finished_epoch':7,'finished_epoch':8,'cpu_identity':{'pid':2,'start_ticks':2}}
        hardware={'finished_epoch':1,'producer_identity':{'pid':1,'start_ticks':1}}
        return root,here,known,report,hardware

    def test_actual_saved_producer_closing_metadata_and_mutations(self):
        import source_page_watchdog_v3 as pages
        with tempfile.TemporaryDirectory()as t:
            base=Path(t);root,here,known,report,hardware=self.closing_fixture(base)
            with mock.patch.object(s.d,'HERE',here),mock.patch.object(s.d,'ROOT',base),mock.patch.object(pages,'KNOWN_PAGES',known):
                self.assertTrue(s.cpu_closing_binding(root,report,hardware))
                bad=copy.deepcopy(report);bad['numerical_admission_finished_epoch']=10
                with self.assertRaises(ValueError):s.cpu_closing_binding(root,bad,hardware)
                bad=copy.deepcopy(report);bad['CPU_closing_source']['full4_sha256']='0'*64
                with self.assertRaises(ValueError):s.cpu_closing_binding(root,bad,hardware)
                raw=root/'cpu-closing-source/before-buffered-page-0.raw';raw.write_bytes(b'x'*4096)
                with self.assertRaises(ValueError):s.cpu_closing_binding(root,report,hardware)


if __name__=='__main__':unittest.main()
