"""Small source/owned-metadata controls; no native/model qualification."""
import ast,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source44_num38_fields_v1 as fields
import source44_num38_zero_case_v1 as case

class Controls(unittest.TestCase):
    def test_private_count_port_preserves_source(self):
        before=Path(fields.software.__file__).read_bytes()
        fn=fields.port()
        self.assertEqual(fn.__globals__['field_contract'],fields.field_contract)
        self.assertEqual(Path(fields.software.__file__).read_bytes(),before)
        self.assertEqual(fields.software.collect_frame.__globals__['field_contract'],fields.software.field_contract)

    def fixture(self,root,extra=None):
        old=(Path(fields.__file__).parent/'layer0_numerical_contract_v1.hpp').read_text()
        rows=fields.re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',old)
        text='\n'.join('{"%s","%s",%s}'%r for r in rows)+'\n'
        text+='\n'.join('{"%s","%s",%s}'%r for r in(fields.EXTRA if extra is None else extra))
        path=root/'source'/fields.CONTRACT;path.parent.mkdir(parents=True);path.write_text(text,encoding='ascii')
        plan=root/'plan.json';plan.write_text(json.dumps({'expected_patched_source_sha256':{fields.CONTRACT:fields.sdk.sha(path)}}),encoding='ascii')
        return plan

    def test_exact_contract_and_preserved_prefix(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);plan=self.fixture(root)
            with patch.multiple(fields.sdk,SDK=root,PLAN=plan,PLAN_SHA=fields.sdk.sha(plan)):
                result=fields.field_contract()
            self.assertEqual(len(result),38);self.assertEqual(sum(r['bytes']for r in result[-5:]),25156)
            self.assertEqual([i for i,r in enumerate(result)if r['provenance'].startswith('DERIVED')],[25,29])

    def test_wrong_rms_extent_refused_even_rehashed(self):
        extra=list(fields.EXTRA);extra[2]=(extra[2][0],extra[2][1],196)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);plan=self.fixture(root,extra)
            with patch.multiple(fields.sdk,SDK=root,PLAN=plan,PLAN_SHA=fields.sdk.sha(plan)):
                with self.assertRaises(ValueError):fields.field_contract()

    def test_generation_and_coverage_are_mandatory(self):
        record={'frame':{'numerical_source_generation':44,'gdn_core_output_observed':True},'source_value_fields':36,'derived_value_fields':2}
        with patch.object(fields,'port',return_value=lambda *args:copy.deepcopy(record)):
            self.assertEqual(fields.collect_frame({},Path('.'),'0'*64),record)
            record['frame']['numerical_source_generation']=43
            with self.assertRaises(ValueError):fields.collect_frame({},Path('.'),'0'*64)
            record['frame']['numerical_source_generation']=44;record['frame']['gdn_core_output_observed']=False
            with self.assertRaises(ValueError):fields.collect_frame({},Path('.'),'0'*64)

    def test_exact_fresh_request_roster(self):
        rows=case.requests();self.assertEqual(len(rows),4)
        self.assertTrue(all(r['ids']==[248045]and r['fresh']==1 and r['max_new']==1 and r['pin']is None for r in rows))

    def test_actual_actor_and_reader_namespace_exports(self):
        import run_source44_num38_zero_v1 as actor
        import validate_source44_num38_zero_v1 as reader
        self.assertTrue(callable(actor.derived()['run_phase']))
        self.assertTrue(callable(reader.derived()['finalized_binding']))
        references={n.attr for n in ast.walk(ast.parse(Path(reader.original.__file__).read_bytes()))if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='a'}
        self.assertTrue(all(hasattr(actor,n)for n in references))
        self.assertIs(actor.time,actor.original.time)

    def test_numeric_transport_prefix_preserved(self):
        import source44_num38_numeric_v1 as numeric
        fn,counts,node=numeric.port()
        self.assertEqual(counts,{'prompt_length':7,'PCL_prefix_slice':1})
        slices=[n for n in ast.walk(node)if isinstance(n,ast.Subscript)and isinstance(n.value,ast.Name)and n.value.id=='line'and isinstance(n.slice,ast.Slice)]
        self.assertEqual([ast.unparse(n.slice)for n in slices],['4:'])

if __name__=='__main__':unittest.main()
