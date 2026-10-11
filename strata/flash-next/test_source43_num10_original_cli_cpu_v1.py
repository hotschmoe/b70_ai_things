import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_source43_num10_original_reference_v1 as c
class Controls(unittest.TestCase):
    def test_owned_F32_narrowing_hash_and_shape_without_values(self):
        x=np.asarray([1.00000001,2.],dtype=np.float64);v=c.array_binding(x);self.assertEqual(v['sha256'],hashlib.sha256(x.astype('<f4').tobytes()).hexdigest());self.assertEqual(v['shape'],[2]);self.assertNotIn('values',v)
        with self.assertRaises(ValueError):c.array_binding([float('nan')])
    def test_compact_primary_and_conditional_target_metrics_are_distinct(self):
        v={'gdn_output_gated':np.zeros(6144,dtype='<f4'),'gdn_block_output':np.zeros(4,dtype='<f4'),'gdn_state_after':np.zeros((2,2,2),dtype='<f4'),'gdn_conv_after':np.zeros((8,3),dtype='<f4')};packet=b'cpu_packet'
        result={'independent':[{'request':1,'lane':'FP64','fields_owned':{'x':np.zeros(4)},'comparisons':{'x':{'numeric_gate_assigned':False}},'packet_exact':{'q':False}}],'conditional':[{'request':1,'lanes':{'native_HC_mixed_to_original_zero_GDN':{'block':np.zeros(4),'gated':np.zeros(6144),'state':{'recurrent':np.zeros((2,2,2)),'conv':np.zeros((3,8))}},'native_gated_to_original_Q81_projection':{'packet':packet,'output':np.zeros(4),'projection':'original Q8_0'}}}]}
        value=c.compact(result,[{'request':1,'values':v,'packets':{'gdn_output_q81':packet}}]);self.assertFalse(value['primary'][0]['packet_exact']['q']);self.assertTrue(value['conditional'][0]['native_gated_to_original_Q81_projection']['packet_bitwise_equal_native']);self.assertTrue(value['conditional'][0]['captured_native_inputs_used_for_local_diagnostic']);self.assertFalse(value['reference_replaced']);self.assertFalse(value['full_model_math_qualified']);json.dumps(value,allow_nan=False)
    def close(self,failure):
        import run_source_upload_oracle_full_v2 as full
        import source43_num10_intake_v3 as intake
        calls=[];lock={'destination':'CPU_NO_MODEL','files':[{'path':'UD-Q4_K_XL/'+str(i)}for i in range(4)]}
        def guard(shards,output,label):
            calls.append(label)
            if failure==label:raise ValueError('actual tiny '+label)
            return {'epoch':110 if label.endswith('before')else 130,'passed':True}
        def scan(lock_path,body,shards,path,boundary):
            calls.append('full4');self.assertEqual(boundary,100);self.assertEqual(len(shards),4);value={'started':120,'finished':125,'passed':failure!='predicate'};Path(path).write_text(json.dumps(value));return value
        with tempfile.TemporaryDirectory()as t,patch.object(c,'read_unique',return_value=lock),patch.object(full,'full_buffered_identity',side_effect=scan),patch.object(intake,'guard',side_effect=guard):
            out=Path(t)
            if failure:
                with self.assertRaises(ValueError):c.closing_identity(out,100)
                self.assertTrue(json.loads((out/'closing-receipt.json').read_text())['errors'])
            else:self.assertTrue(c.closing_identity(out,100)['passed'])
        self.assertEqual(calls,['postCPU-before','full4','postCPU-after'])
    def test_actual_closing_dispatch_success_and_bracket(self):self.close(None)
    def test_page_failure_still_attempts_full4_after_guard_and_retains_failure(self):self.close('postCPU-before')
    def test_full4_predicate_failure_still_writes_closing_receipt(self):self.close('predicate')
    def test_enclosing_failure_retains_new_error_artifact_without_model_construction(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);actor=root/'actor';actor.mkdir();identity=root/'identity';identity.write_text('{}');out=root/'out'
            with patch.object(c,'source_binding',return_value='CPU_SOURCE'),patch.object(c.reference,'enclosed',side_effect=ValueError('actual controlled callback failure')):
                with self.assertRaises(ValueError):c.execute(actor,'a'*64,identity,'b'*64,out)
            report=json.loads((out/'report.json').read_text());self.assertFalse(report['passed']);self.assertFalse(report['actual_reference_computation_attempted']);self.assertIn('callback failure',report['errors'][0])
if __name__=='__main__':unittest.main()
