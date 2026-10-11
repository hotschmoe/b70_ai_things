"""Own bounded raw38 fixtures; no model/provider or native execution."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source44_num38_original_reference_v1 as ref
import run_source44_num38_original_reference_v1 as cli

def fixture(root):
    directory=root/'layer0';directory.mkdir();contract=ref.fields.field_contract();observed=[]
    for i,f in enumerate(contract):
        path=directory/('field-'+str(i));data=bytes(f['bytes']);path.write_bytes(data)
        observed.append({**f,'path':str(path),'sha256':hashlib.sha256(data).hexdigest()})
    rows=[];pid=123;binding='a'*64
    for request in range(1,5):
        frame={'schema':1,'numerical_source_generation':44,'gdn_core_output_observed':True,'request':request,'pid':pid,'binding_sha256':binding,'position':0,'token':248045,'rows':1,'layer':0,'stage':0,'reused':0,'gen_ids':[248045],'completed_nonce':(pid<<32)|request,'request_replay_marker_verified':True,'producer_mapping_verified':True,'complete_preregistered_layout':True,'graph_key':2,'raw_fused_hidden_observed':False,'full_model_math_qualified':False,'fields':contract}
        path=directory/('frame-'+str(request)+'.json');path.write_text(json.dumps(frame),encoding='ascii')
        rows.append({'ordinal':request,'raw':{'ids':[248045],'fresh':1,'pin':None},'layer0':{'frame':frame,'metadata':str(path),'metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'observed':observed}})
    return {'source_generation':44,'materialized_mode':1,'NUM38_enabled':True,'QSA3_enabled':False,'capture_root':str(root),'engine_pid':pid,'binding_sha256':binding,'plan_sha256':'b'*64,'requests':rows},contract

class Controls(unittest.TestCase):
    def test_all38_fields_exact_five_extents_and_owner(self):
        with tempfile.TemporaryDirectory()as tmp:
            descriptor,contract=fixture(Path(tmp));targets=ref.load_targets(descriptor,descriptor['requests'])
            self.assertEqual(len(targets),4);self.assertEqual(set(targets[0]['values']),set(ref.SHAPES));self.assertEqual(len(targets[0]['raw38']),38)
            self.assertEqual({n:targets[0]['values'][n].shape for n in ref.core.NAMES},{n:ref.SHAPES[n]for n in ref.core.NAMES})
            descriptor['source_generation']=43
            with self.assertRaises(ValueError):ref.load_targets(descriptor,descriptor['requests'])

    def test_old_quota_changed_extra_extent_and_owner_refused(self):
        with tempfile.TemporaryDirectory()as tmp:
            descriptor,contract=fixture(Path(tmp));row=descriptor['requests'][0];frame=row['layer0']['frame'];path=Path(row['layer0']['metadata'])
            def save():path.write_text(json.dumps(frame),encoding='ascii');row['layer0']['metadata_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            frame['fields']=contract[:33];save()
            with self.assertRaises(ValueError):ref.load_targets(descriptor,descriptor['requests'])
            frame['fields']=list(contract);frame['fields'][-1]={**contract[-1],'bytes':8};save()
            with self.assertRaises(ValueError):ref.load_targets(descriptor,descriptor['requests'])
            frame['fields']=contract;frame['pid']=999;save()
            with self.assertRaises(ValueError):ref.load_targets(descriptor,descriptor['requests'])

    def test_driver_callback_is_actual44_reader2(self):
        import inspect
        text=inspect.getsource(cli.execute)
        self.assertIn('from validate_source44_num38_zero_v2 import captured_binding',text)
        self.assertIn('reference.enclosed(',text);self.assertIn('closing_identity(output,boundary)',text)

    def test_extra_raw_sha_and_frame_generation_refused(self):
        with tempfile.TemporaryDirectory()as tmp:
            descriptor,contract=fixture(Path(tmp));row=descriptor['requests'][0]
            row['layer0']['frame']['numerical_source_generation']=43
            with self.assertRaises(ValueError):ref.load_targets(descriptor,descriptor['requests'])
            row['layer0']['frame']['numerical_source_generation']=44
            field=next(f for f in row['layer0']['observed']if f['name']=='gdn_rsqrt128')
            Path(field['path']).write_bytes(b'\x00\x00\x80\x3f')
            with self.assertRaises(ValueError):ref.load_targets(descriptor,descriptor['requests'])

    def test_initial_admission_failure_attempts_both_closings(self):
        calls=[]
        def admission(root):calls.append('capture');raise ValueError('actual admission failed')
        def close(boundary):calls.append('publisher');raise ValueError('fixture publisher closed failed')
        with patch.dict(ref._PORTS,source_binding=lambda:'source'):
            with self.assertRaises(ValueError):ref.enclosed(Path('/own-fixture'),lambda:self.fail('model must not run'),admission,close)
        self.assertEqual(calls,['capture','capture','publisher'])

if __name__=='__main__':unittest.main()
