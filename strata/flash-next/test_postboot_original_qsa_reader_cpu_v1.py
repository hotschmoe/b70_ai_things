"""Tiny source/receipt controls; no original model or device actions."""
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from postboot_original_identity_reader_v1 import identity_admission
from postboot_original_model_association_v1 import current_stat
from postboot_original_provider_derivation_v1 import binding
HERE=Path(__file__).resolve().parent

class Controls(unittest.TestCase):
    def test_provider_exact_constructor_only_derivation(self):
        value=binding()
        self.assertFalse(value['reader_arithmetic_or_headers_or_geometry_or_read_bounds_changed'])
        self.assertFalse(value['old_current_stat_gate_passed'])
    def fixture(self,root):
        files=[];rows=[];mapping=[];shards=[]
        for i in range(4):
            p=root/('shard%d.gguf'%i);p.write_bytes(b'fixture');shards.append(p)
            stat=current_stat(p);old=stat.copy();old[0]+=1
            digest=hashlib.sha256(b'fixture').hexdigest()
            files.append({'path':'UD-Q4_K_XL/'+p.name,'sha256':digest,'size':7})
            rows.append({'path':str(p),'passed':True,'sha256':digest,'expected_sha256':digest,
                         'bytes':7,'stat_before':old,'stat_after':old})
            mapping.append({'path':str(p),'historical_stat5':old,'current_stat5':stat})
        lock=root/'lock.json';lock.write_text(json.dumps({'revision':'fixture','files':files}))
        identity=root/'identity.json'
        identity.write_text(json.dumps({'passed':True,'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
                         'model_revision':'fixture','started':2,'finished':3,'rows':rows}))
        original={'path':str(identity),'sha256':hashlib.sha256(identity.read_bytes()).hexdigest(),
                  'started':2,'finished':3,'current_stat_verified':True,
                  'complete_four_publisher_hashes_verified':True}
        association={'mapping':mapping,'current_identity_sha256':'a'*64}
        return identity,lock,shards,original,association
    def test_saved_identity_projection_is_labelled_historical(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));path,lock,shards,original,a=args
            before=path.read_bytes()
            result=identity_admission(path,lock,shards,1,model_association=a,original_binding=original)
            self.assertEqual(result['original_binding'],original)
            self.assertFalse(result['old_current_stat_gate_passed'])
            self.assertFalse(result['current_runtime_qualified'])
            self.assertEqual(path.read_bytes(),before)
    def test_nondevice_change_or_borrowed_saved_binding_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,lock,shards,original,a=self.fixture(Path(d))
            for index in range(1,5):
                changed=json.loads(json.dumps(a));changed['mapping'][0]['current_stat5'][index]+=1
                with self.assertRaises(ValueError):
                    identity_admission(path,lock,shards,1,model_association=changed,original_binding=original)
            bad={**original,'sha256':'f'*64}
            with self.assertRaises(ValueError):
                identity_admission(path,lock,shards,1,model_association=a,original_binding=bad)
    def test_provider_keeps_original_row_arithmetic(self):
        from original_tensor_rows_postboot_v1 import BoundOriginalFull48RowsPostboot
        from original_first_gdn_layer_v1 import OriginalTensorRows
        self.assertIs(BoundOriginalFull48RowsPostboot.rows,OriginalTensorRows.rows)
        self.assertIs(BoundOriginalFull48RowsPostboot.shape,OriginalTensorRows.shape)
    def test_all_public_readers_have_explicit_recheck_and_false_scope(self):
        names=('postboot_num10_historical_reader_v1','postboot_p30_historical_reader_v1',
               'postboot_rms37_v7_historical_reader_v1','postboot_rms37_v8_historical_reader_v1',
               'postboot_owned_hc_v3_historical_reader_v1','postboot_owned_qsa_v4_historical_reader_v1')
        for name in names:
            tree=ast.parse((HERE/(name+'.py')).read_text())
            fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finalized_binding')
            calls=[n.func.id for n in ast.walk(fun) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            self.assertIn('recheck',calls)
            returns=next(n for n in ast.walk(fun) if isinstance(n,ast.Return)).value
            fields={key.value:value for key,value in zip(returns.keys,returns.values)}
            self.assertIs(fields['old_current_stat_gate_passed'].value,False)
            self.assertIs(fields['current_runtime_qualified'].value,False)
    def test_historical_work_cannot_launch_new_computation(self):
        for name in ('postboot_owned_hc_work_v1','postboot_owned_qsa_work_v1'):
            tree=ast.parse((HERE/(name+'.py')).read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
            call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
            self.assertEqual(len(call.body),1);self.assertIsInstance(call.body[0],ast.Raise)

if __name__=='__main__':unittest.main()
