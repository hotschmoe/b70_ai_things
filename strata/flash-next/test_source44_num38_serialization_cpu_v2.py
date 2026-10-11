"""Actual enclosed/serializer paths with owned tiny closing fixtures only."""
import hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import source44_num38_original_reference_v2 as ref
import run_source44_num38_original_reference_v2 as cli
from original_first_gdn_layer_v1 import FirstGdnLayer

class Controls(unittest.TestCase):
    def test_enclosed_capture_keywords_reach_actual_serialization(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);here=root/'strata/flash-next';here.mkdir(parents=True);destination=root/'own'
            files=[]
            for i in range(4):
                name='UD-Q4_K_XL/'+str(i);path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bytes([i]));files.append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':1})
            lock={'destination':'own','revision':'owned-fixture','files':files};(here/'model-lock.json').write_text(json.dumps(lock),encoding='ascii')
            descriptor={'capture_root':str(root/'capture'),'source_generation':44,'materialized_mode':1,'engine_pid':123,'plan_sha256':'b'*64,'binding_sha256':'a'*64,'requests':[],'public_actor_admission':{'report_sha256':'c'*64,'source_plan_sha256':'a'*64}}
            calls=[]
            def capture(_):calls.append('capture');return descriptor
            def close(boundary):
                calls.append('publisher');start=time.time();self.assertGreaterEqual(start,boundary)
                return {'started':start,'finished':time.time(),'lock_sha256':hashlib.sha256((here/'model-lock.json').read_bytes()).hexdigest(),'model_revision':'owned-fixture','passed':True,'rows':[{'path':str(destination/f['path']),'passed':True,'sha256':f['sha256'],'expected_sha256':f['sha256'],'bytes':1,'stat_before':ref.software.sig((destination/f['path']).stat()),'stat_after':ref.software.sig((destination/f['path']).stat())}for f in files]}
            def evaluate(*_):return {'independent':[],'conditional':[],'computation_terminal_epoch':time.time()}
            with patch.dict(ref._PORTS,HERE=here,source_binding=lambda:'source',load_targets=lambda *_:[],evaluate=evaluate):
                result=ref.enclosed(root,lambda:object.__new__(FirstGdnLayer),capture,close)
            self.assertEqual(calls,['capture','capture','publisher']);self.assertNotIn('current43_capture_before',result)
            body={'diagnostics':cli.compact(result,[]),'current44_capture_before':cli.capture_binding(result['current44_capture_before']),'current44_capture_after':cli.capture_binding(result['current44_capture_after'])}
            self.assertEqual(json.loads(json.dumps(body,allow_nan=False))['current44_capture_before']['source_generation'],44)

if __name__=='__main__':unittest.main()
