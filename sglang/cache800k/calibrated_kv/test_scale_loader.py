import copy
import hashlib
import json
import pathlib
import tempfile
import types
import unittest

from scale_loader import install_validated_plan,verify_model_files,try_load


class Layer:
    def __init__(self,i):
        self.layer_id=i
        self.k_scale=self.v_scale=self.k_scale_float=self.v_scale_float=None
        self.buffers={}
    def register_buffer(self,name,value,persistent):
        assert not hasattr(self,name)
        self.buffers[name]=(value,persistent)
        setattr(self,name,value)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.layer=Layer(3)
        self.model=types.SimpleNamespace(named_modules=lambda:[('model.layers.3.attn',self.layer)])
        self.plan={'rows':[{'module_path':'model.layers.3.attn','layer_id':3,'k_scale':0.03125,'v_scale':0.0625}]}
    def install(self,**kwargs):
        args=dict(radix_attention_type=Layer,tensor_factory=lambda value:('fp32',value))
        args.update(kwargs)
        return install_validated_plan(self.model,self.plan,**args)
    def test_persistent_buffers_and_matching_float_mirrors(self):
        self.assertEqual(self.install(),1)
        for label in ['k','v']:
            tensor,persistent=self.layer.buffers[label+'_scale']
            self.assertTrue(persistent)
            self.assertEqual(tensor[1],getattr(self.layer,label+'_scale_float'))
    def test_reinstall_rejected(self):
        self.install()
        with self.assertRaises(ValueError):self.install()
    def test_existing_scale_or_inventory_change_rejected(self):
        self.layer.k_scale_float=1.0
        with self.assertRaises(ValueError):self.install()
        self.layer.k_scale_float=None;self.layer.layer_id=7
        with self.assertRaises(ValueError):self.install()
    def test_allocation_failure_does_not_mutate_layer(self):
        calls=[]
        def factory(value):
            calls.append(value)
            if len(calls)==2:raise MemoryError('fixture allocation failure')
            return value
        with self.assertRaises(MemoryError):self.install(tensor_factory=factory)
        self.assertIsNone(self.layer.k_scale)
        self.assertIsNone(self.layer.v_scale)
        self.assertEqual(self.layer.buffers,{})
    def test_runtime_digest_gate_and_generic_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path=pathlib.Path(temp)/'scales.json'
            args=dict(model=None,scale_path=path,model_root=temp,kv_dtype='fp8_e4m3',
                      is_draft_worker=False,tp_rank=0,tp_size=1,pp_size=1,dcp_size=1,
                      dp_attention=False,prefill_backend='triton',decode_backend='triton',
                      device='xpu',gpu_id=0,model_dtype=None)
            path.write_text('{"schema":"another-schema"}')
            self.assertFalse(try_load(**args))
            path.write_text('{"schema":"b70.qwen38-kv-scales.v2"}')
            with self.assertRaisesRegex(ValueError,'artifact digest'):
                try_load(**args)

    def test_exact_file_hashes_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp);files={}
            for name,data in [('config.json',b'{}'),('model.safetensors',b'fixture'),('model.safetensors.index.json',json.dumps({'weight_map':{'weight':'model.safetensors'}}).encode())]:
                (root/name).write_bytes(data)
                files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            a={'provenance':{'model_files':files}}
            verify_model_files(a,root)
            truncated=copy.deepcopy(a)
            del truncated['provenance']['model_files']['model.safetensors']
            with self.assertRaises(ValueError):verify_model_files(truncated,root)
            (root/'extra.safetensors').write_bytes(b'unknown')
            with self.assertRaises(ValueError):verify_model_files(a,root)
            (root/'extra.safetensors').unlink()
            (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_model_files(a,root)
            bad=copy.deepcopy(a);bad['provenance']['model_files']['../escape']={}
            with self.assertRaises(ValueError):verify_model_files(bad,root)


if __name__=='__main__':unittest.main()
