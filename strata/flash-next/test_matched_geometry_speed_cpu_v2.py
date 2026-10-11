import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import yaml
import matched_geometry_c140_diagnostic_v2 as m


class Controls(unittest.TestCase):
    def test_geometry_equal_topology_distinct(self):
        a, b = m.profile('matched-one'), m.profile('matched-pair')
        self.assertEqual([a[k] for k in ('context','prefill','ple_rows')], [2048,64,65536])
        self.assertEqual([a[k] for k in ('context','prefill','ple_rows')], [b[k] for k in ('context','prefill','ple_rows')])
        self.assertEqual(a['cards'], [0]); self.assertEqual(b['cards'], [0,1])
        self.assertIn('--layer-split', b['extra']); self.assertNotIn('--layer-split', a['extra'])

    def test_original_profiles_unchanged(self):
        self.assertEqual(m.original.PROFILES['two-card-segmented']['context'],8192)
        self.assertEqual(m.original.PROFILES['two-card-segmented']['prefill'],128)
        self.assertEqual(m.original.PROFILES['two-card-segmented']['ple_rows'],1048576)
        self.assertNotIn('matched-pair',m.original.PROFILES)
        self.assertNotIn('PROFILES',m._PORTS)

    def candidate(self):
        old = m.parse(m.OLD_REGISTRY.read_text())
        old['models'] += m.parse(m.DELTA.read_text())
        return old

    def test_exact_append_association(self):
        r = m.registry_association(yaml.safe_dump(self.candidate(),sort_keys=False))
        self.assertEqual(r['old_rows_preserved'],171)
        self.assertFalse(r['old_runtime_or_math_proof_transferred'])

    def test_old171_refuses_association_gate(self):
        with self.assertRaises(ValueError):m.registry_association(m.OLD_REGISTRY.read_text())

    def test_prior_row_change_reject(self):
        v = self.candidate();v['models'][0]['label']='changed'
        with self.assertRaises(ValueError):m.registry_association(yaml.safe_dump(v,sort_keys=False))

    def test_duplicate_top_models_reject(self):
        with self.assertRaises(ValueError):m.registry_association(m.OLD_REGISTRY.read_text()+'\nmodels: []\n')

    def test_only_declared_function_changes(self):
        class Undo(ast.NodeTransformer):
            def visit_Call(self,n):
                n=self.generic_visit(n)
                if isinstance(n.func,ast.Name) and n.func.id=='profile':
                    return ast.copy_location(ast.Subscript(value=ast.Name(id='PROFILES',ctx=ast.Load()),slice=n.args[0],ctx=ast.Load()),n)
                if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='original' and n.func.attr=='validate_prepared':
                    n.func=ast.Name(id='validate_prepared',ctx=ast.Load())
                return n
        tree=ast.parse(Path(m.original.__file__).read_text())
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in ('prepare','validate_prepared','metadata_admission_gate','launch','screen'):
                self.assertEqual(ast.dump(n,include_attributes=False),ast.dump(Undo().visit(m.derive(n)),include_attributes=False))

    def test_config_full_reconstruction_and_mutation(self):
        common=m.original.read(m.ROOT/'strata/flash-next/native-hc-launch-plan.json')
        p=m.profile('matched-pair');runtime={'inherited_math_env':{}}
        args=list(common['engine_arguments_common'])
        for key,value in (('--max-context','2048'),('--prefill','64'),('--ple-row-cache','65536')):args[args.index(key)+1]=value
        args+=p['extra'];env=dict(common['runtime_environment_common']);env.update(p['env']);env.update(ZE_AFFINITY_MASK='0,1',STRATA_DEBUG='1',STRATA_REQUEST_LINES='1')
        cfg={'exe':'/build/strata','args':args,'cwd':'/src','tokenizer':'/pack/tokenizer','model_name':'hotschmoe-dd','aliases':[p['alias']],
             'artifact_identity_manifest':'/results/artifact-identity.json','host':'127.0.0.1','port':18082,'backend':'sycl','env':env,
             'log':'/results/server-engine.log','parallel':1,'repeat_stop_tokens':0,'reasoning_close_retry':False,'reasoning_budget_tokens':0,
             'engine_silence_s':180,'sampling':{'temperature':0,'seed':1}}
        m.config_gate('matched-pair',cfg,runtime,18082)
        for key,value in (('--max-context','8192'),('--prefill','128'),('--ple-row-cache','1048576')):
            bad=copy.deepcopy(cfg);bad['args'][bad['args'].index(key)+1]=value
            with self.assertRaises(ValueError):m.config_gate('matched-pair',bad,runtime,18082)
        for key,value in (('backend','cuda'),('exe','/FOREIGN'),('cwd','/foreign'),('tokenizer','/other'),('parallel',17),('port',18083),('repeat_stop_tokens',1),('sampling',{'temperature':0,'seed':2})):
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):m.config_gate('matched-pair',bad,runtime,18082)

    def fixture(self,root):
        receipt=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f14-source33-20261009/runtime-c111/receipt.json')
        raw=receipt.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),m.RUNTIME_RECEIPT_SHA)
        (root/'runtime.json').write_bytes(raw);runtime=json.loads(raw)
        cfg=m.expected_config('matched-pair',runtime,18082)
        (root/'server-config.json').write_text(json.dumps(cfg))
        meta={'profile':'matched-pair','cards':[0,1],'alias':m.profile('matched-pair')['alias'],'segmented_stage_mirrors':True,
              'runtime_receipt':str(root/'runtime.json'),'runtime_receipt_sha256':m.RUNTIME_RECEIPT_SHA,'runtime':runtime,'port':18082,
              'config_sha256':m.original.sha(root/'server-config.json')}
        (root/'prepared.json').write_text(json.dumps(meta));return meta,cfg

    def test_receipt_card_and_complete_cfg_metadata_join(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);meta,cfg=self.fixture(root);self.assertEqual(m.predevice_metadata(root)['cards'],[0,1])
            for path,value in (('cards',[False,1]),('cards',[0]),('port',True),('alias','foreign')):
                bad=copy.deepcopy(meta);bad[path]=value;(root/'prepared.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):m.predevice_metadata(root)
            for field,value in (('image','sha256:foreign'),('inherited_math_env',{'STRATA_QFUSE':'1'})):
                bad=copy.deepcopy(meta);bad['runtime'][field]=value;(root/'prepared.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):m.predevice_metadata(root)
            for field,value in (('backend','cuda'),('exe','/foreign'),('parallel',17)):
                badcfg=copy.deepcopy(cfg);badcfg[field]=value;(root/'server-config.json').write_text(json.dumps(badcfg))
                bad=copy.deepcopy(meta);bad['config_sha256']=m.original.sha(root/'server-config.json');(root/'prepared.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):m.predevice_metadata(root)

    def test_changed_runtime_receipt_even_with_rehashed_metadata_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);meta,cfg=self.fixture(root)
            bad=copy.deepcopy(meta);bad['runtime']['image']='sha256:foreign'
            (root/'runtime.json').write_text(json.dumps(bad['runtime']));bad['runtime_receipt_sha256']=m.original.sha(root/'runtime.json')
            (root/'prepared.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):m.predevice_metadata(root)

    def test_explicit_original_one_qualification_recipe(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);meta,cfg=self.fixture(root)
            meta.update(profile='one-card-segmented',cards=[0],alias=m.original.PROFILES['one-card-segmented']['alias'])
            cfg=m.expected_config('matched-one',meta['runtime'],meta['port']);cfg['aliases']=[meta['alias']]
            (root/'server-config.json').write_text(json.dumps(cfg));(root/'prepared.json').write_text(json.dumps(meta))
            self.assertEqual(m.one_card_prerequisite_metadata(root/'qualification.json')['cards'],[0])
            with self.assertRaises(ValueError):m.one_card_prerequisite_metadata(root/'parent-qualification.json')
            cfg['backend']='cuda';(root/'server-config.json').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):m.one_card_prerequisite_metadata(root/'qualification.json')

    def test_exact_declared_change_counts(self):
        tree=ast.parse(Path(m.original.__file__).read_text());counts=[0,0]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('prepare','validate_prepared','metadata_admission_gate','launch','screen'):
                result=m.derive(node)
                counts[0]+=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='profile' for n in ast.walk(result))
                counts[1]+=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='original' and n.func.attr=='validate_prepared' for n in ast.walk(result))
        self.assertEqual(counts,[3,1])

    def test_extra_profile_or_prerequisite_rewrite_refused(self):
        n=ast.parse("def metadata_admission_gate(prepared):\n return PROFILES[prepared['profile']]\n").body[0]
        with self.assertRaises(ValueError):m.derive(n)
        n=ast.parse("def launch(a):\n return validate_prepared(a.one_card_receipt)\n").body[0]
        with self.assertRaises(ValueError):m.derive(n)


if __name__=='__main__':unittest.main()
