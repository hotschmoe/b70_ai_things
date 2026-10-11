import ast
import copy
from pathlib import Path
import tempfile
import unittest
import yaml
import prefill_chunk64_512_control_v2 as c


class Controls(unittest.TestCase):
    def test_only_chunk_and_alias_differ(self):
        a=c.expected_config('chunk64',{'inherited_math_env':{}},18082);b=c.expected_config('chunk512',{'inherited_math_env':{}},18082)
        self.assertEqual(c.profile('chunk64')['cards'],[0]);self.assertEqual(c.profile('chunk512')['cards'],[0])
        b['args'][b['args'].index('--prefill')+1]='64';b['aliases']=a['aliases']
        self.assertEqual(a,b);self.assertEqual(a['args'][a['args'].index('--max-context')+1],'2048')
        self.assertEqual(a['args'][a['args'].index('--ple-row-cache')+1],'65536')
        self.assertNotIn('STRATA_PREFILL_FIRST',a['env']);self.assertNotIn('--mtp',a['args'])

    def test_original_parent_unchanged(self):
        self.assertEqual(c.base.profile('matched-one')['prefill'],64)
        self.assertNotIn('chunk512',c.base._PORTS['profile'].__globals__.get('PROFILES',{}))

    def test_exact_constructor_change_undo(self):
        n=next(n for n in ast.parse(Path(c.base.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='expected_config');old=copy.deepcopy(n)
        new=c.expected_node(n);count=0
        for node in ast.walk(new):
            if isinstance(node,ast.Tuple)and len(node.elts)==2 and getattr(node.elts[0],'value',None)=='--prefill':node.elts[1]=ast.Constant(value=64);count+=1
        self.assertEqual(count,1);self.assertEqual(ast.dump(old),ast.dump(new))

    def test_registry_exact171_plus2_proposal(self):
        v=c.base.parse(c.base.OLD_REGISTRY.read_text());v['models']+=c.base.parse(c.DELTA.read_text())
        self.assertEqual(c.registry_association(yaml.safe_dump(v,sort_keys=False))['current_rows'],173)
        with self.assertRaises(ValueError):c.registry_association(c.base.OLD_REGISTRY.read_text())

    def startup(self,free=1900,mirror=16484):
        return ('strata generate: expert cache 8092 slots, 23.62 GiB of VRAM\n'
                'stage mirror: layers0-47 missing16484 mirrored'+str(mirror)+' bytes51660083200 global_used51660083200 remaining17059393536\n'
                'strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n'
                'strata serve: '+str(free)+' MiB of VRAM free with everything loaded\n')

    def test_actual_shaped_memory_gate(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'server-engine.log').write_text(self.startup());m={'profile':'chunk512','runtime':{'inherited_math_env':{}},'port':18082}
            proof=c.memory_admission(root,m);self.assertEqual(proof['resolved_prefill_first_default'],0);self.assertFalse(proof['physical_spill_observed'])
            (root/'server-engine.log').write_text(self.startup(free=900))
            with self.assertRaises(ValueError):c.memory_admission(root,m)
            (root/'server-engine.log').write_text(self.startup(mirror=16000))
            with self.assertRaises(ValueError):c.memory_admission(root,m)

    def test_hidden_prefill_first_or_fallback_refused(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'server-engine.log').write_text(self.startup());m={'profile':'chunk64','runtime':{'inherited_math_env':{'STRATA_PREFILL_FIRST':'256'}},'port':18082}
            with self.assertRaises(ValueError):c.memory_admission(root,m)
            m['runtime']['inherited_math_env']={};(root/'server-engine.log').write_text(self.startup()+'trying a 256-token chunk\n')
            with self.assertRaises(ValueError):c.memory_admission(root,m)


class NamespaceExecution(unittest.TestCase):
    def test_all_recursive_global_loads_resolve(self):
        import builtins, dis, types
        import run_prefill_chunk64_512_control_v2 as runner
        def check(code, namespace):
            for instruction in dis.get_instructions(code):
                if instruction.opname in ('LOAD_GLOBAL','LOAD_NAME'):
                    self.assertTrue(instruction.argval in namespace or hasattr(builtins,instruction.argval),
                                    code.co_name+': '+str(instruction.argval))
            for value in code.co_consts:
                if isinstance(value,types.CodeType):check(value,namespace)
        for name in ('prepare','_VALIDATE','metadata_admission_gate','launch','screen','main'):
            fn=c._PORTS[name];check(fn.__code__,fn.__globals__)
        fn=runner.parent_ports()['main'];check(fn.__code__,fn.__globals__)
        fn=c.base.original.owned_stop;check(fn.__code__,fn.__globals__)

    def test_real_prepare_writes_complete_tiny_metadata(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); engine=root/'engine'; engine.mkdir(); out=root/'out'
            model=root/'models';model.mkdir()
            rows=[]
            for n in range(4):
                f=model/('shard'+str(n));f.write_bytes(b'tiny')
                rows.append({'path':'UD-Q4_K_XL/shard'+str(n),'size':4,'sha256':'tiny'})
            # The original path construction is retained, with tiny owned stand-ins.
            (root/'UD-Q4_K_XL').mkdir()
            for n in range(4):(root/'UD-Q4_K_XL'/('shard'+str(n))).write_bytes(b'tiny')
            registry=root/'evals/configs';registry.mkdir(parents=True)
            (registry/'models.yaml').write_text('served_model_id: '+c.profile('chunk512')['alias'])
            launch=c.base.original.read(c.ROOT/'strata/flash-next/native-hc-launch-plan.json')
            runtime={'passed':True,'base_image':c.base.original.BASE_IMAGE,'gpu_libraries_unchanged':True,
                     'packages_path':str(root/'packages'),'packages_sha256':'tiny','inherited_math_env':{},'python_versions':{}}
            build={'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,
                   'image':c.base.original.BASE_IMAGE,'binary_sha256':{str(engine/'build/strata'):'tiny'},'patched_source_sha256':{}}
            def read(path):
                name=Path(path).name
                if name=='runtime.json':return runtime
                if name=='receipt.json':return build
                if name=='intake.json':return {'metadata_complete':True,'RESULT':{'unexpected_inexact_conversions':[], 'files':{}}}
                if name=='model-lock.json':return {'destination':str(root),'files':rows,'repo':'tiny','revision':'tiny'}
                if name=='native-hc-launch-plan.json':return launch
                raise AssertionError('Unexpected fixture read '+str(path))
            ns=c._PORTS
            with patch.dict(ns,REPO=root,PACK=root,INTAKE=root/'intake.json',FILES=(),PYTHON_SOURCES=(),
                            read=read,sha=lambda p:'tiny',combined_generation_gate=lambda p:{'tiny':True},
                            upload_gate=lambda *a:{'tiny':True},baseline_profile_gate=lambda *a:None,
                            segmented_profile_gate=lambda *a:None,original_page_sentinel=lambda *a:{'tiny':True}):
                ns['prepare'](SimpleNamespace(verify_model_shards=True,runtime_receipt=root/'runtime.json',
                    upload_lifecycle=root/'upload.json',oracle_receipt=root/'oracle.json',engine_root=engine,
                    output=out,profile='chunk512',port=18082))
            saved=json.loads((out/'prepared.json').read_text());cfg=json.loads((out/'server-config.json').read_text())
            self.assertTrue(saved['launch_allowed']);self.assertEqual(saved['cards'],[0])
            self.assertEqual(cfg['args'][cfg['args'].index('--prefill')+1],'512')
            self.assertEqual(cfg['aliases'],[c.profile('chunk512')['alias']])

    def test_real_launch_reaches_lease_boundary(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Boundary(Exception):pass
        seen=[]
        def lease(cards):seen.append(cards);raise Boundary('owned CPU fixture boundary')
        with tempfile.TemporaryDirectory() as t,patch.dict(c._PORTS,validate_prepared=lambda p:{'cards':[0]},leased=lease):
            with self.assertRaises(Boundary):c._PORTS['launch'](SimpleNamespace(prepared=Path(t)))
        self.assertEqual(seen,[[0]])

    def test_original_stop_executes_owned_prior_receipt_path(self):
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);saved={'container':'tiny-owned','prepared_sha256':'tiny','removed':True,
                               'clean_exit':True,'engine_clean_exit_proven':True}
            (root/'stop.json').write_text(json.dumps(saved));(root/'launch.json').write_text(json.dumps(saved))
            with patch.dict(c.base.original.__dict__,absent=lambda name:name=='tiny-owned'):
                self.assertEqual(c.base.original.owned_stop(root),saved)

    def test_exact_derived_function_changes(self):
        old_tree=ast.parse(Path(c.base.original.__file__).read_text())
        new_tree=ast.parse(Path(c.__file__).read_text())
        self.assertIn('namespace=dict(vars(base.original));namespace.update(vars(base));',Path(c.__file__).read_text())
        for old in old_tree.body:
            if isinstance(old,ast.FunctionDef)and old.name in ('prepare','validate_prepared','metadata_admission_gate','launch','screen'):
                derived=c.base.derive(old)
                expected=sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='profile'for n in ast.walk(derived))
                self.assertEqual(expected,{'prepare':1,'validate_prepared':1,'metadata_admission_gate':0,'launch':1,'screen':0}[old.name])
        self.assertIs(c._PORTS['_chunk_profile'],c.profile)


if __name__=='__main__':unittest.main()
