import ast
import copy
from pathlib import Path
import tempfile
import unittest
import yaml
import prefill_chunk64_512_control_v1 as c


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


if __name__=='__main__':unittest.main()
