import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_source', HERE / 'prepare_cpu_reference_source_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class SourceTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'checkout'
        source.mkdir()
        (source / 'CMakeLists.txt').write_text('project(tiny C)\n')
        (source / 'new-untracked.cpp').write_text('int x = 1;\n')
        return source

    def test_payload_and_build_excluded_untracked_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.fixture(Path(t))
            for d in ['models', 'build-cpu', '.git']:
                (s / d).mkdir(); (s / d / 'secret').write_text('never snapshot')
            (s / 'payload.gguf').write_text('never hash')
            self.assertEqual([x['path'] for x in m.manifest(s)['files']], ['CMakeLists.txt', 'new-untracked.cpp'])

    def test_contents_and_modes_change_binding(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.fixture(Path(t)); before = m.manifest(s)
            (s / 'new-untracked.cpp').write_text('int x = 2;\n')
            self.assertNotEqual(before, m.manifest(s))
            before = m.manifest(s); (s / 'new-untracked.cpp').chmod(0o755)
            self.assertNotEqual(before, m.manifest(s))

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.fixture(Path(t)); (s / 'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError): m.manifest(s)

    def test_success_snapshot_and_false_scope(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.fixture(Path(t)); out = Path(t) / 'owned'
            plan = {'current_worktree_manifest_sha256': m.manifest(s)['manifest_sha256'], 'expected_head': 'known'}
            meta = {'head': 'known\n', 'dirty_patch': 'diff test\n', 'status': '?? new-untracked.cpp\n'}
            with patch.object(m, 'git_metadata', return_value=meta.copy()): r = m.prepare(s, out, plan)
            self.assertEqual(m.manifest(out / 'source'), m.manifest(s))
            self.assertTrue(r['snapshot_verified']); self.assertFalse(r['actual_build']); self.assertFalse(r['quality_qualified'])
            with self.assertRaises(ValueError): m.prepare(s, out, plan)

    def test_stale_source_and_head_fail_before_output(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.fixture(Path(t)); out = Path(t) / 'owned'
            with self.assertRaises(ValueError): m.prepare(s, out, {'current_worktree_manifest_sha256':'wrong'})
            self.assertFalse(out.exists())
            plan = {'current_worktree_manifest_sha256':m.manifest(s)['manifest_sha256'], 'expected_head':'known'}
            with patch.object(m, 'git_metadata', return_value={'head':'other'}):
                with self.assertRaises(ValueError): m.prepare(s, out, plan)
            self.assertFalse(out.exists())

    def test_plan_flags_have_real_declarations_and_targets(self):
        plan = json.loads((HERE / 'cpu-reference-source-build-plan-v1.json').read_text())
        source = Path(plan['source'])
        declarations = (source / 'CMakeLists.txt').read_text() + (source / 'ggml/CMakeLists.txt').read_text()
        for flag in plan['configure_argv'][5:]:
            if flag.startswith('-DGGML_') or flag.startswith('-DLLAMA_'):
                self.assertIn(flag.split('=',1)[0][2:], declarations)
        self.assertIn('-DGGML_SYCL=OFF', plan['configure_argv'])
        self.assertIn('-DGGML_BACKEND_DL=OFF', plan['configure_argv'])
        self.assertIn('-DGGML_CPU_REPACK=OFF', plan['configure_argv'])
        self.assertEqual(plan['container_controls']['devices'], [])
        self.assertEqual(plan['container_controls']['network'], 'none')

if __name__ == '__main__': unittest.main()
