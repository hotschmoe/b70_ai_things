import json
from pathlib import Path
import tempfile
import unittest
from kv_campaign_calibrate import merge


class ScaleMergeTest(unittest.TestCase):
    def inputs(self, root):
        files = []
        for rank in (0, 1):
            p = root / f'rank{rank}.json'
            p.write_text(json.dumps({f'layer{i}': {'rank': rank, 'n': 2, 'tokens': 128, 'q_amax': 448., 'k_amax': 224. * (rank + 1), 'v_amax': 0.} for i in range(17)}))
            files.append(p)
        return files

    def test_uses_cross_rank_max_and_zero_floor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            artifact = merge(self.inputs(root), root / 'out.json', 1.1)
            self.assertEqual(artifact['layers']['layer0']['k_scale'], 1.1)
            self.assertEqual(artifact['layers']['layer0']['v_scale'], 1e-6)

    def test_missing_rank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                merge(self.inputs(root)[:1], root / 'out.json', 1.)

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); files = self.inputs(root)
            data = json.loads(files[0].read_text()); data['layer0']['k_amax'] = float('nan')
            files[0].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                merge(files, root / 'out.json', 1.)

    def test_missing_layer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); files = self.inputs(root)
            data = json.loads(files[0].read_text()); del data['layer0']
            files[0].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                merge(files, root / 'out.json', 1.)

    def test_target_only_int4_has_sixteen_layers_and_new_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); files = self.inputs(root)
            for path in files:
                data = json.loads(path.read_text()); del data['layer16']
                path.write_text(json.dumps(data))
            result = merge(files, root / 'out.json', 1.1, 16,
                           'qwen3.8-27b/int4-autoround-gptq-relabel-r212')
            self.assertEqual(len(result['layers']), 16)
            self.assertEqual(result['schema'], 'b70.qwen38-kv-scales.v2')
            with self.assertRaises(ValueError):
                merge(files, root / 'wrong.json', 1.1, 17)


if __name__ == '__main__':
    unittest.main()
