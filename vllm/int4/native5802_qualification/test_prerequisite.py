import json
from pathlib import Path
import tempfile
import unittest
from prerequisite import validate,digest
from freeze200k import freeze
from launch200k import check


class RefusalTests(unittest.TestCase):
    def test_missing_receipt_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(FileNotFoundError):freeze(root,root/'absent','a'*64)
            self.assertEqual(list(root.iterdir()),[])

    def test_changed_receipt_and_wrong_scope_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';p.write_text(json.dumps({'passed':True,'backend_scope':'old100K'}))
            with self.assertRaises(ValueError):validate(p,'a'*64)
            with self.assertRaises(ValueError):validate(p,digest(p))

    def test_template_is_not_launchable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'plan.template.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):check(root)
            self.assertFalse((root/'run').exists())


if __name__=='__main__':unittest.main()
