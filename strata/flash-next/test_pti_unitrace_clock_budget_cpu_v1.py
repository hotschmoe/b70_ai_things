import hashlib,os,subprocess,tempfile,unittest
from pathlib import Path
import prepare_pti_unitrace_clock_budget_v1 as p
class Controls(unittest.TestCase):
    def test_actual_git_apply_and_reverse_on_fresh_tiny_overlay(self):
        with tempfile.TemporaryDirectory()as root:
            root=Path(root);target=root/p.HEADER;target.parent.mkdir(parents=True)
            original=(p.BASE/p.HEADER).read_bytes();target.write_bytes(original)
            patch=p.ROOT/'strata/flash-next/pti-unitrace-clock-budget-v1.patch'
            for args in [('git','apply','--check',str(patch)),('git','apply',str(patch))]:subprocess.run(args,cwd=root,check=True,capture_output=True)
            self.assertEqual(target.read_bytes(),p.changes()[p.HEADER])
            subprocess.run(['git','apply','--reverse',str(patch)],cwd=root,check=True,capture_output=True)
            self.assertEqual(target.read_bytes(),original)
    def test_exact_patch_and_full_byte_undo(self):
        self.assertEqual((p.ROOT/'strata/flash-next/pti-unitrace-clock-budget-v1.patch').read_text(),p.patch_text())
        raw=(p.BASE/p.HEADER).read_bytes();new=p.changes()[p.HEADER]
        self.assertEqual(new.replace(p.NEW.encode(),p.OLD.encode()),raw)
        self.assertEqual(set(p.changes()),{p.HEADER})
        self.assertEqual(raw.count(b'value.size() > 4096'),new.count(b'value.size() > 4096'))
    def test_query_wait_serializer_and_close_fullbytes_preserved(self):
        old=(p.BASE/p.HEADER).read_bytes();new=p.changes()[p.HEADER]
        self.assertEqual(old.split(p.OLD.encode()),new.split(p.NEW.encode()))
        expected=p.base_binding()
        for name in expected:
            if name!=p.HEADER:self.assertEqual(hashlib.sha256((p.BASE/name).read_bytes()).hexdigest(),expected[name])
    def test_production_exact512_boundary_no_large_allocation(self):
        self.assertEqual(p.admission((512<<20)-4096,4096),512<<20)
        for current,row in [((512<<20)-4096,4097),((512<<20)-1,2),(512<<20,1)]:
            with self.assertRaises(ValueError):p.admission(current,row)
    def test_small_exclusive_writer_refuses_before_write_preserves_prefix(self):
        # Actual bounded CPU file operations; separate cap reference fixture only.
        with tempfile.TemporaryDirectory()as root:
            path=Path(root)/'clock.jsonl';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);used=0
            try:
                for row in (b'{"sequence":1}\n',b'{"sequence":2}\n'):
                    end=p.admission(used,len(row),32);self.assertEqual(os.write(fd,row),len(row));used=end
                before=path.read_bytes()
                with self.assertRaises(ValueError):p.admission(used,len(b'{"sequence":3}\n'),32)
                self.assertEqual(path.read_bytes(),before)
                with self.assertRaises(FileExistsError):os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            finally:os.close(fd)
            self.assertEqual(path.read_bytes().count(b'\n'),2)
    def test_typed_extent_and_separate_fixture_cap(self):
        for args in [(True,1),(0,False),(-1,1),(0,1,0),(0,1,(512<<20)+1)]:
            with self.assertRaises(ValueError):p.admission(*args)
        self.assertIn(b'(512u << 20)',p.changes()[p.HEADER]);self.assertNotIn(b'(32u << 20)',p.changes()[p.HEADER])
if __name__=='__main__':unittest.main()
