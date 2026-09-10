"""CPU-only opt-in trace ownership and restrictive-mode checks."""
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

HOOK = Path(__file__).resolve().parents[2] / 'fp8/kv_hooks/b70_tp_host_trace.py'


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('trace_owner_fixture', HOOK)
        self.hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hook)
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / 'trace'
        self.environment = patch.dict(os.environ,
            {'B70_TP_HOST_TRACE_DIR': str(self.directory)}, clear=True)
        self.environment.start()

    def tearDown(self):
        if self.hook._fd is not None:
            os.close(self.hook._fd)
        self.environment.stop()
        self.temp.cleanup()

    def test_optin_actual_owner_and_modes(self):
        owner = (os.getuid(), os.getgid())
        with patch.dict(os.environ, B70_TP_HOST_TRACE_UID=str(owner[0]),
                        B70_TP_HOST_TRACE_GID=str(owner[1])), \
                patch.object(self.hook.os, 'chown', wraps=os.chown) as chown, \
                patch.object(self.hook.os, 'fchown', wraps=os.fchown) as fchown:
            self.hook.emit({'event': 'fixture'})
            self.hook.emit({'event': 'second'})
            chown.assert_called_once_with(self.directory, *owner)
            fchown.assert_called_once_with(self.hook._fd, *owner)
        file = next(self.directory.glob('host-*.jsonl'))
        for path, mode in [(self.directory, 0o700), (file, 0o600)]:
            info = path.stat()
            self.assertEqual((info.st_uid, info.st_gid), owner)
            self.assertEqual(stat.S_IMODE(info.st_mode), mode)
        self.assertEqual(len(file.read_text().splitlines()), 2)

    def test_without_owner_environment_no_chown(self):
        with patch.object(self.hook.os, 'chown') as chown, \
                patch.object(self.hook.os, 'fchown') as fchown:
            self.hook.emit({'event': 'fixture'})
            chown.assert_not_called(); fchown.assert_not_called()

    def test_invalid_or_partial_owner_refused_before_creation(self):
        cases = [{'B70_TP_HOST_TRACE_UID': '1000'},
                 {'B70_TP_HOST_TRACE_UID': '-1', 'B70_TP_HOST_TRACE_GID': '1000'},
                 {'B70_TP_HOST_TRACE_UID': '1000', 'B70_TP_HOST_TRACE_GID': '4294967295'}]
        for env in cases:
            with self.subTest(env=env), patch.dict(os.environ, env):
                with self.assertRaises(ValueError):
                    self.hook.emit({'event': 'fixture'})
                self.assertFalse(self.directory.exists())
                self.assertIsNone(self.hook._fd)

    def test_fchown_failure_closes_unpublished_descriptor(self):
        opened = []
        real_open = os.open
        def capture_open(*args, **kwargs):
            fd = real_open(*args, **kwargs); opened.append(fd); return fd
        with patch.dict(os.environ, B70_TP_HOST_TRACE_UID=str(os.getuid()),
                        B70_TP_HOST_TRACE_GID=str(os.getgid())), \
                patch.object(self.hook.os, 'open', side_effect=capture_open), \
                patch.object(self.hook.os, 'fchown', side_effect=PermissionError('fixture')):
            with self.assertRaises(PermissionError):
                self.hook.emit({'event': 'fixture'})
        self.assertIsNone(self.hook._fd)
        self.assertEqual(len(opened), 1)
        with self.assertRaises(OSError):
            os.fstat(opened[0])


if __name__ == '__main__':
    unittest.main()
