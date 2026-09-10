"""Host-only cleanup classification; no Docker calls."""
import unittest
from launch_reference import is_missing_container


class CleanupTests(unittest.TestCase):
    def test_missing_container_case_variants(self):
        self.assertTrue(is_missing_container(1, 'error: no such object: owned-id'))
        self.assertTrue(is_missing_container(1, 'Error: No such container: owned-id'))

    def test_daemon_failure_is_not_absence(self):
        self.assertFalse(is_missing_container(1, 'Cannot connect to Docker daemon'))
        self.assertFalse(is_missing_container(1, 'dial unix /run/docker.sock: no such file or directory'))
        self.assertFalse(is_missing_container(0, ''))
        self.assertFalse(is_missing_container(0, 'No such container'))

if __name__ == '__main__': unittest.main()
