import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_arm


class ArmTest(unittest.TestCase):
    def exercise(self, job_rc):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'arm'; plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(out=str(out), server=['mock-server'], jobs=[
                dict(name='01', command=['mock-probe'], timeout=1),
                dict(name='02', command=['mock-probe'], timeout=1)])))
            class Server:
                def __init__(self, *args, **kwargs):
                    out.mkdir(); (out / 'jobs').mkdir(); (out / 'READY').touch()
                def poll(self):
                    return None
                def wait(self):
                    assert (out / 'STOP').exists()
                    return 0
            def tick(_):
                for job in (out / 'jobs').glob('*.json'):
                    job.with_suffix('.done').write_text(str(job_rc))
            with patch.object(run_arm.subprocess, 'Popen', Server), patch.object(run_arm.time, 'sleep', tick), patch('sys.argv', ['run_arm', str(plan)]):
                if job_rc:
                    with self.assertRaisesRegex(RuntimeError, 'failed job'):
                        run_arm.main()
                    self.assertFalse((out / 'jobs/02.json').exists())
                    self.assertFalse((out / 'WORKLOADS_PASSED').exists())
                else:
                    run_arm.main()
                    self.assertTrue((out / 'WORKLOADS_PASSED').exists())
            self.assertEqual(plan.with_suffix('.lifecycle-rc').read_text(), '0\n')

    def test_failed_probe_stops_before_next_job_and_waits_for_teardown(self):
        self.exercise(1)

    def test_success_waits_for_teardown(self):
        self.exercise(0)


if __name__ == '__main__':
    unittest.main()
