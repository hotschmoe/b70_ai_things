#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import summarize


def trial(name, minute, reward=None, error=None):
    verifier = None if reward is None else {"rewards": {"reward": reward}}
    return {
        "task_name": name,
        "trial_name": name,
        "started_at": "2026-08-24T00:00:00+00:00",
        "finished_at": f"2026-08-24T00:{minute:02d}:00+00:00",
        "verifier_result": verifier,
        "exception_info": error,
        "agent_execution": {
            "started_at": "2026-08-24T00:00:00+00:00",
            "finished_at": f"2026-08-24T00:{minute:02d}:00+00:00",
        },
        "agent_result": {"n_input_tokens": 100, "n_cache_tokens": 20, "n_output_tokens": 10},
    }


class AnalyzeTest(unittest.TestCase):
    def test_curve_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            payload = {
                "started_at": "2026-08-24T00:00:00+00:00",
                "finished_at": "2026-08-24T01:00:00+00:00",
                "n_total_trials": 4,
                "trial_results": [
                    trial("a", 10, 1.0),
                    trial("b", 20, 0.0),
                    trial("c", 30, 1.0),
                    trial("d", 40, None, {"exception_type": "InfraError"}),
                ],
            }
            (path / "result.json").write_text(json.dumps(payload))
            row = summarize(path)
            self.assertEqual(row["n_correct"], 2)
            self.assertEqual(row["n_errors"], 1)
            self.assertEqual(row["correct_pct_raw"], 50.0)
            self.assertAlmostEqual(row["correct_pct_model_only"], 200 / 3)
            self.assertEqual(row["time_to_25pct_correct_seconds"], 600.0)
            self.assertEqual(row["time_to_50pct_correct_seconds"], 1800.0)
            self.assertEqual(row["tokens"], {"input": 400, "cache": 80, "output": 40})


if __name__ == "__main__":
    unittest.main()
