from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from evals.terminalbench.harbor_pi import qwen_model_definition


REPO = Path(__file__).resolve().parents[3]
RUN_ARM = REPO / "evals" / "terminalbench" / "run_arm.sh"


class PiMetadataTest(unittest.TestCase):
    def test_only_off_and_xhigh_are_supported(self) -> None:
        model = qwen_model_definition("oracle", context_window=4096, max_tokens=1024)
        self.assertEqual(
            model["thinkingLevelMap"],
            {
                "off": "none",
                "minimal": None,
                "low": None,
                "medium": None,
                "high": None,
                "xhigh": "xhigh",
                "max": None,
            },
        )


class PolicyResolutionTest(unittest.TestCase):
    def config(self, thinking: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["THINKING"] = thinking
        return subprocess.run(
            [str(RUN_ARM), "--print-config", "qwen-w8a8-reclaim500"],
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_off_has_no_strict_thinking_cap(self) -> None:
        result = self.config("off")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("thinking=off\n", result.stdout)
        self.assertIn("max_tokens=8192\n", result.stdout)
        self.assertIn("pi_concise_off_prompt.j2\n", result.stdout)
        self.assertIn("thinkcap=\n", result.stdout)
        self.assertNotIn("THINKCAP=4096", result.stdout)

    def test_xhigh_has_recorded_strict_thinking_cap(self) -> None:
        result = self.config("xhigh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("thinking=xhigh\n", result.stdout)
        self.assertIn("max_tokens=16384\n", result.stdout)
        self.assertIn("pi_concise_prompt.j2\n", result.stdout)
        self.assertIn("thinkcap=4096\n", result.stdout)
        self.assertIn("THINKCAP=4096", result.stdout)

    def test_intermediate_level_fails_closed(self) -> None:
        result = self.config("medium")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("THINKING must be off or xhigh", result.stdout)


if __name__ == "__main__":
    unittest.main()
