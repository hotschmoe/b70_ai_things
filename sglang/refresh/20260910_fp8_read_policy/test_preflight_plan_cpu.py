"""CPU guard checks; never invoke a GPU helper."""
import json
from pathlib import Path
import subprocess
import sys

script = Path(__file__).with_name("preflight_plan.py")
plan = json.loads(subprocess.check_output([sys.executable, str(script)]))
assert plan["command"][0] == "bin/gpu-run"
assert "--card" not in plan["command"]
assert plan["command"][-1] == "--run"
assert plan["image"].startswith("sha256:14ee7d")
root = Path(plan["output"])
assert not root.exists(), "test must precede actual launch"
# No inherited lease FDs: helper must fail before mkdir or any GPU command.
failed = subprocess.run([sys.executable, str(script), "--run"],
                        capture_output=True, text=True, close_fds=True)
assert failed.returncode != 0
assert "FileNotFoundError" in failed.stderr and "/fd/8" in failed.stderr
assert not root.exists()
print("PASS plan uses pair lease and unleased execution fails before GPU work")
