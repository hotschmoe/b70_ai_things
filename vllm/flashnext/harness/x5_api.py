"""Opt-in overlay entrypoint; set child PYTHONPATH before vLLM spawn."""
import os
import runpy
from pathlib import Path

def bootstrap():
    here = str(Path(__file__).resolve().parent)
    overlay = here + "/x5_overlay"
    os.environ["PYTHONPATH"] = overlay + ":" + here + ":" + os.environ.get("PYTHONPATH", "")
    runpy.run_path(overlay + "/sitecustomize.py")


if __name__ == "__main__":
    bootstrap()
    from vllm.entrypoints.cli.main import main

    main()
