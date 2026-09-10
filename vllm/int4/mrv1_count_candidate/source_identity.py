"""CPU-only package/source identity, with no Torch import or device access."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

root = Path('/opt/venv/lib/python3.12/site-packages/vllm')
files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(root.rglob('*.py'))}
packages = sorted((d.metadata['Name'], d.version) for d in metadata.distributions())
print(json.dumps(dict(packages=packages, python_sources=files), indent=2, sort_keys=True))
