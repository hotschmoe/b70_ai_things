"""Exact regular-file artifact closure for both new owned QSA phases."""
from pathlib import Path
import hashlib
from owned_layer3_qsa_contract_v1 import require


def artifacts(root, report):
    root = Path(root).resolve(); actual = {}
    for path in root.rglob('*'):
        require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Confined nonsymlink QSA evidence required')
        if path.is_file() and path != root/'report.json':
            actual[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == report['artifact_sha256'], 'Exact complete QSA original evidence tree changed')
    return actual
