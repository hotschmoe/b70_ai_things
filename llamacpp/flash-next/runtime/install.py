"""Install the six hash-pinned official Intel release assets in the image only."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

manifest = Path('/opt/flashnext-runtime/packages.json')
with tempfile.TemporaryDirectory(prefix='flashnext-neo-') as directory:
    paths = []
    for item in json.loads(manifest.read_text()):
        path = Path(directory) / item['file']
        digest = hashlib.sha256()
        with urllib.request.urlopen(item['url'], timeout=120) as response, path.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            raise RuntimeError('SHA256 mismatch: ' + item['file'])
        paths.append(str(path))
    # Install all verified replacements together. dpkg orders configuration by
    # dependencies without apt re-resolving the local GMM package as a download.
    # Do not remove old IGC until the old NEO dependency has been replaced.
    subprocess.run(['dpkg', '--install', *paths], check=True)
    subprocess.run(['dpkg', '--remove', 'libigc2', 'libigdfcl2'], check=True)
    audit = subprocess.run(['dpkg', '--audit'], check=True, capture_output=True, text=True)
    if audit.stdout.strip() or audit.stderr.strip():
        raise RuntimeError('Package audit failed:\n' + audit.stdout + audit.stderr)
    subprocess.run(['ldconfig'], check=True)
