#!/usr/bin/env python3
"""Fetch and hash the recipe's public binary closure and pinned source archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest_path = args.source / 'repro/qwen38-27b-autoround-int4-b70/publication-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = [dict(a, local='release/' + a['name']) for a in manifest['release']['assets']]
    for source in manifest['source_repositories']:
        repo = source['url'].removeprefix('https://github.com/').removesuffix('.git')
        inputs.append({'url': 'https://codeload.github.com/' + repo + '/tar.gz/' + source['commit'],
                       'local': 'source/' + repo.split('/')[-1] + '-' + source['commit'] + '.tar.gz',
                       'commit': source['commit'], 'kind': 'source-archive'})
    def fetch(item):
        path = args.out / item['local']; path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(item['url'], headers={'User-Agent': 'Mozilla/5.0'})
        h = hashlib.sha256(); size = 0
        with urllib.request.urlopen(req, timeout=120) as response, path.open('wb') as out:
            while chunk := response.read(2**20):
                out.write(chunk); h.update(chunk); size += len(chunk)
        if item.get('sha256') and h.hexdigest() != item['sha256']:
            raise ValueError('hash mismatch: ' + item['local'])
        if item.get('size') and size != item['size']:
            raise ValueError('size mismatch: ' + item['local'])
        return dict(item, actual_sha256=h.hexdigest(), actual_bytes=size)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, inputs))
    record = {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'artifacts': results}
    (args.out / 'acquisition.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Verified %d artifacts (%d bytes)' % (len(results), sum(r['actual_bytes'] for r in results)))


if __name__ == '__main__':
    main()
