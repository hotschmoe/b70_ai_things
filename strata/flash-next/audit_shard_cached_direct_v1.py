#!/usr/bin/env python3
"""Preserve differing buffered/prefaulted-direct pages without source mutation."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import time


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--prior-identity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    chunk = 16 << 20
    buffer = mmap.mmap(-1, chunk)
    pattern = b'\xa5' * chunk
    receipt = {'schema': 1, 'CONFIG': 'CPU read-only buffered plus initialized reusable aligned anonymous O_DIRECT destination; no source write, recovery or GPU', 'COMMAND': ' '.join(os.sys.argv), 'source': str(source), 'stat_before': signature(source), 'expected_sha256': a.expected_sha256, 'prior_identity_sha256': hashlib.sha256(a.prior_identity.read_bytes()).hexdigest(), 'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'started_epoch': time.time(), 'finished_epoch': None, 'bytes': 0, 'different_pages': [], 'error': None, 'direct_hash_matches_publisher': False, 'buffered_hash_matches_publisher': False, 'source_qualification_claim': False}
    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='ascii')
    save()
    cached_hash = hashlib.sha256()
    direct_hash = hashlib.sha256()
    fd = None
    try:
        fd = os.open(source, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
        receipt['direct_fd_flags'] = os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC
        with source.open('rb', buffering=0) as cached:
            offset = 0
            while True:
                original_cached = cached.read(chunk)
                if not original_cached:
                    break
                # Fault/write every destination page before each direct read.
                # Untouched destinations can fall back to cached reads on Btrfs.
                buffer[:] = pattern
                size = os.preadv(fd, [buffer], offset)
                if size != len(original_cached):
                    raise ValueError('Buffered/direct extent differs at ' + str(offset))
                original_direct = buffer[:size]
                cached_hash.update(original_cached)
                direct_hash.update(original_direct)
                if original_cached != original_direct:
                    for start in range(0, size, 4096):
                        left = original_cached[start:start + 4096]
                        right = original_direct[start:start + 4096]
                        if left == right:
                            continue
                        absolute = offset + start
                        stem = str(absolute)
                        (out / (stem + '.buffered.bin')).write_bytes(left)
                        (out / (stem + '.prefaulted-direct.bin')).write_bytes(right)
                        differences = [{'page_offset': i, 'absolute_offset': absolute + i, 'buffered': x, 'direct': y, 'xor': x ^ y} for i, (x, y) in enumerate(zip(left, right)) if x != y]
                        receipt['different_pages'].append({'offset': absolute, 'bytes': len(left), 'buffered_sha256': hashlib.sha256(left).hexdigest(), 'direct_sha256': hashlib.sha256(right).hexdigest(), 'differences': differences})
                        save()
                        print('PRESERVED differing page ' + str(absolute), flush=True)
                offset += size
                receipt['bytes'] = offset
                if offset % (8 << 30) == 0:
                    save()
                    print('AUDIT bytes ' + str(offset), flush=True)
        receipt['buffered_sha256'] = cached_hash.hexdigest()
        receipt['direct_sha256'] = direct_hash.hexdigest()
        receipt['direct_hash_matches_publisher'] = receipt['direct_sha256'] == a.expected_sha256
        receipt['buffered_hash_matches_publisher'] = receipt['buffered_sha256'] == a.expected_sha256
    except Exception as error:
        receipt['error'] = str(error)
    finally:
        if fd is not None:
            os.close(fd)
        buffer.close()
        receipt['stat_after'] = signature(source)
        receipt['stat_unchanged'] = receipt['stat_before'] == receipt['stat_after']
        receipt['finished_epoch'] = time.time()
        receipt['VERDICT'] = 'Preserved observations only; source failure and corruption origin remain unqualified. No automatic recovery.'
        save()
    print(json.dumps({k: receipt[k] for k in ['bytes', 'error', 'direct_hash_matches_publisher', 'buffered_hash_matches_publisher', 'stat_unchanged']}), flush=True)
    if receipt['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
